# Copyright (c) Facebook, Inc. and its affiliates.

import os
from copy import deepcopy
from typing import Dict

import torch
import torch.nn.functional as F
from mmf.models.composition import NormalizationLayer
from mmf.models.FashionFAE.base import FashionFAEBaseModel
from mmf.modules.losses import (
    ContrastiveLoss,
    CrossEntropyLoss,
    MSELoss,
    SupervisedContrastiveLoss,
    SoftLabelCrossEntropyLoss,
    ContrastivefsisLoss,
)
from mmf.modules.ot import optimal_transport_dist
from mmf.utils.build import build_image_encoder
from mmf.utils.configuration import get_mmf_cache_dir
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertOnlyNSPHead,
    BertForPreTraining,
    BertPredictionHeadTransform,
)
from transformers.activations import ACT2FN

class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FusionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()
        self.LN = nn.LayerNorm(output_dim)

    def forward(self, vector1, vector2):
        fusion_input = torch.cat((vector1, vector2), dim=1)
        hidden1 = self.gelu(self.fc1(fusion_input))
        fusion_vector = self.fc2(hidden1)
        output_vector = self.LN(fusion_vector + vector1 + vector2)
        return output_vector

class FashionFAEForPretraining(FashionFAEBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.task_for_inference = config.task_for_inference
        self.tasks = config.tasks
        self.double_view = config.get("double_view", False)
        self.no_sharing = config.get("no_sharing", False)
        if self.no_sharing:
            self.bert_2 = deepcopy(self.bert)

        self.heads = nn.ModuleDict()
        self.loss_funcs = nn.ModuleDict()
        self.fusion = FusionMLP(1536, 3072, 768)
        self.training_head_type = config.training_head_type

        if "mpfc" in config.tasks and config.bypass_transformer:
            self.image_tokenizer = build_image_encoder(
                config.image_tokenizer,
                config.direct_features_input,
            )
            self.image_tokenizer = self.image_tokenizer.eval()
            for param in self.image_tokenizer.parameters():
                param.requires_grad = False

        self.init_heads()
        self.init_losses()

    def get_image_embedding(self, *args, **kwargs):
        if self.no_sharing:
            return self.bert_2.get_image_embedding(*args, **kwargs)
        else:
            return self.bert.get_image_embedding(*args, **kwargs)

    def get_text_embedding(self, *args, **kwargs):
        if self.no_sharing:
            return self.bert_2.get_text_embedding(*args, **kwargs)
        else:
            return self.bert.get_text_embedding(*args, **kwargs)

    def get_joint_embedding(self, *args, **kwargs):
        return self.bert.get_joint_embedding(*args, **kwargs)

    def get_mpfc_embedding(self, *args, **kwargs):
        return self.bert.get_mpfc_embedding(*args, **kwargs)

    def init_heads(self):
        if "itm" in self.tasks:
            self.heads["itm"] = BertOnlyNSPHead(self.bert.config)
        if "mlm" in self.tasks:
            bert_masked_lm = BertForPreTraining.from_pretrained(
                self.config.bert_model_name,
                config=self.bert.config,
                cache_dir=os.path.join(
                    get_mmf_cache_dir(), "distributed_{}".format(-1)
                ),
            )
            self.heads["mlm"] = deepcopy(bert_masked_lm.cls.predictions)
            self.bert._tie_or_clone_weights(
                self.heads["mlm"].decoder, self.bert.embeddings.word_embeddings
            )

        if "mpfc" in self.tasks:
            self.heads["mpfc"] = nn.Sequential(
                BertPredictionHeadTransform(self.bert.config),
                nn.Linear(
                    self.bert.config.hidden_size,
                    1024,
                ),
            )

    def init_losses(self):
        self.loss_funcs["itc"] = ContrastiveLoss()
        if "itm" in self.tasks:
            self.loss_funcs["itm"] = CrossEntropyLoss()
        if "mlm" in self.tasks:
            self.loss_funcs["mlm"] = CrossEntropyLoss(ignore_index=-1)
        if "mpfc" in self.tasks:
            self.loss_funcs["mpfc"] = CrossEntropyLoss()
        if "icc" in self.tasks:
            self.loss_funcs["icc"] = ContrastiveLoss()

    @torch.no_grad()
    def get_patch_labels(self, image, chunk_size=8):
        batch_size = image.shape[0]
        assert batch_size % chunk_size == 0
        # We need to override eval() as this image_tokenizer is a submodule
        self.image_tokenizer = self.image_tokenizer.eval()
        indices = []
        for i in range(batch_size // chunk_size):
            _, _, idx = self.image_tokenizer(
                image[i * chunk_size: (i + 1) * chunk_size]
            )
            indices.append(idx)
        indices = torch.cat(indices, dim=0)
        return indices.long()

    @torch.no_grad()
    def get_hard_pairs(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Hard negative pairs mining
        # FIXME: not support multi-gpu mining
        if self.training:
            reset_train = True
            self.eval()
        else:
            reset_train = False

        itc_dict = self._forward_itc(sample_list)
        image_embeddings = itc_dict["scores"]
        text_embeddings = itc_dict["targets"]
        correlations = image_embeddings @ text_embeddings.t()
        batch_size = correlations.shape[0]
        # under double_view mode we have more than one positives
        diag = torch.eye(batch_size).bool()
        if self.double_view:
            bs = batch_size // 2
            diag[:bs, bs:] = diag[:bs, :bs]
            diag[bs:, :bs] = diag[:bs, :bs]
        correlations[diag] = -1
        # FIXME: more complicated sampling strategy
        hard_text_index = torch.argmax(correlations, dim=1)
        combine_index = torch.arange(batch_size).to(image_embeddings.device)
        combine_index_index = torch.rand(batch_size) > 0.5
        combine_index[combine_index_index] = hard_text_index[combine_index_index]

        if reset_train:
            self.train()

        sample_list["input_ids"] = sample_list["input_ids"][combine_index]
        sample_list["segment_ids"] = sample_list["segment_ids"][combine_index]
        sample_list["input_mask"] = sample_list["input_mask"][combine_index]
        if "attention_mask" in sample_list.keys():
            sample_list["attention_mask"] = sample_list["attention_mask"][combine_index]
        sample_list["targets"][combine_index_index] = 0

        return sample_list

    def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        to_be_flattened = ["input_ids", 'segment_ids']
        to_be_flattened_dim = ["image"]
        if sample_list["task"] == "mlm":
            to_be_flattened += ["lm_label_ids", "input_ids_masked"]
        elif sample_list["task"] == "mpfc":
            to_be_flattened += ["image_masks"]
            if "patch_labels" in sample_list.keys():
                to_be_flattened += ["patch_labels"]
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def add_post_flatten_params(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        b, l, _ = sample_list["image"].shape
        device = sample_list["image"].device
        sample_list["visual_embeddings_type"] = torch.zeros(
            (b, l), device=device
        ).long()

        b1, l1, _ = sample_list["imageitc"].shape
        device = sample_list["imageitc"].device
        sample_list["visual_embeddingsitc_type"] = torch.zeros(
            (b1, l1), device=device
        ).long()

        if self.double_view and self.training:
            sample_list["input_ids"] = sample_list["input_ids"].repeat(2, 1)
            sample_list["segment_ids"] = sample_list["segment_ids"].repeat(2, 1)
            sample_list["input_mask"] = sample_list["input_mask"].repeat(2, 1)
            sample_list["targets"] = sample_list["targets"].repeat(2)
            if sample_list["task"] == "mlm":
                sample_list["input_ids_masked"] = sample_list[
                    "input_ids_masked"
                ].repeat(2, 1)
                sample_list["lm_label_ids"] = sample_list["lm_label_ids"].repeat(2, 1)

        if sample_list["task"] in ["itm", "mlm", "mpfc"]:
            sample_list["attention_mask"] = torch.cat(
                (
                    sample_list["input_mask"],
                    torch.ones((b, l), device=device).long(),
                ),
                dim=-1,
            )

        if sample_list["task"] in ["mpfc"]:
            if self.double_view:
                sample_list["image_masks"] = sample_list["image_masks"].repeat(2, 1)

            _, _, input_embedding = self.get_text_embedding(
                sample_list["input_ids"],
                sample_list["segment_ids"],
                sample_list["input_mask"],
            )
            sample_list["inputtext_embeddings"] = input_embedding[0]
            text_cls = sample_list["inputtext_embeddings"][:, 0]

            visual_embeddings, _, _ = self.get_image_embedding(
                sample_list["image"], sample_list["visual_embeddings_type"]
            )
            sample_list["visual_embeddings"] = visual_embeddings
            visual_cls = visual_embeddings[:, 0]

            cls = self.fusion(text_cls, visual_cls)
            cls = cls.float().unsqueeze(1)

            if self.double_view:
                sample_list["image_masks"] = sample_list["image_masks"].repeat(2, 1)
            mask = sample_list["image_masks"] == 0
            fill = sample_list["image_masks"] == 1
            mask = mask.float().unsqueeze(-1)
            fill = fill.float().unsqueeze(-1)
            sample_list["masked_image"] = sample_list["image"][:, 1:] * mask + cls * fill
            sample_list["masked_image"] = torch.cat((cls, sample_list["masked_image"]), dim=1)

        return sample_list

    def _forward_itc(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        visual_embeddings, _, _ = self.get_image_embedding(
            sample_list["imageitc"], sample_list["visual_embeddingsitc_type"]
        )
        visual_embeddings = visual_embeddings.mean(dim=1)
        visual_embeddings = self.bert.contrastive_norm(visual_embeddings)

        text_embeddings, _, input_embedding = self.get_text_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["input_mask"],
        )
        masks = sample_list["input_mask"]
        text_embeddings = text_embeddings * masks.unsqueeze(2)
        text_embeddings = torch.sum(text_embeddings, dim=1) / (
            torch.sum(masks, dim=1, keepdim=True)
        )

        text_embeddings = self.bert.contrastive_norm(text_embeddings)

        loss = {}
        output_dict = {
            "scores": visual_embeddings,
            "targets": text_embeddings,
        }

        if hasattr(sample_list, "dv_image"):
            visual_embeddings_dv, _, _ = self.get_image_embedding(
                sample_list["dv_imageitc"], sample_list["visual_embeddings_type"]
            )
            visual_embeddings_dv = visual_embeddings_dv.mean(dim=1)
            visual_embeddings_dv = self.bert.contrastive_norm(visual_embeddings_dv)
            output_dict2 = {
                "scores": visual_embeddings_dv,
                "targets": text_embeddings,
            }

        if hasattr(sample_list, "dv_image"):
            loss["itc_loss"] = self.loss_funcs["itc"](sample_list, output_dict) + self.loss_funcs["itc"](sample_list, output_dict2)
        else:
            loss["itc_loss"] = self.loss_funcs["itc"](sample_list, output_dict)

        #loss["itc_loss"] = self.loss_funcs["itc"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_itm(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample_list = self.get_hard_pairs(sample_list)
        hidden, pooled_output, _ = self.get_joint_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["image"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        num_visual_tokens = sample_list["image"].shape[1]
        logits = self.heads["itm"](pooled_output)
        reshaped_logits = logits.contiguous().view(-1, 2)
        output_dict = {"scores": reshaped_logits}

        loss = {}
        loss["itm_loss"] = self.loss_funcs["itm"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict

    def _forward_mlm(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sequence_output, _, _ = self.get_joint_embedding(
            sample_list["input_ids_masked"],
            sample_list["segment_ids"],
            sample_list["image"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        num_visual_tokens = sample_list["image"].shape[1]
        sequence_output = sequence_output[:, :-num_visual_tokens]
        logits = (
            self.heads["mlm"](sequence_output)
                .contiguous()
                .view(-1, self.bert.config.vocab_size)
        )
        labels = sample_list["lm_label_ids"].contiguous().view(-1)

        sample_list["targets"] = labels

        text_embeddings, _, input_embedding = self.get_text_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["input_mask"],
        )
        masks = sample_list["input_mask"]
        text_embeddings = text_embeddings * masks.unsqueeze(2)
        text_embeddings = torch.sum(text_embeddings, dim=1) / (
            torch.sum(masks, dim=1, keepdim=True)
        )

        text_embeddings = self.bert.contrastive_norm(text_embeddings)
        sequence_output = sequence_output * masks.unsqueeze(2)
        global_embeddings = torch.sum(sequence_output, dim=1) / (
            torch.sum(masks, dim=1, keepdim=True)
        )
        global_embeddings = self.bert.contrastive_norm(global_embeddings)
        output_dict1 = {
            "scores": text_embeddings,
            "targets": global_embeddings,
        }
        output_dict = {"scores": logits}

        loss = {}
        loss["mlm_loss"] = self.loss_funcs["mlm"](sample_list, output_dict) + self.loss_funcs["itc"](sample_list, output_dict1)

        output_dict["losses"] = loss

        return output_dict

    def _forward_mpfc(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:

        _, _, hidden_list = self.get_mpfc_embedding(
            sample_list["inputtext_embeddings"],
            sample_list["masked_image"],
            sample_list["visual_embeddings_type"],
            sample_list["attention_mask"],
        )
        hidden = hidden_list[1]

        _, num_visual_tokens, visual_dim = sample_list["image"].shape  # num_visual_tokens��Ҫ�޸�

        mask = sample_list["image_masks"] == 1

        hidden = hidden[:, -num_visual_tokens + 1:]
        hidden_masked = (
            hidden[mask.unsqueeze(-1).expand_as(hidden)]
                .contiguous()
                .view(-1, hidden.size(-1))
        )
        logits = self.heads["mpfc"](hidden_masked)

        target = self.get_patch_labels(sample_list["original_image"])
        target_masked = target[mask].contiguous().view(-1)

        visual_embeddings = sample_list["visual_embeddings"].mean(dim=1)
        visual_embeddings = self.bert.contrastive_norm(visual_embeddings)
        global_embeddings = self.bert.contrastive_norm(hidden.mean(dim=1))
        output_dict1 = {
            "scores": visual_embeddings,
            "targets": global_embeddings,
        }
        sample_list["targets"] = target_masked

        output_dict = {"scores": logits}

        loss = {}
        loss["mpfc_loss"] = self.loss_funcs["mpfc"](sample_list, output_dict) + self.loss_funcs["itc"](sample_list, output_dict1)
        output_dict["losses"] = loss

        return output_dict

    def _forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if sample_list["task"] == "itm":
            ouput_dict = self._forward_itm(sample_list)
        elif sample_list["task"] == "mlm":
            ouput_dict = self._forward_mlm(sample_list)
        elif sample_list["task"] == "mpfc":
            ouput_dict = self._forward_mpfc(sample_list)
        else:
            ouput_dict = self._forward_itc(sample_list)

        return ouput_dict
