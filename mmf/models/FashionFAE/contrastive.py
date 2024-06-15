# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

import torch
from mmf.models.composition import NormalizationLayer
from mmf.models.FashionFAE.base import FashionFAEBaseModel
from mmf.modules.losses import ContrastiveLoss
from torch import Tensor
import torch.nn as nn


class FashionFAEForContrastive(FashionFAEBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.loss_funcs = nn.ModuleDict()
        self.loss_funcs["itc"] = ContrastiveLoss()

    def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        to_be_flattened = []
        to_be_flattened_dim = []
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def add_post_flatten_params(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        b, l, _ = sample_list["imageitc"].shape
        device = sample_list["imageitc"].device
        sample_list["visual_embeddingsitc_type"] = torch.zeros(
            (b, l), device=device
        ).long()
        return sample_list

    def _forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        text_embeddings, _, _ = self.bert.get_text_embedding(
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

        visual_embeddings, _, _ = self.bert.get_image_embedding(
            sample_list["imageitc"],
            sample_list["visual_embeddingsitc_type"],
        )

        visual_embeddings = visual_embeddings.mean(dim=1)
        visual_embeddings = self.bert.contrastive_norm(visual_embeddings)
        output_dict = {
            "scores": visual_embeddings,
            "targets": text_embeddings,
        }
        loss = {}

        loss["itc_loss"] = self.loss_funcs["itc"](sample_list, output_dict)
        output_dict["losses"] = loss

        return output_dict
