import copy
import json
import random

import torch
from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.mmf_dataset import MMFDataset
from .utils import pharse_fashiongen_season, pre_caption, process_description, process_attribute
from .database import FashionAllDatabase, FashionGenDatabase, Fashion200kDatabase, BigFACADDatabase, PolyvoreOutfitsDatabase
from transformers.models.bert.tokenization_bert import BertTokenizer

class FashionDataset(MMFDataset):
    def __init__(
        self,
        name: str,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            name,
            config,
            dataset_type,
            index,
            *args,
            **kwargs,
        )
        self._double_view = config.get("double_view", False)
        self._attribute_label = config.get("attribute_label", False)
        self._category_label = config.get("category_label", False)
        self._subcategory_label = config.get("subcategory_label", False)
        self.catemap = self.__load_cate_map()
        self.training_head_type = config.get("training_head_type", 'pretraining')
        self.tokenizer = BertTokenizer.from_pretrained('./bert_config')
        self.max_words=120
        if self.training_head_type == 'contrastive':
            self.max_words=180
        self.__inner_attribute_names = [
            'title',
            'category',
            'subcategory',
            'gender',
            'season',
            'composition'
            ]
        if self.training_head_type == 'classification':
            self.__inner_attribute_names = [
            'title',
            'gender',
            'season',
            'composition'
            ]

    def init_processors(self):
        super().init_processors()
        if self._use_images:
            # Assign transforms to the image_db
            if self._dataset_type == "train":
                self.image_db.transform = self.train_image_processor
            else:
                self.image_db.transform = self.eval_image_processor

    def _get_valid_text_attribute(self, sample_info):
        if "captions" in sample_info:
            return "captions"

        if "sentences" in sample_info:
            return "sentences"

        raise AttributeError("No valid text attribution was found")

    def __load_cate_map(self):
        cate_map = {}
        with open('./categorys_to_sign.txt', mode='r', encoding='utf8') as mapfile:
            lines = mapfile.readlines()
            for line in lines:
                cate, symbol = line.strip().split('\t')
                cate_map[cate] = symbol
        return cate_map

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        text_attr = self._get_valid_text_attribute(sample_info)

        current_sample = Sample()
        sentence = sample_info[text_attr]
        category = sample_info["category"]
        current_sample.text = sentence
        sign_token = self.catemap[category.lower()]
        #if self._dataset_type == "train" and self.training_head_type == 'pretraining':
        if self.training_head_type == 'pretraining' or self.training_head_type == 'contrastive' or self.training_head_type == 'classification':
            ###### process description
            description = sentence
            description_oritokens, description_tokens, des_mask_token, des_mask_sign, des_replace_token, des_replace_sign = \
                process_description(description, tagger=None, tokenizer=self.tokenizer)

            # process attribute
            attributes = {k: sample_info[k] for k in self.__inner_attribute_names}
            attribute_oritokens, attribute_tokens, attr_mask_token, attr_mask_sign, attr_replace_token, attr_replace_sign = \
                process_attribute(attributes, self.tokenizer)

            # process all
            all_tokens = ['[CLS]'] + description_tokens + ['[SEP]'] + attribute_tokens
            ori_tokens = ['[CLS]'] + description_oritokens + ['[SEP]'] + attribute_oritokens
            #all_tokens = ['[CLS]'] + attribute_tokens + ['[SEP]'] + description_tokens
            #ori_tokens = ['[CLS]'] + attribute_oritokens + ['[SEP]'] + description_oritokens
            mask_sign = [0] + des_mask_sign + [0] + attr_mask_sign
            #mask_sign = [0] + attr_mask_sign + [0] + des_mask_sign
            tokens_length = len(all_tokens)
            ori_length = len(ori_tokens)
            assert tokens_length == ori_length
            if tokens_length > self.max_words:
                all_tokens = all_tokens[:self.max_words]
                mask_sign = mask_sign[:self.max_words]
            if ori_length > self.max_words:
                ori_tokens = ori_tokens[:self.max_words]
                
            tokens_length = len(all_tokens)
            ori_length = len(ori_tokens)
            pad_length = self.max_words - tokens_length
            padori_length = self.max_words - ori_length
            pad_tokens = ['[PAD]'] * pad_length
            oripad_tokens = ['[PAD]'] * padori_length
            all_tokens = all_tokens + pad_tokens
            ori_tokens = ori_tokens + oripad_tokens
            segment_ids = [0] * len(all_tokens)

            mask_token = des_mask_token + attr_mask_token
            #mask_token = attr_mask_token + des_mask_token
            mask_token = mask_token[:sum(mask_sign)]

            input_ids = self.tokenizer.convert_tokens_to_ids(all_tokens)
            ori_tokens = self.tokenizer.convert_tokens_to_ids(ori_tokens)
            mask_token_ids = self.tokenizer.convert_tokens_to_ids(mask_token)
            fsislabel = self.tokenizer.convert_tokens_to_ids(sign_token)

            mask_labels = [-1] * self.max_words
            mask_pos = [i for i, a in enumerate(mask_sign) if a == 1]

            assert len(mask_pos) == len(mask_token_ids)

            for p, l in zip(mask_pos, mask_token_ids):
                mask_labels[p] = l

            input_mask = [1] * ori_length + [0] * padori_length

            ####change list to tensor
            current_sample.input_ids_masked = torch.tensor(input_ids, dtype=torch.long)
            current_sample.input_ids = torch.tensor(ori_tokens, dtype=torch.long)
            current_sample.input_mask = torch.tensor(input_mask, dtype=torch.long)
            current_sample.lm_label_ids = torch.tensor(mask_labels, dtype=torch.long)
            current_sample.segment_ids = torch.tensor(segment_ids, dtype=torch.long)
            current_sample.fsislabel = torch.tensor(fsislabel, dtype=torch.long)
        
        else:
            processed_sentence = self.text_processor({"text": sentence, "sign_token": sign_token})
            current_sample.update(processed_sentence)
        
        image_path = sample_info["image_path"]
        if self._dataset_type == "train":
            if not self._double_view:
                image_path = random.choices(image_path)[0]
                current_sample.image = self.image_db.from_path(image_path)["images"][0]
            else:
                # FIXME: don't support features loading under double view mode
                assert self._use_images
                image_path_0, image_path_1 = random.choices(image_path, k=2)
                
                current_sample.image = self.image_db.from_path(image_path_0)["images"][0]
                current_sample.dv_image = self.image_db.from_path(image_path_1)["images"][0]
            if self._category_label:
                current_sample.targets = torch.tensor(sample_info["category_id"], dtype=torch.long)
            elif self._subcategory_label:
                current_sample.targets = torch.tensor(sample_info["subcategory_id"], dtype=torch.long)
            else:
                current_sample.targets = torch.tensor(1, dtype=torch.long)
        else:
            images = self.image_db.from_path(image_path)["images"]
            images = torch.stack(images)
            current_sample.image = images
            current_sample.text_id = torch.tensor(sample_info["id"], dtype=torch.long)
            current_sample.image_id = current_sample.text_id.repeat(len(image_path))
            if "subcategory_id" in sample_info:
                current_sample.text_subcat_id = torch.tensor(sample_info["subcategory_id"], dtype=torch.long)
                current_sample.image_subcat_id = current_sample.text_subcat_id.repeat(len(image_path))
                current_sample.text_cat_id = torch.tensor(sample_info["category_id"], dtype=torch.long)
                current_sample.image_cat_id = current_sample.text_cat_id.repeat(len(image_path))
            if self._category_label:
                current_sample.targets = torch.tensor(sample_info["category_id"], dtype=torch.long).repeat(
                    len(image_path))
                current_sample.input_ids = current_sample.input_ids.repeat(len(image_path), 1)
                current_sample.segment_ids = current_sample.segment_ids.repeat(len(image_path), 1)
                current_sample.input_mask = current_sample.input_mask.repeat(len(image_path), 1)
            elif self._subcategory_label:
                current_sample.targets = torch.tensor(sample_info["subcategory_id"], dtype=torch.long).repeat(
                    len(image_path))
                current_sample.input_ids = current_sample.input_ids.repeat(len(image_path), 1)
                current_sample.segment_ids = current_sample.segment_ids.repeat(len(image_path), 1)
                current_sample.input_mask = current_sample.input_mask.repeat(len(image_path), 1)
            else:
                current_sample.targets = torch.tensor(1, dtype=torch.long)

        current_sample.ann_idx = torch.tensor(idx, dtype=torch.long)

        if hasattr(self, "masked_image_processor"):
            current_sample.image_masks = self.masked_image_processor(current_sample.image)

        return current_sample


class FashionGenDataset(FashionDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "fashiongen",
            config,
            dataset_type,
            index,
            FashionGenDatabase,
            *args,
            **kwargs,
        )


class FashionAllDataset(FashionDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "fashionall",
            config,
            dataset_type,
            index,
            FashionAllDatabase,
            *args,
            **kwargs,
        )
