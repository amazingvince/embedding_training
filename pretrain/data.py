import random
from copy import deepcopy
from dataclasses import dataclass

import torch.utils.data.dataset
from datasets import Dataset
from pretrain.utils import tensorize_batch
from transformers import DataCollatorForWholeWordMask


class DatasetForPretraining(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.dataset = Dataset.load_from_disk(data_dir)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


@dataclass
class DupMAECollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 1024
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def __call__(self, examples):
        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []
        bag_word_weight = []

        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)

        for e in examples:
            # print(e)
            e_trunc = self.tokenizer.build_inputs_with_special_tokens(
                e["input_ids"][:tgt_len]
            )
            tokens = [
                self.tokenizer._convert_id_to_token(tid) for tid in e["input_ids"]
            ]

            self.mlm_probability = self.encoder_mlm_probability
            text_encoder_mlm_mask = self._whole_word_mask(tokens)

            self.mlm_probability = self.decoder_mlm_probability
            mask_set = []
            for _ in range(min(len(tokens), self.max_seq_length // 2)):
                mask_set.append(self._whole_word_mask(tokens))

            text_matrix_attention_mask = []
            for i in range(len(tokens)):
                idx = random.randint(0, min(len(tokens), self.max_seq_length // 2) - 1)
                text_decoder_mlm_mask = deepcopy(mask_set[idx])
                text_decoder_mlm_mask[i] = 1
                text_matrix_attention_mask.append(text_decoder_mlm_mask)

            input_ids_batch.append(torch.tensor(e_trunc))
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))
            e_trunc[0] = -100
            e_trunc[-1] = -100
            decoder_labels_batch.append(torch.tensor(e_trunc))

            encoder_mlm_mask_batch.append(torch.tensor(text_encoder_mlm_mask))
            decoder_matrix_attention_mask_batch.append(
                1 - torch.tensor(text_matrix_attention_mask)
            )

            weight = torch.zeros(size=(self.tokenizer.vocab_size,))
            print(weight.shape)
            for t in e["input_ids"][:tgt_len]:
                weight[t] = 1 / len(e["input_ids"][:tgt_len])
            bag_word_weight.append(weight.unsqueeze(0))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        origin_input_ids_batch = input_ids_batch.clone()
        encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
        encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(
            input_ids_batch, encoder_mlm_mask_batch
        )
        decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
        matrix_attention_mask_batch = tensorize_batch(
            decoder_matrix_attention_mask_batch, 0
        )
        bag_word_weight = torch.cat(bag_word_weight, dim=0)

        batch = {
            "encoder_input_ids": encoder_input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
            "encoder_labels": encoder_labels_batch,
            "decoder_input_ids": origin_input_ids_batch,
            "decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
            "decoder_labels": decoder_labels_batch,
            "bag_word_weight": bag_word_weight,
        }

        return batch


from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    RobertaTokenizer,
    RobertaTokenizerFast,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
)
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


@dataclass
class DupMAECollator_Roberta(DataCollatorForWholeWordMask):
    max_seq_length: int = 1024
    encoder_mlm_probability: float = 0.3
    decoder_mlm_probability: float = 0.5

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=max_seq_length):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        # if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast,
        #                                   RobertaTokenizer, RobertaTokenizerFast,
        #                                   XLMRobertaTokenizer, XLMRobertaTokenizerFast,
        #                                   HerbertTokenizer, HerbertTokenizerFast)):
        #    warnings.warn(
        #        "DataCollatorForWholeWordMask is only suitable for BertTokenizer or RobertaTokenizer-like tokenizers. "
        #        "Please refer to the documentation for more information."
        #    )

        cand_indexes = []
        special_tokens = [
            val
            for key, val in self.tokenizer.special_tokens_map.items()
            if key not in ["unk_token", "mask_token"]
        ]
        is_bert_tokenizer = isinstance(
            self.tokenizer, (BertTokenizer, BertTokenizerFast)
        )
        is_roberta_tokenizer = isinstance(
            self.tokenizer, (RobertaTokenizer, RobertaTokenizerFast)
        )
        is_xlm_roberta_tokenizer = isinstance(
            self.tokenizer, (XLMRobertaTokenizer, XLMRobertaTokenizerFast)
        )
        for i, token in enumerate(input_tokens):
            if token in special_tokens:
                continue

            if is_bert_tokenizer:
                if len(cand_indexes) >= 1 and token.startswith("##"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
            elif is_roberta_tokenizer:
                # If a token doesn't start with Ġ, it's part of the previous token
                if len(cand_indexes) >= 1 and not token.startswith("Ġ"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
            elif is_xlm_roberta_tokenizer:
                # If a token doesn't start with ▁, it's part of the previous token
                if len(cand_indexes) >= 1 and not token.startswith("▁"):
                    cand_indexes[-1].append(i)
                else:
                    cand_indexes.append([i])
            else:
                raise ValueError(
                    "Whole-word masking only implemented for BERT/RoBERTa/XLM-Roberta so far"
                )

        if len(cand_indexes[-1]) == 0:
            cand_indexes = cand_indexes[:-1]

        random.shuffle(cand_indexes)
        num_to_predict = min(
            max_predictions,
            max(1, int(round(len(input_tokens) * self.mlm_probability))),
        )
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError(
                "Length of covered_indexes is not equal to length of masked_lms."
            )
        mask_labels = [
            1 if i in covered_indexes else 0 for i in range(len(input_tokens))
        ]
        return mask_labels

    def __call__(self, examples):
        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []
        bag_word_weight = []

        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)

        for e in examples:
            # print(e)
            e_trunc = self.tokenizer.build_inputs_with_special_tokens(
                e["input_ids"][:tgt_len]
            )
            tokens = [
                self.tokenizer._convert_id_to_token(tid) for tid in e["input_ids"]
            ]

            self.mlm_probability = self.encoder_mlm_probability
            text_encoder_mlm_mask = self._whole_word_mask(tokens)

            self.mlm_probability = self.decoder_mlm_probability
            mask_set = []
            for _ in range(min(len(tokens), self.max_seq_length // 2)):
                mask_set.append(self._whole_word_mask(tokens))

            text_matrix_attention_mask = []
            for i in range(len(tokens)):
                idx = random.randint(0, min(len(tokens), self.max_seq_length // 2) - 1)
                text_decoder_mlm_mask = deepcopy(mask_set[idx])
                text_decoder_mlm_mask[i] = 1
                text_matrix_attention_mask.append(text_decoder_mlm_mask)

            input_ids_batch.append(torch.tensor(e_trunc))
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))
            e_trunc[0] = -100
            e_trunc[-1] = -100
            decoder_labels_batch.append(torch.tensor(e_trunc))

            encoder_mlm_mask_batch.append(torch.tensor(text_encoder_mlm_mask))
            decoder_matrix_attention_mask_batch.append(
                1 - torch.tensor(text_matrix_attention_mask)
            )
            # print("vocab size:", self.tokenizer.vocab_size)

            weight = torch.zeros(100288)
            # print(weight.shape)
            for t in e["input_ids"][:tgt_len]:
                weight[t] = 1 / len(e["input_ids"][:tgt_len])
            bag_word_weight.append(weight.unsqueeze(0))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        origin_input_ids_batch = input_ids_batch.clone()
        encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
        encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(
            input_ids_batch, encoder_mlm_mask_batch
        )
        decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
        matrix_attention_mask_batch = tensorize_batch(
            decoder_matrix_attention_mask_batch, 0
        )
        bag_word_weight = torch.cat(bag_word_weight, dim=0)

        batch = {
            "encoder_input_ids": encoder_input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
            "encoder_labels": encoder_labels_batch,
            "decoder_input_ids": origin_input_ids_batch,
            "decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
            "decoder_labels": decoder_labels_batch,
            "bag_word_weight": bag_word_weight,
        }

        return batch
