from __future__ import annotations
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from pelutils import TT
from pelutils.jsonl import load_jsonl

from daluke.data import Example, Words, Entities, get_special_ids, Words, Entities
from daluke.pretrain.data.build import DatasetBuilder
from .masking import MaskedBatchedExamples
from . import ENTITY_MASK_TOKEN


class DataLoader:

    def __init__(
        self,
        data_dir:           str,
        metadata:           dict,
        entity_vocab:       dict,
        device:             torch.device,
        word_mask_prob:     float,
        word_unmask_prob:   float,
        word_randword_prob: float,
        ent_mask_prob:      float,
        only_load_validation = False,
        vocab_size:         int | None = None,
        token_map:          np.ndarray | None = None,
    ):
        """ Loads a generated json dataset prepared by the preprocessing pipeline """
        self.data_dir = data_dir
        self.metadata = metadata
        self.ent_ids = { info["id"] for info in entity_vocab.values() }
        self.device = device

        self.max_sentence_len = metadata["max-seq-length"]
        self.max_entities = metadata["max-entities"]
        self.max_entity_span = metadata["max-entity-span"]

        self.word_mask_prob = word_mask_prob
        self.word_unmask_prob = word_unmask_prob
        self.word_randword_prob = word_randword_prob
        self.ent_mask_prob = ent_mask_prob
        self.only_load_validation = only_load_validation

        self.tokenizer = AutoTokenizer.from_pretrained(metadata["base-model"])
        self.sep_id, self.cls_id, self.pad_id, self.word_mask_id, __ = get_special_ids(self.tokenizer)
        if token_map is not None:
            self.sep_id, self.cls_id, self.pad_id, self.word_mask_id = token_map[
                [self.sep_id, self.cls_id, self.pad_id, self.word_mask_id]
            ]
        self.ent_mask_id = entity_vocab[ENTITY_MASK_TOKEN]["id"]
        # Don't insert ids that are special tokens when performing random word insertion in the masking
        # The allowed range is dependant on the placement of special ids
        vocab_size = vocab_size or self.tokenizer.vocab_size
        self.random_word_id_range = (self.word_mask_id + 1, vocab_size)\
            if self.word_mask_id < vocab_size-1 else\
                (self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)+1, vocab_size-1)

        with TT.profile("Building examples"):
            self.train_examples, self.val_examples = self.build_examples()

    def __len__(self):
        return len(self.train_examples) + len(self.val_examples)

    def build_examples(self) -> tuple[list[Example], list[Example]]:
        train_examples, val_examples = list(), list()
        with open(os.path.join(self.data_dir, DatasetBuilder.data_file)) as df:
            for seq_data in load_jsonl(df):
                # Backwards compatible to time before validation
                is_validation = seq_data.get("is_validation", False)
                if self.only_load_validation and not is_validation:
                    continue

                ex = Example(
                    words = Words.build(
                        torch.IntTensor(seq_data["word_ids"]),
                        seq_data["word_spans"],
                        max_len = self.max_sentence_len,
                        sep_id  = self.sep_id,
                        cls_id  = self.cls_id,
                        pad_id  = self.pad_id,
                    ),
                    entities = Entities.build(
                        torch.IntTensor(seq_data["entity_ids"]),
                        seq_data["entity_spans"],
                        max_entities    = self.max_entities,
                        max_entity_span = self.max_entity_span,
                    ),
                )
                if is_validation:
                    val_examples.append(ex)
                else:
                    train_examples.append(ex)
        return train_examples, val_examples

    def get_dataloader(self, batch_size: int, sampler: torch.utils.data.Sampler, validation=False):
        return torch.utils.data.DataLoader(
            list(enumerate(self.val_examples if validation else self.train_examples)),
            batch_size  = batch_size,
            sampler     = sampler,
            collate_fn  = self.collate,
            drop_last   = True,
        )

    def collate(self, batch: list[tuple[int, Example]]) -> MaskedBatchedExamples:
        with TT.profile("Masking words and entities"):
            return MaskedBatchedExamples.build(
                [ex for _, ex in batch],
                self.device,
                word_mask_id       = self.word_mask_id,
                ent_mask_id        = self.ent_mask_id,
                word_mask_prob     = self.word_mask_prob,
                word_unmask_prob   = self.word_unmask_prob,
                word_randword_prob = self.word_randword_prob,
                word_id_range      = self.random_word_id_range,
                ent_mask_prob      = self.ent_mask_prob,
                cut_extra_padding  = True,
            )
