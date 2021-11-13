from __future__ import annotations
import ctypes
import os
from typing import Generator

import numpy as np
import torch
from transformers import AutoTokenizer

from pelutils import TT, c_ptr
from pelutils.jsonl import load_jsonl

from daluke.data import Example, Words, Entities, get_special_ids, Words, Entities
from daluke.pretrain.data.build import DatasetBuilder
from .masking import MaskedBatchedExamples
from . import ENTITY_MASK_TOKEN


_lib = ctypes.cdll.LoadLibrary(os.path.join("so", "collate.so"))

class DataLoader:

    def __init__(
        self,
        data_dir:           str,
        metadata:           dict,
        entity_vocab:       dict,  # Only given if a subset of the full vocab
        device:             torch.device,
        word_mask_prob:     float,
        word_unmask_prob:   float,
        word_randword_prob: float,
        ent_mask_prob:      float,
        validation_prop:    float = 0,
        vocab_size:         int | None = None,
        token_map:          np.ndarray | None = None,
    ):
        """ Loads a generated json dataset prepared by the preprocessing pipeline """
        self.data_dir = data_dir
        self.datafile = os.path.join(self.data_dir, DatasetBuilder.data_file)
        self.metadata = metadata
        self.ent_ids  = { info["id"] for info in entity_vocab.values() }
        self.device   = device

        self.max_sentence_len = metadata["max-seq-length"]
        self.max_entities     = metadata["max-entities"]
        self.max_entity_span  = metadata["max-entity-span"]

        self.word_mask_prob     = word_mask_prob
        self.word_unmask_prob   = word_unmask_prob
        self.word_randword_prob = word_randword_prob
        self.ent_mask_prob      = ent_mask_prob

        self.validation_prop    = validation_prop

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
        self.random_word_id_range = (self.word_mask_id + 1, vocab_size-1)\
            if self.word_mask_id < vocab_size-1 else\
                (self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)+1, vocab_size-2)

        self.train_examples = list(range(int(self.validation_prop*len(self)), len(self)))
        self.val_examples = list(range(int(self.validation_prop*len(self))))

    def __len__(self):
        return self.metadata["number-of-items"]

    def get_dataloader(self, batch_size: int, sampler: torch.utils.data.Sampler, validation=False):
        return torch.utils.data.DataLoader(
            np.array(self.val_examples if validation else self.train_examples) * self.metadata["bytes-per-example"],
            batch_size  = batch_size,
            sampler     = sampler,
            collate_fn  = self.collate,
            drop_last   = True,
        )

    def collate(self, batch: list[int]) -> MaskedBatchedExamples:
        batch = sorted(batch)

        with TT.profile("Building sub-batch"):
            with TT.profile("Load data"):
                # Read examples
                batch_matrix = torch.empty((len(batch), self.metadata["bytes-per-example"]), dtype=torch.int)
                max_size_arr = np.empty(2, np.uint64)
                res = _lib.read_batch(
                    ctypes.c_char_p(self.datafile.encode("utf-8")),
                    len(batch),
                    self.metadata["bytes-per-example"],
                    c_ptr(np.array(batch)),
                    ctypes.c_void_p(batch_matrix.data_ptr()),
                    c_ptr(max_size_arr),
                )
                if res == -1:
                    raise FileNotFoundError("Datafile not found at %s" % self.datafile)

            with TT.profile("Mask words and entities"):
                return MaskedBatchedExamples.build(
                    list(self.examples_from_batch_matrix()),
                    self.device,
                    word_mask_id       = self.word_mask_id,
                    ent_mask_id        = self.ent_mask_id,
                    word_mask_prob     = self.word_mask_prob,
                    word_unmask_prob   = self.word_unmask_prob,
                    word_randword_prob = self.word_randword_prob,
                    word_id_range      = self.random_word_id_range,
                    ent_mask_prob      = self.ent_mask_prob,
                )

    def examples_from_batch_matrix(self, batch_matrix: torch.IntTensor) -> Generator[Example, None, None]:
        span_start = 3 + self.max_sentence_len
        entity_start = 3 + 2 * self.max_sentence_len
        entity_span_start = entity_start + self.max_entities

        for row in batch_matrix:
            word_spans = row[span_start:span_start+row[1]]
            word_starts = word_spans // 2**16
            word_ends = word_spans % 2**16


            yield Example(
                Words.build(
                    row[3:3+row[0]+2],
                    list(zip(word_starts, word_ends)),
                    self.max_sentence_len,
                    self.sep_id,
                    self.cls_id,
                    self.pad_id,
                ),
                Entities.build(
                    row[entity_start:entity_start+row[2]],
                    
                )
            )
