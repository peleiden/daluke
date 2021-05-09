from __future__ import annotations
import os
import json
import pickle

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from pelutils import log, TT

from daluke.data import Example, Words, Entities, get_special_ids, Words, Entities
from daluke.pretrain.data import load_jsonl
from daluke.pretrain.data.build import DatasetBuilder
from .masking import MaskedBatchedExamples

class DataLoader:

    def __init__(
        self,
        data_dir: str,
        metadata: dict,
        entity_vocab: dict,
        device:   torch.device,
        ent_mask_id:         int = 2,
        max_sentence_len:    int = 512,
        word_mask_prob:      float = 0.15,
        word_unmask_prob:    float = 0.1,
        word_randword_prob:  float = 0.1,
        ent_mask_prob:       float = 0.15,
    ):
        """
        Loads a generated json dataset prepared by the preprocessing pipeline
        """
        self.data_dir = data_dir
        self.metadata = metadata
        self.ent_ids = { info["id"] for info in entity_vocab.values() }
        self.device = device

        self.max_sentence_len = max_sentence_len
        self.max_entities = metadata["max-entities"]
        self.max_entity_span = metadata["max-entity-span"]

        self.word_mask_prob = word_mask_prob
        self.word_unmask_prob = word_unmask_prob
        self.word_randword_prob = word_randword_prob
        self.ent_mask_prob = ent_mask_prob

        self.tokenizer = AutoTokenizer.from_pretrained(metadata["base-model"])
        self.sep_id, self.cls_id, self.pad_id = get_special_ids(self.tokenizer)
        self.word_mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.ent_mask_id = ent_mask_id
        # Don't insert ids that are special tokens when performing random word insertion in the masking
        self.random_word_id_range = (self.word_mask_id + 1, self.tokenizer.vocab_size)

        log.section("Building examples ...")
        with TT.profile("Building examples"):
            self.examples: list[Example] = self.build_examples()
        log("Got %i examples" % len(self.examples))

    def __len__(self):
        return len(self.examples)

    def build_examples(self) -> list[Example]:
        examples = list()
        for seq_data in log.tqdm(tqdm(
            load_jsonl(os.path.join(self.data_dir, DatasetBuilder.data_file)),
            total=self.metadata["number-of-items"],
        )):
            try:
                # Keep only entities in filtered entity vocab
                seq_data["entity_ids"], seq_data["entity_spans"] = zip(
                    *((id_, span) for id_, span in zip(seq_data["entity_ids"], seq_data["entity_spans"]) if id_ in self.ent_ids)
                )
            except ValueError:
                # Happens if no entities in vocab
                seq_data["entity_ids"] = list()
                seq_data["entity_spans"] = list()
            examples.append(
                Example(
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
                        list(seq_data["entity_spans"]),
                        max_entities    = self.max_entities,
                        max_entity_span = self.max_entity_span,
                    ),
                )
            )
        return examples

    def get_dataloader(self, batch_size: int, sampler: torch.utils.data.Sampler) -> DataLoader:
        return torch.utils.data.DataLoader(
            list(enumerate(self.examples)),
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
