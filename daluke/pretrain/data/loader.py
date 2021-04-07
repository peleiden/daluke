from __future__ import annotations
import os
import json

from transformers import AutoTokenizer
import torch

from pelutils import log

from daluke.data import Example, Words, Entities, get_special_ids, Words, Entities
from daluke.pretrain.data import load_jsonl
from daluke.pretrain.data.build import DatasetBuilder
from .masking import MaskedBatchedExamples

class DataLoader:

    def __init__(
        self,
        data_dir: str,
        metadata: dict,
        word_mask_prob:     float = 0.15,
        word_unmask_prob:   float = 0.1,
        word_randword_prob: float = 0.1,
        ent_mask_prob:      float = 0.15,
        max_sentence_len:   int = 512,
        max_entity_len:     int = 128,
        max_mention:        int = 30,
    ):
        """
        Loads a generated json dataset prepared by the preprocessing pipeline, e.g. like
        ```
        {
            "word_ids":     [[32, 59, 3]], [42, 11]],
            "word_spans":   [[[0, 2], [2, 3], [5, 7]], [[0, 1], [1, 2]]],
            "entity_ids":   [[5], []],
            "entity_spans": [[[0, 3]], []]
        }
        ```
        """
        self.max_sentence_len = max_sentence_len
        self.max_entity_len = max_entity_len
        self.max_mention = max_mention

        self.word_mask_prob = word_mask_prob
        self.word_unmask_prob = word_unmask_prob
        self.word_randword_prob = word_randword_prob
        self.ent_mask_prob = ent_mask_prob

        self.tokenizer = AutoTokenizer.from_pretrained(metadata["base-model"])
        self.sep_id, self.cls_id, self.pad_id = get_special_ids(self.tokenizer)
        self.word_mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.ent_mask_id = 2  # FIXME: Load entity vocab and dont hardcode this!
        # Don't insert ids that are special tokens when performing random word insertion in the masking
        self.random_word_id_range = (self.word_mask_id + 1, self.tokenizer.vocab_size)

        log.section("Creating examples ...")
        self.examples: list[Example] = list()
        # Build data in reverse order to pop more effeciently
        for seq_data in load_jsonl(os.path.join(data_dir, DatasetBuilder.data_file)):
            # Pop to reduce memory usage
            self.examples.append(Example(
                words = Words.build(
                    torch.LongTensor(seq_data["word_ids"]),
                    seq_data["word_spans"],
                    max_len = self.max_sentence_len,
                    sep_id  = self.sep_id,
                    cls_id  = self.cls_id,
                    pad_id  = self.pad_id,
                ),
                entities = Entities.build(
                    torch.LongTensor(seq_data["entity_ids"]),
                    seq_data["entity_spans"],
                    max_len     = self.max_entity_len,
                    max_mention = self.max_mention,
                )
            ))

    def __len__(self):
        return len(self.examples)

    def get_dataloader(self, batch_size: int, sampler: torch.utils.data.Sampler) -> DataLoader:
        # TODO: Maybe dataloader should be created in __init__?
        return torch.utils.data.DataLoader(list(enumerate(self.examples)), batch_size=batch_size, sampler=sampler, collate_fn=self.collate)

    def collate(self, batch: list[tuple[int, Example]]) -> MaskedBatchedExamples:
        return MaskedBatchedExamples.build(
            [ex for _, ex in batch],
            word_mask_id       = self.word_mask_id,
            ent_mask_id        = self.ent_mask_id,
            word_mask_prob     = self.word_mask_prob,
            word_unmask_prob   = self.word_unmask_prob,
            word_randword_prob = self.word_randword_prob,
            word_id_range      = self.random_word_id_range,
            ent_mask_prob      = self.ent_mask_prob,
        )
