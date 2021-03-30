from __future__ import annotations
import os
import json

from transformers import AutoTokenizer
import torch

from pelutils import log

from daluke import daBERT
from daluke.data import Example, Words, Entities, get_special_ids, BatchedExamples, Words, Entities


class DataLoader:

    def __init__(
        self,
        data_dir: str,
        tokenizer_name:   str = daBERT,
        max_sentence_len: int = 512,
        max_entity_len:   int = 128,
        max_mention:      int = 30,
    ):
        # TODO: Update docs with word spans
        """
        Loads a generated json dataset prepared by the preprocessing pipeline, e.g. like
        ```
        {
            "word_ids":     [[32, 59, 3]], [42, 11]],
            "entity_ids":   [[5], []],
            "entity_spans": [[[0, 3]], []]
        }
        ```
        """
        self.max_sentence_len = max_sentence_len
        self.max_entity_len = max_entity_len
        self.max_mention = max_mention

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.sep_id, self.cls_id, self.pad_id = get_special_ids(self.tokenizer)
        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.ent_mask_id = 2  # FIXME: Load entity vocab and dont hardcode this!

        # FIXME: Circular imports, so Builder.data_file can be used
        with open(path := os.path.join(data_dir, "data.json"), "r") as f:
            log.debug("Loading json dataset into memory from %s ..." % path)
            data = json.load(f)
        N = len(data["word_ids"])
        assert N == len(data["entity_ids"]) == len(data["entity_spans"])

        log.debug("Creating examples ...")
        self.examples: list[Example] = [None for _ in range(N)]
        # Build data in reverse order to pop more effeciently
        for i in range(N-1, -1, -1):
            # Pop to reduce memory usage
            word_ids, ent_ids, ent_spans = data["word_ids"].pop(), data["entity_ids"].pop(), data["entity_spans"].pop()
            self.examples[i] = Example(
                words = Words.build(
                    torch.LongTensor(word_ids),
                    max_len = self.max_sentence_len,
                    sep_id  = self.sep_id,
                    cls_id  = self.cls_id,
                    pad_id  = self.pad_id,
                ),
                entities = Entities.build(
                    torch.LongTensor(ent_ids),
                    ent_spans,
                    max_len     = self.max_entity_len,
                    max_mention = self.max_mention,
                )
            )

    def __len__(self):
        return len(self.examples)

    def get_dataloader(self, batch_size: int, sampler: torch.utils.data.Sampler) -> DataLoader:
        # TODO: Maybe dataloader should be created in __init__?
        return torch.utils.data.DataLoader(list(enumerate(self.examples)), batch_size=batch_size, sampler=sampler, collate_fn=self.collate)

    @staticmethod
    def collate(batch: list[tuple[int, Example]]) -> BatchedExamples:
        return BatchedExamples.build([ex for _, ex in batch])
