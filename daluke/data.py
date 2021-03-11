from __future__ import annotations
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Iterator, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from transformers import AutoTokenizer
from danlp.datasets import DDT

from daluke.luke import convert_examples_to_features, InputFeatures

TOKENIZER = "Maltehb/danish-bert-botxo"

class Split(IntEnum):
    TRAIN = 0
    DEV = 1
    TEST = 2

class NERDataset(ABC):
    feature_names = ("ent_start_pos", "ent_end_pos", "word_ids", "ent_ids", "word_seg_ids", "ent_seg_ids",
        "ent_pos_ids", "word_att_mask", "ent_att_mask", "labels")

    null_label: str = None
    labels: tuple[str] = None

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        self.texts: list[list[str]] = None
        self.annotations: list[list[str]] = None
        self.features: list[dict] = None
        self.current_split: Split = None
        self.entity_spans: list[list[int]] = None

    @abstractmethod
    def build(self, split: Split, batch_size: int) -> DataLoader:
        pass

    @property
    def all_labels(self):
        L = list() if self.null_label is None else [self.null_label]
        return L + list(self.labels)

    def _build_features(self, sent_bounds: list[list[int]]) -> list[dict[str, Any]]:
        feature_objects: list[InputFeatures] = convert_examples_to_features(
            list(zip(self.texts, self.annotations, sent_bounds)),
            self.all_labels,
            self.tokenizer,
            max_seq_length=512,    #FIXME: No hardcode
            max_entity_length=128,
            max_mention_length=18,
        )
        # Convert to dict using only the relevant fields, as the data is highly flexible
        fields = ("entity_start_positions", "entity_end_positions", "word_ids", "entity_ids", "word_segment_ids",
            "entity_segment_ids", "entity_position_ids", "word_attention_mask", "entity_attention_mask", "labels")
        self.entity_spans = [f_obj.original_entity_spans for f_obj in feature_objects]
        return [
            {fname: getattr(f_obj, field) for fname, field in zip(self.feature_names, fields)}
                for f_obj in feature_objects
        ]

    def _collate(self, batch: Iterator[tuple[int, dict[str, Any]]]):
        """
        Collect dataset examples into tensors and pad them for sequence classification
        """
        paddings = {"word_ids": self.tokenizer.pad_token_id, "ent_pos_ids": -1, "labels": -1}
        collated = dict()
        for feature in self.feature_names:
            tensors = [torch.tensor(x[1][feature], dtype=torch.long) for x in batch]
            pad_val = paddings.get(feature) or 0
            collated[feature] = nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_val)
        if self.current_split != Split.TRAIN:
            collated.pop("labels")
            collated["spans"] = [self.entity_spans[x[0]] for x in batch]
        return collated

class DaNE(NERDataset):
    null_label = "O"
    labels = ("LOC", "PER", "ORG", "MISC")

    def build(self, split: Split, batch_size: int) -> DataLoader:
        self.current_split = split
        self.texts, self.annotations = DDT().load_as_simple_ner(predefined_splits=True)[split.value]
        # Sadly, we do not have access to where the DaNE sentences are divided into articles, so we let each sentence be an entire text.
        sentence_boundaries = [[0, len(s)] for s in self.texts]

        self.features = self._build_features(sentence_boundaries)
        sampler = RandomSampler(self.features) if split == split.TRAIN else None
        return DataLoader(list(enumerate(self.features)), batch_size=batch_size, collate_fn=self._collate, sampler=sampler)
