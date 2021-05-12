from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any
import math
from itertools import chain

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from danlp.datasets import DDT

from daluke.data import Entities, Example, BatchedExamples, Words, get_special_ids

@dataclass
class NEREntities(Entities):
    start_pos: torch.LongTensor
    end_pos: torch.LongTensor
    labels: torch.LongTensor # Must be long for criterion
    fullword_spans: list[tuple[int, int]] # Must be same order as the spans given to entities

    @classmethod
    def build_from_entities(cls, ent: Entities, labels: torch.LongTensor, fullword_spans: list[tuple[int, int]], max_entities: int):
        out_labels = torch.full((max_entities,), -1, dtype=torch.long)
        out_labels[:len(labels)] = torch.LongTensor(labels)

        # Have to be long tensors as we are working with indeces
        start_pos = torch.full((max_entities,), 0, dtype=torch.long)
        start_pos[:len(ent.spans)] = torch.LongTensor([s for s, _ in ent.spans])

        end_pos = torch.full((max_entities,), 0, dtype=torch.long)
        # -1 as the spans are end-exclusive, but we want to gather the last token
        end_pos[:len(ent.spans)] = torch.LongTensor([e-1 for _, e in ent.spans])
        return cls(ent.ids, ent.attention_mask, ent.N, ent.spans, ent.pos, start_pos, end_pos, out_labels, fullword_spans)

@dataclass
class NERExample(Example):
    """
    A single data example for Named Entity Recognition
    """
    entities: NEREntities
    text_num: int

@dataclass
class NERBatchedExamples(BatchedExamples):
    text_nums: list[int]

    @classmethod
    def build(
        cls,
        examples: list[NERExample],
        device:   torch.device,
        cut_extra_padding: bool=True,
    ):
        words, entities = cls.collate(examples, device=device, cut=cut_extra_padding)
        ent_limit = entities.ids.shape[1]

        text_nums = [ex.text_num for ex in examples]
        ner_entities = NEREntities(
            ids             = entities.ids,
            attention_mask  = entities.attention_mask,
            N               = entities.N,
            spans           = entities.spans,
            pos             = entities.pos,
            start_pos       = torch.stack([ex.entities.start_pos[:ent_limit] for ex in examples]).to(device),
            end_pos         = torch.stack([ex.entities.end_pos[:ent_limit] for ex in examples]).to(device),
            labels          = torch.stack([ex.entities.labels[:ent_limit] for ex in examples]).to(device),
            fullword_spans  = [ex.entities.fullword_spans for ex in examples],
        )
        return cls(words, ner_entities, text_nums)

class Split(IntEnum):
    TRAIN = 0
    DEV = 1
    TEST = 2

class NERDataset(ABC):
    null_label: str | None = None
    labels: tuple[str] | None = None

    def __init__(self,
            entity_vocab: dict,
            base_model: str,
            max_seq_length: int,
            max_entities: int,
            max_entity_span: int,
            device: torch.device,
        ):
        self.device = device
        self.entity_vocab = entity_vocab
        self.max_seq_length = max_seq_length
        self.max_entities = max_entities
        self.max_entity_span = max_entity_span

        self.label_to_idx = {label: i for i, label in enumerate(self.all_labels)}

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.sep_id, self.cls_id, self.pad_id = get_special_ids(self.tokenizer)

        # To be set in build method
        self.examples: list[NERExample] | None = None
        self.texts: list[list[str]] | None = None
        self.annotations: list[list[str]] | None= None

    @abstractmethod
    def build(self, split: Split, batch_size: int) -> DataLoader:
        pass

    @property
    def all_labels(self) -> list[str]:
        L = list() if self.null_label is None else [self.null_label]
        return [*L, *self.labels]

    def _build_examples(self, sent_bounds: list[list[int]]) -> list[NERExample]:
        examples = list()
        for i, (text, annotation, bounds) in enumerate(zip(self.texts, self.annotations, sent_bounds)):
            text_token_ids: list[list[int]] = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            # We might have to split some sentences to respect the maximum sentence length
            bounds = self._add_extra_sentence_boundaries(bounds, text_token_ids)
            # TODO: Consider the current sentence splitting: Do we throw away context in situations where we actually have sentence-document information? (Not relevant for DaNE)
            for j, end in enumerate(bounds):
                start = bounds[j-1] if j else 0
                # Flatten structure of [[subwords], [subwords], ... ]
                word_ids = list(chain(*text_token_ids[start: end]))
                true_entity_fullword_spans = self._segment_entities(annotation[start: end])
                # The cumulative length of each word in units of subwords
                cumlength = np.cumsum([len(t) for t in text_token_ids[start: end]])
                # Save the spans of entities as they are in the token list
                true_entity_subword_spans = {(cumlength[entstart] - len(text_token_ids[start+entstart]), cumlength[entend-1]): ann
                    for (entstart, entend), ann in true_entity_fullword_spans.items()}
                assert all(e-s <= self.max_entity_span for s, e in true_entity_subword_spans),\
                        f"Example {i}, sentence {j} contains an entity longer than limit of {self.max_entity_span} tokens. Text:\n\t{text}"
                assert len(true_entity_subword_spans) < self.max_entities,\
                        f"Example {i}, sentence {j} contains {len(true_entity_subword_spans)} entities, but only {self.max_entities} are allowed. Text:\n\t{text}"

                all_entity_fullword_spans = self._generate_all_entity_spans(true_entity_fullword_spans, text_token_ids[start: end], cumlength)
                all_entity_subword_spans = [(cumlength[s] - len(text_token_ids[start+s]), cumlength[e-1]) for s, e in all_entity_fullword_spans]

                # We dont use the entity id: We just use the id feature for their length
                entity_ids = [self.entity_vocab["[UNK]"]["id"] for _ in range(len(all_entity_subword_spans))]
                subword_entity_labels = torch.LongTensor(
                    [self.label_to_idx[true_entity_subword_spans.get(span, self.null_label)] for span in all_entity_subword_spans]
                )
                # If there are too many possible spans for self.max_entities, we must divide the sequence into multiple examples
                for sub_example in range(int(math.ceil(len(all_entity_subword_spans)/self.max_entities))):
                    substart, subend = self.max_entities * sub_example,  self.max_entities * (sub_example + 1)
                    entities = Entities.build(
                        torch.IntTensor(entity_ids[substart:subend]),
                        all_entity_subword_spans[substart:subend],
                        max_entities    = self.max_entities,
                        max_entity_span = self.max_entity_span,
                    )
                    words = Words.build(
                        torch.IntTensor(word_ids),
                        max_len = self.max_seq_length,
                        sep_id  = self.sep_id,
                        cls_id  = self.cls_id,
                        pad_id  = self.pad_id,
                    )
                    examples.append(
                        NERExample(
                            words    = words,
                            entities = NEREntities.build_from_entities(
                                entities,
                                fullword_spans = all_entity_fullword_spans[substart:subend],
                                labels         = subword_entity_labels[substart:subend],
                                max_entities   = self.max_entities,
                            ),
                            text_num = i,
                        )
                    )

        return examples

    def collate(self, batch: list[tuple[int, NERExample]]) -> NERBatchedExamples:
        return NERBatchedExamples.build(
            [ex for _, ex in batch],
            self.device,
            cut_extra_padding = True,
        )

    def _generate_all_entity_spans(self, true_spans: dict[tuple[int, int], str], token_ids: list[list[int]], cumlength: np.ndarray) -> list[tuple[int, int]]:
        possible_spans = list()
        # Spans are (0, 1), (0, 2), (0, 3), ... (0, N), (1, 2), (1, 3), ... (N-1, N)
        for i in range(len(token_ids)):
            for j in range(i+1, len(token_ids)):
                if (i, j) not in true_spans and (cumlength[j] - cumlength[i]) <= self.max_entity_span:
                    possible_spans.append((i, j))
        # Making special case of single-word sequences as these are missed by end-exclusive for loops
        if len(token_ids) == 1 and (0, 1) not in true_spans:
            possible_spans.append((0, 1))
        # Make sure we include the true spans. Sort it such that the true spans are not always in the last example
        return list(sorted(possible_spans + list(true_spans.keys())))

    def _segment_entities(self, annotation: list[str]) -> dict[tuple[int, int], str]:
        """
        Reads entity annotation in IOB format and finds entity spans mapped to entity type
        """
        spans = dict()
        start = None
        ent_type = None
        for i, ann in enumerate(annotation):
            # Ignore "O"
            if ann == self.null_label:
                assert start is None, "Found O label, but another entity was not ended"
                continue
            tag, typ_ = ann.split("-")
            # Might be end if either (1) last in sentence or (2) followed by explicit start or (3) followed by another annotation type
            if i+1 == len(annotation) or annotation[i+1] == self.null_label or\
                    (next_ann := annotation[i+1].split("-"))[0] == "B" or next_ann[1] != typ_:
                assert ent_type is None or typ_ == ent_type, "Entity seems to change annotation during span - this should not be possible"
                spans[(i if start is None else start, i+1)] = typ_
                # We ended entity, look for a new one
                start = None
                ent_type = None
            # Might be a beginning if it is either (1) explicitly marked (2) starts sentence (3) follows another annotation type
            elif tag == "B" or not i or annotation[i-1] == self.null_label or\
                    annotation[i-1].split("-")[1] != typ_:
                assert start is None, "Found start entity while another one was not ended"
                start = i
                ent_type = typ_
        return spans

    def _add_extra_sentence_boundaries(self, bounds: list[int], text_token_ids: list[list[int]]) -> list[int]:
        # Check whether we should add another sentence bound by splitting one of the sentences
        might_need_split = True
        while might_need_split:
            for i, bound in enumerate(bounds):
                # Sum up subword lengths for this sentence
                sent_start = bounds[i-1] if i else 0
                sentence_cumlength = np.cumsum([len(tokens) for tokens in text_token_ids[sent_start:bound]])
                if sentence_cumlength[-1] + 2 > self.max_seq_length: # +2 for start and end tokens
                    # Use bool cast to int to find the number of words that give a sum under the limit.
                    split_candidate = (sentence_cumlength + 2 < self.max_seq_length).sum()
                    # TODO: Maybe split more intelligently such as checking whether this split candidate breaks up an entity
                    bounds.insert(i, sent_start + split_candidate)
                    break
            else:
                might_need_split = False
        return bounds

class DaNE(NERDataset):
    null_label = "O"
    labels = ("LOC", "PER", "ORG", "MISC")

    def build(self, split: Split, batch_size: int) -> DataLoader:
        self.texts, self.annotations = DDT().load_as_simple_ner(predefined_splits=True)[split.value]
        # Sadly, we do not have access to where the DaNE sentences are divided into articles, so we let each sentence be an entire text.
        sentence_boundaries = [[len(s)] for s in self.texts]
        self.examples = self._build_examples(sentence_boundaries)
        return DataLoader(list(enumerate(self.examples)), batch_size=batch_size, collate_fn=self.collate, shuffle=split==split.TRAIN)
