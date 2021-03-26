from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Union

import torch
from daluke.data import Example, Words, Entities, get_special_ids, BatchedExamples, Words, Entities

@dataclass
class MaskedBatchedExamples(BatchedExamples):
    word_mask_labels: torch.Tensor
    word_mask: torch.BoolTensor
    ent_mask_labels: torch.Tensor
    ent_mask: torch.BoolTensor

    @classmethod
    def build(cls, examples: list[Example],
            word_mask_id: int,
            ent_mask_id: int,
            word_mask_prob: float,
            ent_mask_prob: float,
        ):
        words, entities = cls.stack(examples)

        word_mask_labels, word_mask = mask_word_batch(examples.words, word_mask_prob, word_mask_id)
        ent_mask_labels, ent_mask = mask_ent_batch(examples.entities, word_mask_prob, ent_mask_id)

        return cls(words, entities, word_mask_labels, word_mask, ent_mask_labels, ent_mask)

def mask_ent_batch(ent: Entities, prob: float, mask_id: int) -> (torch.Tensor, torch.BoolTensor):
    mask = torch.zeros_like(ent.ids, dtype=torch.bool)
    # TODO: Can this be vectorized?
    to_masks = (ent.N*prob).round().long()
    for i, (n, t) in enumerate(zip(ent.N, to_masks)):
        throw = torch.multinomial(torch.ones(n), t)
        mask[i, throw] = True
    labels = torch.full_like(ent.ids, -1)
    labels[mask] = ent.ids[mask]
    ent.ids[mask] = mask_id
    return labels, mask

def mask_word_batch(w: Words, prob: float, mask_id: int) -> (torch.Tensor, torch.BoolTensor):
    raise NotImplementedError
    mask = torch.zeros_like(w.ids, dtype=torch.bool)
    # TODO: Can this be vectorized?
    for i, n in enumerate(w.N):
        mask[i, torch.randint(n, int(round(n*prob)) or 1)] = True
    labels = torch.full_like(w.ids, -1)
    labels[mask] = w.ids[mask]
    w.ids[mask] = mask_id
    return labels, mask
