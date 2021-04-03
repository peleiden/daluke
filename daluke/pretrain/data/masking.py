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
            word_unmask_prob: float,
            word_randword_prob: float,
            word_id_range: tuple[int],
            ent_mask_prob: float,
        ):
        words, entities = cls.stack(examples)

        word_mask_labels, word_mask = mask_word_batch(examples.words, word_mask_prob, word_unmask_prob, word_randword_prob, word_id_range, word_mask_id)
        ent_mask_labels, ent_mask = mask_ent_batch(examples.entities, ent_mask_prob, ent_mask_id)

        return cls(words, entities, word_mask_labels, word_mask, ent_mask_labels, ent_mask)

def mask_ent_batch(ent: Entities, prob: float, mask_id: int) -> (torch.Tensor, torch.BoolTensor):
    mask = torch.zeros_like(ent.ids, dtype=torch.bool)
    # TODO: Can this be vectorized?
    to_masks = (ent.N*prob).round().long()
    for i, (n, t) in enumerate(zip(ent.N, to_masks)):
        throw = torch.multinomial(torch.ones(n), t or 1)
        mask[i, throw] = True

    labels = torch.full_like(ent.ids, -1)
    labels[mask] = ent.ids[mask]
    ent.ids[mask] = mask_id
    return labels, mask

def mask_word_batch(
        w: Words,
        prob: float,
        unmask_prob: float,
        randword_prob: float,
        word_id_range: tuple[int],
        mask_id: int,
        ) -> (torch.Tensor, torch.BoolTensor):
    mask = torch.zeros_like(w.ids, dtype=torch.bool)
    unmask_mask, randword_mask = torch.zeros_like(mask), torch.zeros_like(mask)
    masking_throws = torch.rand(w.ids.shape)

    # TODO: Can part of the below lookup be vectorized?
    for i, _ in enumerate(w.N):
        # Look up in the word spans as we do whole-word masking
        n = w.spans[i].shape[0]
        to_mask = w.spans[i][torch.multinomial(torch.ones(n), int(round(n*prob)) or 1)]
        for (start, end), p in zip(to_mask, masking_throws[i]):
            mask[i, start:end] = True
            if p < unmask_prob:
                unmask_mask[i, start:end] = True
            elif p > (1 - randword_prob):
                randword_mask[i, start:end] = True

    labels = torch.full_like(w.ids, -1)
    labels[mask] = w.ids[mask]
    w.ids[mask & ~unmask_mask] = mask_id
    randwords = torch.randint_like(w.ids[randword_mask], *word_id_range)
    w.ids[randword_mask] = randwords

    # TODO: Remove this check when we are convinced that this does not happen, or we resolve it n some way
    for wid, m in zip(w.ids, mask):
        if torch.all(~m):
            raise RuntimeError(f"No word could be selected for masking for example {wid}")
    return labels, mask
