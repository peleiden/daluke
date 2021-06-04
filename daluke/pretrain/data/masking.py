from __future__ import annotations
from dataclasses import dataclass

import torch
from daluke.data import Example, Words, Entities, BatchedExamples, Words, Entities

@dataclass
class MaskedBatchedExamples(BatchedExamples):
    word_mask_labels: torch.LongTensor
    word_mask: torch.BoolTensor
    ent_mask_labels: torch.LongTensor
    ent_mask: torch.BoolTensor

    @classmethod
    def build(
        cls,
        examples: list[Example],
        device:   torch.device,
        word_mask_id: int,
        ent_mask_id: int,
        word_mask_prob: float,
        word_unmask_prob: float,
        word_randword_prob: float,
        word_id_range: tuple[int],
        ent_mask_prob: float,
        cut_extra_padding: bool=True,
    ):
        words, entities = cls.collate(examples, device=device, cut=cut_extra_padding)
        word_mask_labels, word_mask = mask_word_batch(words, word_mask_prob, word_unmask_prob, word_randword_prob, word_id_range, word_mask_id)
        ent_mask_labels, ent_mask = mask_ent_batch(entities, ent_mask_prob, ent_mask_id)
        return cls(words, entities, word_mask_labels, word_mask, ent_mask_labels, ent_mask)

def mask_ent_batch(ent: Entities, prob: float, mask_id: int) -> (torch.Tensor, torch.BoolTensor):
    mask = torch.zeros_like(ent.ids, dtype=torch.bool)
    # TODO: Can this be vectorized?
    to_masks = (ent.N*prob).round().int()
    for i, (n, t) in enumerate(zip(ent.N, to_masks)):
        if not n: continue
        throw = torch.multinomial(torch.ones(n), t or 1)
        mask[i, throw] = True

    labels = ent.ids[mask].long() # Labels should be longs for criterion
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

    labels = w.ids[mask].long() # Labels should be 64-bit for criterion
    w.ids[mask & ~unmask_mask] = mask_id # Take unmasking into account
    w.ids[randword_mask] = torch.randint_like(w.ids[randword_mask], *word_id_range)

    return labels, mask
