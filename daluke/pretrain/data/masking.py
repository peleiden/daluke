from dataclasses import dataclass
import random

import torch
from daluke.data import Example, Words, Entities, get_special_ids, BatchedExamples, Words, Entities

@dataclass
class MaskedBatchedExamples(BatchedExamples):
    word_masks: torch.Tensor
    entity_masks: torch.Tensor

    @classmethod
    def build(cls, features: list[Example], ent_mask_prob: float):
        # Create entity masks first as entity mask positions influence word masking
        entity_masks = torch.stack(tuple(mask_ent(f.entities, ent_mask_prob) for f in features))
        word_masks   = torch.stack(tuple())

        words, entities = cls.stack(features)
        return cls(words, entities, word_masks, entity_masks)

def mask_ent(ent: Entities, prob: float) -> torch.Tensor:
    masks = torch.full_like(ent.ids, -1)
    I = list(range(ent.N))
    random.shuffle(I)
    for i in I[max(1, int(round(ent.N*prob)))]:
        masks[i] = ent.ids[i]
    return masks

def mask_words(words: Words) -> torch.Tensor:
    masks = torch.full_like(words.ids, -1)
    raise NotImplementedError


