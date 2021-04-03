from copy import deepcopy
import torch
from daluke.data import Entities, Words
from daluke.pretrain.data import mask_ent_batch, mask_word_batch

def test_entity_masking():
    ents = Entities(
        ids = torch.LongTensor([
            [42]*100,
            [69]*10+[-1]*90,
        ]),
        segments = None,
        attention_mask = None,
        N = torch.LongTensor([100, 10]),
        spans = None,
        pos = None,
    )
    _, mask = mask_ent_batch(ents, 0.1, 999)
    assert sum(mask[0]) == 10
    assert sum(mask[1]) == 1
    assert sum(ents.ids.ravel() == 999) == 11

def test_word_masking():
    w = Words(
        ids = torch.LongTensor([
            [42]*10,
            [69, 5, 60, 60, 3] + [-1] * 5,
        ]),
        segments = None,
        attention_mask = None,
        N = torch.LongTensor([10, 5]),
        spans = [
            torch.LongTensor([[0, 10]]),
            torch.LongTensor([[0, 1], [2, 4]]),
        ]
    )
    w1 = deepcopy(w)
    _, mask = mask_word_batch(
        w1, 0.5, 0.25, 0.25, (5, 29), 999,
    )
    assert sum(mask[0]) == 10     # Entire 10-long word must be masked
    assert sum(mask[1]) in (1, 2) # Either the first 1-long word or the second 2-long word must be masked
    assert sum(w1.ids.ravel() == 999) <= 12

    w2 = deepcopy(w)
    _, mask = mask_word_batch(
        w2, 1, 1, 0, (5, 29), 999,
    )
    assert sum(mask[0]) == 10     # Entire 10-long word must be masked
    assert sum(mask[1]) == 3      # Both the 1-long and 2-long are masked
    assert torch.equal(w2.ids, w.ids)        # Everything must be unmasked

    w3 = deepcopy(w)
    _, mask = mask_word_batch(
        w3, 1, 0, 1, (69, 70), 999,
    )
    assert sum(mask[0]) == 10
    assert sum(mask[1]) == 3
    assert torch.equal(w3.ids, torch.LongTensor(
        [[69, 69, 69, 69, 69, 69, 69, 69, 69, 69],
         [69,  5, 69, 69,  3, -1, -1, -1, -1, -1]]))

    w4 = deepcopy(w)
    _, mask = mask_word_batch(
        w4, 0.5, 0, 0, (0, 1), 999,
    )
    assert sum(w4.ids[0] == 999) == 10
    assert sum(w4.ids[1] == 999) in (1, 2)
