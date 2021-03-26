import torch
from daluke.data import Entities
from daluke.pretrain.data import mask_ent_batch

def test_entity_masking():
    ents = Entities(
        ids = torch.LongTensor([
            [42]*100,
            [69]*10+[-1]*90,
        ]),
        segments = None,
        attention_mask = None,
        N = torch.LongTensor([100, 10]),
        pos = None,
    )
    _, mask = mask_ent_batch(ents, 0.1, 999)
    assert sum(mask[0]) == 10
    assert sum(mask[1]) == 1
