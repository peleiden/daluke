import torch
from transformers import AutoTokenizer

from daluke.data import Words, Entities, Example
from . import features_from_str
from daluke import daBERT

def test_words():
    words = Words.build(
        torch.IntTensor([22, 48, 99]),
        max_len=10,
    )
    assert torch.equal(words.ids, torch.IntTensor([2, 22, 48,  99,  3,  0,  0,  0,  0,  0]))
    assert torch.equal(words.attention_mask, torch.IntTensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))

def test_entities():
    ents = Entities.build(
        torch.IntTensor([69, 420, 42060]),
        [(0, 2), (2, 3), (3, 3)],
        max_entities=5,
        max_entity_span=4,
    )
    assert torch.equal(ents.ids, torch.IntTensor([69, 420, 42060, 0, 0]))
    assert torch.equal(ents.attention_mask, torch.IntTensor([1, 1, 1, 0, 0]))
    assert torch.equal(ents.pos,
        torch.IntTensor([[ 1,  2, -1, -1],
            [ 3, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1]]
        )
    )

def test_create_features():
    ent_vocab = {"[UNK]": 1, "Danmark": 42}
    res = features_from_str("Jeg hedder Jens Nielsen og er fra Danmark".split(), [(2,4), (7, 8)], ent_vocab, AutoTokenizer.from_pretrained(daBERT))
    assert isinstance(res, Example)
    assert res.words.ids[2].item() == 2567
    assert sum(res.words.attention_mask) == 10
    assert res.entities.ids[1].item() == 42
    assert 3 in res.entities.pos[0]
    assert 4 in res.entities.pos[0]
    assert 8 in res.entities.pos[1]
