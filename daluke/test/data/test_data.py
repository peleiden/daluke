import os

import torch
from transformers import AutoTokenizer

from pelutils import MainTest

from daluke.data import Words, Entities, Example, features_from_str, load_entity_vocab
from daluke import daBERT

def test_words():
    words = Words.build(
        torch.LongTensor([22, 48, 99]),
        max_len=10,
    )
    assert torch.equal(words.ids, torch.LongTensor([2, 22, 48,  99,  3,  0,  0,  0,  0,  0]))
    assert torch.equal(words.segments, torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert torch.equal(words.attention_mask, torch.LongTensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))

def test_entities():
    ents = Entities.build(
        torch.LongTensor([69, 420, 42060]),
        [(0, 2), (2, 3), (3, 3)],
        max_len=5,
        max_mention=4,
    )
    assert torch.equal(ents.ids, torch.LongTensor([69, 420, 42060, 0, 0]))
    assert torch.equal(ents.segments, torch.LongTensor([0, 0, 0, 0, 0]))
    assert torch.equal(ents.attention_mask, torch.LongTensor([1, 1, 1, 0, 0]))
    assert torch.equal(ents.pos,
        torch.LongTensor([[ 1,  2, -1, -1],
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


class TestData(MainTest):

    def test_entity_loader(self):
        path = os.path.join(self.test_dir, "entity.jsonl")
        with open(path, "w") as ev:
            ev.write("\n".join([
                '{"id": 1271, "entities": [["27. oktober", "da"]], "count": 529}',
                '{"id": 1272, "entities": [["Computerprogram", "da"]], "count": 528}',
                '{"id": 1273, "entities": [["Kancelli", "da"]], "count": 527}',
                '{"id": 1274, "entities": [["2. marts", "da"]], "count": 527}',
                '{"id": 1275, "entities": [["Guvern\u00f8r", "da"]], "count": 527}',
            ]))
        ev = load_entity_vocab(path)
        assert ev == {
            "27. oktober":     { "id": 1271, "count": 529 },
            "Computerprogram": { "id": 1272, "count": 528 },
            "Kancelli":        { "id": 1273, "count": 527 },
            "2. marts":        { "id": 1274, "count": 527 },
            "Guvern\u00f8r":   { "id": 1275, "count": 527 },
        }


