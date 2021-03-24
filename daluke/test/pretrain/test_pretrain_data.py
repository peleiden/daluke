import os
import json

from pelutils import MainTest

from daluke.pretrain.data import load_entity_vocab, DataLoader

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

    def test_dataloader(self):
        path = os.path.join(self.test_dir, "data.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "word_ids":     [[32, 59, 3], [42, 11]],
                    "entity_ids":   [[5], []],
                    "entity_spans": [[(0, 3)], []]
                }, f
            )
        dl = DataLoader(path)



