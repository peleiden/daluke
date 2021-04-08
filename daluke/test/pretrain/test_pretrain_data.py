import os
import json

import torch
from pelutils import MainTest

from daluke.pretrain.data import load_entity_vocab, DataLoader
from daluke.data import BatchedExamples
from daluke import daBERT


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
        path = os.path.join(self.test_dir, "data.jsonl")
        ex = {
            "word_ids":     [[32, 59, 3], [42, 11]],
            "word_spans":   [[[0, 2], [2, 3], [5, 7]], [[0, 1], [1, 2]]],
            "entity_ids":   [[5], []],
            "entity_spans": [[(0, 3)], []]
        }
        with open(path, "w") as f:
            for i in range(2):
                f.write(json.dumps(
                    {k: v[i] for k, v in ex.items()}
                ) + "\n")
        dl = DataLoader(self.test_dir, {"base-model": daBERT})
        assert len(dl.examples) == 2
        assert torch.all(dl.examples[1].entities.ids == 0)
        loader = dl.get_dataloader(1, torch.utils.data.RandomSampler(dl.examples))
        i = 0
        for batch in loader:
            i += 1
            assert isinstance(batch, BatchedExamples)
        assert i == 2
