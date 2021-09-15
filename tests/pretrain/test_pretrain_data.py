import os

import torch
from pelutils import MainTest

from daluke import daBERT
from daluke.data import BatchedExamples
from daluke.pretrain.data import load_entity_vocab, DataLoader, calculate_spans
from daluke.pretrain.data.build import DatasetBuilder


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
        path = os.path.join(self.test_dir, DatasetBuilder.data_file)
        with open(path, "w") as f:
            f.write("\n".join([
                '{ "word_ids": [32, 59, 3], "word_spans": [[0, 2], [2, 3], [5, 7]], "entity_ids": [5], "entity_spans": [[0, 3]] }',
                '{ "word_ids": [42, 11], "word_spans": [[0, 1], [1, 2]], "entity_ids": [], "entity_spans": [], "is_validation": true }',
            ]))
        metadata = {
            "number_of_items": 2,
            "max-seq-length": 512,
            "max-entities": 128,
            "max-entity-span": 30,
            "min-sentence-length": 5,
            "base-model": daBERT,
            "tokenizer_class": "BertTokenizerFast",
            "language": "da",
        }
        dl = DataLoader(
            self.test_dir,
            metadata,
            entity_vocab       = {"[MASK]": dict(id=2)},
            device             = torch.device("cpu"),
            word_mask_prob     = 0.1,
            word_unmask_prob   = 0.1,
            word_randword_prob = 0.1,
            ent_mask_prob      = 0.1,
        )
        assert len(dl) == 2
        assert len(dl.train_examples) == 1
        assert len(dl.val_examples) == 1
        assert torch.all(dl.val_examples[0].entities.ids == 0)
        train_loader = dl.get_dataloader(1, torch.utils.data.RandomSampler(dl.train_examples))
        i = 0
        for batch in train_loader:
            i += 1
            assert isinstance(batch, BatchedExamples)
        assert i == 1
        val_loader = dl.get_dataloader(1, torch.utils.data.RandomSampler(dl.train_examples), validation=True)
        i = 0
        for batch in val_loader:
            i += 1
            assert isinstance(batch, BatchedExamples)
        assert i == 1


    def test_word_spans(self):
        tokens = ["jeg", "hed", "##der", "kaj", "ii", "d", ".", "Sto", "##re"]
        word_spans = [(0, 1), (1, 3), (3, 4), (4, 5), (5, 6), (7, 9)]
        assert calculate_spans(tokens) == word_spans
