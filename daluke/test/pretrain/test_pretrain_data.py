import os
import json

import torch
from pelutils import MainTest

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
                '{ "word_ids": [42, 11], "word_spans": [[0, 1], [1, 2]], "entity_ids": [], "entity_spans": [] }',
            ]))
        metadata = {
            "number_of_items": 2,
            "max_seq_length": 512,
            "max_entity_length": 128,
            "max_mention_length": 30,
            "min_sentence_length": 5,
            "base-model": "Maltehb/danish-bert-botxo",
            "tokenizer_class": "BertTokenizerFast",
            "language": "da",
        }
        dl = DataLoader(self.test_dir, metadata)
        assert len(dl.examples) == 2
        assert torch.all(dl.examples[1].entities.ids == 0)
        loader = dl.get_dataloader(1, torch.utils.data.RandomSampler(dl.examples))
        i = 0
        for batch in loader:
            i += 1
            assert isinstance(batch, BatchedExamples)
        assert i == 2

    def test_word_spans(self):
        tokens = ["jeg", "hed", "##der", "kaj", "ii", "d", ".", "Sto", "##re"]
        word_spans = [(0, 1), (1, 3), (3, 4), (4, 5), (5, 6), (7, 9)]
        assert calculate_spans(tokens) == word_spans
