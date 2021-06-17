import torch
from daluke import example_from_str, masked_example_from_str, ner_example_from_str


METADATA = {
    "number-of-items": 122,
    "number-of-entities": 1336,
    "max-seq-length": 512,
    "max-entities": 128,
    "max-entity-span": 30,
    "min-sentence-length": 5,
    "base-model": "Maltehb/danish-bert-botxo",
    "tokenizer-class": "BertTokenizerFast",
    "language": "da"
}

ENTITY_VOCAB = {
  "[PAD]": { "id": 0, "count": 0 },
  "[UNK]": { "id": 1, "count": 0 },
  "[MASK]": { "id": 2, "count": 0 },
  "Danmark": { "id": 3, "count": 27787 },
  "USA": { "id": 4, "count": 25768 },
}

def test_examples_from_str():
    res = example_from_str("Hej med dig, kommer du fra Danmark eller Norge?", [(6, 7)], ENTITY_VOCAB, METADATA)
    assert torch.all(res.words.ids == torch.IntTensor([[   2, 2175,   61,  224,  911,  492,   86,  145,  739,  151, 5669, 3766, 3]]))
    assert torch.all(res.entities.ids == torch.IntTensor([[3]]))
    assert res.entities.spans == [[(8,9)]]

    res = example_from_str("USA USA USA USA", [(0, 1), (1, 2), (2, 3), (3,4)], ENTITY_VOCAB, METADATA)
    assert torch.all(res.words.ids == torch.IntTensor([[2, 2493, 2493, 2493, 2493, 3]]))
    assert torch.all(res.entities.ids == torch.IntTensor([[4, 4, 4, 4]]))
    assert res.entities.spans == [[(1,2), (2, 3), (3, 4), (4, 5)]]

    res = example_from_str("Goddaw, jeg har ingen entititer", [], ENTITY_VOCAB, METADATA)
    assert torch.all(res.entities.ids == torch.IntTensor([[]]))
    assert res.entities.spans == [[]]

def test_masked_example_from_str():
    res = masked_example_from_str("Hej med [MASK]", [], ENTITY_VOCAB, METADATA)
    assert torch.all(res.word_mask == torch.BoolTensor([False, False, False, True, False]))

    res = masked_example_from_str("USA [MASK]", [(0,1), (1,2)], ENTITY_VOCAB, METADATA)
    assert torch.all(res.word_mask == torch.BoolTensor([False, False, True, False]))
    assert torch.all(res.entities.ids == torch.IntTensor([4, 2]))

def test_ner_example_from_str():
    res = ner_example_from_str("Hej med dig", METADATA)
    assert res.entities.fullword_spans == [[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]]
    assert torch.all(res.entities.ids == torch.IntTensor([1,1,1,1,1,1]))
