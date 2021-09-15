import torch
from transformers import AutoTokenizer
from daluke import example_from_str, masked_example_from_str, ner_example_from_str, daBERT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DummyDaLUKE:
    metadata = {
        "number-of-items": 122,
        "number-of-entities": 1336,
        "max-seq-length": 512,
        "max-entities": 128,
        "max-entity-span": 30,
        "min-sentence-length": 5,
        "base-model": daBERT,
        "tokenizer-class": "BertTokenizerFast",
        "language": "da"
    }
    entity_vocab = {
    "[PAD]": { "id": 0, "count": 0 },
    "[UNK]": { "id": 1, "count": 0 },
    "[MASK]": { "id": 2, "count": 0 },
    "Danmark": { "id": 3, "count": 27787 },
    "USA": { "id": 4, "count": 25768 },
    }
    tokenizer = AutoTokenizer.from_pretrained(daBERT)

def test_examples_from_str():
    res = example_from_str("Hej med dig, kommer du fra Danmark eller Norge?", [(6, 7)], DummyDaLUKE())
    assert torch.all(res.words.ids == torch.IntTensor([[   2, 2175,   61,  224,  911,  492,   86,  145,  739,  151, 5669, 3766, 3]]).to(device))
    assert torch.all(res.entities.ids == torch.IntTensor([[3]]).to(device))
    assert res.entities.spans == [[(8,9)]]

    res = example_from_str("USA USA USA USA", [(0, 1), (1, 2), (2, 3), (3,4)], DummyDaLUKE())
    assert torch.all(res.words.ids == torch.IntTensor([[2, 2493, 2493, 2493, 2493, 3]]).to(device))
    assert torch.all(res.entities.ids == torch.IntTensor([[4, 4, 4, 4]]).to(device))
    assert res.entities.spans == [[(1,2), (2, 3), (3, 4), (4, 5)]]

    res = example_from_str("Goddaw, jeg har ingen entititer", [], DummyDaLUKE())
    assert torch.all(res.entities.ids == torch.IntTensor([[]]).to(device))
    assert res.entities.spans == [[]]

def test_masked_example_from_str():
    res = masked_example_from_str("Hej med [MASK]", [], DummyDaLUKE())
    assert torch.all(res.word_mask == torch.BoolTensor([False, False, False, True, False]).to(device))

    res = masked_example_from_str("USA [MASK]", [(0,1), (1,2)], DummyDaLUKE())
    assert torch.all(res.word_mask == torch.BoolTensor([False, False, True, False]).to(device))
    assert torch.all(res.entities.ids == torch.IntTensor([4, 2]).to(device))

def test_ner_example_from_str():
    res = ner_example_from_str("Hej med dig", DummyDaLUKE())
    assert res.entities.fullword_spans == [[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]]
    assert torch.all(res.entities.ids == torch.IntTensor([1,1,1,1,1,1]).to(device))
