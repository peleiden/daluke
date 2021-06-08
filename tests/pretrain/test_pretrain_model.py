import torch
from transformers import AutoConfig

from daluke import daBERT
from daluke.data import Example, Words, Entities
from daluke.pretrain.data import MaskedBatchedExamples
from daluke.pretrain.model import PretrainTaskDaLUKE

def test_forward_pass():
    data = MaskedBatchedExamples.build([
        Example(
            Words.build(torch.IntTensor([232, 13, 13, 21, 5, 1]), spans=[[0,1], [1, 3], [5, 6]]),
            Entities.build(torch.IntTensor([5]), [[1,3]], 128, 30),
        ),
        Example(
            Words.build(torch.IntTensor([29, 28, 5000, 22, 11, 55, 1, 1, 1, 1]), spans=[[0, 1], [1, 2], [2, 5], [5, 10]]),
            Entities.build(torch.IntTensor([11, 5]), [[0, 2], [2, 10]], 128, 30),
        ),
        ],
        torch.device("cpu"),
        42,
        11,
        0.1,
        0.1,
        0.1,
        (5, 42),
        0.1,
    )
    cfg = AutoConfig.from_pretrained(daBERT)
    model = PretrainTaskDaLUKE(cfg, 66, 79)
    word_scores, ent_scores = model(data)
    assert word_scores.shape[0] >= 2
    assert ent_scores.shape[0] == 2

    assert word_scores.shape[1] == 32_000
    assert ent_scores.shape[1] == 66
