import torch

from daluke.data import Words, Entities, Features

def test_words():
    words = Words.build(
        torch.LongTensor([22, 48, 2]),
        max_len=10,
    )

    assert torch.equal(words.ids, torch.LongTensor([3, 22, 48,  2,  3,  0,  0,  0,  0,  0]))
    assert torch.equal(words.segments, torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert torch.equal(words.attention_mask, torch.LongTensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))



def test_entities():
    ents = Entities.build(
        torch.LongTensor([69, 420, 42060]),
        torch.LongTensor([[0, 1, -1, -1], [2, -1, -1, -1], [-1, -1, -1, -1]]),
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
