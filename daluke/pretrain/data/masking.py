from __future__ import annotations
from dataclasses import dataclass
import ctypes

from pelutils import TT
import torch

from daluke.data import Example, Words, Entities, BatchedExamples, Words, Entities


try:
    _lib = ctypes.CDLL("so/masking.so")
    _lib_avail = True
except OSError:
    _lib_avail = False

def _c_ptr(arr: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(arr.data_ptr())

@dataclass
class MaskedBatchedExamples(BatchedExamples):
    word_mask_labels: torch.LongTensor
    word_mask: torch.BoolTensor
    ent_mask_labels: torch.LongTensor
    ent_mask: torch.BoolTensor

    @classmethod
    def build(
        cls,
        examples: list[Example],
        device:   torch.device,
        word_mask_id: int,
        ent_mask_id: int,
        word_mask_prob: float,
        word_unmask_prob: float,
        word_randword_prob: float,
        word_id_range: tuple[int],
        ent_mask_prob: float,
        cut_extra_padding: bool = True,
    ):
        with TT.profile("Combine to batch"):
            words, entities = cls.collate(examples, device=torch.device("cpu"), cut=cut_extra_padding)
        with TT.profile("Mask words"):
            word_mask_labels, word_mask = mask_word_batch(words, word_mask_prob, word_unmask_prob, word_randword_prob, word_id_range, word_mask_id)
        with TT.profile("Mask entities"):
            ent_mask_labels, ent_mask = mask_ent_batch(entities, ent_mask_prob, ent_mask_id)
        with TT.profile("Send to %s" % device):
            words.ids               = words.ids.to(device)
            words.attention_mask    = words.attention_mask.to(device)
            words.N                 = words.N.to(device)
            entities.ids            = entities.ids.to(device)
            entities.attention_mask = entities.attention_mask.to(device)
            entities.pos            = entities.pos.to(device)
            entities.N              = entities.N.to(device)
            word_mask_labels        = word_mask_labels.to(device)
            word_mask               = word_mask.to(device)
            ent_mask_labels         = ent_mask_labels.to(device)
            ent_mask                = ent_mask.to(device)
        return cls(words, entities, word_mask_labels, word_mask, ent_mask_labels, ent_mask)

def mask_ent_batch(ent: Entities, prob: float, mask_id: int) -> tuple[torch.Tensor, torch.BoolTensor]:
    mask = torch.zeros_like(ent.ids, dtype=torch.bool)
    # TODO: Can this be vectorized?
    to_masks = (ent.N*prob).round().int()
    for i, (n, t) in enumerate(zip(ent.N, to_masks)):
        if not n: continue
        throw = torch.multinomial(torch.ones(n), t or 1)
        mask[i, throw] = True

    labels = ent.ids[mask].long()  # Labels should be longs for criterion
    ent.ids[mask] = mask_id
    return labels, mask

def mask_word_batch(
    wo: Words,
    prob: float,
    unmask_prob: float,
    randword_prob: float,
    word_id_range: tuple[int],
    mask_id: int,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    if not _lib_avail:
        raise ModuleNotFoundError("masking.so needed for pretraining not found. Compile using `make`")
    mask = torch.zeros_like(wo.ids, dtype=torch.bool)
    labels = torch.empty(wo.ids.numel(), dtype=torch.int32)
    n_spans = torch.IntTensor([len(span) for span in wo.spans])

    max_num_spans = n_spans.max().item()
    spans = torch.empty(len(wo.ids), max_num_spans, 2, dtype=torch.int32)
    for i, span in enumerate(wo.spans):
        spans[i, :len(span)] = torch.IntTensor(span)

    num_labels = _lib.mask_words(
        ctypes.c_int(mask_id),
        ctypes.c_int(word_id_range[0]),
        ctypes.c_int(word_id_range[1]),
        ctypes.c_double(prob),
        ctypes.c_double(unmask_prob),
        ctypes.c_double(randword_prob),
        ctypes.c_int(wo.ids.shape[0]),
        ctypes.c_int(wo.ids.shape[1]),
        ctypes.c_int(max_num_spans),
        _c_ptr(wo.ids),
        _c_ptr(mask),
        _c_ptr(spans),
        _c_ptr(n_spans),
        _c_ptr(labels),
    )
    labels = labels[:num_labels].long()

    return labels, mask
