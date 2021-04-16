from __future__ import annotations

import torch

from daluke.pretrain.data import MaskedBatchedExamples

def accuracy_from_preds(word_preds: torch.Tensor, ent_preds: torch.Tensor, batch: MaskedBatchedExamples) -> tuple[float, float]:
    w_acc = (word_preds.argmax(dim=1) == batch.word_mask_labels).sum() / word_preds.shape[0]
    e_acc = (ent_preds.argmax(dim=1) == batch.ent_mask_labels).sum() / ent_preds.shape[0]
    return w_acc.item(), e_acc.item()

