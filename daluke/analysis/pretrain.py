from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from pelutils import DataStorage
from pelutils.ds import no_grad

from daluke.pretrain.model import PretrainTaskDaLUKE

@dataclass
class TrainResults(DataStorage):
    parameter_update:  int         # 0-indexed completed parameter updates
    runtime:           np.ndarray  # Runtime by pu
    lr:                np.ndarray  # Learning rate by pu

    losses:            np.ndarray  # Total loss by pu
    scaled_loss:       np.ndarray  # Scaled loss by pu. Zeros only if not using AMP

    top_k:             list[int]   # Which accuracies to save, e.g. [1, 5, 10, 50]
    w_losses:          np.ndarray  # Word pred. loss by pu
    e_losses:          np.ndarray  # Entity pred. lossby pu
    w_accuracies:      np.ndarray  # Masked word pred. accuracy, param updates x len(top_k)
    e_accuracies:      np.ndarray  # Masked ent. pred. accuracy, param updates x len(top_k)

    val_param_updates: np.ndarray  # What param updates the validation set is tested
    val_losses:        np.ndarray  # Validation: Total loss, len(val_param_updates)
    val_w_losses:      np.ndarray  # Validation: Word pred. loss
    val_e_losses:      np.ndarray  # Validation: Entity pred. loss
    val_w_accuracies:  np.ndarray  # Validation: Masked word pred. accuracy, len(val_param_updates) x len(top_k)
    val_e_accuracies:  np.ndarray  # Validation: Masked ent. pred. accuracy, len(val_param_updates) x len(top_k)

    orig_params:       np.ndarray  # Array of all parameters in original model
    param_diff_1:      np.ndarray  # 1-norm distance to original parameters by pu
    param_diff_2:      np.ndarray  # 2-norm distance to original parameters by pu

    # Weights that do not exist in base model. Keys in state_dict
    luke_exclusive_params: set[str]
    # Some attention matrices do not exist in base model but have been set from it. Subset of luke_exclusive_params
    att_mats_from_base:    set[str]

    subfolder = None  # Set at runtime
    json_name = "pretrain_results.json"
    ignore_missing = True

    def __post_init__(self):
        assert self.top_k == sorted(self.top_k), "Top k accuracy list must be monotonically increasing"


@no_grad
def top_k_accuracy(
    labels: torch.Tensor,
    scores: torch.Tensor,
    top_k: list[int],
) -> np.ndarray:
    """ Calculate top k accuracies for given predictions """
    largest_k = top_k[-1]
    idcs = torch.arange(len(labels))
    k_scores = np.zeros(len(top_k))
    for k in range(largest_k):
        argmax = scores.argmax(dim=1)
        trues = argmax == labels
        k_scores[[i for i, k_ in enumerate(top_k) if k < k_]] += trues.sum().item()
        scores[idcs, argmax] = -float("inf")

    return k_scores / len(labels)


@no_grad
def validate_model(
    model: PretrainTaskDaLUKE,
    data: torch.utils.data.DataLoader,
    word_criterion: Callable,
    entity_criterion: Callable,
    top_k: list[int]
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    w_losses, e_losses = list(), list()
    w_accuracies, e_accuracies = list(), list()
    for batch in data:
        word_preds, ent_preds = model(batch)
        w_losses.append(float(word_criterion(word_preds, batch.word_mask_labels)))
        e_losses.append(float(entity_criterion(ent_preds, batch.ent_mask_labels)))
        w_accuracies.append(
            top_k_accuracy(batch.word_mask_labels, word_preds, top_k)
        )
        e_accuracies.append(
            top_k_accuracy(batch.ent_mask_labels, ent_preds, top_k)
        )
    # Mean over batch axis
    w_accuracies = np.mean(np.vstack(w_accuracies), axis=0)
    e_accuracies = np.mean(np.vstack(e_accuracies), axis=0)
    return np.mean(w_losses), np.mean(e_losses), w_accuracies, e_accuracies
