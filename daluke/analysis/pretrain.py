from __future__ import annotations
from dataclasses import dataclass
import heapq

import numpy as np
import torch

from pelutils import log, DataStorage, get_timestamp
from pelutils.ds import no_grad


@dataclass
class TrainResults(DataStorage):
    losses:       np.ndarray  # Total loss, epochs x param updates
    w_losses:     np.ndarray  # Word pred. loss, epochs x param updates
    e_losses:     np.ndarray  # Entity pred. loss, epochs x param updates
    scaled_loss:  np.ndarray  # Scaled loss, epochs x param updates. Only zeros if not using amp
    runtime:      np.ndarray  # Runtime, epochs x param updates
    lr:           np.ndarray  # Learning rate, epochs x param updates
    epoch:        int

    top_k:        list[int]   # Which accuracies to save, e.g. [1, 5, 10, 50]
    w_accuracies: np.ndarray  # Masked word pred. accuracy, epochs x param updates x len(top_k)
    e_accuracies: np.ndarray  # Masked ent. pred. accuracy, epochs x param updates x len(top_k)

    orig_params:  np.ndarray  # Array of all parameters in original model
    param_diff_1: np.ndarray  # 1-norm distance to original parameters, epochs x param updates
    param_diff_2: np.ndarray  # 2-norm distance to original parameters, epochs x param updates

    # Weights that do not exist in base model. Keys in state_dict
    luke_exclusive_params: set[str]
    # Query matrices that do not exist in base model but have been set from it. Subset of luke_exclusive_params
    q_mats_from_base:      set[str]

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



