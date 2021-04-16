from dataclasses import dataclass

import numpy as np
import torch

from pelutils import log, DataStorage, get_timestamp


@dataclass
class TrainResults(DataStorage):
    losses: np.ndarray    # Total loss, epochs x param updates
    w_losses: np.ndarray  # Word pred. loss, epochs x param updates
    e_losses: np.ndarray  # Entity pred. loss, epochs x param updates
    runtime: np.ndarray   # Runtime, epochs x param updates
    epoch: int

    w_accuracies: np.ndarray
    e_accuracies: np.ndarray

    subfolder = get_timestamp(for_file=True) + "_pretrain_results"
    json_name = "pretrain_results.json"
