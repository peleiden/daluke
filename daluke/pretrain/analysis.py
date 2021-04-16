from dataclasses import dataclass

import numpy as np
import torch

from pelutils import log, DataStorage, get_timestamp


@dataclass
class TrainResults(DataStorage):
    losses: np.ndarray
    # w_losses: np.ndarray  # TODO
    # e_losses: np.ndarray
    epoch: int

    subfolder = get_timestamp(for_file=True) + "_pretrain_results"
