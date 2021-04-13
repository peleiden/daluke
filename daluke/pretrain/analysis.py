from dataclasses import dataclass

import numpy as np
import torch

from pelutils import log, DataStorage


@dataclass
class TrainResults(DataStorage):
    losses: np.ndarray
    epoch: int
    accumulate_step: int


    subfolder = "train_results"
