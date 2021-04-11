from dataclasses import dataclass

import numpy as np

from pelutils.datahandler import DataStorage

@dataclass
class TrainResults(DataStorage):
    losses: np.ndarray
    epoch: int
    accumulate_step: int


    subfolder = "train_results"
