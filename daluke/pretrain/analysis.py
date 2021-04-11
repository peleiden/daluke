from dataclasses import dataclass

import numpy as np

from pelutils.datahandling import DataStorage

@dataclass
class TrainResults(DataStorage):
    losses: np.ndarray
    epoch: int
    accumulate_step: int
