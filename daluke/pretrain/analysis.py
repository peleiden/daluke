from dataclasses import dataclass

import numpy as np

from pelutils import DataStorage

@dataclass
class TrainResults(DataStorage):
    losses: np.ndarray
