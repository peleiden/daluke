from dataclasses import dataclass

import numpy as np
import torch

from pelutils import log, DataStorage


def gpu_usage() -> str:
    """ Logs resource usage on GPU """
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1024**3
        cache = torch.cuda.memory_reserved(0) / 1024**3
        return "VRAM usage: %4.2f GB, cache usage: %4.2f GB" % (alloc, cache)
    return ""

from pelutils.datahandler import DataStorage

@dataclass
class TrainResults(DataStorage):
    losses: np.ndarray
    epoch: int
    accumulate_step: int


    subfolder = "train_results"
