""" https://pytorch.org/hub/pytorch_fairseq_roberta/ """
import os
from typing import Type

from pelutils import log

import torch
import torch.hub
import torch.nn as nn

from daluke import cuda


FNAME = "roberta.pt"

def load(loc: str) -> Type[nn.Module]:
    path = os.path.join(loc, FNAME)
    log.debug("Loading RoBERTa from %s" % path)
    return torch.load(path, map_location=cuda)


def download(loc: str):
    os.makedirs(loc, exist_ok=True)
    path = os.path.join(loc, FNAME)
    log.debug("Saving RoBERTa to %s" % path)
    roberta = torch.hub.load("pytorch/fairseq", "roberta.large")
    torch.save(roberta.state_dict(), path)


if __name__ == "__main__":
    download("local_workspace")
