""" https://pytorch.org/hub/pytorch_fairseq_roberta/ """
import os
from typing import Type

from pelutils import log

import torch
import torch.hub

from daluke import cuda


FNAME = "roberta.pt"

def load(*, large=True, force_reload=False):
    model = "large" if large else "base"
    roberta = torch.hub.load("pytorch/fairseq", "roberta.%s" % model)
    roberta.eval()
    return roberta


if __name__ == "__main__":
    roberta = load(large = True, force_reload=True)
    print(type(roberta))
    s = "$GME 🚀🚀🚀"
    tokens = roberta.encode(s)
    print(tokens)
