from dataclasses import dataclass

import torch

@dataclass
class Words:
    ids: torch.Tensor
    pos: torch.Tensor
    attention_mask: torch.Tensor

@dataclass
class Entities(Words):
    types: torch.Tensor

@dataclass
class Example:
    """
    Data to be forward passed to daLUKE
    """
    words: Words
    entities: Entities
