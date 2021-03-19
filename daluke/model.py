import numpy as np
import torch
from torch import nn

class DaLUKE(nn.Module):
    ent_emb_size = 256
    hidden_size  = 768

    def __init__(self,
        ent_vocab_size: int,

    ):
        self.ent_embeds = nn.Embedding(ent_vocab_size, self.ent_emb_size, padding_idx=0)


