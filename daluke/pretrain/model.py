from typing import Callable

import torch
from torch import nn

from transformers import AutoTokenizer, AutoModelForPreTraining
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertPreTrainingHeads,
)

from daluke.model import DaLUKE
from daluke.pretrain.data import MaskedBatchedExamples


class PretrainTaskDaLUKE(DaLUKE):
    """
    DaLUKE for the LUKE pre-training task consisting of masked language modelling and entity masking
    """
    def __init__(self,
        bert_config: BertConfig,
        ent_vocab_size: int,
        ent_emb_size: int,
    ):
        super().__init__(bert_config, ent_vocab_size, ent_emb_size)

        self.mask_word_scorer = BertPreTrainingHeads(self.bert_config)
        self.mask_entity_scorer = EntityPreTrainingHeads(self.bert_config, self.ent_vocab_size, self.ent_emb_size)
        # FIXME: Set decoder weights

    def forward(self, ex: MaskedBatchedExamples):
        word_hidden, entity_hidden = super().forward(ex)

class EntityPreTrainingHeads(nn.Module):
    def __init__(self, bert_config: BertConfig, ent_vocab_size: int, ent_emb_size: int):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(bert_config.hidden_size, ent_emb_size),
            bert_config.hidden_act,
            nn.LayerNorm(ent_emb_size, eps=bert_config.layer_norm_eps),
        )
        self.decode = nn.Linear(ent_emb_size, ent_vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(ent_vocab_size))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decode(self.transform(hidden)) + self.bias
