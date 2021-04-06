from typing import Callable
from transformers.activations import get_activation

import torch
from torch import nn

from transformers import AutoTokenizer, AutoModelForPreTraining
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertLMPredictionHead,
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

        self.mask_word_scorer = BertLMPredictionHead(bert_config)
        self.mask_entity_scorer = EntityPreTrainingHeads(bert_config, ent_vocab_size, self.ent_emb_size)

        # Needed for reshaping in forward pass
        self.hiddsize, self.wsize, self.esize = bert_config.hidden_size, bert_config.vocab_size, ent_vocab_size

        self.mask_entity_scorer.decode.weight = self.entity_embeddings.ent_embeds.weight # TODO: Understand

    def forward(self, ex: MaskedBatchedExamples):
        word_hidden, ent_hidden = super().forward(ex)

        word_hidden_masked = word_hidden[ex.word_mask].view(-1, self.hiddsize)
        word_scores = self.mask_word_scorer(word_hidden_masked).view(-1, self.wsize)

        ent_hidden_masked = ent_hidden[ex.ent_mask].view(-1, self.hiddsize)
        ent_scores = self.mask_entity_scorer(ent_hidden_masked).view(-1, self.esize)

        return word_scores, ent_scores


class EntityPreTrainingHeads(nn.Module):
    def __init__(self, bert_config: BertConfig, ent_vocab_size: int, ent_emb_size: int):
        super().__init__()
        self.transform = nn.Linear(bert_config.hidden_size, ent_emb_size)
        self.act = get_activation(bert_config.hidden_act)
        self.lnorm = nn.LayerNorm(ent_emb_size, eps=bert_config.layer_norm_eps)

        self.decode = nn.Linear(ent_emb_size, ent_vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(ent_vocab_size))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decode(self.lnorm(self.act(self.transform(hidden)))) + self.bias
