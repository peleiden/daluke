from __future__ import annotations
from collections import OrderedDict
from pprint import pformat
from typing import Callable

import torch
from torch import nn

from pelutils import log
from transformers import AutoTokenizer, AutoModelForPreTraining
from transformers.activations import get_activation
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertLMPredictionHead,
    BertEncoder,
)

from daluke.model import DaLUKE
from daluke.pretrain.data import MaskedBatchedExamples


class PretrainTaskDaLUKE(DaLUKE):
    """
    DaLUKE for the LUKE pre-training task consisting of masked language modelling and entity masking
    """
    def __init__(
        self,
        bert_config: BertConfig,
        ent_vocab_size: int,
        ent_embed_size: int,
    ):
        super().__init__(bert_config, ent_vocab_size, ent_embed_size)

        self.mask_word_scorer = BertLMPredictionHead(bert_config)
        self.mask_entity_scorer = EntityPreTrainingHeads(bert_config, ent_vocab_size, self.ent_embed_size)

        # Needed for reshaping in forward pass
        self.hiddsize, self.wsize, self.esize = bert_config.hidden_size, bert_config.vocab_size, ent_vocab_size

        self.mask_entity_scorer.decode.weight = self.entity_embeddings.ent_embeds.weight  # TODO: Understand

    def forward(self, ex: MaskedBatchedExamples):
        word_hidden, ent_hidden = super().forward(ex)

        word_hidden_masked = word_hidden[ex.word_mask].view(-1, self.hiddsize)
        word_scores = self.mask_word_scorer(word_hidden_masked).view(-1, self.wsize)

        ent_hidden_masked = ent_hidden[ex.ent_mask].view(-1, self.hiddsize)
        ent_scores = self.mask_entity_scorer(ent_hidden_masked).view(-1, self.esize)

        return word_scores, ent_scores

class EntityPreTrainingHeads(nn.Module):
    def __init__(self, bert_config: BertConfig, ent_vocab_size: int, ent_embed_size: int):
        super().__init__()
        self.transform = nn.Linear(bert_config.hidden_size, ent_embed_size)
        self.act = get_activation(bert_config.hidden_act)
        self.lnorm = nn.LayerNorm(ent_embed_size, eps=bert_config.layer_norm_eps)

        self.decode = nn.Linear(ent_embed_size, ent_vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(ent_vocab_size))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decode(self.lnorm(self.act(self.transform(hidden)))) + self.bias

class BertAttentionPretrainTaskDaLUKE(PretrainTaskDaLUKE):
    """
    DaLUKE using the normal attention from BERT instead of the Entity-Aware Attention.
    """
    # TODO: Load bert weights
    def __init__(self,
        bert_config: BertConfig,
        ent_vocab_size: int,
        ent_embed_size: int,
    ):
        super().__init__(bert_config, ent_vocab_size, ent_embed_size)

        self.encoder = BertEncoder(bert_config)

    def forward(self, ex: MaskedBatchedExamples) -> tuple[torch.Tensor, torch.Tensor]:
        word_size = ex.words.ids.size(1)

        # Exactly same as in DaLUKE
        word_hidden    = self.word_embeddings(ex.words.ids)
        entity_hidden  = self.entity_embeddings(ex.entities.ids, ex.entities.pos)
        attention_mask = torch.cat((ex.words.attention_mask, ex.entities.attention_mask), dim=1).unsqueeze(1).unsqueeze(2)
        attention_mask = 10_000.0 * (attention_mask - 1.0)

        # BERT encoder
        encodings = self.encoder(torch.cat([word_hidden, entity_hidden], dim=1), attention_mask)
        word_hidden, ent_hidden = encodings[0][:, :word_size, :], encodings[0][:, word_size:, :]

        # Exactly same as in PretrainTaskDaLUKE
        word_hidden_masked = word_hidden[ex.word_mask].view(-1, self.hiddsize)
        word_scores = self.mask_word_scorer(word_hidden_masked).view(-1, self.wsize)
        ent_hidden_masked = ent_hidden[ex.ent_mask].view(-1, self.hiddsize)
        ent_scores = self.mask_entity_scorer(ent_hidden_masked).view(-1, self.esize)

        return word_scores, ent_scores

def load_base_model_weights(daluke: PretrainTaskDaLUKE, base_model_state_dict: OrderedDict, bert_attention: bool) -> set:
    """
    Load a base model into this model. Assumes BERT for now
    Returns the set of keys that were not tansfered from base model
    """

    # Mappings from bert to daLUKE naming scheme
    one2one_map = {
        "embeddings": "word_embeddings",
        "query": "Q_w",
        "key": "K",
        "value": "V",
    } if not bert_attention else {
        "embeddings": "word_embeddings",
    }
    multipart_map = {
        "attention.self": "attention",
        "attention.output": "self_output",
        "layer": "",
        "cls.predictions": "mask_word_scorer",
    } if not bert_attention else {
        "cls.predictions": "mask_word_scorer",
    }
    # Be safe in case some parts are subsets of others
    multipart_map = { ".%s." % key: ".%s." % value if value else "." for key, value in multipart_map.items() }
    # Remove these prefixes
    base_model_prefixes = ("bert.", "roberta.")

    state_dict = base_model_state_dict.copy()
    # Remove base model naming from keys
    for bert_key in tuple(state_dict):
        # daluke_key is the key for this model and will be changed to fit naming scheme
        # Add extra dot to make the maps work
        daluke_key = "." + bert_key

        # Replace multilevel diffs
        for bert_part in multipart_map:
            if bert_part in daluke_key:
                daluke_key = daluke_key.replace(bert_part, multipart_map[bert_part])

        # Replace base model prefix if it matches any
        for prefix in base_model_prefixes:
            if daluke_key.startswith("." + prefix):
                daluke_key = "." + daluke_key[1+len(prefix):]

        # Fix 1:1 differences in naming scheme
        parts = [new_part if (new_part := one2one_map.get(part)) else part for part in daluke_key.split(".")]
        daluke_key = ".".join(parts)

        # Set new key if different
        if daluke_key[1:] != bert_key:
            daluke_key = daluke_key[1:]  # Remove leading dot
            # Overwrite bert naming scheme with daluke naming scheme
            state_dict[daluke_key] = state_dict[bert_key]
            del state_dict[bert_key]

    missing_keys    = list()
    unexpected_keys = list()
    error_msgs      = list()

    # Create shallow copy, so load can modify it
    state_dict = state_dict.copy()

    def load(module: nn.Module, *, prefix: str):
        """ Load weights recursively. Heavily inspired by torch.nn.Module.load_state_dict """
        module._load_from_state_dict(
            state_dict, prefix, dict(), True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix=prefix + name + ".")

    # Load base model parameters into daluke
    load(daluke, prefix="")

    if error_msgs:
        raise RuntimeError("Errors in loading state_dict for %s:\n" % daluke.__class__.__name__ + "\n\t".join(error_msgs))

    return set(missing_keys)
