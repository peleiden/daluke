from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertSelfOutput,
    BertOutput,
    BertIntermediate,
)

from daluke.data import BatchedExamples

class DaLUKE(nn.Module):
    """
    Language Understanding with Knowledge-based Embeddings in Danish.
    Returns entity and word representations.
    """
    def __init__(self,
        bert_config: BertConfig,
        ent_vocab_size: int,
        ent_emb_size: int,
    ):
        """
        bert_config:    Used for the BERT Pooler
        ent_vocab_size: Necessary for the entity embeddings
        ent_emb_size:   Dimension of entity embedding
        """
        super().__init__()
        self.ent_emb_size = ent_emb_size
        self.word_embeddings   = BertEmbeddings(bert_config)
        self.entity_embeddings = EntityEmbeddings(bert_config, ent_vocab_size, self.ent_emb_size)
        self.encoder = nn.ModuleList(
            [EntityAwareLayer(bert_config) for _ in range(bert_config.num_hidden_layers)]
        )

    def forward(self, ex: BatchedExamples) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given a data class of word and entity ids and other tokens, return embeddings of both
        """
        word_hidden    = self.word_embeddings(ex.words.ids, ex.words.segments)
        entity_hidden  = self.entity_embeddings(ex.entities.ids, ex.entities.pos, ex.entities.segments)

        attention_mask = torch.cat((ex.words.attention_mask, ex.entities.attention_mask), dim=1).unsqueeze(1).unsqueeze(2)
        attention_mask = 10_000.0 * (attention_mask - 1.0) # TODO: Explain

        for encode in self.encoder:
            word_hidden, entity_hidden = encode(word_hidden, entity_hidden, attention_mask)
        return word_hidden, entity_hidden

class EntityAwareLayer(nn.Module):
    """
    Transformer layer where the attention is replaced by the entity-aware method
    """
    def __init__(self, bert_config: BertConfig):
        super().__init__()
        self.attention    = EntitySelfAttention(bert_config.hidden_size, bert_config.num_attention_heads, bert_config.attention_probs_dropout_prob)
        self.self_output  = BertSelfOutput(bert_config)
        self.intermediate = BertIntermediate(bert_config)
        self.output       = BertOutput(bert_config)

    def forward(self, word_hidden: torch.Tensor, entity_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        word_size = word_hidden.size(1)
        total_hidden = torch.cat((word_hidden, entity_hidden), dim=1)
        self_attention = self.attention(word_hidden, entity_hidden, attention_mask)
        self_out  = self.self_output(self_attention, total_hidden)
        out = self.output(self.intermediate(self_out), self_out)
        return out[:, :word_size, :], out[:, word_size:, :]

class EntitySelfAttention(nn.Module):
    """
    As LUKE uses both entities and words, this self-attention takes the token type into account.
    """
    def __init__(self, hidden_size: int, num_heads: int, drop_prob: float):
        """
        Sets up the four query matrices used in the Entity-aware Self-attention:
            Q_w used between two words
            Q_e used between two entities
            Q_w2e used from word to entity
            Q_e2w used from entity to word
        Alse sets up key (K) and value (V) layers as in normal self-attention
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        # Four query matrices, the key and the value
        self.Q_w, self.Q_e, self.Q_w2e, self.Q_e2w, self.K, self.V = (
            nn.Linear(hidden_size, self.num_heads*self.head_size)
            for _ in range(6)
        )
        self.drop = nn.Dropout(drop_prob)

    def forward(self, word_hidden: torch.Tensor, entity_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Query and key is divided into the four cases (word2word, entity2entity, word2entity, entity2word), but is otherwise
        used in normal transformer way to compute attention from which context is returned
        """
        word_size = word_hidden.size(1)
        total_hidden = torch.cat((word_hidden, entity_hidden), dim=1)

        # Queries are given input dependant on the domain FROM which they map
        queries = self.Q_w(word_hidden), self.Q_e(entity_hidden), self.Q_w2e(word_hidden), self.Q_e2w(entity_hidden)
        # Key layers divided dependant on domain TO which they map
        key = self.reshape_to_matrix(self.K(total_hidden))
        key_2w = key[:, :, :word_size, :].transpose(-1, -2)
        key_2e = key[:, :, word_size:, :].transpose(-1, -2)

        # Attention matrices computed as query*key and then concatenated
        A_w, A_e, A_w2e, A_e2w = (self.reshape_to_matrix(q) @ k for q, k in zip(queries, (key_2w, key_2e, key_2e, key_2w)))
        attention = torch.cat(
            [torch.cat(a, dim=3) for a in ((A_w, A_w2e), (A_e2w, A_e))],
            dim=2,
        )

        # Attention is transformed to probability and matmul'ed with value layer, creating context
        attention = self.drop(
            F.softmax(attention/self.num_heads**0.5 + attention_mask, dim=-1)
        )
        value = self.reshape_to_matrix(self.V(total_hidden))
        context = (attention @ value).permute(0, 2, 1, 3).contiguous()
        return context.view(
            *context.size()[:-2], self.num_heads*self.head_size
        )

    def reshape_to_matrix(self, layer_out: torch.Tensor) -> torch.Tensor:
        """
        Make the layer outputs usable for matrix products
        """
        return layer_out.view(
            *layer_out.size()[:-1], self.num_heads, self.head_size
        ).permute(0, 2, 1, 3)

class EntityEmbeddings(nn.Module):
    """
    Embeds entitites from the entity vocabulary
    """
    def __init__(self, bert_config: BertConfig, ent_vocab_size: int, ent_emb_size: int):
        super().__init__()
        h = bert_config.hidden_size

        self.ent_embeds = nn.Embedding(ent_vocab_size, ent_emb_size, padding_idx=0)
        self.pos_embeds = nn.Embedding(bert_config.max_position_embeddings, h)
        self.typ_embeds = nn.Embedding(bert_config.type_vocab_size, h)

        self.ent_embeds_dense = nn.Linear(ent_emb_size, h) if ent_emb_size != h else None

        self.lnorm = nn.LayerNorm(h, eps=bert_config.layer_norm_eps)
        self.drop  = nn.Dropout(bert_config.hidden_dropout_prob)

    def forward(self, entity_ids: torch.Tensor, pos_ids: torch.Tensor, typ_ids: torch.Tensor):
        """
        Takes
        entity_ids: Vector of length X holding the vocab. ids of entities
        pos_ids: (X, max_position_embeddings) holding the position of entity i in the sequence
        typ_ids: Vector of length (X) holding types of entity tokens (0 or 1)

        Output embeddings of shape (X, H), H: hidden layer size
        """
        ent_embeds = self.ent_embeds(entity_ids)
        ent_embeds = self.ent_embeds_dense(ent_embeds) if self.ent_embeds_dense is not None else ent_embeds

        pos_embeds = self.pos_embeds(pos_ids.clamp(min=0))
        pos_embed_mask = (pos_ids != -1).type_as(pos_embeds).unsqueeze(-1)
        pos_embeds = (pos_embeds*pos_embed_mask).sum(dim=-2) / pos_embed_mask.sum(dim=-2).clamp(min=1e-7)

        typ_embeds = self.typ_embeds(typ_ids)
        return self.drop(self.lnorm(ent_embeds + pos_embeds + typ_embeds))
