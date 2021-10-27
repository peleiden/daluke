from __future__ import annotations
from collections import namedtuple
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertSelfOutput,
    BertOutput,
    BertIntermediate,
)
from pelutils import Table, thousand_seps
try:
    from sklearn.decomposition import PCA as scipy_PCA
    sklearn_available = True
except ImportError:
    sklearn_available = False

from daluke.data import BatchedExamples


ENTITY_EMBEDDING_KEY = "entity_embeddings.ent_embeds.weight"

def all_params(model: torch.nn.Module) -> torch.Tensor:
    """ Returns an array of all model parameters, given either from a Module or a state_dict """
    return torch.cat([x.detach().view(-1) for n, x in model.state_dict().items() if n != "word_embeddings.position_ids"])

def all_params_groups_to_slices(model: DaLUKE, num_blocks: int) -> tuple[dict[str, slice], Table]:
    state_dict = model.state_dict()
    if (del_key := "word_embeddings.position_ids") in state_dict:
        del state_dict[del_key]
    slices = dict()
    keys = {
        "word_embeddings":    "Word embeddings",
        "entity_embeddings":  "Entity embeddings",
        "mask_word_scorer":   "Masked word scorer",
        "mask_entity_scorer": "Masked entity scorer",
        **{ "encoder.%i" % i: "Encoder layer %i" % i for i in range(num_blocks) },
    }
    idx = 0
    n = len(model)
    t = Table()
    t.add_header(["Group name", "Name", "Start index", "Stop index", "Number of parameters"])
    for group_name, nice_name in keys.items():
        numel = sum(state_dict.pop(n).numel() for n in tuple(state_dict.keys()) if n.startswith(group_name+"."))
        slices[nice_name] = slice(idx, idx+numel)
        t.add_row([
            group_name,
            nice_name,
            thousand_seps(slices[nice_name].start),
            thousand_seps(slices[nice_name].stop),
            thousand_seps(numel) + " (%5.2f %%)" % (100*numel/n),
        ], [1, 1, 0, 0, 0])
        idx += numel
    numel = sum(p.numel() for p in state_dict.values())
    slices["Other"] = slice(idx, idx+numel)
    return slices, t

def get_ent_embed(state_dict: dict) -> torch.nn.Module:
    return state_dict[ENTITY_EMBEDDING_KEY]

def get_ent_embed_size(state_dict: dict) -> int:
    return get_ent_embed(state_dict).shape[1]

class DaLUKE(nn.Module):
    """ Language Understanding with Knowledge-based Embeddings in Danish.
    Returns contextualized entity and word representations. """
    def __init__(
        self,
        bert_config: BertConfig,
        ent_vocab_size: int,
        ent_embed_size: int,
        ent_hidden_size:        Optional[int]=None,
        ent_intermediate_size:  Optional[int]=None,
    ):
        """
        bert_config:    Used for the BERT Pooler
        ent_vocab_size: Necessary for the entity embeddings
        ent_embed_size: Dimension of entity embedding
        ent_hidden_size: Dimension of entity representations, if None, is set to word hidden size
        ent_intermediate_size: Dimension of linear layers between attention blocks. If none, set to same multiple of hidden size as for words.
            If ent_hidden_size is None, this variable has no effect.
        """
        super().__init__()
        if ent_hidden_size is None:
            ent_hidden_size = bert_config.hidden_size
            ent_intermediate_size = bert_config.intermediate_size
        if ent_intermediate_size is None:
            ent_intermediate_size = ent_hidden_size * bert_config.intermediate_size // bert_config.hidden_size

        self.ent_embed_size    = ent_embed_size
        self.ent_hidden_size   = ent_hidden_size
        self.word_embeddings   = BertEmbeddings(bert_config)
        self.entity_embeddings = EntityEmbeddings(bert_config, ent_vocab_size, self.ent_embed_size, ent_hidden_size)
        self.encoder = nn.ModuleList(
            [EntityAwareLayer(bert_config, ent_hidden_size, ent_intermediate_size) for _ in range(bert_config.num_hidden_layers)]
        )

    def forward(self, ex: BatchedExamples) -> tuple[torch.Tensor, torch.Tensor]:
        """ Given a data class of word and entity ids and other tokens, return embeddings of both """
        word_hidden    = self.word_embeddings(ex.words.ids)
        entity_hidden  = self.entity_embeddings(ex.entities.ids, ex.entities.pos)

        attention_mask = torch.cat((ex.words.attention_mask, ex.entities.attention_mask), dim=1)\
            .unsqueeze(1)\
            .unsqueeze(2)
        attention_mask = 10_000.0 * (attention_mask - 1.0)
        for encode in self.encoder:
            word_hidden, entity_hidden = encode(word_hidden, entity_hidden, attention_mask)
        return word_hidden, entity_hidden

    @staticmethod
    def _weight_reduce_pca(l: int, W: torch.FloatTensor) -> list[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        """
        if not sklearn_available:
            raise ModuleNotFoundError("Using PCA for weight initialization requires isntallation of the optional requirement `scikit-learn`")

        W_w2e = scipy_PCA(n_components=l).fit_transform(W.numpy())
        W_e2w = W_w2e.T
        W_e = scipy_PCA(n_components=l).fit_transform(W_w2e)
        return torch.from_numpy(W_e), torch.from_numpy(W_e2w), torch.from_numpy(W_w2e)

    def init_special_attention(self, pca: bool) -> set[str]:
        """
        As the attention layers of DaLUKE have four query matrices and the BERT-based transformers only have one,
        we might want to init the other three to this one word-to-word query matrix
        Returns keys in state_dict set in this method
        """
        keys = set()
        for i, layer in enumerate(self.encoder):
            if pca and self.ent_hidden_size < layer.attention.Q_w.weight.data.shape[0]:
                # In the low dimensional case, we must create low-dim Q, K, V using PCA
                Q_e, Q_w2e, Q_e2w = self._weight_reduce_pca(
                        self.ent_hidden_size,
                        layer.attention.Q_w.weight.data.detach().clone(),
                )
                _, _, K_e = self._weight_reduce_pca(
                        self.ent_hidden_size,
                        layer.attention.K.weight.data.detach().clone()
                )
                V_e, V_w2e, V_e2w = self._weight_reduce_pca(
                        self.ent_hidden_size,
                        layer.attention.V.weight.data.detach().clone(),
                )
                layer.attention.K_e.weight.data = K_e
                layer.attention.V_e.weight.data = V_e
                layer.attention.V_w2e.weight.data = V_w2e
                layer.attention.V_e2w.weight.data = V_e2w
            else:
                Q_e, Q_w2e, Q_e2w = (layer.attention.Q_w.weight.data.detach().clone() for _ in range(3))
            layer.attention.Q_e.weight.data = Q_e
            layer.attention.Q_w2e.weight.data = Q_w2e
            layer.attention.Q_e2w.weight.data = Q_e2w

            if not pca:
                layer.attention.Q_e.bias.data = layer.attention.Q_w.bias.data.detach().clone()
                layer.attention.Q_w2e.bias.data = layer.attention.Q_w.bias.data.detach().clone()
            layer.attention.Q_e2w.bias.data = layer.attention.Q_w.bias.data.detach().clone()

            for key in "Q_e", "Q_w2e", "Q_e2w":
                keys = set.union(keys, {f"encoder.{i}.attention.{key}.weight", f"encoder.{i}.attention.{key}.bias"})

        return keys

    @staticmethod
    def init_weights(module: nn.Module, std: float):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            if module.embedding_dim == 1:  # Embedding for bias parameters
                module.weight.data.zero_()
            else:
                module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __len__(self):
        """ Number of model parameters. Further docs here: https://pokemondb.net/pokedex/numel """
        return sum(x.numel() for n, x in self.state_dict().items() if n != "word_embeddings.position_ids")

class EntityAwareLayer(nn.Module):
    """
    Transformer layer where the attention is replaced by the entity-aware method
    """
    def __init__(self, bert_config: BertConfig, ent_hidden_size: int, ent_intermediate_size: int):
        super().__init__()
        self.attention    = EntitySelfAttention(
            bert_config.hidden_size,
            bert_config.num_attention_heads,
            bert_config.attention_probs_dropout_prob,
            ent_hidden_size,
        )
        self.self_output  = BertSelfOutput(bert_config)
        self.intermediate = BertIntermediate(bert_config)
        self.output       = BertOutput(bert_config)

        self.ent_low_dim  = ent_hidden_size != bert_config.hidden_size
        if self.ent_low_dim:
            ent_config = self._get_ent_config(ent_hidden_size, ent_intermediate_size, bert_config)
            self.ent_self_output = BertSelfOutput(ent_config)
            self.ent_intermediate = BertIntermediate(ent_config)
            self.ent_output = BertOutput(ent_config)

    def forward(self, word_hidden: torch.Tensor, entity_hidden: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        word_size = word_hidden.size(1)
        word_attention, entity_attention = self.attention(word_hidden, entity_hidden, attention_mask)
        if not self.ent_low_dim:
            self_out  = self.self_output(torch.cat((word_attention, entity_attention), dim=1), torch.cat((word_hidden, entity_hidden), dim=1))
            out = self.output(self.intermediate(self_out), self_out)
            return out[:, :word_size, :], out[:, word_size:, :]
        # If we DO have lower entity dimension, the have to FP representations separately
        word_self_out  = self.self_output(word_attention, word_hidden)
        word_out = self.output(self.intermediate(word_self_out), word_self_out)
        ent_self_out  = self.ent_self_output(entity_attention, entity_hidden)
        ent_out = self.ent_output(self.ent_intermediate(ent_self_out), ent_self_out)
        return word_out, ent_out

    @staticmethod
    def _get_ent_config(ent_hidden_size: int, ent_intermediate_size: int, bert_config: BertConfig) -> NamedTuple:
        # Mock Bert config
        dummy_ent_config = namedtuple("BertConfig", "hidden_size intermediate_size layer_norm_eps hidden_dropout_prob hidden_act")
        return dummy_ent_config(
            ent_hidden_size,
            ent_intermediate_size,
            bert_config.layer_norm_eps,
            bert_config.hidden_dropout_prob,
            bert_config.hidden_act
        )

class EntitySelfAttention(nn.Module):
    """
    As LUKE uses both entities and words, this self-attention takes the token type into account.
    """
    def __init__(self, hidden_size: int, num_heads: int, drop_prob: float, ent_hidden_size: int):
        """
        Sets up the four query matrices used in the Entity-aware Self-attention:
            Q_w used between two words
            Q_e used between two entities
            Q_w2e used from word to entity
            Q_e2w used from entity to word
        Alse sets up key (K) and value (V) layers as in normal self-attention
        """
        super().__init__()
        self.ent_hidden_size = ent_hidden_size
        self.num_heads = num_heads
        self.ent_low_dim = hidden_size != self.ent_hidden_size
        self.head_size = hidden_size // num_heads
        self.ent_head_size = ent_hidden_size // num_heads

        self.dropout = nn.Dropout(drop_prob)

        # Four query matrices, the key and the value
        self.Q_w    = nn.Linear(hidden_size, hidden_size)
        self.Q_e    = nn.Linear(ent_hidden_size, ent_hidden_size)
        self.Q_w2e  = nn.Linear(hidden_size, ent_hidden_size)
        self.Q_e2w  = nn.Linear(ent_hidden_size, hidden_size)

        self.K      = nn.Linear(hidden_size, hidden_size)
        self.V      = nn.Linear(hidden_size, hidden_size)
        if self.ent_low_dim:
            self.K_e   = nn.Linear(ent_hidden_size, ent_hidden_size)
            self.V_e   = nn.Linear(ent_hidden_size, ent_hidden_size)
            self.V_w2e = nn.Linear(hidden_size, ent_hidden_size)
            self.V_e2w = nn.Linear(ent_hidden_size, hidden_size)

    def forward(self, word_hidden: torch.Tensor, entity_hidden: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Query and key is divided into the four cases (word2word, entity2entity, word2entity, entity2word), but is otherwise
        used in normal transformer way to compute attention from which context is returned
        """
        if self.ent_low_dim:
            return self.entity_low_dim_forward(word_hidden, entity_hidden, attention_mask)

        word_size = word_hidden.size(1)
        # Queries are given input dependant on the domain from which they map
        queries = self.Q_w(word_hidden), self.Q_e(entity_hidden), self.Q_w2e(word_hidden), self.Q_e2w(entity_hidden)
        total_hidden = torch.cat((word_hidden, entity_hidden), dim=1)
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
        attention = self.dropout(
            F.softmax(attention/self.head_size**0.5 + attention_mask, dim=-1)
        )
        value = self.reshape_to_matrix(self.V(total_hidden))
        context = (attention @ value).permute(0, 2, 1, 3).contiguous()
        out_hidden = context.view(
            *context.shape[:-2], self.num_heads*self.head_size
        )
        return out_hidden[:, :word_size, :], out_hidden[:, word_size:, :]

    def entity_low_dim_forward(self, word_hidden: torch.Tensor, entity_hidden: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the novel method were entity representations live in lower dimensional space.
        """
        word_size = word_hidden.size(1)
        queries = self.Q_w(word_hidden), self.Q_e(entity_hidden), self.Q_w2e(word_hidden), self.Q_e2w(entity_hidden)

        attmask_w = attention_mask[:, :, :, :word_size]
        attmask_e = attention_mask[:, :, :, word_size:]
        key_2w  = self.reshape_to_matrix(self.K(word_hidden)).transpose(-1, -2)
        key_2e  = self.reshape_to_matrix(self.K_e(entity_hidden), entity=True).transpose(-1, -2)

        # The four attention dot products
        attention_w = self.dropout(F.softmax(
            self.reshape_to_matrix(queries[0]) @ key_2w /  self.head_size**0.5 + attmask_w,
        dim=-1))
        attention_e = self.dropout(F.softmax(
            self.reshape_to_matrix(queries[1], entity=True) @ key_2e / self.ent_head_size**0.5 + attmask_e,
        dim=-1))
        attention_w2e = self.dropout(F.softmax(
            self.reshape_to_matrix(queries[2], entity=True) @ key_2e / self.ent_head_size**0.5 + attmask_e,
        dim=-1)).transpose(-2, -1)
        attention_e2w = self.dropout(F.softmax(
            self.reshape_to_matrix(queries[3]) @ key_2w / self.head_size**0.5 + attmask_w,
        dim=-1)).transpose(-2, -1)
        # TODO: Evaluate whether the two above transposes are justified: It is necessary for the shapes to work,
        # and makes intuitive sense, but it is not present in normal attention and we cannot see a reason why
        # it should appear now.

        value_w   = self.reshape_to_matrix(self.V(word_hidden))
        value_e   = self.reshape_to_matrix(self.V_e(entity_hidden), entity=True)
        value_w2e = self.reshape_to_matrix(self.V_w2e(word_hidden), entity=True)
        value_e2w = self.reshape_to_matrix(self.V_e2w(entity_hidden))

        # Build output from each part matrix
        context_w = (attention_w @ value_w + attention_e2w @ value_e2w).permute(0, 2, 1, 3).contiguous()
        context_e = (attention_e @ value_e + attention_w2e @ value_w2e).permute(0, 2, 1, 3).contiguous()
        # Reshape to expected representation array
        out_w  = context_w.view(*context_w.shape[:-2], self.num_heads*self.head_size)
        out_e = context_e.view(*context_e.shape[:-2], self.num_heads*self.ent_head_size)
        return out_w, out_e

    def reshape_to_matrix(self, layer_out: torch.Tensor, entity=False) -> torch.Tensor:
        """
        Make the layer outputs usable for matrix products in the multihead setting.
        """
        return layer_out.view(
            *layer_out.size()[:-1], self.num_heads, self.ent_head_size if entity else self.head_size
        ).permute(0, 2, 1, 3)

class EntityEmbeddings(nn.Module):
    """
    Embeds entitites from the entity vocabulary
    """
    def __init__(self, bert_config: BertConfig, ent_vocab_size: int, ent_embed_size: int, ent_hidden_size: int):
        super().__init__()
        self.ent_embeds = nn.Embedding(ent_vocab_size, ent_embed_size, padding_idx=0)
        self.pos_embeds = nn.Embedding(bert_config.max_position_embeddings, ent_hidden_size)
        self.typ_embeds = nn.Embedding(bert_config.type_vocab_size, ent_hidden_size)

        self.ent_embeds_dense = nn.Linear(ent_embed_size, ent_hidden_size) if ent_embed_size != ent_hidden_size else None

        self.lnorm = nn.LayerNorm(ent_hidden_size, eps=bert_config.layer_norm_eps)
        self.drop  = nn.Dropout(bert_config.hidden_dropout_prob)

    def forward(self, entity_ids: torch.Tensor, pos_ids: torch.Tensor, typ_ids: torch.Tensor=None):
        """
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

        if typ_ids is None:
            typ_ids = torch.zeros_like(entity_ids)
        typ_embeds = self.typ_embeds(typ_ids)
        return self.drop(self.lnorm(ent_embeds + pos_embeds + typ_embeds))

def ent_dims_from_state_dict(state_dict: dict) -> tuple[int, int]:
    """
    From a state dict, return the appropriate values of `ent_hidden_size` and `ent_intermediate_size`
    which can be used for instantiating the model.
    """
    # We assume that all transformer blocks have same dim.
    # The entity output maps from hidden to intermediate so gives us exactly the shape we need
    return state_dict["encoder.0.ent_output.dense.weight"].shape
