#######################################################################################################
# Everything in this file belongs is copied from https://github.com/studio-ousia/luke
# and is as such under the Apache License 2.0. The code has been copied to here with minimal changes.
# See https://github.com/studio-ousia/luke/blob/master/LICENSE for more deteails.
#######################################################################################################
import itertools
import unicodedata
import math

import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import (
    BertSelfOutput,
    BertIntermediate,
    BertOutput
)

class EntityEmbeddings(nn.Module):
    def __init__(self, config):
        super(EntityEmbeddings, self).__init__()
        self.config = config

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)
        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, entity_ids: torch.LongTensor, position_ids: torch.LongTensor, token_type_ids: torch.LongTensor = None
    ):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.config.entity_emb_size != self.config.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        position_embeddings = self.position_embeddings(position_ids.clamp(min=0))
        position_embedding_mask = (position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = torch.sum(position_embeddings, dim=-2)
        position_embeddings = position_embeddings / position_embedding_mask.sum(dim=-2).clamp(min=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class EntityAwareSelfAttention(nn.Module):
    def __init__(self, config):
        super(EntityAwareSelfAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.w2e_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.e2e_query = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_x_shape).permute(0, 2, 1, 3)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_size = word_hidden_states.size(1)

        w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states))
        w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states))
        e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))
        e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))

        key_layer = self.transpose_for_scores(self.key(torch.cat([word_hidden_states, entity_hidden_states], dim=1)))

        w2w_key_layer = key_layer[:, :, :word_size, :]
        e2w_key_layer = key_layer[:, :, :word_size, :]
        w2e_key_layer = key_layer[:, :, word_size:, :]
        e2e_key_layer = key_layer[:, :, word_size:, :]

        w2w_attention_scores = torch.matmul(w2w_query_layer, w2w_key_layer.transpose(-1, -2))
        w2e_attention_scores = torch.matmul(w2e_query_layer, w2e_key_layer.transpose(-1, -2))
        e2w_attention_scores = torch.matmul(e2w_query_layer, e2w_key_layer.transpose(-1, -2))
        e2e_attention_scores = torch.matmul(e2e_query_layer, e2e_key_layer.transpose(-1, -2))

        word_attention_scores = torch.cat([w2w_attention_scores, w2e_attention_scores], dim=3)
        entity_attention_scores = torch.cat([e2w_attention_scores, e2e_attention_scores], dim=3)
        attention_scores = torch.cat([word_attention_scores, entity_attention_scores], dim=2)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        value_layer = self.transpose_for_scores(
            self.value(torch.cat([word_hidden_states, entity_hidden_states], dim=1))
        )
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer[:, :word_size, :], context_layer[:, word_size:, :]


class EntityAwareAttention(nn.Module):
    def __init__(self, config):
        super(EntityAwareAttention, self).__init__()
        self.self = EntityAwareSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_self_output, entity_self_output = self.self(word_hidden_states, entity_hidden_states, attention_mask)
        hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1)
        self_output = torch.cat([word_self_output, entity_self_output], dim=1)
        output = self.output(self_output, hidden_states)
        return output[:, : word_hidden_states.size(1), :], output[:, word_hidden_states.size(1) :, :]


class EntityAwareLayer(nn.Module):
    def __init__(self, config):
        super(EntityAwareLayer, self).__init__()

        self.attention = EntityAwareAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        word_attention_output, entity_attention_output = self.attention(
            word_hidden_states, entity_hidden_states, attention_mask
        )
        attention_output = torch.cat([word_attention_output, entity_attention_output], dim=1)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output[:, : word_hidden_states.size(1), :], layer_output[:, word_hidden_states.size(1) :, :]


class EntityAwareEncoder(nn.Module):
    def __init__(self, config):
        super(EntityAwareEncoder, self).__init__()
        self.layer = nn.ModuleList([EntityAwareLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask):
        for layer_module in self.layer:
            word_hidden_states, entity_hidden_states = layer_module(
                word_hidden_states, entity_hidden_states, attention_mask
            )
        return word_hidden_states, entity_hidden_states

class InputFeatures(object):
    def __init__(
        self,
        example_index,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_start_positions,
        entity_end_positions,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        original_entity_spans,
        labels,
    ):
        self.example_index = example_index
        self.word_ids = word_ids
        self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_start_positions = entity_start_positions
        self.entity_end_positions = entity_end_positions
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids
        self.entity_segment_ids = entity_segment_ids
        self.entity_attention_mask = entity_attention_mask
        self.original_entity_spans = original_entity_spans
        self.labels = labels

def convert_examples_to_features(
    examples, label_list, tokenizer, max_seq_length, max_entity_length, max_mention_length
):
    max_num_subwords = max_seq_length - 2
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    def tokenize_word(text):
        return tokenizer.tokenize(text)

    for example_index, example in enumerate(examples):
        tokens = [tokenize_word(w) for w in example[0]]
        subwords = [w for li in tokens for w in li]

        subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))
        token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))
        subword_start_positions = frozenset(token2subword)
        subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in example[2]]

        entity_labels = {}
        start = None
        cur_type = None
        for n, label in enumerate(example[1]):
            if label == "O" or n in example[2]:
                if start is not None:
                    entity_labels[(token2subword[start], token2subword[n])] = label_map[cur_type]
                    start = None
                    cur_type = None

            if label.startswith("B"):
                if start is not None:
                    entity_labels[(token2subword[start], token2subword[n])] = label_map[cur_type]
                start = n
                cur_type = label[2:]

            elif label.startswith("I"):
                if start is None:
                    start = n
                    cur_type = label[2:]
                elif cur_type != label[2:]:
                    entity_labels[(token2subword[start], token2subword[n])] = label_map[cur_type]
                    start = n
                    cur_type = label[2:]

        if start is not None:
            entity_labels[(token2subword[start], len(subwords))] = label_map[cur_type]

        for n in range(len(subword_sentence_boundaries) - 1):
            doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]

            left_length = doc_sent_start
            right_length = len(subwords) - doc_sent_end
            sentence_length = doc_sent_end - doc_sent_start
            half_context_length = int((max_num_subwords - sentence_length) / 2)

            if left_length < right_length:
                left_context_length = min(left_length, half_context_length)
                right_context_length = min(right_length, max_num_subwords - left_context_length - sentence_length)
            else:
                right_context_length = min(right_length, half_context_length)
                left_context_length = min(left_length, max_num_subwords - right_context_length - sentence_length)

            doc_offset = doc_sent_start - left_context_length
            target_tokens = subwords[doc_offset : doc_sent_end + right_context_length]

            word_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + target_tokens + [tokenizer.sep_token])
            word_attention_mask = [1] * (len(target_tokens) + 2)
            word_segment_ids = [0] * (len(target_tokens) + 2)

            entity_start_positions = []
            entity_end_positions = []
            entity_ids = []
            entity_attention_mask = []
            entity_segment_ids = []
            entity_position_ids = []
            original_entity_spans = []
            labels = []

            for entity_start in range(left_context_length, left_context_length + sentence_length):
                doc_entity_start = entity_start + doc_offset
                if doc_entity_start not in subword_start_positions:
                    continue
                for entity_end in range(entity_start + 1, left_context_length + sentence_length + 1):
                    doc_entity_end = entity_end + doc_offset
                    if doc_entity_end not in subword_start_positions:
                        continue

                    if entity_end - entity_start > max_mention_length:
                        continue

                    entity_start_positions.append(entity_start + 1)
                    entity_end_positions.append(entity_end)
                    entity_ids.append(1)
                    entity_attention_mask.append(1)
                    entity_segment_ids.append(0)

                    position_ids = list(range(entity_start + 1, entity_end + 1))
                    position_ids += [-1] * (max_mention_length - entity_end + entity_start)
                    entity_position_ids.append(position_ids)

                    original_entity_spans.append(
                        (subword2token[doc_entity_start], subword2token[doc_entity_end - 1] + 1)
                    )

                    labels.append(entity_labels.get((doc_entity_start, doc_entity_end), 0))
                    entity_labels.pop((doc_entity_start, doc_entity_end), None)

            if len(entity_ids) == 1:
                entity_start_positions.append(0)
                entity_end_positions.append(0)
                entity_ids.append(0)
                entity_attention_mask.append(0)
                entity_segment_ids.append(0)
                entity_position_ids.append(([-1] * max_mention_length))
                original_entity_spans.append(None)
                labels.append(-1)

            split_size = math.ceil(len(entity_ids) / max_entity_length)
            for i in range(split_size):
                entity_size = math.ceil(len(entity_ids) / split_size)
                start = i * entity_size
                end = start + entity_size
                features.append(
                    InputFeatures(
                        example_index=example_index,
                        word_ids=word_ids,
                        word_attention_mask=word_attention_mask,
                        word_segment_ids=word_segment_ids,
                        entity_start_positions=entity_start_positions[start:end],
                        entity_end_positions=entity_end_positions[start:end],
                        entity_ids=entity_ids[start:end],
                        entity_position_ids=entity_position_ids[start:end],
                        entity_segment_ids=entity_segment_ids[start:end],
                        entity_attention_mask=entity_attention_mask[start:end],
                        original_entity_spans=original_entity_spans[start:end],
                        labels=labels[start:end],
                    )
                )

        assert not entity_labels

    return features

def is_punctuation(char):
    # obtained from:
    # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
