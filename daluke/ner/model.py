from __future__ import annotations
import json
import os
import tarfile
import tempfile

import numpy as np
import torch
from torch import nn
from transformers.models.bert.modeling_bert import (
        BertEncoder,
        BertPooler,
        BertConfig,
        BertEmbeddings
)

from pelutils import log

from daluke.model import DaLUKE
from daluke.collect_modelfile import VOCAB_FILE, METADATA_FILE, MODEL_OUT
from daluke.ner.data import NERDataset

ENTITY_EMBEDDING_KEY = "module.entity_embeddings.ent_embeds.weight"

class NERDaLUKE(DaLUKE):
    """
    Named Entity Recognition using the BERT based model LUKE using Entity Aware Attention
    """
    def __init__(self,
        output_shape: int,
        bert_config: BertConfig,
        ent_vocab_size: int,
        ent_embed_size: int,
    ):
        """
        Build the architecture and setup the config
        """
        super().__init__(bert_config, ent_vocab_size, ent_embed_size)
        self.output_shape = output_shape

        self.drop = nn.Dropout(bert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_config.hidden_size*3, self.output_shape)

    def forward(self, ex: BatchedExamples) -> torch.Tensor:
        """
        Classify NER by passing the word and entity id's through the encoder
        and running the linear classifier on the output
        """
        # Forward pass through encoder, saving the embeddings of words and entitites
        word_hidden, ent_hidden = super().forward(ex)

        entity_ends = ex.entities.pos.max(dim=2)
        dummy_pos = ex.enties.pos.copy()
        hid_w_size = hidden_state_w.size()[-1]

        ent_start_pos = ent_start_pos.unsqueeze(-1).expand(-1, -1, hid_w_size)
        ent_end_pos = ent_end_pos.unsqueeze(-1).expand(-1, -1, hid_w_size)

        all_starts = torch.gather(hidden_state_w, -2, ent_start_pos)
        all_ends = torch.gather(hidden_state_w, -2, ent_end_pos)

        features = torch.cat([all_starts, all_ends, hidden_state_ent], dim=2)
        features = self.drop(features)
        return self.classifier(features)

def span_probs_to_preds(span_probs: dict[tuple[int], np.ndarray], seq_len: int, dataset: NERDataset) -> list[str]:
    positives = list()
    for span, probs in span_probs.items():
        max_idx = probs.argmax()
        if (max_label := dataset.all_labels[max_idx]) != dataset.null_label:
            positives.append((probs[max_idx], span, max_label))
    preds = [dataset.null_label for _ in range(seq_len)]
    # Sort after max probability
    for _, span, label in reversed(sorted(positives)):
        if all(l == dataset.null_label for l in preds[span[0]:span[1]]):
            # Follow IOUB2 scheme: Set all to "I-X" unless first which is "B-X"
            for i in range(*span):
                preds[i] = f"I-{label}"
            preds[span[0]] = f"B-{label}"
    return preds

def mutate_for_ner(state_dict: dict, mask_id: int) -> (dict, int):
    """
    For NER, we don't need the entire entity vocabulary layer: Only entity and not entity are considered
    """
    ent_embed = state_dict[ENTITY_EMBEDDING_KEY]
    mask_embed = ent_embed[mask_id].unsqueeze(0)
    state_dict[ENTITY_EMBEDDING_KEY] = torch.cat((ent_embed[:1], mask_embed))

    return state_dict, ent_embed.shape[1]
