from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertConfig

from daluke.model import DaLUKE, get_ent_embed, ENTITY_EMBEDDING_KEY
from daluke.ner.data import NERDataset, NERBatchedExamples


class NERDaLUKE(DaLUKE):
    """
    Named Entity Recognition using the BERT based model LUKE using Entity Aware Attention
    """
    def __init__(self,
        output_shape: int,
        bert_config: BertConfig,
        ent_vocab_size: int,
        ent_embed_size: int,
        dropout: float,
        words_only: bool,
        entities_only: bool,
    ):
        """
        Build the architecture and setup the config
        """
        super().__init__(bert_config, ent_vocab_size, ent_embed_size)
        self.output_shape = output_shape
        self.drop = nn.Dropout(dropout if dropout is not None else bert_config.hidden_dropout_prob)

        self.words_only = words_only
        self.entities_only = entities_only
        if self.words_only:
            concat_size = 2
        elif self.entities_only:
            concat_size = 1
        else:
            concat_size = 3
        self.classifier = nn.Linear(concat_size*bert_config.hidden_size, self.output_shape)

    def forward(self, ex: NERBatchedExamples) -> torch.Tensor:
        """
        Classify NER by passing the word and entity id's through the encoder
        and running the linear classifier on the output
        """
        # Forward pass through encoder, saving the contextualized representations of words and entitites
        word_representations, ent_representations = super().forward(ex)
        # We gather the starting and ending words in each entity span
        start_word_representations, end_word_representations = self.collect_start_and_ends(word_representations, ex)
        if self.words_only:
            features = torch.cat([start_word_representations, end_word_representations], dim=2)
        elif self.entities_only:
            features = ent_representations
        else:
            features = torch.cat([start_word_representations, end_word_representations, ent_representations], dim=2)
        features = self.drop(features)
        return self.classifier(features)

    @staticmethod
    def collect_start_and_ends(word_representations: torch.Tensor, ex: NERBatchedExamples) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the word representations for first and last entity as is done for BERT NER
        """
        hidden_word_size = word_representations.size()[-1]

        ent_start_pos = ex.entities.start_pos.unsqueeze(-1).expand(-1, -1, hidden_word_size)
        ent_end_pos = ex.entities.end_pos.unsqueeze(-1).expand(-1, -1, hidden_word_size)

        return torch.gather(word_representations, -2, ent_start_pos), torch.gather(word_representations, -2, ent_end_pos)

def span_probs_to_preds(span_probs: dict[tuple[int, int], np.ndarray], seq_len: int, dataset: NERDataset) -> list[str]:
    positives = list()
    for span, probs in span_probs.items():
        max_idx = probs.argmax()
        try:
            if (max_label := dataset.all_labels[max_idx]) != dataset.null_label:
                positives.append((probs[max_idx], span, max_label))
        # If the model predicted a class which is not in the dataset, we count it as O(utside annotation)
        except IndexError:
            pass
    # Initialize all predictions to null predictions
    preds = [dataset.null_label for _ in range(seq_len)]
    # Sort after max probability
    for _, span, label in reversed(sorted(positives)):
        if all(l == dataset.null_label for l in preds[span[0]:span[1]]):
            # Follow IOUB2 scheme: Set all to "I-X" apart from first which is "B-X"
            for i in range(*span):
                preds[i] = f"I-{label}"
            preds[span[0]] = f"B-{label}"
    return preds

def mutate_for_ner(state_dict: dict, mask_id: int) -> (dict, int):
    """
    For NER, we don't need the entire entity vocabulary layer: Only entity and not entity are considered
    """
    ent_embed = get_ent_embed(state_dict)
    mask_embed = ent_embed[mask_id].unsqueeze(0)
    state_dict[ENTITY_EMBEDDING_KEY] = torch.cat((ent_embed[:1], mask_embed))

    return state_dict, ent_embed.shape[1]
