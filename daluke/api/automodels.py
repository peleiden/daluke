from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from transformers import AutoTokenizer
import numpy as np
import torch
import torch.nn.functional as F

from pelutils.ds import no_grad

from daluke.pretrain.data.masking import BatchedExamples
from daluke.pretrain.data import MaskedBatchedExamples
from daluke.ner.data import NERBatchedExamples
from daluke.api.fetch_model import fetch_model, Models

class AutoDaLUKE(ABC):
    """
    Class that bundles up internal state necessary for inference using DaLUKE.
    """
    model_weight_type: Models

    def __init__(self):
        self.model, self.metadata, self.entity_vocab = fetch_model(self.model_weight_type)
        self.tokenizer = AutoTokenizer.from_pretrained(self.metadata["base-model"])


    @abstractmethod
    def predict(self, example: BatchedExamples) -> Any:
        """
        Perform inference using this trained DaLUKE model
        """

class AutoRepresentationDaLUKE(AutoDaLUKE):
    """ Inference for contextualized word and entity representations """
    model_weight_type: Models = Models.DaLUKE

    @no_grad
    def predict(self, example: BatchedExamples) -> tuple[torch.Tensor, torch.Tensor]:
        """ Contextualized word and entity representations from a example created using BatchedExamples """
        return super(type(self.model), self.model).forward(example)

class AutoMLMDaLUKE(AutoDaLUKE):
    """ Inference for predicting masked words. Masked entities are currently not predicted. """
    model_weight_type: Models = Models.DaLUKE

    @no_grad
    def predict(self, example: MaskedBatchedExamples) -> torch.Tensor:
        """ Probability distribution over mask word candidates """
        word_scores, __ = self.model(example)
        return F.softmax(word_scores, dim=1)

class AutoNERDaLUKE(AutoDaLUKE):
    """ Inference for NER """
    model_weight_type: Models = Models.DaLUKE_NER

    @no_grad
    def predict(self, example: NERBatchedExamples) -> dict[tuple[int, int], np.ndarray]:
        """
        Returns probability distribution over NER classes of every span in example
        """
        scores = self.model(example)
        probs = F.softmax(scores, dim=2)
        # Probability distribution for every possible span in the example
        span_probs = dict()
        for i, spans in enumerate(example.entities.fullword_spans):
            span_probs.update({
                span: probs[i, j].detach().cpu().numpy() for j, span in enumerate(spans) if span
            })
        return span_probs
