from __future__ import annotations
from typing import Any, Type

import torch
import torch.nn as nn
from transformers import AutoConfig

import daluke.ner.data as datasets
from daluke.model import get_ent_embed_size
from daluke.ner.data import NERDataset
from daluke.ner.model import NERDaLUKE
from daluke.pretrain.model import load_base_model_weights

def load_dataset(args: dict[str, Any], metadata: dict[str, Any], device: torch.device) -> NERDataset:
    dataset_cls: Type[NERDataset] = getattr(datasets, args["dataset"])
    dataset = dataset_cls(
        base_model      = metadata["base-model"],
        max_seq_length  = metadata["max-seq-length"],
        max_entities    = metadata["max-entities"] if args.get("max_entities") is None else args["max_entities"],
        max_entity_span = metadata["max-entity-span"] if args.get("max_entity_span") is None else args["max_entity_span"],
        device          = device,
    )
    dataset.load(
        plank_path   = args.get("plank_path"),
        wikiann_path = args.get("wikiann_path"),
    )
    return dataset

def load_model(
    state_dict: dict,
    dataset: NERDataset,
    metadata: dict[str, Any],
    device: torch.device,
    bert_attention: bool=False,
    entity_embedding_size: int=None,
    dropout: float=None,
) -> NERDaLUKE:
    bert_config = AutoConfig.from_pretrained(metadata["base-model"])
    model = NERDaLUKE(
        metadata.get("output-size", len(dataset.all_labels)),
        bert_config,
        ent_vocab_size = 2, # Same reason as mutate_for_ner
        ent_embed_size = entity_embedding_size if entity_embedding_size is not None else get_ent_embed_size(state_dict),
        dropout = dropout,
        words_only = metadata.get("NER-words-only", False),
        entities_only = metadata.get("NER-entities-only", False),
    )
    model.load_state_dict(state_dict, strict=False)
    if bert_attention:
        load_base_model_weights(model, state_dict, bert_attention=False)
        model.init_queries()
    return model.to(device)
