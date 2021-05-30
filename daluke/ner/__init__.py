from __future__ import annotations
from typing import Any, Type

import torch
from transformers import AutoConfig

import daluke.ner.data as datasets
from daluke.ner.data import NERDataset
from daluke.ner.model import NERDaLUKE, get_ent_embed

def load_dataset(entity_vocab: list[dict], args: dict[str, Any], metadata: dict[str, Any], device: torch.device) -> NERDataset:
    dataset_cls: Type[NERDataset] = getattr(datasets, args["dataset"])
    dataset = dataset_cls(
        entity_vocab,
        base_model      = metadata["base-model"],
        max_seq_length  = metadata["max-seq-length"],
        max_entities    = metadata["max-entities"] if args.get("max_entities") is None else args["max_entities"],
        max_entity_span = metadata["max-entity-span"] if args.get("max_entity_span") is None else args["max_entity_span"],
        device          = device,
    )
    dataset.load(
        plank_path   = args["plank_path"],
        wikiann_path = args["wikiann_path"],
    )
    return dataset

def load_model(state_dict: dict, dataset: NERDataset, metadata: dict[str, Any], device: torch.device, entity_embedding_size: int=None) -> NERDaLUKE:
    bert_config = AutoConfig.from_pretrained(metadata["base-model"])
    model = NERDaLUKE(
        len(dataset.all_labels),
        bert_config,
        ent_vocab_size = 2, # Same reason as mutate_for_ner
        ent_embed_size = entity_embedding_size if entity_embedding_size is not None else get_ent_embed(state_dict).shape[1],
    )
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)
