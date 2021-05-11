#!/usr/bin/env python3
from __future__ import annotations
import os

import torch
from transformers import AutoConfig
from pelutils import log, Levels, Parser

import daluke.ner.data as datasets

from daluke.ner.model import NERDaLUKE, get_ent_embed
from daluke.ner.data import NERDataset, Split
from daluke.ner.evaluation import evaluate_ner

from daluke.serialize import load_from_archive
from daluke.serialize import OUT_FILE as train_out

EVAL_BATCH_SIZE = 32

ARGUMENTS = {
    "model": {
        "help": "directory or .tar.gz file containing fine-tuned model, metadata and entity vocab",
        "default": os.path.join("local_data", train_out)
    },
    "quieter":    {"help": "Don't show debug logging", "action": "store_true"},
    "cpu":        {"help": "Run experiment on cpu",    "action": "store_true"},
    "dataset":    {"help": "Which dataset to use. Currently, only DaNE supported", "default": "DaNE"},
}

def run_experiment(args: dict[str, str]):
    device = torch.device("cpu") if args["cpu"] or not torch.cuda.is_available() else torch.device("cuda")
    entity_vocab, metadata, state_dict = load_from_archive(args["model"])

    log("Loading dataset ...")
    dataset = getattr(datasets, args["dataset"])
    dataset: NERDataset = dataset(
        entity_vocab,
        base_model      = metadata["base-model"],
        max_seq_length  = metadata["max-seq-length"],
        max_entities    = metadata["max-entities"],
        max_entity_span = metadata["max-entity-span"],
        device          = device,
    )
    dataloader = dataset.build(Split.TEST, EVAL_BATCH_SIZE)

    log("Loading model ...")
    bert_config = AutoConfig.from_pretrained(metadata["base-model"])
    ent_embed_size = get_ent_embed(state_dict).shape[1]
    model = NERDaLUKE(len(dataset.all_labels), bert_config, ent_vocab_size=2, ent_embed_size=ent_embed_size)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    log("Starting evaluation of daLUKE for NER")
    results = evaluate_ner(model, dataloader, dataset, device)
    results.save(args["location"])

if __name__ == '__main__':
    with log.log_errors:
        parser = Parser(ARGUMENTS, name="daLUKE-NER-eval", multiple_jobs=False)
        experiments = parser.parse()
        parser.document_settings()
        log.configure(
            os.path.join(parser.location, "daluke_eval_ner.log"), "Finetune daLUKE for Danish NER",
            print_level=Levels.INFO if experiments[0]["quieter"] else Levels.DEBUG
        )
        for exp in experiments:
            run_experiment(exp)
