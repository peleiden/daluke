#!/usr/bin/env python3
from __future__ import annotations
import os

import torch
from pelutils import log, Levels, Parser

import daluke.data as datasets

from daluke import cuda
from daluke.train_ner import OUT_FILE as train_out
from daluke.model import load_from_archive, DaLukeNER
from daluke.data import NERDataset, Split
from daluke.eval import evaluate_ner

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
    device = torch.device("cpu") if args["cpu"] else cuda
    entity_vocab, metadata, state_dict = load_from_archive(args["model"])

    log.debug("Loading dataset ...")
    dataset = getattr(datasets, args["dataset"])
    dataset: NERDataset = dataset()
    dataloader = dataset.build(Split.TEST, EVAL_BATCH_SIZE)

    log.debug("Loading model ...")
    model = DaLukeNER(metadata["model_config"], output_shape=len(dataset.all_labels))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    log("Starting evaluation of daLUKE for NER")
    results = evaluate_ner(model, dataloader, dataset, device)
    results.save(args["location"])

if __name__ == '__main__':
    with log.log_errors:
        parser = Parser(ARGUMENTS, name="daLUKE-NER-eval", multiple_jobs=True)
        experiments = parser.parse()
        parser.document_settings()
        log.configure(
            os.path.join(parser.location, "daluke_eval_ner.log"), "Finetune daLUKE for Danish NER",
            print_level=Levels.INFO if experiments[0]["quieter"] else Levels.DEBUG
        )
        for exp in experiments:
            run_experiment(exp)
