#!/usr/bin/env python3
from __future__ import annotations
import os
from typing import Any

import torch
from pelutils import log, Levels, Parser

from daluke.ner import load_dataset, load_model
from daluke.ner.data import Split
from daluke.ner.evaluation import evaluate_ner, type_distribution
from daluke.ner.run import DATASET_ARGUMENTS

from daluke.serialize import load_from_archive, TRAIN_OUT

FP_SIZE = 32

ARGUMENTS = {
    "model": {
        "help": ".tar.gz file containing fine-tuned model, metadata and entity vocab",
        "default": None,
    },
    "max-entity-span": {
        "help": "Max. length of spans used in data. If not given, use the one in pre-training metadata",
        "default": None,
        "type": int,
    },
    "max-entities": {
        "help": "Max. enitites in each example. If not given, use the one in pre-training metadata",
        "default": None,
        "type": int,
    },
    "quieter":    {"help": "Don't show debug logging", "action": "store_true"},
    **DATASET_ARGUMENTS
}

def run_experiment(args: dict[str, Any]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, metadata, state_dict = load_from_archive(args["model"])

    log("Loading dataset ...")
    dataset = load_dataset(args, metadata, device)
    dataloader = dataset.build(Split.TEST, FP_SIZE)

    log("Loading model ...")
    model = load_model(state_dict, dataset, metadata, device)

    # Print some important information to stdout
    log.debug(model)
    dataset.document(dataloader, Split.TEST)
    type_distribution(dataset.data[Split.TEST].annotations)

    log("Starting evaluation of daLUKE for NER")
    results = evaluate_ner(model, dataloader, dataset, device, Split.TEST)

    results.save(args["location"])
    type_distribution(results.preds)

if __name__ == '__main__':
    with log.log_errors:
        parser = Parser(ARGUMENTS, name="daluke-ner-eval", multiple_jobs=False)
        exp = parser.parse()
        parser.document_settings()
        log.configure(
            os.path.join(parser.location, "daluke-eval-ner.log"), "Finetuning evaluation of daLUKE for Danish NER",
            print_level=Levels.INFO if exp["quieter"] else Levels.DEBUG,
        )
        run_experiment(exp)
