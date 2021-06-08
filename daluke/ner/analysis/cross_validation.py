#!/usr/bin/env python3
from __future__ import annotations
import os
from typing import Any

import torch
from pelutils import log, Levels, Parser, set_seeds

from daluke.ner import load_dataset, load_model
from daluke.ner.model import mutate_for_ner
from daluke.ner.run import DATASET_ARGUMENTS

from daluke.serialize import load_from_archive

ARGUMENTS = {
    "model": {
        "help": ".tar.gz file containing fine-tuned model, metadata and entity vocab",
        "default": None,
    },
    "quieter":    {"help": "Don't show debug logging", "action": "store_true"},
    **DATASET_ARGUMENTS
}

def run_experiment(args: dict[str, Any]):
    set_seeds(seed=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    entity_vocab, metadata, state_dict = load_from_archive(args["model"])
    state_dict, ent_embed_size = mutate_for_ner(state_dict, mask_id=entity_vocab["[MASK]"]["id"])

    log(f"Loading dataset {args['dataset']} ...")
    dataset = load_dataset(args, metadata, device)

    log("Loading model ...")
    model = load_model(state_dict, dataset, metadata, device, entity_embedding_size=ent_embed_size)

if __name__ == '__main__':
    with log.log_errors:
        parser = Parser(ARGUMENTS, name="crossval-eval", multiple_jobs=False)
        exp = parser.parse()
        parser.document_settings()
        log.configure(
            os.path.join(parser.location, "daluke-crossval.log"), "Cross-validate NER results",
            print_level=Levels.INFO if exp["quieter"] else Levels.DEBUG,
        )
        run_experiment(exp)
