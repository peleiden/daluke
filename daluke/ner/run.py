#!/usr/bin/env python3
from __future__ import annotations
import os
from typing import Type

import torch
from transformers import AutoConfig
from pelutils import log, Levels, Parser, EnvVars

import daluke.ner.data as datasets

from daluke.serialize import load_from_archive, save_to_archive, COLLECT_OUT, TRAIN_OUT
from daluke.ner.model import NERDaLUKE, mutate_for_ner
from daluke.ner.training import TrainNER
from daluke.ner.data import NERDataset, Split


ARGUMENTS = {
    "model": {
        "help": ".tar.gz file containing the model, metadata, and entity vocab as generated by collect_modelfile",
        "default": os.path.join("local_data", COLLECT_OUT)
    },
    "lr":               {"default": 1e-5, "type": float},
    "epochs":           {"default": 5, "type": int},
    "batch-size":       {"default": 16, "type": int},
    "max-entity-span":  {"help": "Max. length of spans used in data. If not given, use the one in pre-training metadata",
                            "default": None, "type": int},
    "max-entities":     {"help": "Max. enitites in each example. If not given, use the one in pre-training metadata",
                            "default": None, "type": int},
    "warmup-prop":      {"default": 0.06, "type": float},
    "weight-decay":     {"default": 0.01, "type": float},

    "dataset": {"help": "Which dataset to use. Currently, only DaNE supported", "default": "DaNE"},
    "eval": {"help": "Run evaluation on dev. set after each epoch", "action": "store_true"},
    "quieter": {"help": "Don't show debug logging", "action": "store_true"},
}

def run_experiment(args: dict[str, str]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    entity_vocab, metadata, state_dict = load_from_archive(args["model"])
    state_dict, ent_embed_size = mutate_for_ner(state_dict, mask_id=entity_vocab["[MASK]"]["id"])

    log("Loading dataset ...")
    dataset_cls: Type[NERDataset] = getattr(datasets, args["dataset"])
    dataset = dataset_cls(
        entity_vocab,
        base_model      = metadata["base-model"],
        max_seq_length  = metadata["max-seq-length"],
        max_entities    = metadata["max-entities"] if args["max_entities"] is None else args["max_entities"],
        max_entity_span = metadata["max-entity-span"] if args["max_entity_span"] is None else args["max_entity_span"],
        device          = device,
    )
    dataloader = dataset.build(Split.TRAIN, args["batch_size"])
    dev_dataloader = dataset.build(Split.DEV, args["batch_size"]) if args["eval"] else None

    log("Loading model ...")
    bert_config = AutoConfig.from_pretrained(metadata["base-model"])
    model = NERDaLUKE(
        len(dataset.all_labels),
        bert_config,
        ent_vocab_size = 2, # Same reason as mutate_for_ner
        ent_embed_size = ent_embed_size,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    log(f"Starting training of DaLUKE for NER on {args['dataset']}")
    training = TrainNER(
        model,
        dataloader,
        dataset,
        device          = device,
        epochs          = args["epochs"],
        lr              = args["lr"],
        warmup_prop     = args["warmup_prop"],
        weight_decay    = args["weight_decay"],
        dev_dataloader  = dev_dataloader,
    )
    log.debug(training.model)
    log.debug(training.scheduler)
    log.debug(training.optimizer)
    dataset.document(dataloader, Split.TRAIN)

    results = training.run()

    os.makedirs(args["location"], exist_ok=True)
    results.save(args["location"])
    outpath = os.path.join(args["location"], TRAIN_OUT)
    save_to_archive(outpath, entity_vocab, metadata, model)
    log("Training complete, saved model archive to", outpath)

if __name__ == '__main__':
    with log.log_errors, EnvVars(TOKENIZERS_PARALLELISM=str(not "Tue").lower()):
        parser = Parser(ARGUMENTS, name="daLUKE-NER-finetune", multiple_jobs=False)
        experiments = parser.parse()
        parser.document_settings()
        log.configure(
            os.path.join(parser.location, "daluke_train_ner.log"), "Finetune daLUKE for Danish NER",
            print_level=Levels.INFO if experiments[0]["quieter"] else Levels.DEBUG,
        )
        for exp in experiments:
            run_experiment(exp)
