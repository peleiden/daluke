#!/usr/bin/env python3
from __future__ import annotations
import json
from itertools import product
from abc import ABC, abstractmethod
import os
from copy import deepcopy
from typing import Any

import torch
from pelutils import log, Levels, Parser, set_seeds

from daluke.ner import load_dataset, load_model
from daluke.ner.model import NERDaLUKE, mutate_for_ner
from daluke.ner.run import ARGUMENTS as train_arguments
from daluke.ner.training import TrainNER
from daluke.ner.data import NERDataset,  Split
from daluke.ner.evaluation import NER_Results, evaluate_ner

from daluke.serialize import load_from_archive

EVAL_BATCH = 32

ARGUMENTS = {
    "sampler":     {"help": "What method to use to sample hyper params. Currently, only set-product", "default": "set-product"},
    "params":      {"help": "Path to json containing possible values of parameters", "default": None},
    **train_arguments
}

def f(d: dict) -> str:
    return json.dumps(d, indent=4)

class Sampler(ABC):
    @abstractmethod
    def __init__(self, param_lists: dict[str, list]):
        """ Create internal state """

    @abstractmethod
    def sample(self) -> dict[str, Any]:
        """ Return a dictionary of chosen arguments """

class SetProduct(Sampler):
    def __init__(self, param_lists: dict[str, list]):
        self.param_lists = param_lists
        self.product = list(product(*param_lists.values()))
        log(f"Created {len(self.product)} value combinations", list(self.param_lists.keys()), "\n".join(str(p) for p in self.product))

    def sample(self) -> None | dict[str, Any]:
        if not self.product:
            return None
        values = self.product.pop(0)
        return dict(zip(self.param_lists.keys(), values))

SAMPLERS = {
    "set-product": SetProduct,
}

def objective_function(model: NERDaLUKE, dataset: NERDataset, args: dict[str, Any]) -> NER_Results:
    dataloader = dataset.build(Split.TRAIN, args["batch_size"])
    dev_dataloader = dataset.build(Split.DEV, EVAL_BATCH)
    device = next(model.parameters()).device
    training = TrainNER(
        model,
        dataloader,
        dataset,
        device         = device,
        epochs         = args["epochs"],
        lr             = args["lr"],
        warmup_prop    = args["warmup_prop"],
        weight_decay   = args["weight_decay"],
        dev_dataloader = dev_dataloader,
        loss_weight    = args["batch_size"]
    )
    res = training.run()

    log.debug("Evaluating")
    best_res = res.running_dev_evaluations[res.best_epoch]
    log(f"Best model achieved {best_res.statistics['micro avg']['f1-score']} in mic-F1")
    return best_res

def optimize(model: NERDaLUKE, dataset: NERDataset, args: dict[str, Any], sampler: Sampler):
    results, tried_params = list(), list()
    best = None
    i = 0
    while (sampled_params := sampler.sample()) is not None:
        log.section(f"Sampling #{i}: chose", f(sampled_params))
        result = objective_function(deepcopy(model), dataset, {**args, **sampled_params})
        score = result.statistics["micro avg"]["f1-score"]
        if best is None or score > results[best].statistics["micro avg"]["f1-score"]:
            log(f"Found new best at F1 of {score}")
            best = i
        result.save(out := os.path.join(args['location'], f"res-optim{i}"))
        log.debug(f"Saved results to {out}")
        results.append(result)
        tried_params.append(sampled_params)
        i += 1
    log(f"Best result found with parameters", f(tried_params[best]), "resulting in", f(results[best].statistics))

def run_experiment(args: dict[str, Any]):
    set_seeds(seed=0)
    # Remove subfolder so we can control location directly
    NER_Results.subfolder = ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    entity_vocab, metadata, state_dict = load_from_archive(args["model"])
    state_dict, ent_embed_size = mutate_for_ner(state_dict, mask_id=entity_vocab["[MASK]"]["id"])

    log("Setting up sampler")
    with open(args["params"], "r") as f:
        param_lists = json.load(f)
    sampler = SAMPLERS[args["sampler"]](param_lists)

    log(f"Loading dataset {args['dataset']} ...")
    dataset = load_dataset(args, metadata, device)

    log("Loading model ...")
    model = load_model(state_dict, dataset, metadata, device, entity_embedding_size=ent_embed_size)

    optimize(model, dataset, args, sampler)

if __name__ == '__main__':
    with log.log_errors:
        parser = Parser(ARGUMENTS, name="hyper-optim.log", multiple_jobs=False)
        exp = parser.parse()
        parser.document_settings()
        log.configure(
            os.path.join(parser.location, "daluke-hyper-optim.log"), "Search for hyper parameters for daLUKE",
            print_level=Levels.INFO if exp["quieter"] else Levels.DEBUG,
        )
        run_experiment(exp)
