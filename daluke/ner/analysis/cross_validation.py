#!/usr/bin/env python3
from __future__ import annotations
from itertools import chain
import os
from copy import deepcopy
from typing import Any

import torch
import numpy as np
from pelutils import log, Levels, Parser, set_seeds

from daluke.ner import load_dataset, load_model
from daluke.ner.model import NERDaLUKE, mutate_for_ner
from daluke.ner.run import ARGUMENTS as train_arguments
from daluke.ner.training import TrainNER, type_distribution
from daluke.ner.data import NERDataset, Sequences, Split
from daluke.ner.evaluation import NER_Results, evaluate_ner

from daluke.serialize import load_from_archive

EVAL_BATCH = 32

ARGUMENTS = {
    "k": {"help": "number of folds", "type": int, "default": 5},
    **train_arguments
}

def merge_data(splits: list[Sequences]) -> Sequences:
    return Sequences(
        texts               = list(chain(*[s.texts for s in splits])),
        annotations         = list(chain(*[s.annotations for s in splits])),
        sentence_boundaries = list(chain(*[s.sentence_boundaries for s in splits])),
    )

def random_divide(full_data: Sequences, k: int) -> list[Sequences]:
    I = np.arange(N := len(full_data.texts))
    np.random.shuffle(I)
    cv_splits = list()
    split_size = int(N * 1/k)
    for i in range(k):
        split = Sequences(
            texts               = list(),
            annotations         = list(),
            sentence_boundaries = list(),
        )
        # Add rest of data to last split
        extra = N if i == (k-1) else 0
        for j in I[i*split_size: (i+1)*split_size + extra]:
            split.texts.append(full_data.texts[j])
            split.annotations.append(full_data.annotations[j])
            split.sentence_boundaries.append(full_data.sentence_boundaries[j])
        cv_splits.append(split)
    return cv_splits

def cross_validate(model: NERDaLUKE, dataset: NERDataset, k: int, train_args: dict[str, Any]) -> list[NER_Results]:
    cv_splits = random_divide(
        merge_data(list(dataset.data.values())), k
    )
    results = list()
    log(f"Split into {k} subdatasets with lengths {[len(c.texts) for c in cv_splits]}")
    for i, test_data in enumerate(cv_splits):
        log.section(f"Cross-validation split {i}")
        train_data = merge_data([s for j, s in enumerate(cv_splits) if j != i])
        # Create split specific model and data
        split_model = deepcopy(model)
        split_dataset = deepcopy(dataset)
        split_dataset.data[Split.TRAIN] = train_data
        split_dataloader = split_dataset.build(Split.TRAIN, train_args["batch_size"])


        log("Training")
        split_dataset.document(split_dataloader, Split.TRAIN)
        type_distribution(split_dataset.data[Split.TRAIN].annotations)
        trainer = TrainNER(
            split_model,
            split_dataloader,
            split_dataset,
            device         = next(split_model.parameters()).device,
            epochs         = train_args["epochs"],
            lr             = train_args["lr"],
            warmup_prop    = train_args["warmup_prop"],
            weight_decay   = train_args["weight_decay"],
            dev_dataloader = None, # Don't eval
            loss_weight    = train_args["loss_weight"]
        )
        trainer.run()

        split_dataset.data[Split.TEST] = test_data
        split_test_dataloader = split_dataset.build(Split.TEST, EVAL_BATCH)

        log("Evaluation")
        split_dataset.document(split_dataloader, Split.TEST)
        type_distribution(split_dataset.data[Split.TEST].annotations)
        results.append(
            evaluate_ner(split_model, split_test_dataloader, split_dataset, trainer.device, Split.TEST, also_no_misc=False)
        )
    return results

def run_experiment(args: dict[str, Any]):
    set_seeds(seed=0)
    # Remove subolder so we can control location directly
    NER_Results.subfolder = ""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    entity_vocab, metadata, state_dict = load_from_archive(args["model"])
    state_dict, ent_embed_size = mutate_for_ner(state_dict, mask_id=entity_vocab["[MASK]"]["id"])

    log(f"Loading dataset {args['dataset']} ...")
    dataset = load_dataset(args, metadata, device)

    log("Loading model ...")
    model = load_model(
        state_dict,
        dataset,
        metadata,
        device,
        entity_embedding_size=ent_embed_size,
        bert_attention = args["bert_attention"],
        dropout = args["dropout"]
    )

    cv_results = cross_validate(model, dataset, args["k"], args)

    log(f"Saving results to {args['location']}")
    for i, r in enumerate(cv_results):
        r.save(os.path.join(args["location"], f"res-cv{i}"))
    log("Micro avg. F1 estimate", np.mean([r.statistics["micro avg"]["f1-score"] for r in cv_results]))

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
