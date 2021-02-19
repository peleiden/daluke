#!/usr/bin/env python3
from __future__ import annotations
import os

from pelutils import log, Parser

from models import setup_models, ALL_MODELS
from data import setup_datasets, ALL_DATASETS
from evaluation import Evaluator

ALL_MODEL_NAMES   = " ".join(m.name for m in ALL_MODELS)
ALL_DATASET_NAMES = " ".join(d.name for d in ALL_DATASETS)

def main():
    options = {
        "models":
            {"default": "all", "type": str,
                "help": f"Models to test. Either `all` or space-seperated list of following models:\n{ALL_MODEL_NAMES}"
            },
        "datasets":
            {"default": "all", "type": str,
                "help": f"Datasets to test on. Either `all` or space-seperated list of following datasets:\n{ALL_DATASET_NAMES}"
            },
        "daner_path":
            {"default": "daner", "type": str,
                "help": "Path to the cloned repository ITUnlp/daner. Only needed if testing daner"
            },
    }
    parser = Parser(options, name="NER_Test")
    experiments = parser.parse()

    log.configure(
        os.path.join(parser.location, "danish_ner.log"), "Benchmark Danish NER models",
    )

    for experiment in experiments:
        run_experiment(experiment)

def run_experiment(args: dict[str, str]):
    if args["models"] == "all":
        args["models"] = ALL_MODEL_NAMES
    if args["datasets"] == "all":
        args["datasets"] = ALL_DATASET_NAMES

    models = setup_models(args["models"].split(), args["location"], daner_path=args["daner_path"])
    log(f"Succesfully set up {len(models)} models")

    datasets = setup_datasets(args["datasets"].split())
    log(f"Sucessfully acquired {len(datasets)} NER datasets")

    for model in models:
        for dataset in datasets:
            e = Evaluator(model, dataset)
            e.run()
            e.result.save(os.path.join(args["location"], "-".join((model.name, dataset.name))))

if __name__ == '__main__':
    with log.log_errors:
        main()
