#!/usr/bin/env python3
from __future__ import annotations
import os

from pelutils import log, Parser

from models import setup_models, ALL_MODELS
from data import setup_datasets, ALL_DATASETS
from evaluation import Evaluator

ALL_MODEL_NAMES   = " ".join(m.name for m in ALL_MODELS)
ALL_DATASET_NAMES = " ".join(d.name for d in ALL_DATASETS)

ARGUMENTS = {
    "models":
        {"default": "all", "type": str,
            "help": f"Models to test. Either `all` or space-seperated list of following models:\n{ALL_MODEL_NAMES}"
        },
    "datasets":
        {"default": "all", "type": str,
            "help": f"Datasets to test on. Either `all` or space-seperated list of following datasets:\n{ALL_DATASET_NAMES}"
        },
    "daner":
        {"default": "daner", "type": str,
            "help": "Path to the cloned repository ITUnlp/daner. Only needed if testing daner"
        },
    "wikiann":
        {"default": "wikiann", "type": str,
            "help": "Path to folder containing WikiANN-da data set. Only needed if testing on dataset WikiANN.\n"\
                    "Dataset was downloaded from https://github.com/afshinrahimi/mmner"
        },
    "plank":
        {"default": "plank", "type": str,
            "help": "Path to the folder containing B. Plank data set. Only needed if testing on dataset Plank.\n"\
                    "Dataset was downloaded from https://github.com/bplank/danish_ner_transfer"
        },
}
def main():
    parser = Parser(ARGUMENTS, name="NER_Test", multiple_jobs=False)
    exp = parser.parse()

    log.configure(
        os.path.join(parser.location, "danish-ner.log"), "Benchmark Danish NER models",
    )

    run_experiment(exp)

def run_experiment(args: dict[str, str]):
    if args["models"] == "all":
        args["models"] = ALL_MODEL_NAMES
    if args["datasets"] == "all":
        args["datasets"] = ALL_DATASET_NAMES

    models = setup_models(args["models"].split(), args["location"], daner_path=args["daner"])
    log(f"Succesfully set up {len(models)} models")

    datasets = setup_datasets(args["datasets"].split(), wikiann_path=args["wikiann"], plank_path=args["plank"])
    log(f"Sucessfully acquired {len(datasets)} NER datasets")

    for model in models:
        for dataset in datasets:
            e = Evaluator(model, dataset)
            res = e.run()
            res.save(os.path.join(args["location"], "-".join((model.name, dataset.name))))

if __name__ == '__main__':
    with log.log_errors:
        main()
