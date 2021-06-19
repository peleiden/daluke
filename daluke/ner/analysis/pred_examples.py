from __future__ import annotations
import os

import torch

import click

from pelutils import log, Levels, Table

from daluke.ner import load_dataset
from daluke.ner.data import Split

from daluke.ner.evaluation import NER_Results
from daluke.ner.analysis.representation_examples import DUMMY_METADATA

classes = lambda s: {c.split("-")[-1] if "-" in c else c for c in s}

@click.command()
@click.argument("path")
@click.argument("pred")
@click.argument("truth")
def main(path: str, pred: str, truth: str):
    log.configure(
        os.path.join(path, f"prediction-examples-{pred}-{truth}.log"),
        print_level=Levels.DEBUG
    )
    log(f"Looking for examples where model predicted {pred}, but the truth was {truth}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = NER_Results.load(path)
    data = load_dataset(dict(dataset="DaNE"), DUMMY_METADATA, device).data[Split.TEST]
    for preds, truths, text in zip(res.preds, data.annotations, data.texts):
        if pred in classes(preds) and truth in classes(truths):
            t = Table()
            t.add_row(["Text:"] + text)
            t.add_row(["Truth:"] + truths)
            t.add_row(["Pred:"]  + preds)
            log(str(t).replace("|", ""), with_info=False)

if __name__ == "__main__":
    with log.log_errors:
        main()
