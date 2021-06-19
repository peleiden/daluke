from __future__ import annotations
import os
from itertools import chain

import torch
import numpy as np

import click

from pelutils import log, Levels, Table

from daluke.ner import load_dataset

from daluke.ner.data import Split
from daluke.ner.evaluation import NER_Results, confusion_matrix, _format_confmat
from daluke.ner.analysis.representation_examples import DUMMY_METADATA

try:
    from reproduction.danish_ner.results import NER_TestResults
except ImportError:
    raise ImportError("This code requires access to the reproduction module")


def sequence_covar(preds: list[list[str]], other_preds: list[list[str]]) -> np.numeric:
    return (np.array(list(chain(*preds))) == np.array(list(chain(*other_preds)))).mean()

@click.command()
@click.argument("daluke-path")
@click.argument("other-path")
@click.option("--show", is_flag=True)
def main(daluke_path: str, other_path: str, show: bool):
    other_name = os.path.split(other_path)[-1]
    log.configure(
        os.path.join(daluke_path, f"comparison_with_{other_name}.log"),
        print_level=Levels.DEBUG
    )

    daluke_res = NER_Results.load(daluke_path)
    other_res = NER_TestResults.load(other_path)
    if show:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = load_dataset(dict(dataset="DaNE"), DUMMY_METADATA, device).data[Split.TEST]
        for da_preds, ot_preds, truths, text in zip(daluke_res.preds, other_res.predictions, data.annotations, data.texts):
            if da_preds != ot_preds:
                t = Table()
                t.add_row(["Text:"] + text)
                t.add_row(["Truth:"] + truths)
                t.add_row(["DaLUKE pred:"]  + da_preds)
                t.add_row([f"{other_name} pred:"]  + ot_preds)
                log(str(t).replace("|", ""), with_info=False)

    log(f"Confusion matrix with DaLUKE results ↓ and results from {other_name} →")
    log(
        _format_confmat(confusion_matrix(daluke_res.preds, other_res.predictions, ["LOC", "PER", "ORG", "MISC", "O"]))
    )
    log(f"Covar. {sequence_covar(daluke_res.preds, other_res.predictions)}")

    # for preds, truths, text in zip(res.preds, data.annotations, data.texts):
    #     if pred in classes(preds) and truth in classes(truths):
    #         t = Table()
    #         t.add_row(["Text:"] + text)
    #         t.add_row(["Truth:"] + truths)
    #         t.add_row(["Pred:"]  + preds)
    #         log(text)
    #         log(t, with_info=False)

if __name__ == "__main__":
    with log.log_errors:
        main()
