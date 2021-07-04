from __future__ import annotations
import os

import click

import torch
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from pelutils.logger import log
from pelutils.ds.plot import figsize_std
from daluke.ner import load_dataset
from daluke.ner.data import Split, DaNE
from daluke.ner.evaluation import NER_Results, Split
from daluke.plot import setup_mpl

setup_mpl()

DEFAULT_METADATA = { #TODO: Make this user-customizable
    "max-seq-length":  512,
    "max-entities":    128,
    "max-entity-span": 30,
    "base-model":      "Maltehb/danish-bert-botxo",
}

COLORS = ["grey", "red", "gold", "blue", "green"]
CLASSES = (DaNE.null_label, *DaNE.labels)

def calibration_plot(preds, truths, location):
    _, ax = plt.subplots(figsize=figsize_std)
    for i, c in enumerate(CLASSES):
        if not i: continue # Exclude "O"

        log(f"Calibration for {c}")
        c_preds = [p[i] for p in preds]
        c_truths = [int(t==i) for t in truths]

        p_true, p_pred = calibration_curve(c_truths, c_preds, n_bins=4)
        ax.plot(p_pred, p_true, marker="o", linewidth=3, label=c, color=COLORS[i], linestyle="--")

    ax.set_ylim((0, 1))
    ax.set_xlim((0, 1))
    ax.plot([(0,0), (1,1)], color="black", linewidth=2)
    ax.set_title("Calibration of DaLUKE on DaNE test")
    ax.grid()
    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Predicted probability")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(location, f"calibration.png"))
    plt.close()

@click.command()
@click.argument("location")
def make_cal_plots(location: str):
    log.configure(os.path.join(location, "calibration-plot.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = NER_Results.load(location)

    log("Loading data")
    dataset = load_dataset(dict(dataset="DaNE"), DEFAULT_METADATA, device)
    dataloader = dataset.build(Split.TEST, 1, shuffle=False)
    log("Fetching probs and labels")
    truths = [dict() for _ in range(len(results.span_probs))]
    for _, ex in dataloader.dataset:
        truths[ex.text_num].update({s: l for s, l in zip(ex.entities.fullword_spans, ex.entities.labels)})
    flat_preds, flat_truths = list(), list()
    for p, t in zip(results.span_probs, truths):
        for k, probs in p.items():
            flat_preds.append(probs)
            flat_truths.append(t[k])
    log("Calibration plot")
    calibration_plot(flat_preds, flat_truths, location)

if __name__ == "__main__":
    with log.log_errors:
        make_cal_plots()
