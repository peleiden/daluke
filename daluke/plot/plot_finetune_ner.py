from __future__ import annotations
import os

import click
from pelutils.logger import log
from pelutils.ds.plot import figsize_std, tab_colours

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from daluke.ner.training import TrainResults
from daluke.plot import setup_mpl

setup_mpl()

def loss_plot(location: str):
    res = TrainResults.load()
    _, ax1 = plt.subplots(figsize=figsize_std)

    # Loss axis
    x = np.arange(len(res.losses)) + 1
    ax1.semilogy(x, res.losses, color=tab_colours[0], label="NER classification loss", linewidth=1)

    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Cross Entropy Loss (log. axis)")

    h, l = ax1.get_legend_handles_labels()
    # Accuracy axis
    if res.running_dev_evaluations:
        ax2 = ax1.twinx()
        x2 = (1 + np.arange(len(res.running_dev_evaluations))) * len(res.losses) // len(res.running_dev_evaluations)
        x2 = [1, *x2]

        ax2.plot(x2, 100*np.array([0]+[e.statistics["micro avg"]["f1-score"] for e in res.running_dev_evaluations]),
            color=tab_colours[1], linewidth=3, linestyle="-.", label="Dev. set F1", marker=".", markersize=10,
        )
        ax2.plot(x2, 100*np.array([0]+[s["micro avg"]["f1-score"] for s in res.running_train_statistics]),
            color=tab_colours[2], linewidth=3, linestyle="-.", label="Train. set F1", marker=".", markersize=10,
        )

        ax2.set_ylim([0, 110])
        ax2.set_ylabel("Running evaluation micro avg. F1 [%]")
        h2, l2 = ax2.get_legend_handles_labels()
        h += h2
        l += l2

    ax1.legend(h, l)
    ax1.set_title("NER Fine-tuning of daLUKE")
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(location, "finetune-plots", "loss.png"))
    plt.close()

def running_f1_detail_plot(location: str):
    res = TrainResults.load()
    if not res.running_train_statistics: return
    _, ax = plt.subplots(figsize=figsize_std)
    x = range(len(res.running_train_statistics)+1)
    cols = iter(tab_colours)
    train_stats = [(0,0,0), *[(s["micro avg"]["f1-score"], s["micro avg"]["precision"], s["micro avg"]["recall"]) for s in res.running_train_statistics]]
    dev_stats = [(0,0,0), *[(e.statistics["micro avg"]["f1-score"], e.statistics["micro avg"]["precision"], e.statistics["micro avg"]["recall"]) for e in res.running_dev_evaluations]]
    for stats, name in zip((train_stats, dev_stats), ("Train", "Dev")):
        ax.plot(x, [100*f1_score for f1_score, _, _ in stats], color=next(cols), linewidth=3, markersize=10, marker=".", label=f"{name} F1")
        ax.plot(x, [100*prec for _, prec, _ in stats], color=next(cols), linewidth=3, markersize=10, marker=".", linestyle="-.", label=f"{name} precision", alpha=.5)
        ax.plot(x, [100*rec for _, _, rec in stats], color=next(cols), linewidth=3, markersize=10, marker=".", linestyle="-.", label=f"{name} recall", alpha=.5)
    ax.set_xlabel("Epoch")
    ax.set_xticks(x)
    ax.set_ylim([0, 110])
    ax.set_ylabel("Running evaluation micro avg. F1 [%]")
    ax.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(location, "finetune-plots", "f1s.png"))
    plt.close()

def _do_prediction_distribution_plot(location: str, true_type_distribution: dict[str, int], pred_distributions: list[dict[str, int]], dataset: str):
    types = sorted(list(set(true_type_distribution.keys()) - {"O"}))
    type_sequences = {t: list() for t in types}
    for dist in pred_distributions:
        for t in types:
            type_sequences[t].append(dist.get(t, 0))
    _, ax = plt.subplots(figsize=figsize_std)

    x = np.arange(1, len(pred_distributions)+1)
    for i, t in enumerate(types):
        ax.plot(
            x,
            type_sequences[t],
            label=f"'{t}' predictions",
            color=tab_colours[i],
            linewidth=2,
            marker=".",
            markersize=20,
        )
        ax.axhline(y=true_type_distribution[t], color=tab_colours[i], linestyle="--", alpha=.8)
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([0], [0], color="black", linestyle="--")]
    l += ["True annotation counts"]
    ax.legend(h, l)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("# spans predicted")
    ax.set_title(f"Entity predictions of {dataset}. set during training")
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(location, "finetune-plots", f"{dataset}-dists.png"))
    plt.close()

def prediction_distribution_plots(location: str):
    res = TrainResults.load()
    _do_prediction_distribution_plot(location, res.dev_true_type_distribution, res.dev_pred_distributions, "dev")
    _do_prediction_distribution_plot(location, res.train_true_type_distribution, res.train_pred_distributions, "train")

@click.command()
@click.argument("location")
def make_finetuning_plots(location: str):
    log.configure(os.path.join(location, "finetune-plots", "finetune-plot.log"), "Training plot of NER fine-tune")
    TrainResults.subfolder = location
    log("Loss and accuracy plot")
    loss_plot(location)
    log("Prediction distribution plots")
    prediction_distribution_plots(location)
    log("Detailed plot of running F1's")
    running_f1_detail_plot(location)

if __name__ == "__main__":
    with log.log_errors:
        make_finetuning_plots()
