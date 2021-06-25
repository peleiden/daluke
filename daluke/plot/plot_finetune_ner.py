from __future__ import annotations
import os

import click
from pelutils.logger import log
from pelutils.ds.plot import figsize_std, tab_colours, update_rc_params, rc_params

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from daluke.ner.training import TrainResults
from daluke.plot.plot_pretraining import PretrainingPlots

update_rc_params(rc_params)


def loss_plot(location: str):
    res = TrainResults.load()
    batches_per_epoch = len(res.losses) // (res.epoch+1)
    _, ax1 = plt.subplots(figsize=figsize_std)
    res.losses

    # Loss axis
    x = np.arange(len(res.losses)) + 1
    ax1.semilogy(x/batches_per_epoch, res.losses, color="gray", alpha=0.3)
    x, y = PretrainingPlots.rolling_avg(10, x, res.losses)
    ax1.semilogy(x/batches_per_epoch, y, color=tab_colours[0], label="Loss (Rolling Avg.)")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")

    h, l = ax1.get_legend_handles_labels()
    # Accuracy axis
    if res.running_dev_evaluations:
        ax2 = ax1.twinx()
        x = range(len(res.running_train_statistics)+1)

        ax2.plot(
            x, 100*np.array([0]+[e.statistics["micro avg"]["f1-score"] for e in res.running_dev_evaluations]),
            color=tab_colours[1], ms=15, label="Dev. Set F1", marker=".", lw=3,
        )
        ax2.plot(
            x, 100*np.array([0]+[s["micro avg"]["f1-score"] for s in res.running_train_statistics]),
            color=tab_colours[2], ms=15, label="Training Set F1", marker=".", lw=3,
        )

        ax2.set_ylim([0, 110])
        ax2.set_ylabel("Micro Avg. F1 [%]")
        h2, l2 = ax2.get_legend_handles_labels()
        h += h2
        l += l2

    ax1.legend(h, l, loc="lower right", framealpha=1, edgecolor=(0, 0, 0, 1))
    ax1.set_title("NER Fine-tuning of DaLUKE")
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(location, "finetune-plots", "loss.png"))
    plt.close()

def running_f1_detail_plot(location: str):
    res = TrainResults.load()
    if not res.running_train_statistics:
        return
    _, ax = plt.subplots(figsize=figsize_std)
    x = range(len(res.running_train_statistics)+1)
    cols = iter(tab_colours)
    train_stats = [(0,0,0), *[(s["micro avg"]["f1-score"], s["micro avg"]["precision"], s["micro avg"]["recall"]) for s in res.running_train_statistics]]
    dev_stats = [(0,0,0), *[(e.statistics["micro avg"]["f1-score"], e.statistics["micro avg"]["precision"], e.statistics["micro avg"]["recall"]) for e in res.running_dev_evaluations]]
    for stats, name in zip((train_stats, dev_stats), ("Training Set", "Dev. Set")):
        ax.plot(x, [100*f1_score for f1_score, _, _ in stats], color=next(cols), linewidth=3, ms=15, marker=".", label=f"{name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Micro Avg. F1 [%]")
    ax.set_xlim(left=0)
    ax.set_ylim([0, 110])
    plt.title("Running Performance of Fine-tuning")
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
            label=f"{t} predictions",
            color=tab_colours[i],
            linewidth=2,
            marker=".",
            markersize=20,
        )
        ax.axhline(y=true_type_distribution[t], color=tab_colours[i], linestyle="--", alpha=.8, lw=2)
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([0], [0], color="black", linestyle="--")]
    l += ["True annotation counts"]
    ax.legend(h, l)

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("# Spans Predicted")
    ax.set_title(f"Entity Predictions on {dataset.capitalize()}. Set During Fine-tuning")
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
