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
    linewidth = 2

    # Loss axis
    x = np.arange(len(res.losses)) + 1
    ax1.plot(x, res.losses, color=tab_colours[0], label="NER classification loss", linewidth=linewidth)

    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Cross Entropy Loss")

    h, l = ax1.get_legend_handles_labels()
    # Accuracy axis
    if res.running_evaluations:
        ax2 = ax1.twinx()
        x2 = (1 + np.arange(len(res.running_evaluations))) * len(res.losses) // len(res.running_evaluations)
        x2 = [1, *x2]

        ax2.plot(x2, 100*np.array([0]+[e.statistics["micro avg"]["f1-score"] for e in res.running_evaluations]),
            color=tab_colours[1], linewidth=linewidth, linestyle="-.", label="Micro avg. F1"
        )
        ax2.plot(x2, 100*np.array([0]+[e.statistics["micro avg"]["precision"] for e in res.running_evaluations]),
            color=tab_colours[2], linewidth=linewidth, linestyle="-.", label="Micro avg. precision", alpha=.5
        )
        ax2.plot(x2, 100*np.array([0]+[e.statistics["micro avg"]["recall"] for e in res.running_evaluations]),
            color=tab_colours[3], linewidth=linewidth, linestyle="-.", label="Micro avg. recall", alpha=.5
        )

        ax2.set_ylim([0, 110])
        ax2.set_ylabel("Running evaluation on dev. set [%]")
        #Micro avg. F1 [%]")
        h2, l2 = ax2.get_legend_handles_labels()
        h += h2
        l += l2

    ax1.legend(h, l)
    ax1.set_title("Fine-tuning loss")
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(location, "ner-plots", "loss.png"))
    plt.close()

def prediction_distribution_plot(location: str):
    res = TrainResults.load()
    types = sorted(list(set(res.true_type_distribution.keys()) - {"O"}))
    type_sequences = {t: list() for t in types}
    for dist in res.pred_distributions:
        for t in types:
            type_sequences[t].append(dist.get(t, 0))
    _, ax = plt.subplots(figsize=figsize_std)

    x = np.arange(1, len(res.pred_distributions)+1)
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
        ax.axhline(y=res.true_type_distribution[t], color=tab_colours[i], linestyle="--", alpha=.8)
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([0], [0], color="black", linestyle="--")]
    l += ["True annotation counts"]
    ax.legend(h, l)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("# spans predicted")
    ax.set_title("Entity predictions of dev. set during training")
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(location, "ner-plots", "dists.png"))
    plt.close()

@click.command()
@click.argument("location")
def make_finetuning_plots(location: str):
    log.configure(os.path.join(location, "ner-plots", "finetune-plot.log"), "Training plot of NER fine-tune")
    TrainResults.subfolder = location
    log("Loss and accuracy plot")
    loss_plot(location)
    log("Prediction distribution plot")
    prediction_distribution_plot(location)

if __name__ == "__main__":
    with log.log_errors:
        make_finetuning_plots()
