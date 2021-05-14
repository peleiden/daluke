import os

import click
from pelutils.logger import log
from pelutils.ds.plot import figsize_std, tab_colours

import matplotlib.pyplot as plt
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

    # Accuracy axis
    if res.running_f1 is not None:
        ax2 = ax1.twinx()
        ax2.plot(100*np.array(res.running_f1), color=tab_colours[1], linewidth=linewidth, linestyle="-.", label="Running evaluation on dev. set")
        ax2.set_ylim([0, 110])
        ax2.set_ylabel("Micro avg. F1 [%]")


    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2)
    plt.title("Fine-tuning loss")
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(location, "ner-plots", "loss.png"))
    plt.close()

def prediction_distribution_plot(location: str):
    res = TrainResults.load()
    _, ax = plt.subplots(figsize=figsize_std)

@click.command()
@click.argument("location")
def make_finetuning_plots(location: str):
    log.configure(os.path.join(location, "ner-plots", "finetune-plot.log"), "Training plot of NER fine-tune")
    TrainResults.subfolder = location
    log("Loss and accuracy plot")
    loss_plot(location)
    log("Prediction distribution plot")

if __name__ == "__main__":
    with log.log_errors:
        make_finetuning_plots()
