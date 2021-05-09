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
    _, ax = plt.subplots(figsize=figsize_std)

    # Loss axis
    x = np.arange(len(res.losses)) + 1
    ax.plot(x, res.losses, color=tab_colours[0], label="NER classification", linewidth=2)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Cross Entropy Loss")

    ax.legend()
    plt.title("Fine-tuning loss")
    plt.grid()
    plt.tight_layout()

    plt.savefig(os.path.join(location, "ner-plots", "loss.png"))
    plt.close()

@click.command()
@click.argument("location")
def make_finetuning_plot(location: str):
    log.configure(os.path.join(location, "ner-plots", "finetune-plot.log"), "Training plot of NER fine-tune")
    TrainResults.subfolder = location
    log("Loss plot")
    loss_plot(location)

if __name__ == "__main__":
    with log.log_errors:
        make_finetuning_plot()
