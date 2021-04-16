import os

import click
from pelutils.logger import log
from pelutils.ds.plot import figsize_std

import matplotlib.pyplot as plt

from daluke.pretrain.analysis import TrainResults


def _save(location: str, name: str):
    plt.tight_layout()
    plt.savefig(os.path.join(location, "plots", name))
    plt.close()

def loss_plot(location: str):
    res = TrainResults.load()
    fig, ax1 = plt.subplots(figsize=figsize_std)
    ax1.plot(res.losses.ravel(), label="Weighted loss")
    ax1.plot(res.w_losses.ravel(), label="Word loss")
    ax1.plot(res.e_losses.ravel(), label="Entity loss")
    plt.plot(res.losses)
    plt.ylim([0, 1.1 * max(res.losses)])
    plt.title("Pretraining loss and accuracy")
    plt.xlabel("Number of parameter updates")
    plt.ylabel("Loss")
    plt.grid()
    _save(location, "loss.png")

@click.command()
@click.argument("location")
def make_pretraining_plots(location: str):
    log.configure(os.path.join(location, "plots", "plots.log"), "Pretraining plots")
    TrainResults.subfolder = location
    loss_plot(location)

if __name__ == "__main__":
    with log.log_errors:
        make_pretraining_plots()
