import os

import click
from pelutils.logger import log
from pelutils.ds.plot import figsize_std, tab_colours

import matplotlib.pyplot as plt
import numpy as np

from daluke.pretrain.analysis import TrainResults
from daluke.plot import setup_mpl
setup_mpl()


def _save(location: str, name: str):
    plt.tight_layout()
    plt.savefig(os.path.join(location, "plots", name))
    plt.close()

def loss_plot(location: str):
    res = TrainResults.load()
    fig, ax1 = plt.subplots(figsize=figsize_std)

    # Positions of epoch ends on x axis
    epochs = np.arange(res.epoch+1) * res.losses.shape[1]

    lw = 2
    dot_size = 30

    # Loss axis
    x = np.arange(res.losses.size) + 1
    ax1.plot(x, res.losses.ravel(),   color=tab_colours[0], label="Weighted loss", lw=lw)
    ax1.plot(x, res.w_losses.ravel(), color=tab_colours[1], label="Word loss",     lw=lw, ls="--")
    ax1.plot(x, res.e_losses.ravel(), color=tab_colours[2], label="Entity loss",   lw=lw, ls="--")
    ax1.scatter(epochs, res.losses[:, 0],   s=dot_size, color=tab_colours[0])
    ax1.scatter(epochs, res.w_losses[:, 0], s=dot_size, color=tab_colours[1])
    ax1.scatter(epochs, res.e_losses[:, 0], s=dot_size, color=tab_colours[2])
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Batches")
    ax1.set_ylabel("Loss")

    # Accuracy axis
    ax2 = ax1.twinx()
    ax2.plot(x, 100*res.w_accuracies[..., 0].ravel(), color=tab_colours[3], label="Masked word accuracy",   lw=lw, ls="-.")
    ax2.plot(x, 100*res.e_accuracies[..., 0].ravel(), color=tab_colours[4], label="Masked entity accuracy", lw=lw, ls="-.")
    ax2.scatter(epochs, 100*res.w_accuracies[:, 0, 0], s=dot_size, color=tab_colours[3])
    ax2.scatter(epochs, 100*res.e_accuracies[:, 0, 0], s=dot_size, color=tab_colours[4])
    ax2.set_ylim([0, 110])
    ax2.set_ylabel("Accuracy [%]")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2)
    plt.title("Pretraining loss and accuracy")
    plt.grid()
    _save(location, "loss.png")

def runtime_plot(location: str):
    res = TrainResults.load()
    runtime = res.runtime.ravel()
    x = np.arange(runtime.size+1)

    fig, ax1 = plt.subplots(figsize=figsize_std)

    ax1.plot(x[1:], runtime, color=tab_colours[0], label="Runtime")
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Batches")
    ax1.set_ylabel("Runtime [s]")

    # Accuracy axis
    ax2 = ax1.twinx()
    ax2.plot(x, [0, *(res.param_diff_1 / res.param_diff_2).ravel()**2], color=tab_colours[1], label="Est. big param. changes")
    ax2.set_ylabel("Estimated number of big parameter changes")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2)
    plt.title("Runtime and parameter change")
    plt.grid()
    _save(location, "runtime.png")

@click.command()
@click.argument("location")
def make_pretraining_plots(location: str):
    log.configure(os.path.join(location, "plots", "plots.log"), "Pretraining plots")
    TrainResults.subfolder = location
    log("Loss plot")
    loss_plot(location)
    log("Runtime plot")
    runtime_plot(location)

if __name__ == "__main__":
    with log.log_errors:
        make_pretraining_plots()
