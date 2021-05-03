import os

import click
from pelutils.logger import log
from pelutils.ds.plot import figsize_std, tab_colours

import matplotlib.pyplot as plt
import numpy as np

from daluke.pretrain.analysis import TrainResults
from daluke.pretrain.train import Hyperparams
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
    ax1.set_xlabel("Batch")
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

    plt.figure(figsize=figsize_std)
    plt.plot(res.runtime.ravel())
    plt.ylim(bottom=0)
    plt.xlabel("Batch")
    plt.ylabel("Runtime [s]")
    plt.title("Runtime")
    plt.grid()
    _save(location, "runtime.png")

def parameter_plot(location: str):
    res = TrainResults.load()
    norm1 = res.param_diff_1.ravel()
    norm2 = res.param_diff_2.ravel()
    D_big = (norm1 / norm2) ** 2

    fig, ax1 = plt.subplots(figsize=figsize_std)

    ax1.plot(norm1, color=tab_colours[0], label="1-norm")
    ax1.plot(norm2, color=tab_colours[1], label="2-norm")
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Distance to original parameters")

    # Accuracy axis
    ax2 = ax1.twinx()
    ax2.plot(D_big, color=tab_colours[2], label=r"$D_{\operatorname{big}}$")
    ax2.set_ylabel("Estimated number of big parameter changes")
    ax2.set_ylim(bottom=0)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2)
    plt.title("Parameter changes")
    plt.grid()
    _save(location, "parameters.png")

def accuracy_plot(location: str):
    res = TrainResults.load()

    plt.figure(figsize=figsize_std)
    for i, k in enumerate(res.top_k):
        plt.plot(100*res.w_accuracies[..., i].ravel(), label="Words, $k=%i$" % k)
    for i, k in enumerate(res.top_k):
        plt.plot(100*res.e_accuracies[..., i].ravel(), label="Entities, $k=%i$" % k)
    plt.ylim([0, 110])
    plt.xlabel("Batch")
    plt.ylabel("Accuracy [%]")
    plt.title("Top-k accuracy")
    plt.legend(loc=2)
    plt.grid()

    _save(location, "accuracy.png")

@click.command()
@click.argument("location")
def make_pretraining_plots(location: str):
    log.configure(os.path.join(location, "plots", "plots.log"), "Pretraining plots")
    TrainResults.subfolder = location
    log("Loss plot")
    loss_plot(location)
    log("Runtime plot")
    runtime_plot(location)
    log("Parameter plot")
    parameter_plot(location)
    log("Accuracy plot")
    accuracy_plot(location)

if __name__ == "__main__":
    with log.log_errors:
        make_pretraining_plots()
