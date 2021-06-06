from __future__ import annotations
import os

import click
from pelutils.logger import log
from pelutils.ds.plot import figsize_std, figsize_wide, tab_colours
from scipy.stats import norm

import matplotlib.pyplot as plt
import numpy as np
import torch

from daluke.pretrain.train import MODEL_OUT
from daluke.analysis.pretrain import TrainResults
from daluke.plot import setup_mpl
setup_mpl()

DOTSIZE = 30


def _save(location: str, name: str):
    plt.tight_layout()
    plt.savefig(os.path.join(location, "plots", name))
    plt.close()

def loss_plot(location: str):
    res = TrainResults.load()
    _, ax1 = plt.subplots(figsize=figsize_std)

    # Positions of epoch ends on x axis
    epochs = np.arange(res.epoch+1) * res.losses.shape[1]

    lw = 2

    # Loss axis
    x = np.arange(res.losses.size) + 1
    ax1.plot(x, res.losses.ravel(),   color=tab_colours[0], label="Weighted loss", lw=lw)
    ax1.plot(x, res.w_losses.ravel(), color=tab_colours[1], label="Word loss",     lw=lw, ls="--")
    ax1.plot(x, res.e_losses.ravel(), color=tab_colours[2], label="Entity loss",   lw=lw, ls="--")
    ax1.scatter(epochs, res.losses[:, 0],   s=DOTSIZE, color=tab_colours[0])
    ax1.scatter(epochs, res.w_losses[:, 0], s=DOTSIZE, color=tab_colours[1])
    ax1.scatter(epochs, res.e_losses[:, 0], s=DOTSIZE, color=tab_colours[2])
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss During Pretraining")
    plt.grid()
    plt.legend(loc=1)
    _save(location, "loss.png")

def scaled_loss_plot(location: str):
    res = TrainResults.load()

    epochs = np.arange(res.epoch+1) * res.scaled_loss.shape[1]
    plt.figure(figsize=figsize_std)
    plt.plot(res.scaled_loss.ravel(), color=tab_colours[0])
    plt.scatter(epochs, res.scaled_loss[:, 0], s=DOTSIZE, color=tab_colours[0])
    plt.title("Scaled Loss")
    plt.yscale("log")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid()
    _save(location, "scaled_loss.png")

def runtime_plot(location: str):
    res = TrainResults.load()
    plt.figure(figsize=figsize_std)

    plt.plot(res.runtime.ravel())
    plt.xlabel("Batch")
    plt.ylabel("Runtime [s]")
    plt.title("Runtime")
    plt.grid()
    _save(location, "runtime.png")

def parameter_plot(location: str):
    res = TrainResults.load()
    norm1 = res.param_diff_1.ravel()
    norm2 = res.param_diff_2.ravel()
    D_big = ((norm1 / norm2) ** 2) / res.orig_params.size
    epochs = np.arange(res.epoch+1) * res.param_diff_1.shape[1]

    _, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize_wide)

    # Norms
    ax1.plot(norm1, color=tab_colours[0], label="1-Norm")
    ax1.scatter(epochs, res.param_diff_1[:, 0], s=DOTSIZE, color=tab_colours[0])
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("1-Norm")
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    ax1_ = ax1.twinx()
    ax1_.plot(norm2, color=tab_colours[1], label="2-Norm")
    ax1_.scatter(epochs, res.param_diff_2[:, 0], s=DOTSIZE, color=tab_colours[1])
    ax1_.set_ylabel("2-Norm")
    ax1_.set_ylim(bottom=0)

    h1, l1 = ax1.get_legend_handles_labels()
    h1_, l1_ = ax1_.get_legend_handles_labels()
    ax1.legend(h1+h1_, l1+l1_)
    ax1.grid()

    # D
    ax2.plot(100*D_big, color=tab_colours[2])
    ax2.scatter(epochs, 100*D_big[epochs], s=DOTSIZE, color=tab_colours[2])
    ax2.set_title(r"$D_{\operatorname{big}}$")
    ax2.set_xlabel("Batch")
    ax2.set_ylabel(r"$D_{\operatorname{big}}$ [%]")
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.grid()

    plt.title("Parameter changes")
    _save(location, "parameters.png")

def _rolling_avg(neighbours: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.empty(2*neighbours+1)
    w[:neighbours+1] = np.arange(1, neighbours+2)
    w[neighbours+1:] = np.arange(neighbours, 0, -1)
    w = w / w.sum()
    y = y.copy()
    for i in range(neighbours, len(x)-neighbours):
        y[i] = y[i-neighbours:i+neighbours+1] @ w
    return x[neighbours:-neighbours], y[neighbours:-neighbours]

def accuracy_plot(location: str):
    res = TrainResults.load()

    plt.figure(figsize=figsize_wide)
    epochs = np.arange(res.epoch+1) * res.w_accuracies.shape[1]
    for i, (data, label) in enumerate(((res.w_accuracies, "Word"), (res.e_accuracies, "Entity"))):
        plt.subplot(1, 2, i+1)
        colours = iter(tab_colours)
        for j, k in enumerate(res.top_k):
            c = next(colours)
            if k <= 10:
                plt.plot(100*data[..., j].ravel(), color=c, label="$k=%i$" % k)
                plt.scatter(epochs, 100*data[:, 0, j], color=c, s=DOTSIZE)
            else:
                plt.plot(100*data[..., j].ravel(), alpha=0.3, color="gray")
                n = 5
                x = np.arange(data[..., j].size)
                x, y = _rolling_avg(n, x, 100*data[..., j].ravel())
                plt.plot(x, y, color=c, label="$k=%i$" % k)
                plt.scatter(x[epochs[1:]]-n, y[epochs[1:]-n], color=c, s=DOTSIZE)

        plt.xlim(left=0)
        plt.ylim([0, 110])
        plt.title("Top-k %s Accuracy" % label)
        plt.xlabel("Batch")
        if i == 0:
            plt.ylabel("Accuracy [%]")
        plt.legend(loc=2)
        plt.grid()

    _save(location, "accuracy.png")

def lr_plot(location: str):
    res = TrainResults.load()

    plt.figure(figsize=figsize_std)
    plt.plot(res.lr.flat)
    plt.xlabel("Batch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid()

    _save(location, "lr.png")

def _bins(data, spacing=lambda x, b: np.linspace(min(x), max(x), b), bins=10, weight: float=1):
    bins = spacing(data, bins+1)
    hist, edges = np.histogram(data, bins=bins, density=True)
    x = (edges[1:] + edges[:-1]) / 2
    xx, yy = x[hist>0], hist[hist>0]
    return xx, yy * weight

def _sample(samples: int, x: torch.Tensor) -> torch.Tensor:
    # Sample from a one-dimensional tensor
    return x[np.linspace(0, len(x)-1, min(samples, len(x)), dtype=int)]

def _normal_binning(x: np.ndarray, b: int) -> np.ndarray:
    """ Creates bins that fit nicely to a normally distributed variable """
    unispace = np.linspace(0, 1, b+2)[1:-1]
    dist = norm(x.mean(), x.std())
    return dist.ppf(unispace)

def weight_plot(location: str):
    res = TrainResults.load()
    bins = 50
    samples = 10 ** 6

    def plot_dist(epoch: int):
        model_state_dict = torch.load(os.path.join(location, MODEL_OUT.format(i=epoch)), map_location=torch.device("cpu"))
        del model_state_dict["word_embeddings.position_ids"]
        from_base = set.difference(set(model_state_dict), set.difference(res.luke_exclusive_params, res.q_mats_from_base))
        from_base_params = torch.cat([p.view(-1) for n, p in model_state_dict.items() if n in from_base])
        not_from_base_params = torch.cat([p.view(-1) for n, p in model_state_dict.items() if n not in from_base])
        model_params = torch.cat([x.view(-1) for x in model_state_dict.values()])

        plt.plot(
            *_bins(
                _sample(samples, model_params),
                spacing=_normal_binning,
                bins=bins,
            ),
            label="DaLUKE",
        )
        plt.plot(
            *_bins(
                _sample(samples, from_base_params),
                spacing=_normal_binning,
                bins=bins,
                weight=len(from_base_params)/len(model_params),
            ),
            label=r"DaLUKE $\cap$ da-BERT",
        )
        plt.plot(
            *_bins(
                _sample(samples, not_from_base_params),
                spacing=_normal_binning,
                bins=bins,
                weight=len(not_from_base_params)/len(model_params),
            ),
            label=r"DaLUKE $\backslash$ da-BERT",
        )
        plt.title("Model Parameter Distribution After %i Epochs" % (epoch+1))
        plt.xlabel("Size of Parameter")
        if epoch == -1:
            plt.ylabel("Probability Density")
        plt.yscale("log")
        plt.legend(loc=1)
        plt.grid()

    plt.figure(figsize=figsize_wide)
    plt.subplot(121)
    plot_dist(-1)
    plt.subplot(122)
    plot_dist(res.epoch)

    _save(location, "weights.png")

@click.command()
@click.argument("location")
def make_pretraining_plots(location: str):
    log.configure(os.path.join(location, "plots", "plots.log"), "Pretraining plots")
    TrainResults.subfolder = location
    log("Loss plot")
    loss_plot(location)
    log("Scaled loss plot")
    scaled_loss_plot(location)
    log("Runtime plot")
    runtime_plot(location)
    log("Parameter plot")
    parameter_plot(location)
    log("Weight distribution plot")
    weight_plot(location)
    log("Accuracy plot")
    accuracy_plot(location)
    log("Learning rate plot")
    lr_plot(location)

if __name__ == "__main__":
    with log.log_errors:
        make_pretraining_plots()
