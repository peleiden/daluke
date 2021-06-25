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


class PretrainingPlots:

    def __init__(self, location: str, label_epochs: False):
        self.location = location
        self.label_epochs = label_epochs
        self.res = TrainResults.load()
        self.x = np.arange(self.res.losses.size) + 1
        if label_epochs:
            self.x = self.x / self.res.losses.shape[1]
            self.xlabel = "Epoch"
        else:
            self.xlabel = "Batch"

    def _save(self, name: str):
        plt.tight_layout()
        plt.savefig(os.path.join(self.location, "plots", name))
        plt.close()

    def loss_plot(self):
        _, ax1 = plt.subplots(figsize=figsize_std)

        # Positions of epoch ends on x axis
        epochs = np.arange(self.res.epoch+1) * self.res.losses.shape[1]

        lw = 2

        # Loss axis
        ax1.plot(self.x, self.res.losses.ravel(),   color=tab_colours[0], label="Weighted loss", lw=lw)
        ax1.plot(self.x, self.res.w_losses.ravel(), color=tab_colours[1], label="Word loss",     lw=lw, ls="--")
        ax1.plot(self.x, self.res.e_losses.ravel(), color=tab_colours[2], label="Entity loss",   lw=lw, ls="--")
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel(self.xlabel)
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss During Pretraining")
        plt.grid()
        plt.legend(loc=1)
        self._save("loss.png")

    def scaled_loss_plot(self):
        epochs = np.arange(self.res.epoch+1) * self.res.scaled_loss.shape[1]
        plt.figure(figsize=figsize_std)
        plt.plot(self.x, self.res.scaled_loss.ravel(), color=tab_colours[0])
        plt.title("Scaled Loss")
        plt.yscale("log")
        plt.xlabel(self.xlabel)
        plt.ylabel("Loss")
        plt.grid()
        self._save("scaled_loss.png")

    def runtime_plot(self):
        plt.figure(figsize=figsize_std)

        plt.plot(self.x, self.res.runtime.ravel())
        plt.xlabel(self.xlabel)
        plt.ylabel("Runtime per Batch [s]")
        plt.title("Runtime")
        plt.grid()
        self._save("runtime.png")

    def parameter_plot(self):
        norm1 = self.res.param_diff_1.ravel()
        norm2 = self.res.param_diff_2.ravel()
        D_big = ((norm1 / norm2) ** 2) / self.res.orig_params.size

        _, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize_wide)

        # Norms
        ax1.plot(self.x, norm1, color=tab_colours[0], label="1-Norm")
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel(self.xlabel)
        ax1.set_ylabel("1-Norm")
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)

        ax1_ = ax1.twinx()
        ax1_.plot(self.x, norm2, color=tab_colours[1], label="2-Norm")
        ax1_.set_ylabel("2-Norm")
        ax1_.set_ylim(bottom=0)

        h1, l1 = ax1.get_legend_handles_labels()
        h1_, l1_ = ax1_.get_legend_handles_labels()
        ax1.legend(h1+h1_, l1+l1_)
        ax1.grid()

        # D
        ax2.plot(self.x, 100*D_big, color=tab_colours[2])
        ax2.set_title(r"$D_{\operatorname{big}}$")
        ax2.set_xlabel(self.xlabel)
        ax2.set_ylabel(r"$D_{\operatorname{big}}$ [%]")
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)
        ax2.grid()

        plt.title("Parameter changes")
        self._save("parameters.png")

    @staticmethod
    def rolling_avg(neighbours: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = np.empty(2*neighbours+1)
        w[:neighbours+1] = np.arange(1, neighbours+2)
        w[neighbours+1:] = np.arange(neighbours, 0, -1)
        w = w / w.sum()
        y = y.copy()
        for i in range(neighbours, len(x)-neighbours):
            y[i] = y[i-neighbours:i+neighbours+1] @ w
        return x[neighbours:-neighbours], y[neighbours:-neighbours]

    def accuracy_plot(self):
        plt.figure(figsize=figsize_wide)
        for i, (data, label) in enumerate(((self.res.w_accuracies, "Word"), (self.res.e_accuracies, "Entity"))):
            plt.subplot(1, 2, i+1)
            colours = iter(tab_colours)
            for j, k in enumerate(self.res.top_k):
                c = next(colours)
                plt.plot(self.x, 100*data[..., j].ravel(), alpha=0.3, color="gray")
                n = 7 if label == "Word" else 15
                x, y = self.rolling_avg(n, self.x, 100*data[..., j].ravel())
                plt.plot(x, y, color=c, label="$k=%i$" % k)

            plt.xlim(left=0)
            plt.ylim((0, 110))
            plt.title("Top-k %s Accuracy" % label)
            plt.xlabel(self.xlabel)
            if i == 0:
                plt.ylabel("Accuracy [%]")
            plt.legend(loc=2)
            plt.grid()

        self._save("accuracy.png")

    def lr_plot(self):
        plt.figure(figsize=figsize_std)
        plt.plot(self.x, self.res.lr.flat)
        plt.xlabel(self.xlabel)
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate")
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.grid()

        self._save("lr.png")

    @staticmethod
    def _bins(data, spacing=lambda x, b: np.linspace(min(x), max(x), b), bins=10, weight: float=1):
        bins = spacing(data, bins+1)
        hist, edges = np.histogram(data, bins=bins, density=True)
        x = (edges[1:] + edges[:-1]) / 2
        xx, yy = x[hist>0], hist[hist>0]
        return xx, yy * weight

    @staticmethod
    def _sample(samples: int, x: torch.Tensor) -> torch.Tensor:
        # Sample from a one-dimensional tensor
        return x[np.linspace(0, len(x)-1, min(samples, len(x)), dtype=int)]

    @staticmethod
    def _normal_binning(x: torch.Tensor, bins: int) -> np.ndarray:
        """ Creates bins that fits nicely to a normally distributed variable
        Bins are smaller close to the mean of x """
        dist = norm(x.mean(), 2*x.std())
        p = min(dist.cdf(x.min()), 1-dist.cdf(x.max()))
        uniform_spacing = np.linspace(p, 1-p, bins)
        return dist.ppf(uniform_spacing)

    @staticmethod
    def _cat(x: torch.Tensor) -> torch.Tensor:
        try:
            return torch.cat(x)
        except RuntimeError:
            return torch.Tensor()

    def weight_plot(self):
        bins = 200
        samples = 10 ** 8

        def plot_dist(self, epoch: int):
            model_state_dict = torch.load(os.path.join(self.location, MODEL_OUT.format(i=epoch)), map_location=torch.device("cpu"))
            del model_state_dict["word_embeddings.position_ids"]
            from_base = set.difference(set(model_state_dict), set.difference(self.res.luke_exclusive_params, self.res.q_mats_from_base))
            from_base_params = self._cat([p.view(-1) for n, p in model_state_dict.items() if n in from_base])
            model_params = self._cat([x.view(-1) for x in model_state_dict.values()])
            not_from_base_params = self._cat([p.view(-1) for n, p in model_state_dict.items() if n not in from_base])

            # Ensure same binning for all plots
            binning = lambda x, b: self._normal_binning(self._sample(samples, model_params), bins)

            plt.plot(
                *self._bins(
                    self._sample(samples, model_params),
                    spacing=binning,
                    bins=bins,
                ),
                label="DaLUKE",
            )
            plt.plot(
                *self._bins(
                    self._sample(samples, from_base_params),
                    spacing=binning,
                    bins=bins,
                    weight=len(from_base_params)/len(model_params),
                ),
                label=r"DaLUKE $\cap$ da-BERT",
            )
            plt.plot(
                *self._bins(
                    self._sample(samples, not_from_base_params),
                    spacing=binning,
                    bins=bins,
                    weight=len(not_from_base_params)/len(model_params),
                ),
                label=r"DaLUKE $\backslash$ da-BERT",
            )
            plt.title("Model Parameter Distribution After %i Epochs" % (epoch+1))
            plt.xlabel("Size of Parameter")
            if epoch == -1:
                plt.ylabel("Probability Density")
            plt.legend(loc=1)
            plt.grid()
            plt.xlim([-0.2, 0.2])
            plt.ylim(bottom=0)

        plt.figure(figsize=figsize_wide)
        plt.subplot(121)
        plot_dist(self, -1)
        plt.subplot(122)
        plot_dist(self, self.res.epoch)

        self._save("weights.png")

@click.command()
@click.argument("location")
def make_pretraining_plots(location: str):
    log.configure(os.path.join(location, "plots", "plots.log"), "Pretraining plots")
    TrainResults.subfolder = location
    plotter = PretrainingPlots(location, True)
    log("Loss plot")
    plotter.loss_plot()
    log("Scaled loss plot")
    plotter.scaled_loss_plot()
    log("Runtime plot")
    plotter.runtime_plot()
    log("Parameter plot")
    plotter.parameter_plot()
    log("Weight distribution plot")
    plotter.weight_plot()
    log("Accuracy plot")
    plotter.accuracy_plot()
    log("Learning rate plot")
    plotter.lr_plot()

if __name__ == "__main__":
    with log.log_errors:
        make_pretraining_plots()
