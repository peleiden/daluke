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
from daluke.plot import double_running_avg, setup_mpl
setup_mpl()

DOTSIZE = 30


class PretrainingPlots:

    def __init__(self, location: str):
        self.location = location
        self.res = TrainResults.load()
        self.x = np.arange(self.res.parameter_update + 1) + 1
        self.lim = self.res.parameter_update + 1
        self.val_lim = self.res.val_param_updates < self.lim
        self.val_x = self.res.val_param_updates[self.val_lim] + 1
        self.xlabel = "Parameter update"

    def _save(self, name: str):
        plt.tight_layout()
        plt.savefig(os.path.join(self.location, "plots", name))
        plt.close()

    def loss_plot(self):
        plt.figure(figsize=figsize_std)

        # Loss axis
        plt.plot(self.x, self.res.losses[:self.lim],   color=tab_colours[0], label="Weighted loss")
        plt.plot(self.x, self.res.w_losses[:self.lim], color=tab_colours[1], label="Word loss")
        plt.plot(self.x, self.res.e_losses[:self.lim], color=tab_colours[2], label="Entity loss")
        plt.scatter(self.val_x, self.res.val_losses[self.val_lim],   color=tab_colours[0], label="Val. weighted loss", edgecolors="black")
        plt.scatter(self.val_x, self.res.val_w_losses[self.val_lim], color=tab_colours[1], label="Val. word loss",     edgecolors="black")
        plt.scatter(self.val_x, self.res.val_e_losses[self.val_lim], color=tab_colours[2], label="Val. entity loss",   edgecolors="black")
        plt.xlabel(self.xlabel)
        plt.ylabel("Loss")
        plt.title("Loss During Pretraining")
        plt.grid()
        plt.legend()
        self._save("loss.png")

    def scaled_loss_plot(self):
        plt.figure(figsize=figsize_std)

        plt.plot(self.x, self.res.scaled_loss[:self.lim], color=tab_colours[0])
        plt.title("Scaled Loss")
        plt.yscale("log")
        plt.xlabel(self.xlabel)
        plt.ylabel("Loss")
        plt.grid()
        self._save("scaled_loss.png")

    def runtime_plot(self):
        plt.figure(figsize=figsize_std)

        plt.plot(self.x, self.res.runtime[:self.lim])
        plt.xlabel(self.xlabel)
        plt.ylabel("Runtime per Batch [s]")
        plt.title("Runtime")
        plt.grid()
        self._save("runtime.png")

    def parameter_plot(self):
        norm1 = self.res.param_diff_1[:self.lim]
        norm2 = self.res.param_diff_2[:self.lim]
        D_big = ((norm1 / norm2) ** 2) / self.res.orig_params.size

        _, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize_wide)

        # Norms
        ax1.plot(self.x, norm1, color=tab_colours[0], label="1-Norm")
        ax1.set_xlabel(self.xlabel)
        ax1.set_ylabel("1-Norm")

        ax1_ = ax1.twinx()
        ax1_.plot(self.x, norm2, color=tab_colours[1], label="2-Norm")
        ax1_.set_ylabel("2-Norm")

        h1, l1 = ax1.get_legend_handles_labels()
        h1_, l1_ = ax1_.get_legend_handles_labels()
        ax1.legend(h1+h1_, l1+l1_)
        ax1.grid()

        # D
        ax2.plot(self.x, 100*D_big, color=tab_colours[2])
        ax2.set_title(r"$D_{\operatorname{big}}$")
        ax2.set_xlabel(self.xlabel)
        ax2.set_ylabel(r"$D_{\operatorname{big}}$ [%]")
        ax1_.set_ylim(bottom=0)
        ax2.grid()

        plt.title("Parameter changes")
        self._save("parameters.png")

    def accuracy_plot(self):
        plt.figure(figsize=figsize_wide)
        for i, (data_train, data_val, label) in enumerate(
            ((self.res.w_accuracies, self.res.val_w_accuracies, "Word"), (self.res.e_accuracies, self.res.val_e_accuracies, "Entity"))
        ):
            plt.subplot(1, 2, i+1)
            colours = iter(tab_colours)
            for j, k in enumerate(self.res.top_k):
                c = next(colours)
                plt.plot(self.x, 100*data_train[:self.lim, j], alpha=0.3, color="gray")
                n = 2 if label == "Word" else 3
                x, y = double_running_avg(self.x, 100*data_train[:self.lim, j], inner_neighbors=n, samples=400)
                plt.plot(x, y, color=c, label="$k=%i$" % k)
                plt.scatter(self.val_x, 100*data_val[self.val_lim, j], s=DOTSIZE, c=c, label="Validation, $k=%i$" % k, edgecolors="black")

            plt.ylim((-5, 105))
            plt.title("Top-k %s Accuracy" % label)
            plt.xlabel(self.xlabel)
            if i == 0:
                plt.ylabel("Accuracy [%]")
            plt.legend()
            plt.grid()

        self._save("accuracy.png")

    def lr_plot(self):
        plt.figure(figsize=figsize_std)
        plt.plot(self.x, self.res.lr[:self.lim])
        plt.xlabel(self.xlabel)
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate")
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

        def plot_dist(self, pu: int):
            model_state_dict = torch.load(os.path.join(self.location, MODEL_OUT.format(i=pu)), map_location=torch.device("cpu"))
            del model_state_dict["word_embeddings.position_ids"]
            from_base = set.difference(set(model_state_dict), set.difference(self.res.luke_exclusive_params, self.res.att_mats_from_base))
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
            plt.title("Model Parameter Distribution After %i Parameter Updates" % (pu+1))
            plt.xlabel("Size of Parameter")
            plt.ylabel("Probability Density")
            plt.legend(loc=1)
            plt.grid()
            plt.xlim([-0.4, 0.4])
            plt.ylim(bottom=0)

        plt.figure(figsize=figsize_wide)
        plt.subplot(121)
        plot_dist(self, -1)
        plt.subplot(122)
        plot_dist(self, self.res.parameter_update)

        self._save("weights.png")

@click.command()
@click.argument("location")
def make_pretraining_plots(location: str):
    log.configure(os.path.join(location, "plots", "plots.log"), "Pretraining plots")
    TrainResults.subfolder = location
    plotter = PretrainingPlots(location)
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
