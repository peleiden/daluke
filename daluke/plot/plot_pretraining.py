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
from daluke.plot import double_running_avg, setup_mpl, setup_mpl_small_legend
setup_mpl()

DOTSIZE = 24


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
        plt.title("Loss during pretraining")
        plt.grid()
        plt.legend()
        self._save("loss.png")

    def scaled_loss_plot(self):
        plt.figure(figsize=figsize_std)

        plt.plot(self.x, self.res.scaled_loss[:self.lim], color=tab_colours[0])
        plt.title("Scaled loss")
        plt.yscale("log")
        plt.xlabel(self.xlabel)
        plt.ylabel("Loss")
        plt.grid()
        self._save("scaled_loss.png")

    def runtime_plot(self):
        plt.figure(figsize=figsize_std)

        plt.plot(self.x, self.res.runtime[:self.lim])
        plt.xlabel(self.xlabel)
        plt.ylabel("Runtime per batch [s]")
        plt.title("Runtime")
        plt.grid()
        self._save("runtime.png")

    def parameter_change_plot(self):
        setup_mpl_small_legend()
        plt.figure(figsize=figsize_std)

        x = self.x[::self.res.paramdiff_every]

        # Norms
        for key, n2 in self.res.paramdiff_1.items():
            ls = "-." if key.startswith("Encoder") else "-"
            slc = self.res.groups_to_slices[key]
            plt.plot(x, n2[np.arange(n2.size)*self.res.paramdiff_every<self.lim] / (slc.stop-slc.start), ls=ls, label=key)

        plt.title("Parameter changes")
        plt.xlabel(self.xlabel)
        plt.ylabel("Avg. absolute change")
        plt.legend()
        plt.grid()

        self._save("parameters.png")
        setup_mpl()

    def accuracy_plot(self):
        plt.figure(figsize=figsize_wide)
        for i, (data_train, data_val, label) in enumerate(
            ((self.res.w_accuracies, self.res.val_w_accuracies, "Word"), (self.res.e_accuracies, self.res.val_e_accuracies, "Entity"))
        ):
            plt.subplot(1, 2, i+1)
            colours = iter(tab_colours)
            for j, k in enumerate(self.res.top_k):
                c = next(colours)
                plt.plot(self.x, 100*data_train[:self.lim, j], alpha=0.4, color="gray")
                n = 1 if label == "Word" else 2
                x, y = double_running_avg(self.x, 100*data_train[:self.lim, j], inner_neighbors=n, outer_neighbors=1, samples=400)
                plt.plot(x, y, color=c, label="$k=%i$" % k)
                plt.scatter(self.val_x, 100*data_val[self.val_lim, j], s=DOTSIZE, c=c, label="Validation, $k=%i$" % k, edgecolors="black")

            plt.ylim((-5, 105))
            plt.title("Top-k %s accuracy" % label)
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
        plt.ylabel("Learning rate")
        m = np.ceil(np.log10(self.res.lr.max()))
        plt.ticklabel_format(axis="y", scilimits=(m, m))
        plt.title("Learning rate")
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
        bins = 300
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
                label=r"DaLUKE $\cap$ base model",
            )
            plt.plot(
                *self._bins(
                    self._sample(samples, not_from_base_params),
                    spacing=binning,
                    bins=bins,
                    weight=len(not_from_base_params)/len(model_params),
                ),
                label=r"DaLUKE $\backslash$ base model",
            )
            plt.title("Model parameter distribution after %i parameter updates" % (pu+1))
            plt.xlabel("Size of parameter")
            plt.ylabel("Density")
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
    plotter.parameter_change_plot()
    log("Weight distribution plot")
    plotter.weight_plot()
    log("Accuracy plot")
    plotter.accuracy_plot()
    log("Learning rate plot")
    plotter.lr_plot()

if __name__ == "__main__":
    with log.log_errors:
        make_pretraining_plots()
