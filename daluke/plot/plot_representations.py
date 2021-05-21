import os

import click
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from pelutils.logger import log
from pelutils.ds.plot import tab_colours, figsize_std

from daluke.ner.analysis.representation_geometry import GeometryResults
from daluke.plot import setup_mpl

setup_mpl()

def _scatter_transformed(Z1: np.ndarray, Z2: np.ndarray, labels: np.ndarray, axis):
    colors = ["grey", "red", "yellow", "blue", "green"]
    cdict = dict(enumerate(colors))

    nulls = labels == 0
    axis.scatter(Z1[nulls], Z2[nulls], c=colors[0], alpha=.25)
    axis.scatter(Z1[~nulls], Z2[~nulls], c=[cdict[l] for l in labels[~nulls]], alpha=.5)

def pca_explained_plot(location: str):
    lambdas = GeometryResults.load().principal_components
    show_k = 100

    _, ax = plt.subplots(figsize=figsize_std)
    ax.plot(range(0, show_k+1), [0, *np.cumsum(lambdas)[:show_k]/np.sum(lambdas)*100], color=tab_colours[0], linestyle="--", marker=".")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Data variance explained [%]")
    ax.set_title("PCA on daLUKE contextualized entity representations for DaNE")
    ax.set_ylim(bottom=0, top=110)

    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(location, "geometry-plots", "variance_explained.png"))
    plt.close()

def pca_matrix_plot(location: str):
    res = GeometryResults.load()
    V = res.pca_transformed
    N = 4

    _, axes = plt.subplots(N-1, N-1, figsize=(20, 20))
    # TODO: Dont create unused subplots
    # j, i are switched around to get lower triangle
    for (j, i) in combinations(range(N), 2):
        ax = axes[i-1, j]
        _scatter_transformed(V[:, j], V[:, i], res.labels, ax)
        ax.set_xlabel(f"PC {j+1}")
        ax.set_ylabel(f"PC {i+1}")

    plt.tight_layout()
    plt.savefig(os.path.join(location, "geometry-plots", "pca_matrix.png"))
    plt.close()

def umap_plot(location: str):
    res = GeometryResults.load()
    _, ax = plt.subplots(figsize=figsize_std)
    _scatter_transformed(res.umap_transformed[:, 0], res.umap_transformed[:, 1], res.labels[:len(res.umap_transformed)], ax)
    ax.set_title("UMAP space of daLUKE contextualized entity representations for DaNE")

    plt.tight_layout()
    plt.savefig(os.path.join(location, "geometry-plots", "umap.png"))
    plt.close()

def tsne_plot(location: str):
    res = GeometryResults.load()
    _, ax = plt.subplots(figsize=figsize_std)
    _scatter_transformed(res.tsne_transformed[:, 0], res.tsne_transformed[:, 1], res.labels[:len(res.tsne_transformed)], ax)
    ax.set_title("t-SNE space of daLUKE contextualized entity representations for DaNE")

    plt.tight_layout()
    plt.savefig(os.path.join(location, "geometry-plots", "tsne.png"))
    plt.close()

@click.command()
@click.argument("location")
def make_representation_plots(location: str):
    log.configure(os.path.join(location, "geometry-plots", "representations-plots.log"), "Visualizing contextualized represenations on NER")
    GeometryResults.subfolder = location
    log("PCA matrix plot")
    pca_matrix_plot(location)
    log("PCA explained plot")
    pca_explained_plot(location)
    log("UMAP plot")
    umap_plot(location)
    log("t-SNE plot")
    tsne_plot(location)

if __name__ == "__main__":
    with log.log_errors:
        make_representation_plots()
