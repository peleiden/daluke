import os

import click
from pelutils.logger import log
from pelutils.ds.plot import figsize_std, tab_colours

import matplotlib.pyplot as plt
import numpy as np

from daluke.ner.analysis.representation_geometry import GeometryResults
from daluke.plot import setup_mpl

setup_mpl()

def pca_explained_plot(location: str):
    lambdas = GeometryResults.load().principal_components.numpy()
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

@click.command()
@click.argument("location")
def make_representation_plots(location: str):
    log.configure(os.path.join(location, "geometry-plots", "representations-plots.log"), "Visualizing contextualized represenations on NER")
    GeometryResults.subfolder = location
    log("PCA matrix plot")
    pca_matrix_plot(location)
    log("PCA explained plot")
    pca_explained_plot(location)

if __name__ == "__main__":
    with log.log_errors:
        make_representation_plots()
