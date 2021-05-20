import os

import click
from pelutils.logger import log
from pelutils.ds.plot import figsize_std, tab_colours

import matplotlib.pyplot as plt

from daluke.ner.analysis.representation_geometry import GeometryResults
from daluke.plot import setup_mpl

setup_mpl()

def pca_explained_plot(location: str):
    res = GeometryResults.load()

def pca_matrix_plot(location: str):
    res = GeometryResults.load()

@click.command()
@click.argument("location")
def make_representation_plots(location: str):
    log.configure(os.path.join(location, "representation-plots", "representations-plots.log"), "Visualizing contextualized represenations on NER")
    GeometryResults.subfolder = location
    log("PCA matrix plot")
    pca_matrix_plot(location)
    log("PCA explained plot")
    pca_matrix_plot(location)

if __name__ == "__main__":
    with log.log_errors:
        make_representation_plots()
