from __future__ import annotations
import os

import torch
import numpy as np

import click

from pelutils import log, Levels, set_seeds

from daluke.ner import load_dataset
from daluke.ner.data import Sequences, Split

from daluke.ner.analysis.representation_geometry import GeometryResults

# OF_INTEREST = {
#     "pca_transformed": 1,
# }
OF_INTEREST = {
    # {field: axis}
    "pca_transformed" : 3,
    "tsne_transformed": 0,
    "umap_transformed": 1,
}

DUMMY_METADATA = {
    "max-seq-length": None,
    "max-entities": None,
    "max-entity-span": None,
    "base-model": "Maltehb/danish-bert-botxo",
}

def _show_examples(res: GeometryResults, X: np.ndarray, I: np.ndarray, data: Sequences):
    for i, idx in enumerate(I):
        num, span = res.content[idx]["text_num"], res.content[idx]["span"]
        t, a = [*data.texts[num]], data.annotations[num]
        t.insert(span[0], "{")
        t.insert(span[1]+1, "}")
        t = " ".join(t)
        log(f"{i} ({X[idx]}) {a[span[0]].split('-')[1] if '-' in a[span[0]] else a[span[0]]}: {t}\n", with_info=False)

@click.command()
@click.argument("path")
@click.option("--n", type=int, default=25)
def main(path: str, n: int):
    log.configure(
        os.path.join(path, "geometry-examples.log"), "daLUKE examples",
        print_level=Levels.DEBUG
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hardcoded to train
    data = load_dataset(dict(dataset="DaNE"), DUMMY_METADATA, device).data[Split.TRAIN]
    set_seeds()
    GeometryResults.subfolder = ""
    res = GeometryResults.load(path)
    for field, axis in OF_INTEREST.items():
        log.section(field)
        X = getattr(res, field)
        order = X[:, axis].argsort()

        log(f"Examples where dim. {axis} is high")
        _show_examples(res, X, order[::-1][:n], data)
        log(f"Examples where dim. {axis} is low")
        _show_examples(res, X, order[:n], data)

if __name__ == "__main__":
    with log.log_errors:
        main()
