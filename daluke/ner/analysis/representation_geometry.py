from __future__ import annotations
from dataclasses import dataclass
import os

import torch
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE

import click
from tqdm import tqdm

from pelutils import log, DataStorage, Levels

from daluke.serialize import load_from_archive, COLLECT_OUT
from daluke.ner import load_model, load_dataset
from daluke.ner.model import mutate_for_ner
from daluke.ner.data import Split

FP_SIZE = 32

@dataclass
class GeometryResults(DataStorage):
    pca_transformed: np.ndarray
    umap_transformed: np.ndarray
    tsne_transformed: np.ndarray

    labels: np.ndarray
    principal_components: np.ndarray

    subfolder = "geometry"

def collect_representations(modelpath: str, device: torch.device, target_device) -> tuple[np.ndarray, np.ndarray]:
    entity_vocab, metadata, state_dict = load_from_archive(modelpath)
    state_dict, ent_embed_size = mutate_for_ner(state_dict, mask_id=entity_vocab["[MASK]"]["id"])
    log("Loading dataset")
    # Note: We dont fill out dict as we dont allow changing max-entities and max-entity-span here. If this results in an error for any dataset, we must change this.
    dataset = load_dataset(entity_vocab,  dict(dataset="DaNE"), metadata, device)
    dataloader = dataset.build(Split.TRAIN, FP_SIZE,)
    log("Loading model")
    model = load_model(state_dict, dataset, metadata, device, entity_embedding_size=ent_embed_size)
    model.eval()

    log("Forward passing examples")
    batch_representations, labels = list(), list()
    for batch in tqdm(dataloader):
        # Use super class as we want the represenations
        word_representations, entity_representations = super(type(model), model).forward(batch)
        start_word_representations, end_word_representations = model.collect_start_and_ends(word_representations, batch)
        representations = torch.cat([start_word_representations, end_word_representations, entity_representations], dim=2)
        # We dont want padding
        mask = batch.entities.attention_mask.bool()
        batch_representations.append(
            representations[mask].contiguous().to(target_device)
        )
        labels.append(
            batch.entities.labels[mask].contiguous().to(target_device)
        )
    return torch.cat(batch_representations).numpy(), torch.cat(labels).numpy()

def pca(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    A is (# data points, # dimensions).
    k is number of eigenvalues used for projection
    """
    log.debug("Calculating covariance matrix")
    A_c = A - A.mean(0)
    # As # data points >>> # dimensions (~1M vs. 2k), we do covariance of features
    covar = (A_c.T @ A_c) / (A_c.shape[1]-1)
    log.debug("Calculating eigenvalues ...")
    lambdas, Q = np.linalg.eigh(covar)
    # Want it in eigenvalue-descending order
    lambdas, Q = lambdas[::-1], np.flip(Q, axis=1)
    log.debug("Transforming to PC space")
    P = Q[:, :k]
    Z = A_c @ P
    return Z, lambdas

def umap(A: np.ndarray) -> np.ndarray:
    reducer = UMAP()
    A_c = A - A.mean(0)
    return reducer.fit_transform(A_c)

def tsne(A: np.ndarray) -> np.ndarray:
    reducer = TSNE()
    A_c = A - A.mean(0)
    return reducer.fit_transform(A_c)

@click.command()
@click.argument("path")
@click.option("--model", default = os.path.join("local_data", COLLECT_OUT))
@click.option("--n-components", default = 10, type=int)
def main(path: str, model: str, n_components: int):
    log.configure(
        os.path.join(path, "geometry-analysis.log"), "daLUKE embedding geometry analysis",
        print_level=Levels.DEBUG
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        representations, labels = collect_representations(model, device, target_device=torch.device("cpu"))
    log("Performing principal component analysis")
    pca_transformed, principal_components = pca(representations, n_components)
    log("Running the UMAP algorithm")
    umap_transformed = umap(representations)
    log("Running the t-SNE algorithm")
    tsne_transformed = tsne(representations)

    log(
        "Saved analysis results to",
        GeometryResults(
            pca_transformed      = pca_transformed,
            umap_transformed     = umap_transformed,
            tsne_transformed     = tsne_transformed,
            labels               = labels,
            principal_components = principal_components,
        ).save(path),
    )

if __name__ == "__main__":
    with log.log_errors:
        main()
