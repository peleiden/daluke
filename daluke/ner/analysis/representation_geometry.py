from __future__ import annotations
from dataclasses import dataclass
import os

import torch
import click
from tqdm import tqdm

from pelutils import log, DataStorage

from daluke.serialize import load_from_archive, COLLECT_OUT
from daluke.ner import load_model, load_dataset
from daluke.ner.model import mutate_for_ner
from daluke.ner.data import Split

FP_SIZE = 32

@dataclass
class GeometryResults(DataStorage):
    pca_transformed: torch.Tensor
    principal_components: torch.Tensor

    subfolder = "geometry"

def collect_representations(modelpath: str, device: torch.device, target_device) -> torch.Tensor:
    entity_vocab, metadata, state_dict = load_from_archive(modelpath)
    state_dict, ent_embed_size = mutate_for_ner(state_dict, mask_id=entity_vocab["[MASK]"]["id"])
    log("Loading dataset")
    # Note: We dont fill out dict as we dont allow changing max-entities and max-entity-span here. If this results in an error for any dataset, we must change this.
    dataset = load_dataset(entity_vocab,  dict(dataset="DaNE"), metadata, device)
    dataloader = dataset.build(Split.TRAIN, FP_SIZE)
    log("Loading model")
    model = load_model(state_dict, dataset, metadata, device, entity_embedding_size=ent_embed_size)
    model.eval()

    log("Forward passing examples")
    batch_representations = list()
    for batch in tqdm(dataloader):
        # Use super class as we want the represenations
        word_representations, entity_representations = super(type(model), model).forward(batch)
        start_word_representations, end_word_representations = model.collect_start_and_ends(word_representations, batch)
        representations = torch.cat([start_word_representations, end_word_representations, entity_representations], dim=2)
        batch_representations.append(
            # Flatten batch dimensions
            representations.view(-1, representations.shape[-1]).contiguous().to(target_device)
        )
    return torch.cat(batch_representations)

def pca(A: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A is (# data points, # dimensions).
    k is number of eigenvalues used for projection
    """
    log.debug("Calculating covariance matrix")
    A_c = A - A.mean(dim=0)
    # As # data points >>> # dimensions (~1M vs. 2k), we do covariance of features
    covar = (A_c.T @ A_c) / (A_c.shape[0]-1)
    log.debug("Calculating eigenvalues ...")
    lambdas, Q = torch.linalg.eigh(covar)
    # Want it in eigenvalue-descending order
    lambdas, Q = lambdas.flip(0), Q.flip(1)
    log.debug("Transforming to PCA")
    P = Q[:, :k]
    Z = A_c @ P
    return Z, lambdas

@click.command()
@click.argument("path")
@click.option("--model", default = os.path.join("local_data", COLLECT_OUT))
@click.option("--n-components", default = 10, type=int)
@click.option("--pca-on-cpu", is_flag=True)
def main(path: str, model: str, n_components: int, pca_on_cpu: bool):
    log.configure(
        os.path.join(path, "geometry-analysis.log"), "daLUKE embedding geometry analysis",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        representations = collect_representations(model, device, target_device=torch.device("cpu") if pca_on_cpu else device)
    log("Performing principal component analysis")
    pca_transformed, principal_components = pca(representations, n_components)

    log(
        "Saved analysis results to",
        GeometryResults(
            pca_transformed      = pca_transformed,
            principal_components = principal_components,
        ).save(path),
    )

if __name__ == "__main__":
    with log.log_errors:
        main()
