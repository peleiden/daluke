import json
import os

import torch
from torch.utils.data import RandomSampler

import click
from pelutils import log
from pelutils.ds import no_grad
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForPreTraining

from daluke.pretrain.data import DataLoader
from daluke.pretrain.data.build import DatasetBuilder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command()
@click.argument("datadir")
@click.argument("--ff-size", default=32)
@no_grad
def word_preds(datadir: str, ff_size: int):
    log.configure(os.path.join(datadir, "dabert-word-preds.log"), "daBERT word predictions")
    log("Loading metadata")
    with open(os.path.join(datadir, DatasetBuilder.metadata_file)) as f:
        metadata = json.load(f)
    log("Loading model")
    dabert = AutoModelForPreTraining.from_pretrained("Maltehb/danish-bert-botxo")
    log("Loading data")
    dataloader = DataLoader(
        datadir,
        metadata,
        dict(),
        device,
    )
    loader = dataloader.get_dataloader(ff_size, RandomSampler([ex.words for ex in dataloader.examples]))
    log("Forward passing")
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        breakpoint()

