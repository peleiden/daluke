import json
import os


import click
import numpy as np
import torch
from pelutils import log
from pelutils.ds import no_grad
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForPreTraining

from daluke.pretrain.data import DataLoader
from daluke.pretrain.data.build import DatasetBuilder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.command()
@click.argument("datadir")
@click.option("--ff-size", default=32)
@no_grad
def word_preds(datadir: str, ff_size: int):
    log.configure(os.path.join(datadir, "dabert-word-preds.log"), "daBERT word predictions")
    log("Loading metadata")
    with open(os.path.join(datadir, DatasetBuilder.metadata_file)) as f:
        metadata = json.load(f)
    log("Loading model")
    dabert = AutoModelForPreTraining.from_pretrained("Maltehb/danish-bert-botxo").to(device)
    log("Loading data")
    dataloader = DataLoader(
        datadir,
        metadata,
        dict(),
        device,
    )
    loader = dataloader.get_dataloader(ff_size, None)
    log("Forward passing")
    correct_preds = np.zeros(len(loader))
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        logits = dabert(batch.words.ids).prediction_logits
        masked_logits = logits[batch.word_mask]
        preds = masked_logits.argmax(dim=1)
        correct_preds[i] = (preds == batch.word_mask_labels).float().mean().cpu()
    log(
        "MLM token prediction accuracy",
        "  Mean: %.4f %%" % (100 * correct_preds.mean()),
        "  Std.: %.4f %%" % (100 * correct_preds.std(ddof=1)),
    )


if __name__ == "__main__":
    with log.log_errors:
        word_preds()
