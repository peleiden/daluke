from __future__ import annotations
import json
import os
import tarfile
import tempfile

import torch
from torch import nn

from pelutils import log

OUT_FILE  = "daluke.tar.gz"
MODEL_OUT = "model.bin"

VOCAB_FILE    = "entity-vocab.json"
METADATA_FILE = "metadata.json"

def load_from_archive(model: str) -> (list[dict], dict, dict):
    """
    Reads the directory or luke.tar.gz archive containing the model, the entity vocabulary and metadata
    """
    if os.path.isdir(model):
        with open(os.path.join(model, VOCAB_FILE), "r") as vfile:
            entity_vocab = json.load(vfile)
        with open(os.path.join(model, METADATA_FILE), "r") as metafile:
            metadata = json.load(metafile)
        with open(os.path.join(model, MODEL_OUT), "rb") as mfile:
            state_dict = torch.load(mfile, map_location="cpu")
        return entity_vocab, metadata, state_dict
    with tarfile.open(model, "r:gz") as tar:
        log.debug(f"Extracting {VOCAB_FILE} and {METADATA_FILE} ...")
        with tar.extractfile(tar.getmember(VOCAB_FILE)) as vfile:
            entity_vocab = json.load(vfile)
        with tar.extractfile(tar.getmember(METADATA_FILE)) as metafile:
            metadata = json.load(metafile)
        log.debug(f"Extracting {MODEL_OUT} ...")
        with tar.extractfile(tar.getmember(MODEL_OUT)) as mfile:
            state_dict = torch.load(mfile, map_location="cpu")
    return entity_vocab, metadata, state_dict

def save_to_archive(outfile: str, entity_vocab: list[dict], metadata: dict, model: nn.Module):
    log.debug(f"Compressing {VOCAB_FILE} and {METADATA_FILE} ...")
    with tarfile.open(outfile, "w:gz") as tar:
        with tempfile.NamedTemporaryFile("w+") as tmp:
            tmp.write(json.dumps(entity_vocab))
            tmp.seek(0)
            tar.add(tmp.name, arcname=VOCAB_FILE)
        with tempfile.NamedTemporaryFile("w+") as tmp:
            tmp.write(json.dumps(metadata))
            tmp.seek(0)
            tar.add(tmp.name, arcname=METADATA_FILE)
        log.debug(f"Compressing {MODEL_OUT} ...")
        with tempfile.NamedTemporaryFile("wb+") as tmp:
            model.to(torch.device("cpu"))
            torch.save(model.state_dict(), tmp)
            tar.add(tmp.name, arcname=MODEL_OUT)
