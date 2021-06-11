from __future__ import annotations
import json
import os
import shutil
import tarfile
import tempfile
from subprocess import Popen, PIPE

import torch
from torch import nn

from pelutils import log

TRAIN_OUT  = "daluke_ner.tar.gz"
TRAIN_OUT_BEST = "daluke_ner_best.tar.gz"
MODEL_OUT = "model.bin"

VOCAB_FILE    = "entity-vocab.json"
METADATA_FILE = "metadata.json"

def load_from_archive(model: str) -> tuple[list[dict], dict[str, int | str], dict]:
    """
    Reads the *.tar.gz archive containing the model, the entity vocabulary and metadata
    """
    if not os.path.isfile(model):
        raise FileNotFoundError(f"Model archive file '{model}' not found.")
    if shutil.which("tar"):
        # tar exists in system and is used for decompressing
        log.debug(f"Extracting {VOCAB_FILE}, {METADATA_FILE}, and {MODEL_OUT} using system tar tool...")
        tmpdir = os.path.join(os.path.split(model)[0], "tmpdir")
        try:
            os.makedirs(tmpdir, exist_ok=True)
            p = Popen(
                ["tar", "-xf", model, "-C", tmpdir],
                stdout=PIPE,
                stderr=PIPE,
            )
            p.wait()
            with open(os.path.join(tmpdir, VOCAB_FILE), "r") as vfile:
                entity_vocab = json.load(vfile)
            with open(os.path.join(tmpdir, METADATA_FILE), "r") as metafile:
                metadata = json.load(metafile)
            with open(os.path.join(tmpdir, MODEL_OUT), "rb") as mfile:
                state_dict = torch.load(mfile, map_location="cpu")
        finally:
            shutil.rmtree(tmpdir)
        return entity_vocab, metadata, state_dict
    else:
        # Use Python's slow build-in tar
        with tarfile.open(model, "r:gz") as tar:
            log.debug(f"Extracting {VOCAB_FILE} and {METADATA_FILE} using build-in tar module (this will take a while)...")
            with tar.extractfile(tar.getmember(VOCAB_FILE)) as vfile:
                entity_vocab = json.load(vfile)
            with tar.extractfile(tar.getmember(METADATA_FILE)) as metafile:
                metadata = json.load(metafile)
            log.debug(f"Extracting {MODEL_OUT} ...")
            with tar.extractfile(tar.getmember(MODEL_OUT)) as mfile:
                state_dict = torch.load(mfile, map_location="cpu")
        return entity_vocab, metadata, state_dict

def save_to_archive(outfile: str, entity_vocab: list[dict], metadata: dict, model: nn.Module):
    outdir = os.path.split(outfile)[0]
    if shutil.which("tar"):
        log.debug(f"Compressing {VOCAB_FILE}, {METADATA_FILE}, and {MODEL_OUT} using system tar tool...")
        tmpdir = os.path.join(outdir, "tmpdir")
        gp = lambda f: os.path.join(tmpdir, f)
        os.makedirs(tmpdir, exist_ok=True)
        try:
            with open(gp(VOCAB_FILE), "w+") as tmp:
                json.dump(entity_vocab, tmp)
            with open(gp(METADATA_FILE), "w+") as tmp:
                json.dump(metadata, tmp)
            with open(gp(MODEL_OUT), "wb+") as tmp:
                torch.save(model.cpu().state_dict(), tmp)
            p = Popen(
                ["tar", "-czvf", outfile, "-C", tmpdir, VOCAB_FILE, METADATA_FILE, MODEL_OUT],
                stdout=PIPE,
                stderr=PIPE,
            )
            p.wait()
        finally:
            shutil.rmtree(tmpdir)
    else:
        log.debug(f"Compressing {VOCAB_FILE}, {METADATA_FILE}, and {MODEL_OUT} using build-in tar module (this will take a while)...")
        with tarfile.open(outfile, "w:gz") as tar:
            with tempfile.NamedTemporaryFile("w+") as tmp:
                tmp.write(json.dumps(entity_vocab))
                tmp.seek(0)
                tar.add(tmp.name, arcname=VOCAB_FILE)
            with tempfile.NamedTemporaryFile("w+") as tmp:
                tmp.write(json.dumps(metadata))
                tmp.seek(0)
                tar.add(tmp.name, arcname=METADATA_FILE)
            with tempfile.NamedTemporaryFile("wb+") as tmp:
                torch.save(model.cpu().state_dict(), tmp)
                tar.add(tmp.name, arcname=MODEL_OUT)
