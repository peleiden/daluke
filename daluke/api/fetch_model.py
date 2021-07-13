from __future__ import annotations
import enum
import os
import pathlib
import wget

import torch
from pelutils import log, Levels
from transformers import AutoConfig

from daluke.serialize import load_from_archive
from daluke.model import DaLUKE, get_ent_embed_size
from daluke.ner.model import NERDaLUKE
from daluke.pretrain.model import PretrainTaskDaLUKE

_download_dir = os.path.join(str(pathlib.Path.home()), ".daluke")
_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Models(enum.Enum):
    DaLUKE = "https://nx5746.your-storageshare.de/s/NAWo2nHzD5rqLmA/download"
    DaLUKE_NER = "https://nx5746.your-storageshare.de/s/Y62i9bKTgCwfZsZ/download"


# Where models are saved
_model_files = { key: os.path.join(_download_dir, key.name + ".tar.gz") for key in Models }
# Status files that only exist if a download is partially completed
_status_files = { key: os.path.join(_download_dir, "." + key.name) for key in Models }


def should_download(model: Models) -> bool:
    # Download if a status file exists, indicating a partially done previous download or if the model is not saved
    if os.path.isfile(_status_files[model]):
        return True
    elif not os.path.isfile(_model_files[model]):
        return True
    return False


def fetch_model(model: Models, force_download=False) -> tuple[DaLUKE, dict, dict]:
    # Make sure .tar.gz model file exists
    os.makedirs(_download_dir, exist_ok=True)
    if should_download(model) or force_download:
        log.debug("Downloading %s to %s" % (model, _model_files[model]))
        # Create status file
        pathlib.Path(_status_files[model]).touch()
        # Download
        wget.download(model.value, out=_model_files[model])
        # Remove status file
        os.remove(_status_files[model])
    else:
        log.debug("Using cached model %s at %s" % (model, _model_files[model]))

    # Read model state dict along with metadata and entity vocab
    # This is done in a seperate working directory
    log.debug("Loading entity vocab, metadata, and state dict")
    cwd = os.getcwd()
    os.chdir(_download_dir)
    entity_vocab, metadata, state_dict = load_from_archive(_model_files[model])
    os.chdir(cwd)

    # Load model
    log.debug("Creating model")
    bert_config = AutoConfig.from_pretrained(metadata["base-model"])
    if model == Models.DaLUKE:
        net = PretrainTaskDaLUKE(bert_config, len(entity_vocab), get_ent_embed_size(state_dict))
    elif model == Models.DaLUKE_NER:
        net = NERDaLUKE(
            output_shape = 5,  # Always use misc in this case
            bert_config = bert_config,
            ent_vocab_size = 2,
            ent_embed_size = get_ent_embed_size(state_dict),
            dropout = 0,
            words_only = False,
            entities_only = False,
        )

    net.load_state_dict(state_dict)
    net.eval()

    return net.to(_device), metadata, entity_vocab
