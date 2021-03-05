from __future__ import annotations
import json
import tarfile

import torch
from torch import nn
from transformers.models.bert.modeling_bert import(
        BertEncoder,
        BertPooler,
        BertConfig,
        BertEmbeddings
)

from pelutils import log

from daluke.collect_modelfile import VOCAB_FILE, METADATA_FILE, MODEL_OUT


LABELS = ("NIL", "LOC", "PER", "ORG", "MISC") #TODO: Make dependant on dataset

class DaLukeNER(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config     = BertConfig(**config_dict)
        self.encoder    = BertEncoder(self.config) #FIXME: Should be the new LUKE encoder
        self.pooler     = BertPooler(self.config)
        self.embeddings = BertEmbeddings(self.config)
        self.entity_embeddings = None #FIXME

        self.drop = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size*3, len(LABELS))


def load_from_archive(modelfile: str) -> (list[dict], dict, dict):
    """
    Reads the luke.tar.gz archive containing the model, the entity vocabulary and the
    """
    with tarfile.open(modelfile, "r:*") as tar:
        log.debug(f"Extracting {VOCAB_FILE} and {METADATA_FILE} ...")
        with tar.extractfile(tar.getmember(VOCAB_FILE)) as vfile:
            entity_vocab = [json.loads(l) for l in vfile.read().splitlines()]
        with tar.extractfile(tar.getmember(METADATA_FILE)) as metafile:
            metadata = json.load(metafile)
        log.debug(f"Extracting {MODEL_OUT} ...")
        with tar.extractfile(tar.getmember(MODEL_OUT)) as mfile:
            state_dict = torch.load(mfile, map_location="cpu")
    return entity_vocab, metadata, state_dict
