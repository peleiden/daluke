from __future__ import annotations
import os
from typing import Any

from pelutils.parse import Parser
from pelutils.logger import log

import torch
import torch.multiprocessing as mp

from daluke import daBERT
from daluke.pretrain.train import train, Hyperparams


ARGUMENTS = {
    "entity-vocab-file": { "default": "entity_vocab.json", "type": str, "help": "Location of entity vocabulary" },
    "epochs":            { "default": Hyperparams.epochs, "type": int, "help": "Number of passes through the entire data set"},
    "batch-size":        { "default": Hyperparams.batch_size, "type": int },
    "lr":                { "default": Hyperparams.lr, "type": float, "help": "Initial learning rate" },
    "ent-embed-size":    { "default": Hyperparams.ent_embed_size, "type": int, "help": "Dimension of the entity embeddings" },
    "weight-decay":      { "default": Hyperparams.weight_decay, "type": float, "help": "The decay factor in the AdamW optimizer" },
    "warmup-prop":       { "default": Hyperparams.warmup_prop, "type": float, "help": "Proportion of training steps used for optimizer warmup" },
    "quiet":             { "action": "store_true", "help": "Don't show debug logging" },
}

def _run_training(rank: int, world_size: int, args: dict[str, Any]):
    """ Wrapper function for train for easy use with mp.spawn """
    return train(
        rank,
        world_size,
        location        = args.pop("location"),
        name            = args.pop("name"),
        quiet           = args.pop("quiet"),
        ent_vocab_file  = args.pop("entity_vocab_file"),
        params          = Hyperparams(**args),
    ),

def run(args: dict[str, Any]):
    """ Initializes training on multiple GPU's """
    mp.spawn(
        _run_training,
        args   = (torch.cuda.device_count(), args),
        nprocs = torch.cuda.device_count(),
        join   = True,
    )

if __name__ == '__main__':
    with log.log_errors:
        parser = Parser(ARGUMENTS, name="daluke-pretrain", multiple_jobs=False)
        args = parser.parse()[0]
        parser.document_settings()
        if torch.cuda.device_count() > 1:
            run(args)
        else:
            _run_training(-1, 1, args)
