#!/usr/bin/env python3
from __future__ import annotations
import os
import re as reee
from typing import Any

from pelutils import EnvVars, get_timestamp
from pelutils.parse import Parser
from pelutils.logger import log

import torch
import torch.multiprocessing as mp

from daluke import daBERT
from daluke.pretrain.train import train, Hyperparams
from daluke.analysis.pretrain import TrainResults


ARGUMENTS = {
    "resume":             { "action": "store_true", "help": "Resume from training given by name. If name if not given, resumes from newest non-named pretraining" },
    "name":               { "default": "", "type": str, "help": "Name of pretraining" },
    "epochs":             { "default": Hyperparams.epochs, "type": int, "help": "Number of passes through the entire data set"},
    "batch-size":         { "default": Hyperparams.batch_size, "type": int, "help": "Number of sequences per parameter update" },
    "lr":                 { "default": Hyperparams.lr, "type": float, "help": "Initial learning rate" },
    "ff-size":            { "default": Hyperparams.ff_size, "type": int, "help": "Size of each feed forward" },
    "ent-embed-size":     { "default": Hyperparams.ent_embed_size, "type": int, "help": "Dimension of the entity embeddings" },
    "weight-decay":       { "default": Hyperparams.weight_decay, "type": float, "help": "The decay factor in the AdamW optimizer" },
    "warmup-prop":        { "default": Hyperparams.warmup_prop, "type": float, "help": "Proportion of training steps used for optimizer warmup" },
    "word-ent-weight":    { "default": Hyperparams.word_ent_weight, "type": float, "help": "0 for only entities, 1 for only words, 0.5 for equal weighting" },
    "bert-fix-prop":      { "default": Hyperparams.bert_fix_prop, "type": float, "help": "Share of epochs for which to fix base model weights" },
    "ent-min-mention":    { "default": Hyperparams.ent_min_mention, "type": int, "help": "How many times an entity at least should mentioned to be kept. 0 for no limit" },
    "fp16":               { "action": "store_true", "help": "Use automatic mixed precision" },
    "entity-loss-weight": { "action": "store_true", "help": "Weigh MLM entity loss by entity count" },
    "save-every":         { "default": 1, "type": int, "help": "Save progress after this many epochs" },
    "bert-attention":     { "action": "store_true", "help": "Use the original BERT attention mechanism instead of the entity aware LUKE variant" },
    "quiet":              { "action": "store_true", "help": "Don't show debug logging" },
}

def _run_training(rank: int, world_size: int, explicit_args: list[set[str]], args: dict[str, Any]):
    """ Wrapper function for train for easy use with mp.spawn """
    return train(
        rank,
        world_size,
        resume         = args.pop("resume"),
        location       = args.pop("location"),
        name           = args.pop("name"),
        quiet          = args.pop("quiet"),
        save_every     = args.pop("save_every"),
        bert_attention = args.pop("bert_attention"),
        explicit_args  = explicit_args[0],
        params         = Hyperparams(**args),
    )

def _run_distributed(explicit_args: list[set[str]], args: dict[str, Any]):
    """ Initializes training on multiple GPU's """
    mp.spawn(
        _run_training,
        args   = (torch.cuda.device_count(), explicit_args, args),
        nprocs = torch.cuda.device_count(),
        join   = True,
    )

if __name__ == '__main__':
    with log.log_errors, EnvVars(OMP_NUM_THREADS=1):
        parser = Parser(ARGUMENTS, name="daluke-pretrain", multiple_jobs=False)
        args = parser.parse()

        if args["resume"] and not args["name"]:
            # Load last created save
            args["name"] = next(
                p for p in sorted(os.listdir(args["location"]), reverse=True)
                if os.path.isdir(os.path.join(args["location"], p)) and reee.fullmatch(r"pretrain-results_[\-_0-9]+", p)
            )
        else:
            if not args["name"]:
                args["name"] = "pretrain-results_" + get_timestamp(for_file=True)
        if not args["resume"]:
            parser.document_settings(args["name"])

        if torch.cuda.device_count() > 1:
            _run_distributed(parser.explicit_args, args)
        else:
            _run_training(-1, 1, parser.explicit_args, args)
