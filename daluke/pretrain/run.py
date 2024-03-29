#!/usr/bin/env python3
from __future__ import annotations
import os
import re as reee
from shutil import rmtree
from typing import Any

from pelutils import EnvVars, get_timestamp
from pelutils.parse import Parser
from pelutils.logger import log

import torch
import torch.multiprocessing as mp

from daluke.pretrain.train import train, Hyperparams


ARGUMENTS = {
    "resume":                { "action":  "store_true", "help": "Resume from training given by name. If name if not given, resumes from newest non-named pretraining" },
    "name":                  { "default": "", "type": str, "help": "Name of pretraining" },
    "parameter-updates":     { "default": Hyperparams.parameter_updates,     "type": int,   "help": "Number of parameter updates to the model throughout training"},
    "batch-size":            { "default": Hyperparams.batch_size,            "type": int,   "help": "Number of sequences per parameter update" },
    "lr":                    { "default": Hyperparams.lr,                    "type": float, "help": "Initial learning rate" },
    "ff-size":               { "default": Hyperparams.ff_size,               "type": int,   "help": "Size of each feed forward" },
    "ent-embed-size":        { "default": Hyperparams.ent_embed_size,        "type": int,   "help": "Dimension of the entity embeddings" },
    "ent-hidden-size":       { "default": Hyperparams.ent_hidden_size,       "type": int,   "help": "Dimension of entity representations" },
    "weight-decay":          { "default": Hyperparams.weight_decay,          "type": float, "help": "The decay factor in the AdamW optimizer" },
    "warmup-prop":           { "default": Hyperparams.warmup_prop,           "type": float, "help": "Proportion of training steps used for optimizer warmup" },
    "word-ent-weight":       { "default": Hyperparams.word_ent_weight,       "type": float, "help": "0 for only entities, 1 for only words, 0.5 for equal weighting" },
    "bert-fix-prop":         { "default": Hyperparams.bert_fix_prop,         "type": float, "help": "Share of epochs for which to fix base model weights" },
    "ent-min-mention":       { "default": Hyperparams.ent_min_mention,       "type": int,   "help": "How many times an entity at least should mentioned to be kept. 0 for no limit" },
    "word-mask-prob":        { "default": Hyperparams.word_mask_prob,        "type": float, "help": "Prop. of full words to mask in MLM" },
    "word-unmask-prob":      { "default": Hyperparams.word_unmask_prob,      "type": float, "help": "Chance of masked word to be forward passed as unmasked" },
    "word-randword-prob":    { "default": Hyperparams.word_randword_prob,    "type": float, "help": "Chance of masked word to be forward passed as another random word" },
    "ent-mask-prob":         { "default": Hyperparams.ent_mask_prob,         "type": float, "help": "Prop of entities to mask in MEM" },
    "fp16":                  { "action":  "store_true", "help": "Use automatic mixed precision" },
    "entity-loss-weight":    { "action":  "store_true", "help": "Weigh MLM entity loss by entity count" },
    "bert-attention":        { "action":  "store_true", "help": "Use the original BERT attention mechanism instead of the entity aware LUKE variant" },
    "lukeinit":              { "action":  "store_true", "help": "Initiliaze model weights the same way as in LUKE (excluding base model weights)" },
    "pcainit":               { "action":  "store_true", "help": "Use PCA approach for initializing weight matrices for entities of lower dim." },
    "no-base-model":         { "action":  "store_true", "help": "Do not use base model for initialization" },
    "save-every":            { "default": 500, "type": int, "help": "Save progress after this many parameter updates" },
    "validate-every":        { "default": 100, "type": int, "help": "How many parameter updates are performed between each validation. 0 for no validation" },
    "post-command":          { "default": str(), "help": "Format: <x>h<y>:::<command> for exiting and running command after x hours and y minutes. E.g. 23h30:::sudo rm -rf /" },
    "quiet":                 { "action":  "store_true", "help": "Don't show debug logging" },
    "max-workers":           { "default": torch.cuda.device_count(), "type": int, "help": "Maximum number of cuda devices to use" },
}

def _run_training(rank: int, world_size: int, explicit_args: list[set[str]], args: dict[str, Any]):
    """ Wrapper function for train for easy use with mp.spawn """
    del args["max_workers"]
    with log.log_errors, EnvVars(OMP_NUM_THREADS=1):
        train(
            rank,
            world_size,
            resume         = args.pop("resume"),
            location       = args.pop("location"),
            name           = args.pop("name"),
            quiet          = args.pop("quiet"),
            save_every     = args.pop("save_every"),
            validate_every = args.pop("validate_every"),
            post_command   = args.pop("post_command"),
            explicit_args  = explicit_args[0],
            params         = Hyperparams(**args),
        )

def _run_distributed(explicit_args: list[set[str]], args: dict[str, Any]):
    """ Initializes training on multiple GPU's """
    n_devices = min(torch.cuda.device_count(), args["max_workers"])
    mp.spawn(
        _run_training,
        args   = (n_devices, explicit_args, args),
        nprocs = n_devices,
        join   = True,
    )

if __name__ == '__main__':
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
        if os.path.isdir(p := os.path.join(args["location"], args["name"])):
            rmtree(p)
        parser.document_settings(args["name"])

    if torch.cuda.device_count() > 1 and args["max_workers"] > 1:
        _run_distributed(parser.explicit_args, args)
    else:
        _run_training(-1, 1, parser.explicit_args, args)
