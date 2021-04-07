from __future__ import annotations
import os
from dataclasses import dataclass
import json

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
import numpy as np

from transformers import AutoConfig, AutoModelForPreTraining, AdamW, get_linear_schedule_with_warmup
from pelutils.logger import log, Levels

from daluke import daBERT
from .data import DataLoader, load_entity_vocab
from .data.build import DatasetBuilder
from .model import PretrainTaskDaLUKE

PORT = "3090" # Are we sure this port is in stock?
NO_DECAY =  {"bias", "LayerNorm.weight"}

def setup(rank: int, world_size: int):
    if rank != -1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = PORT
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup(rank: int):
    if rank != -1:
        dist.destroy_process_group()

def is_master(rank: int) -> bool:
    """ Determine if master node """
    return rank < 1

def get_optimizer_params(params: list, do_decay: bool) -> list:
    """ Returns the parameters that should be tracked by optimizer with or without weight decay"""
    # Only include the parameter if do_decay has reverse truth value of the parameter being in no_decay
    include = lambda n: not do_decay == any(nd in n for nd in NO_DECAY)
    return [p for n, p in params if p.requires_grad and include(n)]

@dataclass
class Hyperparams:
    epochs: int = 20
    batch_size: int = 2048
    lr: float = 1e-4
    grad_accumulate: int = 1024
    ent_embed_size: int = 256
    weight_decay: float = 0.1
    warmup_prop: float = 0.06

    def __post__init(self):
        # Test input correctness
        assert self.epochs > 0
        assert self.lr > 0, "Learning rate must be larger than 0"
        assert isinstance(self.ent_embed_size, int) and self.ent_embed_size > 0
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.grad_accumulate, int) and self.grad_accumulate > 0
        assert 1 > self.weight_decay >= 0
        assert 1 > self.warmup_prop >= 0

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)

def train(
    rank: int,
    world_size: int,
    *,
    location: str,
    name: str,
    quiet: bool,
    params: Hyperparams,
):
    # Get filepath within path context
    fpath = lambda path: os.path.join(location, path)

    # Setup logger
    log.configure(
        fpath(f"{name}{'-' + rank if rank != -1 else ''}.log"),
        "DaLUKE pretraining on node %i" % rank,
        log_commit  = True,
        print_level = (Levels.INFO if quiet else Levels.DEBUG) if is_master(rank) else None,
    )
    log("Starting pre-training with the following hyperparameters", params)

    log.section("Reading metadata and entity vocabulary")
    with open(fpath(DatasetBuilder.metadata_file)) as f:
        metadata = json.load(f)
    with open(fpath(DatasetBuilder.entity_vocab_file)) as f:
        entity_vocab = json.load(f)
    log("Loaded metadata:", json.dumps(metadata, indent=4))
    log(f"Loaded entity vocabulary of {len(entity_vocab)} entities")

    # Setup multi-gpu if used and get device
    setup(rank, world_size)

    log.info("Setting up model ...")

    bert_config = AutoConfig.from_pretrained(daBERT)
    if rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PretrainTaskDaLUKE(
            bert_config,
            ent_vocab_size = len(entity_vocab),
            ent_embed_size = Hyperparams.ent_embed_size,
        )
    else:
        device = torch.device("cuda", index=rank)
        raise NotImplementedError # TODO: Instantiate model and wrap in DDP with device_ids=[rank]
    model.to(device)

    # TODO: Only initialize model parameters if no existing model given (for resuming training)
    # Load parameters from base model
    # TODO: Does this require some .to(device) magic?
    base_model = AutoModelForPreTraining.from_pretrained(metadata["base-model"])
    model.load_base_model_weights(base_model)
    del base_model  # Clear base model weights from memory

    dataloader = DataLoader(location, metadata)
    sampler = RandomSampler if is_master(rank) else DistributedSampler # TODO: Is this a random sampler?

    # TODO: How to handle fixing of BERT parameters
    # FIXME: Not enough dedotated WAM for the following 10-ish lines
    num_updates = int(np.ceil(len(dataloader) / params.batch_size * params.epochs))
    model_params = list(model.named_parameters())
    optimizer = AdamW(
        [{"params": get_optimizer_params(model_params, do_decay=True),  "weight_decay": params.weight_decay},
         {"params": get_optimizer_params(model_params, do_decay=False), "weight_decay": 0}],
        lr = params.lr,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, int(params.warmup_prop * num_updates), num_updates)
    loss = nn.CrossEntropyLoss(ignore_index=-1)
    model.train()
    for i in range(params.epochs):
        for batch in dataloader.get_dataloader(params.batch_size, sampler(dataloader.examples)):
            preds = model(batch)

    # Clean up multi-gpu if used
    cleanup(rank)
