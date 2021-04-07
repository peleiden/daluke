from __future__ import annotations
import os
from dataclasses import dataclass
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoConfig
from pelutils.logger import log, Levels

from daluke.pretrain.data import DataLoader
from daluke.pretrain.data.build import DatasetBuilder

PORT = "3090"

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

@dataclass
class Hyperparams:
    lr: float = 1e-4
    batch_size: int = 2048
    grad_accumulate: int = 1024
    ent_embed_size: int = 256

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
    # Setup logger
    log.configure(
        os.path.join(location, f"{name}-{rank if rank != -1 else ''}.log"),
        "DaLUKE pretraining on node %i" % rank,
        log_commit  = True,
        print_level = (Levels.INFO if quiet else Levels.DEBUG) if is_master(rank) else None,
    )
    log("Running with the following hyperparameters", params)

    log.section("Reading metadata")
    with open(os.path.join(location, DatasetBuilder.metadata_file)) as f:
        metadata = json.load(f)

    # Test input correctness
    assert params.lr > 0, "Learning rate must be larger than 0"

    # Setup multi-gpu if used and get device
    setup(rank, world_size)
    if rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: Instantiate model
    else:
        device = torch.device("cuda", index=rank)
        # TODO: Instantiate model and wrap in DDP with device_ids=[rank]

    # TODO: Get device and use it for dataloader and model
    # TODO: Use DistributedSampler if rank != -1
    data = DataLoader(location, metadata)
    # TODO: Get out-name from module constant in pretrain.data.build
    # TODO: Pass dataset metadata to DataLoader
    bert_config = AutoConfig.from_pretrained(metadata["base-model"])
    # TODO: Get transformer name from metadata

    bert_model = AutoModelForPreTraining.from_pretrained(metadata["base-model"])
    bert_model.parameters()



    # Clean up multi-gpu if used
    cleanup(rank)
