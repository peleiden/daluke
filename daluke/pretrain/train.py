from __future__ import annotations
import os
import contextlib
import json
from dataclasses import dataclass
from math import ceil

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np

from transformers import AutoConfig, AutoModelForPreTraining, AdamW, get_linear_schedule_with_warmup
from pelutils.logger import log, Levels
from pelutils.ds import reset_cuda

from . import TT
from .data import DataLoader, load_entity_vocab
from .data.build import DatasetBuilder
from .model import PretrainTaskDaLUKE, BertAttentionPretrainTaskDaLUKE, load_base_model_weights
from .analysis import TrainResults

PORT = "3090"  # Are we sure this port is in stock?
NO_DECAY =  { "bias", "LayerNorm.weight" }

MODEL_OUT = "daluke_epoch{i}.pt"
OPTIMIZER_OUT = "optim_epoch{i}.pt"
SCHEDULER_OUT = "scheduler_epoch{i}.pt"


def setup(rank: int, world_size: int):
    if rank != -1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = PORT
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup(rank: int):
    if rank != -1:
        dist.destroy_process_group()

def get_optimizer_params(params: list, do_decay: bool) -> list:
    """ Returns the parameters that should be tracked by optimizer with or without weight decay"""
    # Only include the parameter if do_decay has reverse truth value of the parameter being in no_decay
    include = lambda n: not do_decay == any(nd in n for nd in NO_DECAY)
    return [p for n, p in params if p.requires_grad and include(n)]

def save_training(loc: str, model: PretrainTaskDaLUKE, res: TrainResults, optimizer: Optimizer, scheduler) -> list[str]:
    paths = list()
    # Save tracked statistics
    paths.extend(res.save(loc))
    # Save model
    paths.append(os.path.join(loc, TrainResults.subfolder, MODEL_OUT.format(i=res.epoch)))
    torch.save(model.state_dict(), paths[-1])
    # Save optimizer and scheduler states (these are dymanic over time)
    paths.append(os.path.join(loc, TrainResults.subfolder, OPTIMIZER_OUT.format(i=res.epoch)))
    torch.save(optimizer.state_dict(), paths[-1])
    paths.append(os.path.join(loc, TrainResults.subfolder, SCHEDULER_OUT.format(i=res.epoch)))
    torch.save(scheduler.state_dict(), paths[-1])
    return paths

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
        # assert self.batch_size % self.grad_accumulate == 0,\
        #     "Batch size (%i) must be divisible by gradient accumulation steps (%i)" % (self.batch_size, self.grad_accumulate)
        assert 1 > self.weight_decay >= 0
        assert 1 > self.warmup_prop >= 0

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)

def train(
    rank: int,
    world_size: int,
    *,
    resume_from: str,
    location: str,
    name: str,
    quiet: bool,
    save_every: int,
    bert_attention: bool,
    params: Hyperparams,
):
    # Get filepath within path context
    fpath = lambda path: os.path.join(location, path)

    # Setup multi-gpu if used and get device
    setup(rank, world_size)

    is_master = rank < 1  # Are we on the main node?
    is_distributed = rank != -1  # Are we performing distributed computing?
    num_workers = torch.distributed.get_world_size() if is_distributed else 1

    # Setup logger
    log.configure(
        fpath(f"{name}{'-%s' % rank if is_distributed else ''}.log"),
        "DaLUKE pretraining on node %i" % rank,
        log_commit  = True,
        print_level = (Levels.INFO if quiet else Levels.DEBUG) if is_master else None,
        append      = resume_from,  # Append to existing log file if we are resuming training
    )
    log.section("Starting pretraining with the following hyperparameters", params)
    if resume_from:
        log("Resuming from %s" % resume_from)

    log.section("Reading metadata and entity vocabulary")
    with open(fpath(DatasetBuilder.metadata_file)) as f:
        metadata = json.load(f)
    with open(fpath(DatasetBuilder.entity_vocab_file)) as f:
        entity_vocab = json.load(f)
    log("Loaded metadata:", json.dumps(metadata, indent=4))
    log(f"Loaded entity vocabulary of {len(entity_vocab)} entities")

    # Device should be cuda:rank or just cuda if single gpu, else cpu
    if is_distributed:
        device = torch.device("cuda", index=rank)
        torch.cuda.set_device(rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and training results
    data = DataLoader(location, metadata, device)
    # Update batch size to account for gradient accumulation and number of gpus used
    # Number of examples given at forward pass
    worker_batch_size = ceil(params.batch_size / (params.grad_accumulate * num_workers))
    log("Forward pass batch size for this worker: %i" % worker_batch_size)
    sampler = (DistributedSampler if is_distributed else RandomSampler)(data.examples)

    loader = data.get_dataloader(worker_batch_size, sampler)
    # Number of parameter updates each epoch
    num_updates_epoch = ceil(len(data) / (worker_batch_size * num_workers))
    # Total number of parameter updates
    num_updates_all = num_updates_epoch * params.epochs
    # Number of feed forwards each epoch
    # Drop last batch if not divisible by gradient accumulation steps
    num_batches = len(loader) - len(loader) % params.grad_accumulate
    if resume_from:
        TrainResults.subfolder = resume_from
        res = TrainResults.load(location)
    else:
        res = TrainResults(
            losses = list(),
            epoch = 0,
        )

    # Build model, possibly by loading previous weights
    log.section("Setting up model ...")
    bert_config = AutoConfig.from_pretrained(metadata["base-model"])
    assert bert_config.max_position_embeddings == metadata["max-seq-length"], \
        f"Model should respect sequence length; embeddings are of lenght {bert_config.max_position_embeddings}, but max. seq. len. is set to {metadata['max-seq-length']}"
    log("Bert config", bert_config.to_json_string())

    model_cls = BertAttentionPretrainTaskDaLUKE if bert_attention else PretrainTaskDaLUKE
    model = model_cls(
        bert_config,
        ent_vocab_size = len(entity_vocab),
        ent_embed_size = Hyperparams.ent_embed_size,
    ).to(device)
    # TODO: Maybe init fresh model weights manually (they do)
    # Load parameters from base model
    with TT.profile("Loading base model parameters from %s" % metadata["base-model"]):
        base_model = AutoModelForPreTraining.from_pretrained(metadata["base-model"])
        new_weights = load_base_model_weights(model, base_model)

    model_params = list(model.named_parameters())
    # TODO: Re-enable training of BERT weights at some point during the training
    # Fix BERT weights during training
    for n, p in model_params:
        if n not in new_weights:
            p.requires_grad = False
    del base_model  # Clear base model weights from memory

    if is_distributed:
        model = DDP(
            model,
            device_ids=[rank],
            # TODO: Understand reasoning behind following two flags that are copied from LUKE
            broadcast_buffers=False,
            # find_unused_parameters=True,
        )

    if resume_from:
        mpath = fpath(MODEL_OUT.format(i=res.epoch))
        model.load_state_dict(torch.load(mpath))
        log(f"Resuming training saved at epoch {res.epoch} and loaded model from {mpath}")
    # TODO: Consider whether this AdamW is sufficient or we should tune it in some way to LUKE
    optimizer = AdamW(
        [{"params": get_optimizer_params(model_params, do_decay=True),  "weight_decay": params.weight_decay},
         {"params": get_optimizer_params(model_params, do_decay=False), "weight_decay": 0}],
        lr = params.lr,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, int(params.warmup_prop * num_updates_all), num_updates_all)
    if resume_from:
        optimizer.load_state_dict(torch.load(fpath(OPTIMIZER_OUT.format(i=res.epoch))))
        scheduler.load_state_dict(torch.load(fpath(SCHEDULER_OUT.format(i=res.epoch))))
        res.epoch += 1 # We saved the data at epoch i, but should now commence epoch i+1

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    log.section(f"Training of daLUKE for {params.epochs} epochs")
    model.train()
    for i in range(params.epochs):
        TT.profile("Epoch")
        log("Starting epoch %i" % i)
        res.epoch = i
        if is_distributed:
            sampler.set_epoch(i)

        accumulate_step = 0
        grad_updates = 0
        batch_loss = 0
        losses = list()
        for j, batch in enumerate(loader):
            if j == num_batches:
                break
            TT.profile("Batch")

            word_preds, ent_preds = model(batch)

            # Compute and backpropagate loss
            word_loss = criterion(word_preds, batch.word_mask_labels)
            ent_loss = criterion(ent_preds, batch.ent_mask_labels)
            loss = word_loss + ent_loss
            loss /= params.grad_accumulate

            accumulate_step += 1
            # Only sync parameters on grad updates
            if is_distributed and accumulate_step != params.grad_accumulate:
                sync_context = model.no_sync()
            else:
                sync_context = contextlib.ExitStack()
            with sync_context:
                loss.backward()
                batch_loss += loss.item()

            # Performs parameter update every for every `grad_accumulate`'th batch
            if accumulate_step == params.grad_accumulate:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            reset_cuda()
            log.debug(f"Batch {j}/{num_batches-1} (ep. {i}). Loss: {loss.item()}")
            if accumulate_step == params.grad_accumulate:
                losses.append(batch_loss)
                res.losses.append(batch_loss)  # Note: This only saves loss from main node
                grad_updates += 1
                log.debug("Performed gradient update. Loss: %f" % batch_loss)
                batch_loss = 0
                accumulate_step = 0
            TT.end_profile()

        log(f"Completed epoch {i}/{params.epochs-1} with mean loss {np.mean(losses)}")
        # Save results and model
        if is_master and (i+1) % save_every == 0:
            paths = save_training(location, model, res, optimizer, scheduler)
            log.debug("Saved progress to", *paths)

        TT.end_profile()

    log.debug("Time distribution", TT)

    # Clean up multi-gpu if used
    cleanup(rank)
