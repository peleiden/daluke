from __future__ import annotations
import os
import contextlib
import json
from dataclasses import dataclass

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np

from transformers import AutoConfig, AutoModelForPreTraining, AdamW, get_polynomial_decay_schedule_with_warmup
from pelutils import DataStorage, thousand_seps, TT
from pelutils.logger import log, Levels

from .data import DataLoader, load_entity_vocab
from .data.build import DatasetBuilder
from .model import PretrainTaskDaLUKE, BertAttentionPretrainTaskDaLUKE, load_base_model_weights
from ..model import all_params
from ..analysis.pretrain import TrainResults, top_k_accuracy

PORT = "3090"  # Are we sure this port is in stock?
NO_DECAY =  { "bias", "LayerNorm.weight" }

MODEL_OUT = "daluke_epoch{i}.pt"
OPTIMIZER_OUT = "optim_epoch{i}.pt"
SCHEDULER_OUT = "scheduler_epoch{i}.pt"
SCALER_OUT = "scaler_epoch{i}.pt"


@dataclass
class Hyperparams(DataStorage):
    epochs:             int   = 20
    batch_size:         int   = 2048
    lr:                 float = 1e-4
    ff_size:            int   = 16
    ent_embed_size:     int   = 256
    weight_decay:       float = 0.01
    warmup_prop:        float = 0.06
    word_ent_weight:    float = 0.5
    bert_fix_prop:      float = 0.5
    fp16:               bool  = False
    ent_min_mention:    int   = 0
    entity_loss_weight: bool  = False
    bert_attention:     bool  = False
    word_mask_prob:     float = 0.15
    word_unmask_prob:   float = 0.1
    word_randword_prob: float = 0.1
    ent_mask_prob:      float = 0.15
    lukeinit:           bool  = False
    no_base_model:      bool  = False

    subfolder = None  # Set at runtime
    json_name = "params.json"
    ignore_missing = True

    def __post_init__(self):
        # Test parameter validity
        assert self.epochs > 0
        assert self.lr > 0, "Learning rate must be larger than 0"
        assert isinstance(self.ent_embed_size, int) and self.ent_embed_size > 0
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        assert isinstance(self.ff_size, int) and 0 < self.ff_size <= self.batch_size
        assert 0 <= self.weight_decay < 1
        assert 0 <= self.warmup_prop < 1
        assert 0 <= self.word_ent_weight <= 1

        assert 0 <  self.word_mask_prob < 1
        assert 0 <= self.word_unmask_prob <= 1
        assert 0 <= self.word_randword_prob <= 1

        assert 0 < self.ent_mask_prob < 1
        assert isinstance(self.fp16, bool)
        if self.fp16:
            assert torch.cuda.is_available(), "Half-precision cannot be used without CUDA access"
        assert isinstance(self.ent_min_mention, int) and self.ent_min_mention >= 0

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)

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
    include = lambda n: do_decay != any(nd in n for nd in NO_DECAY)
    return [p for n, p in params if p.requires_grad and include(n)]

def clean_saved_epoch(loc: str, epoch: int) -> list[str]:
    paths = {
        os.path.join(loc, TrainResults.subfolder, MODEL_OUT.format(i=epoch)),
        os.path.join(loc, TrainResults.subfolder, OPTIMIZER_OUT.format(i=epoch)),
        os.path.join(loc, TrainResults.subfolder, SCHEDULER_OUT.format(i=epoch)),
        os.path.join(loc, TrainResults.subfolder, SCALER_OUT.format(i=epoch)),
    }
    removed_paths = paths.copy()
    for path in paths:
        if os.path.isfile(path):
            os.remove(path)
        else:
            removed_paths.remove(path)
    return list(removed_paths)

def save_training(
    loc:       str,
    params:    Hyperparams,
    model:     PretrainTaskDaLUKE,
    res:       TrainResults,
    optimizer: Optimizer,
    scheduler,
    scaler   = None,
    epoch    = None,
) -> list[str]:
    epoch = epoch if epoch is not None else res.epoch
    paths = list()
    # Save tracked statistics
    paths += res.save(loc)
    paths += params.save(loc)
    # Save model
    paths.append(os.path.join(loc, TrainResults.subfolder, MODEL_OUT.format(i=epoch)))
    torch.save(model.state_dict(), paths[-1])
    # Save optimizer and scheduler states (these are dymanic over time)
    paths.append(os.path.join(loc, TrainResults.subfolder, OPTIMIZER_OUT.format(i=epoch)))
    torch.save(optimizer.state_dict(), paths[-1])
    paths.append(os.path.join(loc, TrainResults.subfolder, SCHEDULER_OUT.format(i=epoch)))
    torch.save(scheduler.state_dict(), paths[-1])
    # Save scaler if using fp16
    if scaler:
        paths.append(os.path.join(loc, TrainResults.subfolder, SCALER_OUT.format(i=epoch)))
        torch.save(scaler.state_dict(), paths[-1])
    return paths

def train(
    rank:           int,
    world_size:     int,
    *,
    resume:         bool,
    location:       str,
    name:           str,
    quiet:          bool,
    save_every:     int,
    explicit_args:  set[str],
    params:         Hyperparams,
):
    # Get filepath within path context
    fpath = lambda path: os.path.join(location, path) if isinstance(path, str) else os.path.join(location, *path)

    # Setup multi-gpu if used and get device
    setup(rank, world_size)

    is_master = rank < 1  # Are we on the main node?
    is_distributed = rank != -1  # Are we performing distributed computing?
    num_workers = torch.distributed.get_world_size() if is_distributed else 1

    # Update locations
    TrainResults.subfolder = name
    Hyperparams.subfolder = name

    # Setup logger
    log.configure(
        os.path.join(location, name, "pretraining-worker=%s.log" % (rank if is_distributed else 0)),
        "DaLUKE pretraining on node %i" % rank,
        log_commit  = True,
        print_level = (Levels.INFO if quiet else Levels.DEBUG) if is_master else None,
        append      = resume,  # Append to existing log file if we are resuming training
    )
    if resume:
        log("Resuming from %s" % name)
        # Load results and hyperparameters from earlier training
        res = TrainResults.load(location)
        loaded_params = Hyperparams.load(location)
        # Overwrite ff-size if given explicitly
        if "ff_size" in explicit_args:
            loaded_params.ff_size = params.ff_size
        params = loaded_params
    log.section("Starting pretraining with the following hyperparameters", params)
    log("Training using %i workers" % num_workers)

    log.section("Reading metadata and entity vocabulary")
    with open(fpath(DatasetBuilder.metadata_file)) as f:
        metadata = json.load(f)
    with open(fpath(DatasetBuilder.entity_vocab_file)) as f:
        entity_vocab = json.load(f)
    log("Loaded metadata:", json.dumps(metadata, indent=4))
    log(f"Loaded entity vocabulary of {len(entity_vocab)} entities")
    if params.ent_min_mention:
        log("Removing entities with less than %i mentions" % params.ent_min_mention)
        entity_vocab = { ent: info for ent, info in entity_vocab.items()
            if info["count"] >= params.ent_min_mention or ent in {"[PAD]", "[UNK]", "[MASK]"} }
        log("After filtering, entity vocab now has %i entities" % len(entity_vocab))

    # Device should be cuda:rank or just cuda if single gpu, else cpu
    if is_distributed:
        device = torch.device("cuda", index=rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if params.entity_loss_weight:
        log("Setting up loss function with entity loss weighting")
        # Don't weigh special tokens
        weights = torch.Tensor([0, 0, 0, *(1 / info["count"] for info in entity_vocab.values() if info["count"])]).to(device)
        entity_criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        log("Setting up loss function without entity loss weighting")
        entity_criterion = nn.CrossEntropyLoss()
    word_criterion = nn.CrossEntropyLoss()

    # Load dataset and training results
    log("Building dataset")
    data = DataLoader(
        location,
        metadata,
        entity_vocab,
        device,
        params.word_mask_prob,
        params.word_unmask_prob,
        params.word_randword_prob,
        params.ent_mask_prob,
    )
    sampler = (DistributedSampler if is_distributed else RandomSampler)(data.examples)
    log("Built %i examples" % len(data))

    loader = data.get_dataloader(params.ff_size, sampler)
    # Number of parameter updates each epoch
    grad_accumulation_steps = params.batch_size // (params.ff_size * num_workers)
    num_updates_epoch = len(loader) // grad_accumulation_steps
    assert num_updates_epoch > 0, "Batch size cannot be larger than number of sequences in dataset"
    # Total number of parameter updates
    num_updates_all = num_updates_epoch * params.epochs
    log(
        "Batches generated:               %i" % len(loader),
        "Subbatches per parameter update: %i" % grad_accumulation_steps,
        "Parameter updates per epoch:     %i" % num_updates_epoch,
        "Parameter updates in total:      %i" % num_updates_all,
    )
    if resume and num_updates_epoch != res.losses.shape[1]:
        raise ValueError(
            f"The number of parameter updates per epoch ({num_updates_epoch}) does not match previously calculated ({res.losses.shape[1]}).\n"
            "This can be caused by changes to sub-batch size and number of GPU's.\n"
            "Please adjust your settings and try again."
        )

    if not resume:
        top_k = [1, 3, 5, 10, 25, 50]
        res = TrainResults(
            losses       = np.zeros((0, num_updates_epoch)),
            w_losses     = np.zeros((0, num_updates_epoch)),
            e_losses     = np.zeros((0, num_updates_epoch)),
            scaled_loss  = np.zeros((0, num_updates_epoch)),
            lr           = np.zeros((0, num_updates_epoch)),
            top_k        = top_k,
            w_accuracies = np.full((0, num_updates_epoch, len(top_k)), np.nan),
            e_accuracies = np.full((0, num_updates_epoch, len(top_k)), np.nan),
            orig_params  = None,  # Set later
            param_diff_1 = np.zeros((0, num_updates_epoch)),
            param_diff_2 = np.zeros((0, num_updates_epoch)),
            runtime      = np.zeros((0, num_updates_epoch)),
            epoch        = 0,
            luke_exclusive_params = None,  # Set later
            q_mats_from_base      = None,  # Set later
        )

    save_epochs = set(range(-1, params.epochs, save_every))

    # Build model, possibly by loading previous weights
    log.section("Setting up model ...")
    bert_config = AutoConfig.from_pretrained(metadata["base-model"])
    assert bert_config.max_position_embeddings == metadata["max-seq-length"],\
        f"Model should respect sequence length; embeddings are of lenght {bert_config.max_position_embeddings}, "\
        f"but max. seq. len. is set to {metadata['max-seq-length']}"
    log("Bert config", bert_config.to_json_string())

    log("Initializing model")
    model_cls = BertAttentionPretrainTaskDaLUKE if params.bert_attention else PretrainTaskDaLUKE
    model = model_cls(
        bert_config,
        ent_vocab_size = len(entity_vocab),
        ent_embed_size = params.ent_embed_size,
    ).to(device)
    if params.lukeinit:
        model.apply(lambda module: model.init_weights(module, bert_config.initializer_range))
    # Load parameters from base model
    if not params.no_base_model:
        log("Loading base model parameters")
        with TT.profile("Loading base model parameters"):
            base_model = AutoModelForPreTraining.from_pretrained(metadata["base-model"])
            new_weights = load_base_model_weights(model, base_model.state_dict(), params.bert_attention)
    else:
        new_weights = set(model.state_dict())
    # Initialize self-attention query matrices to BERT word query matrices
    q_mat_keys = set()
    if not params.bert_attention and not params.no_base_model:
        q_mat_keys = model.init_queries()
    if not resume:
        res.luke_exclusive_params = new_weights
        res.q_mats_from_base = q_mat_keys
        if is_master:
            res.orig_params = all_params(model).cpu().numpy()
    log("Pretraining model initialized with %s parameters" % thousand_seps(len(model)))

    model_params = list(model.named_parameters())
    # Fix BERT weights during training
    def fix_base_model_params(fix: bool):
        """ Fixes or unfixes base model parameters """
        for n, p in model_params:
            if n not in new_weights:
                p.requires_grad = not fix
    fix_base_model_params(True)

    # Unfixes params at this epoch
    unfix_base_model_params_epoch = round(params.bert_fix_prop * params.epochs)
    log("Unfixing base model params after %i epochs" % unfix_base_model_params_epoch)

    if resume:
        mpath = fpath((TrainResults.subfolder, MODEL_OUT.format(i=res.epoch)))
        model.load_state_dict(torch.load(mpath, map_location=device))
        log(f"Resuming training saved at epoch {res.epoch} and loaded model from {mpath}")
    if is_distributed:
        model = DDP(model, device_ids=[rank])

    # TODO: Consider whether this AdamW is sufficient or we should tune it in some way to LUKE
    optimizer = AdamW(
        [{"params": get_optimizer_params(model_params, do_decay=True),  "weight_decay": params.weight_decay},
         {"params": get_optimizer_params(model_params, do_decay=False), "weight_decay": 0}],
        lr = params.lr,
    )
    scaler = amp.GradScaler() if params.fp16 else None
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        int(params.warmup_prop * num_updates_all),
        num_updates_all,
        lr_end = params.lr / 10,
        power  = 3 ** 0.5,
    )
    if resume:
        optimizer.load_state_dict(torch.load(fpath((TrainResults.subfolder, OPTIMIZER_OUT.format(i=res.epoch))), map_location=device))
        scheduler.load_state_dict(torch.load(fpath((TrainResults.subfolder, SCHEDULER_OUT.format(i=res.epoch))), map_location=device))
        if params.fp16:
            scaler.load_state_dict(torch.load(fpath((TrainResults.subfolder, SCALER_OUT.format(i=res.epoch))), map_location=device))
        res.epoch += 1  # We saved the data at epoch i, but should now commence epoch i+1

    log.section(f"Training of daLUKE for {params.epochs} epochs")
    model.zero_grad()  # To avoid tracking of model parameter manipulation
    model.train()

    # Save initial parameters
    if is_master and not resume:
        paths = save_training(location, params, model.module if is_distributed else model, res, optimizer, scheduler, scaler, -1)
        log.debug("Saved initial state to", *paths)

    for i in range(res.epoch, params.epochs):
        TT.profile("Epoch")
        log("Starting epoch %i" % i)
        res.epoch = i
        if i >= unfix_base_model_params_epoch:
            log("Unfixing base model params")
            fix_base_model_params(False)
        if is_distributed:
            sampler.set_epoch(i)

        # Allocate room for results for this epoch
        res.losses       = np.vstack((res.losses,       np.zeros(num_updates_epoch)))
        res.w_losses     = np.vstack((res.w_losses,     np.zeros(num_updates_epoch)))
        res.e_losses     = np.vstack((res.e_losses,     np.zeros(num_updates_epoch)))
        res.scaled_loss  = np.vstack((res.scaled_loss,  np.zeros(num_updates_epoch)))
        res.lr           = np.vstack((res.lr,           np.zeros(num_updates_epoch)))
        res.w_accuracies = np.concatenate((res.w_accuracies, np.full((1, num_updates_epoch, len(res.top_k)), np.nan)))
        res.e_accuracies = np.concatenate((res.e_accuracies, np.full((1, num_updates_epoch, len(res.top_k)), np.nan)))
        res.param_diff_1 = np.vstack((res.param_diff_1, np.zeros(num_updates_epoch)))
        res.param_diff_2 = np.vstack((res.param_diff_2, np.zeros(num_updates_epoch)))
        res.runtime      = np.vstack((res.runtime,      np.zeros(num_updates_epoch)))

        batch_iter = iter(loader)

        # Parameter updates, each consisting of gradients from multiple batches
        for j in range(num_updates_epoch):
            TT.profile("Parameter update")

            # Losses and accuracies for this parameter update
            t_loss, w_loss, e_loss, s_loss = 0, 0, 0, 0
            w_accuracies = np.full((grad_accumulation_steps, len(res.top_k)), np.nan)
            e_accuracies = np.full((grad_accumulation_steps, len(res.top_k)), np.nan)

            # Loop over enough batches to make a parameter update
            for k in range(grad_accumulation_steps):
                TT.profile("Sub-batch")
                batch = next(batch_iter)

                TT.profile("FP and gradients")
                with amp.autocast() if params.fp16 else contextlib.ExitStack():
                    word_preds, ent_preds = model(batch)
                    # Compute and backpropagate loss
                    word_loss = word_criterion(word_preds, batch.word_mask_labels)
                    ent_loss = entity_criterion(ent_preds, batch.ent_mask_labels)
                loss = params.word_ent_weight * word_loss + (1 - params.word_ent_weight) * ent_loss
                loss /= grad_accumulation_steps

                # Only sync parameters on grad updates, aka last pass of this loop
                with model.no_sync() if is_distributed and k < grad_accumulation_steps - 1 else contextlib.ExitStack():
                    if params.fp16:
                        scaled_loss = scaler.scale(loss)
                        scaled_loss.backward()
                        s_loss += scaled_loss.item()
                    else:
                        loss.backward()

                t_loss += loss.item()
                w_loss += word_loss.item() / grad_accumulation_steps
                e_loss += ent_loss.item() / grad_accumulation_steps

                if torch.cuda.is_available() and is_distributed:
                    torch.cuda.synchronize(rank if is_distributed else None)

                TT.end_profile()

                # Save accuracy for statistics
                with TT.profile("Accuracy"):
                    # Only calculate more than top 10 every 20th subbatch
                    top_k = res.top_k if k % 20 == 0 else [x for x in res.top_k if x <= 10]
                    w_accuracies[k, :len(top_k)] = top_k_accuracy(batch.word_mask_labels, word_preds, top_k)
                    e_accuracies[k, :len(top_k)] = top_k_accuracy(batch.ent_mask_labels, ent_preds, top_k)

                TT.end_profile()

            # Update model parameters
            with TT.profile("Parameter updates"):
                if params.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                model.zero_grad()

            # Calculate how much gradient has changed
            with TT.profile("Parameter changes"), torch.no_grad():
                if is_master:
                    orig_pars = torch.from_numpy(res.orig_params).to(device)
                    current_pars = all_params(model.module if is_distributed else model)
                    res.param_diff_1[i, j] = torch.abs(current_pars-orig_pars).sum().item()
                    res.param_diff_2[i, j] = torch.sqrt(((current_pars-orig_pars)**2).sum()).item()
                    del orig_pars

            res.losses[i, j] = t_loss
            res.w_losses[i, j] = w_loss
            res.e_losses[i, j] = e_loss
            res.scaled_loss[i, j] = s_loss
            res.lr[i, j] = scheduler.get_last_lr()[0]
            res.w_accuracies[i, j] = np.nanmean(w_accuracies, axis=0)
            res.e_accuracies[i, j] = np.nanmean(e_accuracies, axis=0)
            log.debug(
                "Performed parameter update %i / %i (ep. %i)" % (j, num_updates_epoch-1, i),
                f"Loss (total, word, entity, scaled): {t_loss:10.5f}, {w_loss:10.5f}, {e_loss:10.5f}, {s_loss:10.5f}",
                f"Accuracy (word, entity):     {100*res.w_accuracies[i, j, 0]:7.3f} %,  {100*res.e_accuracies[i, j, 0]:7.3f} %",
            )

            res.runtime[i, j] = TT.end_profile()

        TT.end_profile()
        log(
            f"Completed epoch {i:2} / {params.epochs-1}",
            f"Mean loss (total, word, entity, scaled): {res.losses[i].mean():10.5f}, {res.w_losses[i].mean():10.5f}, "
            f"{res.e_losses[i].mean():10.5f}, {res.scaled_loss[i].mean()}",
            f"Mean accuracy (word, entity):     {100*res.w_accuracies[i, :, 0].mean():7.3f} %,  {100*res.e_accuracies[i, :, 0].mean():7.3f} %",
            "Runtime: %s" % thousand_seps(res.runtime[-1].sum()),
            "Time distribution so far",
            TT,
        )
        # Save results and model
        if is_master:
            paths = save_training(location, params, model.module if is_distributed else model, res, optimizer, scheduler, scaler)
            log.debug("Saved progress to", *paths)
            if i > 0 and i - 1 not in save_epochs:
                cleaned = clean_saved_epoch(location, i-1)
                log.debug("Cleaned temporary files at", *cleaned)

    log.debug("Time distribution", TT)

    # Clean up multi-gpu if used
    cleanup(rank)
