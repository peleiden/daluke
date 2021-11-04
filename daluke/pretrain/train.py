from __future__ import annotations
from dataclasses import dataclass
from math import ceil
from typing import Iterator
import contextlib
import json
import os
import time

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.utils.data import RandomSampler, SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from transformers import AutoConfig, AutoModelForPreTraining, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from pelutils import DataStorage, thousand_seps, TT
from pelutils.logger import log, Levels
from pelutils.ds import unique

from . import close_tt
from daluke.data import get_special_ids, token_map_to_token_reduction
from .data import DataLoader
from .data.build import DatasetBuilder
from .model import PretrainTaskDaLUKE, BertAttentionPretrainTaskDaLUKE, load_base_model_weights, copy_with_reduced_state_dict
from ..model import all_params, all_params_groups_to_slices
from ..analysis.pretrain import TrainResults, top_k_accuracy, validate_model

PORT = "3090"  # Are we sure this port is in stock?
NO_DECAY =  { "bias", "LayerNorm.weight" }

MODEL_OUT = "daluke_pu_{i}.pt"
OPTIMIZER_OUT = "optim_pu_{i}.pt"
SCHEDULER_OUT = "scheduler_pu_{i}.pt"
SCALER_OUT = "scaler_pu_{i}.pt"


MIN_EXAMPLES_PER_PARAMDIFF = 20_000

@dataclass
class Hyperparams(DataStorage):
    parameter_updates:     int        = 20_000  # Shortened to pu in much of the code
    batch_size:            int        = 4080
    lr:                    float      = 1e-4
    ff_size:               int        = 16
    ent_embed_size:        int        = 256
    ent_hidden_size:       int | None = None
    ent_intermediate_size: int | None = None
    weight_decay:          float      = 0.01
    warmup_prop:           float      = 0.06
    word_ent_weight:       float      = 0.5
    bert_fix_prop:         float      = 0.5
    fp16:                  bool       = False
    ent_min_mention:       int        = 0
    entity_loss_weight:    bool       = False
    bert_attention:        bool       = False
    word_mask_prob:        float      = 0.15
    word_unmask_prob:      float      = 0.1
    word_randword_prob:    float      = 0.1
    ent_mask_prob:         float      = 0.15
    lukeinit:              bool       = False
    no_base_model:         bool       = False
    pcainit:               bool       = False

    subfolder = None  # Set at runtime
    json_name = "params.json"
    ignore_missing = True

    def __post_init__(self):
        # Test parameter validity
        assert self.parameter_updates > 0
        assert self.lr > 0, "Learning rate must be larger than 0"
        assert isinstance(self.ent_embed_size, int) and self.ent_embed_size > 0
        assert self.ent_hidden_size is None or isinstance(self.ent_hidden_size, int) and self.ent_hidden_size > 0
        if isinstance(self.ent_hidden_size, int):
            assert not self.bert_attention, "When using BERT attention, entity hidden size cannot be specified"
        assert self.ent_intermediate_size is None or isinstance(self.ent_intermediate_size, int) and self.ent_intermediate_size > 0
        if isinstance(self.ent_intermediate_size, int):
            assert isinstance(self.ent_hidden_size, int), "Set entity hidden size explicitly before controlling entity intermediate size"
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

def get_optimizer_params(params: Iterator[tuple[str, Parameter]], do_decay: bool) -> list:
    """ Returns the parameters that should be tracked by optimizer with or without weight decay"""
    # Only include the parameter if do_decay has reverse truth value of the parameter being in no_decay
    include = lambda n: do_decay != any(nd in n for nd in NO_DECAY)
    return [p for n, p in params if p.requires_grad and include(n)]

def clean_saved_pu(loc: str, pu: int) -> list[str]:
    paths = {
        os.path.join(loc, TrainResults.subfolder, MODEL_OUT.format(i=pu)),
        os.path.join(loc, TrainResults.subfolder, OPTIMIZER_OUT.format(i=pu)),
        os.path.join(loc, TrainResults.subfolder, SCHEDULER_OUT.format(i=pu)),
        os.path.join(loc, TrainResults.subfolder, SCALER_OUT.format(i=pu)),
    }
    removed_paths = paths.copy()
    for path in paths:
        if os.path.isfile(path):
            os.remove(path)
        else:
            removed_paths.remove(path)
    return list(removed_paths)

def log_memory_stats(device: torch.device):
    if torch.cuda.is_available():
        log.debug("\n".join(torch.cuda.memory_summary(device=device, abbreviated=True).split("\n")[:-9]))

def save_training(
    loc:       str,
    params:    Hyperparams,
    model:     PretrainTaskDaLUKE,
    res:       TrainResults,
    optimizer: Optimizer,
    scheduler,
    scaler   = None,
    pu       = None,
) -> list[str]:
    pu = pu if pu is not None else res.parameter_update
    paths = list()
    # Save tracked statistics
    paths += res.save(loc)
    paths += params.save(loc)
    # Save model
    paths.append(os.path.join(loc, TrainResults.subfolder, MODEL_OUT.format(i=pu)))
    torch.save(model.state_dict(), paths[-1])
    # Save optimizer and scheduler states (these are dymanic over time)
    paths.append(os.path.join(loc, TrainResults.subfolder, OPTIMIZER_OUT.format(i=pu)))
    torch.save(optimizer.state_dict(), paths[-1])
    paths.append(os.path.join(loc, TrainResults.subfolder, SCHEDULER_OUT.format(i=pu)))
    torch.save(scheduler.state_dict(), paths[-1])
    # Save scaler if using fp16
    if scaler:
        paths.append(os.path.join(loc, TrainResults.subfolder, SCALER_OUT.format(i=pu)))
        torch.save(scaler.state_dict(), paths[-1])
    return paths

def parse_post_command(post_command: str) -> tuple[int, str] | tuple[None, None]:
    """ Returns the epoch time at which it should stop and the system command for resuming """
    if post_command:
        t, cmd = post_command.split(":::")
        h, m = (int(x) for x in t.split("h"))
        return time.time() + h * 3600 + m * 60, cmd
    return None, None

def save_progress(
    location: str,
    pu: int,
    tmp_saved_pu: int | None,
    save_pus: set[int],
    params: Hyperparams,
    module: nn.Module,
    res: TrainResults,
    optimizer: Optimizer,
    scheduler,
    scaler=None,
):
    paths = save_training(location, params, module, res, optimizer, scheduler, scaler)
    log.debug("Saved progress to", *paths)
    if pu > 0 and tmp_saved_pu not in save_pus and tmp_saved_pu is not None:
        # Clean temporarily saved files
        cleaned = clean_saved_pu(location, tmp_saved_pu)
        if cleaned:
            log.debug("Cleaned temporary files at", *cleaned)

def fix_base_model_params(new_weights: set[str], model: torch.nn.Module, fix: bool):
    """ Fixes or unfixes base model parameters """
    for n, p in model.named_parameters():
        if n not in new_weights:
            p.requires_grad = not fix

def train(
    rank:           int,
    world_size:     int,
    *,
    resume:         bool,
    location:       str,
    name:           str,
    quiet:          bool,
    save_every:     int,
    validate_every: int,
    post_command:   str,
    explicit_args:  set[str],
    params:         Hyperparams,
):
    # Get filepath within path context
    fpath = lambda path: os.path.join(location, path) if isinstance(path, str) else os.path.join(location, *path)

    # Setup multi-gpu if used
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

    post_time, post_command = parse_post_command(post_command)
    execute_post_command = False
    if post_time:
        log("Quitting in %.2f h and running command '%s'" % ((post_time-time.time())/3600, post_command))

    if resume:
        log("Resuming from %s" % name)
        # Load results and hyperparameters from earlier training
        res = TrainResults.load(location)
        # Close unended profiles
        close_tt(res.tt)
        TT.fuse(res.tt)
        res.tt = TT
        tmp_saved_pu = res.parameter_update
        loaded_params = Hyperparams.load(location)
        # Overwrite ff-size if given explicitly
        if "ff_size" in explicit_args:
            loaded_params.ff_size = params.ff_size
        params = loaded_params
    else:
        tmp_saved_pu = None
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
    if torch.cuda.is_available():
        log.debug("This worker runs on a %s" % torch.cuda.get_device_name(device))

    if params.entity_loss_weight:
        log("Setting up loss function with entity loss weighting")
        # Don't weigh special tokens
        weights = torch.Tensor([0, 0, 0, *(1 / info["count"] for info in entity_vocab.values() if info["count"])]).to(device)
        entity_criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        log("Setting up loss function without entity loss weighting")
        entity_criterion = nn.CrossEntropyLoss()
    word_criterion = nn.CrossEntropyLoss()
    loss_calculator = lambda w, e: params.word_ent_weight * w + (1 - params.word_ent_weight) * e

    # Load dataset and training results
    bert_config = AutoConfig.from_pretrained(metadata["base-model"])
    if metadata["reduced-vocab"]:
        token_map_file = fpath(DatasetBuilder.token_map_file)
        log("Loading token map from '%s'" % token_map_file)
        token_map = np.load(token_map_file)
        tokenizer = AutoTokenizer.from_pretrained(metadata["base-model"])
        *__, unk_id = get_special_ids(tokenizer)
        token_reduction = token_map_to_token_reduction(token_map, unk_id)
    else:
        token_map = None

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
        vocab_size=metadata["vocab-size"],
        token_map=token_map,
        ent_min_mention=params.ent_min_mention,
    )
    sampler = (DistributedSampler if is_distributed else RandomSampler)(data.train_examples)
    log("Built %i examples" % len(data))

    loader = data.get_dataloader(params.ff_size, sampler)
    val_loader = data.get_dataloader(params.ff_size, SequentialSampler(data.val_examples), validation=True)

    # Number of subbatches in each parameter update (batch)
    grad_accumulation_steps = params.batch_size // (params.ff_size * num_workers)
    # How many full batches can be made from the dataset
    batches_in_data = len(data) // params.batch_size
    log(
        "Parameter updates:               %i" % params.parameter_updates,
        "Subbatches per parameter update: %i" % grad_accumulation_steps,
        "Subbatches generated:            %i" % len(loader),
        "Batches needed to cover dataset: %i" % batches_in_data,
    )

    if not resume:
        # Calculate parameter differences, when at least 20k examples have been seen
        paramdiff_every = ceil(MIN_EXAMPLES_PER_PARAMDIFF / params.batch_size)
        log("Recalculating parameter differences every %i'th parameter update" % paramdiff_every)
        top_k = [1, 3, 10]
        log("Calculating top %s accuracies" % top_k)
        if validate_every:
            val_updates = unique(np.array(
                np.arange(-1, params.parameter_updates, validate_every).tolist() + [params.parameter_updates-1]
            ))[1:]
        else:
            val_updates = np.array([], dtype=int)
        res = TrainResults(
            runtime           = np.zeros(params.parameter_updates),
            lr                = np.zeros(params.parameter_updates),
            parameter_update  = 0,

            losses            = np.zeros(params.parameter_updates),
            scaled_loss       = np.zeros(params.parameter_updates),

            top_k             = top_k,
            w_losses          = np.zeros(params.parameter_updates),
            e_losses          = np.zeros(params.parameter_updates),
            w_accuracies      = np.zeros((params.parameter_updates, len(top_k))),
            e_accuracies      = np.zeros((params.parameter_updates, len(top_k))),

            val_param_updates = val_updates,
            val_losses        = np.zeros(len(val_updates)),
            val_w_losses      = np.zeros(len(val_updates)),
            val_e_losses      = np.zeros(len(val_updates)),
            val_w_accuracies  = np.zeros((len(val_updates), len(top_k))),
            val_e_accuracies  = np.zeros((len(val_updates), len(top_k))),

            paramdiff_every   = paramdiff_every,
            groups_to_slices  = None,  # Set later
            orig_params       = None,
            paramdiff_1       = None,

            luke_exclusive_params = None,  # Set later
            att_mats_from_base    = None,  # Set later

            tt = TT,
        )

    save_pus = set(range(-1, params.parameter_updates, save_every)).union({params.parameter_updates-1})
    log("Saving model at parameter updates: %s" % sorted(save_pus),
        "Validating at parameter updates: %s" % res.val_param_updates.tolist())

    # Build model, possibly by loading previous weights
    log.section("Setting up model")
    bert_config = AutoConfig.from_pretrained(metadata["base-model"])
    if params.ent_hidden_size is None:
        params.ent_hidden_size = bert_config.hidden_size
    else:
        assert params.ent_hidden_size <= bert_config.hidden_size,\
            "Entity hidden size (%i) cannot be larger than hidden size in '%s' (%i)" % (
                params.hidden_size,
                metadata["base-model"],
                bert_config.hidden_size,
            )

    log("Initializing model")
    model_cls = BertAttentionPretrainTaskDaLUKE if params.bert_attention else PretrainTaskDaLUKE
    model = model_cls(
        bert_config,
        ent_vocab_size        = len(entity_vocab),
        ent_embed_size        = params.ent_embed_size,
        ent_hidden_size       = params.ent_hidden_size,
        ent_intermediate_size = params.ent_intermediate_size,
    ).to(device)
    bert_config.vocab_size = metadata["vocab-size"]
    log("Bert config", bert_config.to_json_string())

    if params.lukeinit:
        log("Initializing weights in accordance with LUKE")
        model.apply(lambda module: model.init_weights(module, bert_config.initializer_range))
    # Load parameters from base model
    if not params.no_base_model:
        log("Loading base model parameters")
        with TT.profile("Loading base model parameters"):
            base_model = AutoModelForPreTraining.from_pretrained(metadata["base-model"])
            new_weights = load_base_model_weights(
                model,
                base_model.state_dict(),
                params.bert_attention,
            )
            if metadata["reduced-vocab"]:
                log("Removing unneeded token weights")
                reduced_model = model_cls(
                    bert_config,
                    ent_vocab_size        = len(entity_vocab),
                    ent_embed_size        = params.ent_embed_size,
                    ent_hidden_size       = params.ent_hidden_size,
                    ent_intermediate_size = params.ent_intermediate_size,
                ).to(device)
                copy_with_reduced_state_dict(token_reduction, model, reduced_model)
                model = reduced_model
    else:
        new_weights = set(model.state_dict())
    # Initialize self-attention query matrices to BERT word query matrices
    att_mat_keys = set()
    if not params.bert_attention and not params.no_base_model:
        log("Initializing new attention matrices with%s PCA" % ("" if params.pcainit else "out"))
        att_mat_keys = model.init_special_attention(params.pcainit, device)
    if not resume:
        res.luke_exclusive_params = new_weights
        res.att_mats_from_base = att_mat_keys
        if is_master:
            res.orig_params = all_params(model).cpu().numpy()
    log("Pretraining model initialized with %s parameters" % thousand_seps(len(model)))

    # Unfixes params at this parameter update
    unfix_base_model_params_pu = round(params.bert_fix_prop * params.parameter_updates)
    log("Unfixing base model params after %i parameter updates" % unfix_base_model_params_pu)

    if resume:
        mpath = fpath((TrainResults.subfolder, MODEL_OUT.format(i=res.parameter_update)))
        log("Loading model from '%s'" % mpath)
        model.load_state_dict(torch.load(mpath, map_location=device))
        log(f"Resuming training saved at parameter update {res.parameter_update}")
    else:
        res.groups_to_slices, t = all_params_groups_to_slices(model, bert_config.num_hidden_layers)
        log("Parameter groups and positions", t)
        res.paramdiff_1 = { name: np.zeros(ceil(params.parameter_updates/res.paramdiff_every)) for name in res.groups_to_slices }
    if is_distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    non_ddp_model = model.module if is_distributed else model

    log("Setting up optimizer, scaler, and learning rate scheduler")
    optimizer = AdamW(
        [{"params": get_optimizer_params(non_ddp_model.named_parameters(), do_decay=True),  "weight_decay": params.weight_decay},
         {"params": get_optimizer_params(non_ddp_model.named_parameters(), do_decay=False), "weight_decay": 0}],
        lr = params.lr,
    )
    scaler = amp.GradScaler() if params.fp16 else None
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(params.warmup_prop * params.parameter_updates),
        params.parameter_updates,
    )
    if resume:
        optimizer.load_state_dict(torch.load(fpath((TrainResults.subfolder, OPTIMIZER_OUT.format(i=res.parameter_update))), map_location=device))
        scheduler.load_state_dict(torch.load(fpath((TrainResults.subfolder, SCHEDULER_OUT.format(i=res.parameter_update))), map_location=device))
        if params.fp16:
            scaler.load_state_dict(torch.load(fpath((TrainResults.subfolder, SCALER_OUT.format(i=res.parameter_update))), map_location=device))
        res.parameter_update += 1  # We saved the data at pu i, but should now commence pu i+1

    log.debug("Time distribution before starting training", TT)
    log_memory_stats(device)

    log.section(f"Training DaLUKE for {params.parameter_updates} parameter updates")
    model.zero_grad()  # To avoid tracking of model parameter manipulation
    model.train()

    # Start with transfer learned weights locked
    fix_base_model_params(res.luke_exclusive_params, non_ddp_model, True)
    fixed_params = True

    # Save initial parameters
    if is_master and not resume:
        with TT.profile("Saving progress"):
            paths = save_training(location, params, model.module if is_distributed else model,
                res, optimizer, scheduler, scaler, -1)
            log.debug("Saved initial state to", *paths)

    batch_iter = iter(loader)
    for i in range(res.parameter_update, params.parameter_updates):
        TT.profile("Parameter update")
        res.parameter_update = i
        if i >= unfix_base_model_params_pu and is_distributed:
            model.find_unused_parameters = False
        if i >= unfix_base_model_params_pu and fixed_params:
            log("Unfixing base model params")
            fix_base_model_params(res.luke_exclusive_params, model, False)
            fixed_params = False
        if is_distributed and i % batches_in_data == 0:
            sampler.set_epoch(i // batches_in_data)

        # Losses and accuracies for this parameter update
        t_loss, w_loss, e_loss, s_loss = 0, 0, 0, 0
        w_accuracies = np.zeros((grad_accumulation_steps, len(res.top_k)))
        e_accuracies = np.zeros((grad_accumulation_steps, len(res.top_k)))

        # Loop over enough batches to make a parameter update
        for j in range(grad_accumulation_steps):
            TT.profile("Sub-batch")
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(loader)
                batch = next(batch_iter)

            TT.profile("FP and gradients")
            with amp.autocast() if params.fp16 else contextlib.ExitStack():
                word_preds, ent_preds = model(batch)
                # Compute and backpropagate loss
                word_loss = word_criterion(word_preds, batch.word_mask_labels)
                ent_loss = entity_criterion(ent_preds, batch.ent_mask_labels)
            loss = loss_calculator(word_loss, ent_loss)
            loss /= grad_accumulation_steps

            # Only sync parameters on grad updates, aka last pass of this loop
            with model.no_sync() if is_distributed and j < grad_accumulation_steps - 1 else contextlib.ExitStack():
                if params.fp16:
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                    s_loss += scaled_loss.item()
                else:
                    loss.backward()

            t_loss += loss.item()
            w_loss += word_loss.item() / grad_accumulation_steps
            e_loss += ent_loss.item() / grad_accumulation_steps

            if torch.cuda.is_available():
                torch.cuda.synchronize(rank if is_distributed else None)

            TT.end_profile()

            # Save accuracy for statistics
            if is_master:
                with TT.profile("Training accuracy"):
                    w_accuracies[j] = top_k_accuracy(batch.word_mask_labels, word_preds, res.top_k)
                    e_accuracies[j] = top_k_accuracy(batch.ent_mask_labels, ent_preds, res.top_k)

            TT.end_profile()

        # Update model parameters
        with TT.profile("Parameter step"):
            if params.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            model.zero_grad()

        # Calculate how much gradient has changed
        if is_master and i % res.paramdiff_every == 0:
            with torch.no_grad(), TT.profile("Parameter changes"):
                log.debug("Calculating parameter changes")
                orig_pars = torch.from_numpy(res.orig_params).to(device)
                current_pars = all_params(model.module if is_distributed else model)
                absdiff = torch.abs(current_pars-orig_pars)
                for blockname, slice_ in res.groups_to_slices.items():
                    j = i // res.paramdiff_every
                    res.paramdiff_1[blockname][j] = absdiff[slice_].sum().item()
                del orig_pars, current_pars

        res.losses[i]       = t_loss
        res.w_losses[i]     = w_loss
        res.e_losses[i]     = e_loss
        res.scaled_loss[i]  = s_loss
        res.lr[i]           = scheduler.get_last_lr()[0]
        res.w_accuracies[i] = np.mean(w_accuracies, axis=0)
        res.e_accuracies[i] = np.mean(e_accuracies, axis=0)
        res.runtime[i]      = TT.end_profile()
        log.debug(
            "Performed parameter update %i / %i in %.2f s" % (i, params.parameter_updates-1, res.runtime[i]),
            f"  Loss (total, word, entity, scaled): {t_loss:9.4f}, {w_loss:9.4f}, {e_loss:9.4f}, {s_loss:.4f}",
            f"  Accuracy (word, entity): {100*res.w_accuracies[i, 0]:7.2f} %, {100*res.e_accuracies[i, 0]:7.2f} %",
        )

        if i in res.val_param_updates and is_master:
            TT.profile("Model validation")
            log("Validating model")
            vi = res.val_param_updates.tolist().index(i)
            res.val_w_losses[vi], res.val_e_losses[vi], res.val_w_accuracies[vi], res.val_e_accuracies[vi] =\
                validate_model(model, val_loader, word_criterion, entity_criterion, res.top_k)
            res.val_losses[vi] = loss_calculator(res.val_w_losses[vi], res.val_e_losses[vi])
            log(
                "Validation loss:",
                "  Total:  %9.4f" % res.val_losses[vi],
                "  Word:   %9.4f" % res.val_w_losses[vi],
                "  Entity: %9.4f" % res.val_e_losses[vi],
                "Validation accuracy:",
                "  Word:   %7.2f %%" % (100 * res.val_w_accuracies[vi, 0]),
                "  Entity: %7.2f %%" % (100 * res.val_e_accuracies[vi, 0]),
            )
            model.train()
            TT.end_profile()
            log.debug("Time distribution so far", TT)

        # Save results and model
        if is_master and i in save_pus:
            with TT.profile("Saving progress"):
                save_progress(location, i, tmp_saved_pu, save_pus, params,
                    model.module if is_distributed else model, res, optimizer, scheduler, scaler)
        if i in save_pus:
            log_memory_stats(device)

        # If timed out, save, quit, and run resume command
        if post_time and time.time() > post_time:
            log_memory_stats(device)
            log.section("Time limit reached. Quitting and running command '%s'" % post_command)
            with TT.profile("Saving progress"):
                save_progress(location, i, tmp_saved_pu, save_pus, params,
                    model.module if is_distributed else model, res, optimizer, scheduler, scaler)
            execute_post_command = True
            break

    log.debug("Time distribution", TT)

    # Clean up multi-gpu if used
    cleanup(rank)

    if is_master and execute_post_command:
        os.system(post_command)
