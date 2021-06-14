from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy

import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup

from pelutils import log, DataStorage

from .evaluation import evaluate_ner, type_distribution, NER_Results
from .data import Split, NERDataset

@dataclass
class TrainResults(DataStorage):
    epoch: int
    losses: list[float]
    best_epoch: int | None
    running_train_statistics: list[dict]
    train_pred_distributions: list[dict[str, int]]
    train_true_type_distribution: dict[str, int]
    running_dev_evaluations: list[NER_Results]
    dev_pred_distributions: list[dict[str, int]]
    dev_true_type_distribution: dict[str, int]

    subfolder = "train-results"


class TrainNER:
    # These layers should not be subject to weight decay
    no_decay = {"bias", "LayerNorm.weight"}

    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        dataset: NERDataset,
        device: torch.device,
        epochs: int,
        lr: float,
        warmup_prop: float,
        weight_decay: float,
        loss_weight: bool,
        dev_dataloader: torch.utils.data.DataLoader | None = None,
    ):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.dataset = dataset
        self.dev_dataloader = dev_dataloader
        self.epochs = epochs
        # Create optimizer
        params = list(model.named_parameters())
        optimizer_parameters = [
             {"params": self._get_optimizer_params(params, do_decay=True), "weight_decay": weight_decay},
             {"params": self._get_optimizer_params(params, do_decay=False), "weight_decay": 0.0}
        ]
        self.optimizer = AdamW(
            optimizer_parameters,
            lr           = lr,
            betas        = (0.9, 0.98),
            correct_bias = False,
        )
        # Create LR scheduler
        num_updates = epochs * len(self.dataloader)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(warmup_prop * num_updates), num_updates)
        if loss_weight:
            counts = torch.zeros(len(dataset.all_labels))
            for _, e in self.dataloader.dataset:
                # Do count on the non-padded labels
                for label, count in zip(*e.entities.labels[:e.entities.N].unique(return_counts=True)):
                    counts[label] += count
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=1/counts.to(device) if loss_weight else None)
        self.best_model = None

    def run(self) -> TrainResults:
        res = TrainResults(
            epoch                        = 0,
            losses                       = list(),
            best_epoch                   = None,
            running_train_statistics     = list(),
            running_dev_evaluations      = list(),
            dev_pred_distributions       = list(),
            dev_true_type_distribution   = dict(),
            train_pred_distributions     = list(),
            train_true_type_distribution = dict()
        )
        for i in range(self.epochs):
            res.epoch = i
            self.model.train()
            for j, batch in enumerate(self.dataloader):
                scores = self.model(batch)
                loss = self.criterion(scores.view(-1, self.model.output_shape), batch.entities.labels.view(-1))
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

                res.losses.append(loss.item())
                log.debug(f"Epoch {i} / {self.epochs-1}, batch: {j} / {len(self.dataloader)-1}. LR: {self.scheduler.get_last_lr()[0]:.2e} Loss: {loss.item():.5f}.")

            # Perform running evaluation
            if self.dev_dataloader is not None:
                log("Evaluating on development set ...")
                dev_results = evaluate_ner(self.model, self.dev_dataloader, self.dataset, self.device, Split.DEV, also_no_misc=False)
                res.running_dev_evaluations.append(dev_results)
                res.dev_pred_distributions.append(type_distribution(dev_results.preds))

                log("Evaluating on training set ...")
                train_results = evaluate_ner(self.model, self.dataloader, self.dataset, self.device, Split.TRAIN, also_no_misc=False)
                res.running_train_statistics.append(train_results.statistics)
                res.train_pred_distributions.append(type_distribution(train_results.preds))
                if res.best_epoch is None or\
                        (dev_results.statistics["micro avg"]["f1-score"]) > res.running_dev_evaluations[res.best_epoch].statistics["micro avg"]["f1-score"]:
                    log(f"Found new best model at epoch {i}")
                    self.best_model = deepcopy(self.model)
                    res.best_epoch = i
        return res

    def _get_optimizer_params(self, params: list, do_decay: bool) -> list:
        # Only include the parameter if do_decay has reverse truth value of the parameter being in no_decay
        save = lambda n: not do_decay == any(nd in n for nd in self.no_decay)
        return [p for n, p in params if p.requires_grad and save(n)]
