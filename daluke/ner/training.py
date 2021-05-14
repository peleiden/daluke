from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup

from pelutils import log, DataStorage

from .evaluation import evaluate_ner, pred_distribution

@dataclass
class TrainResults(DataStorage):
    losses: list

class TrainNER:
    # These layers should not be subject to weight decay
    no_decay = {"bias", "LayerNorm.weight"}

    def __init__(self,
            model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            dataset: NERDataset,
            device: torch.device,
            epochs: int,
            lr: float = 1e-5,
            warmup_prop: float = 0.06,
            weight_decay: float = 0.01,
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
        self.optimizer = AdamW(
            [{"params": self._get_optimizer_params(params, do_decay=True), "weight_decay": weight_decay},
             {"params": self._get_optimizer_params(params, do_decay=False), "weight_decay": 0.0}],
            lr           = lr,
        )
        # Create LR scheduler
        num_updates = epochs * len(self.dataloader)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(warmup_prop * num_updates), num_updates)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def run(self):
        self.model.train()
        losses = list()
        for i in range(self.epochs):
            for j, batch in enumerate(self.dataloader):
                scores = self.model(batch)
                loss = self.criterion(scores.view(-1, self.model.output_shape), batch.entities.labels.view(-1))
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

                losses.append(loss.item())
                log.debug(f"Epoch {i} / {self.epochs-1}, batch: {j} / {len(self.dataloader)-1}. Loss: {loss.item():.5f}.")
            if self.dev_dataloader is not None:
                log("Evaluating on development set ...")
                dev_results = evaluate_ner(self.model, self.dev_dataloader, self.dataset, self.device, Split.DEV, also_no_misc=False)
                pred_distribution(dev_results)
                self.model.train()
                # TODO: Save running dev scores and plot afterwards

        return TrainResults(
            losses = losses,
        )

    def _get_optimizer_params(self, params: list, do_decay: bool) -> list:
        # Only include the parameter if do_decay has reverse truth value of the parameter being in no_decay
        save = lambda n: not do_decay == any(nd in n for nd in self.no_decay)
        return [p for n, p in params if p.requires_grad and save(n)]
