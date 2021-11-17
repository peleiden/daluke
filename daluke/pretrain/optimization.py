from __future__ import annotations
from typing import Iterator

from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from transformers import AdamW

NO_DECAY =  { "bias", "LayerNorm.weight" }

def get_optimizer_params(params: Iterator[tuple[str, Parameter]], do_decay: bool) -> list:
    """ Returns the parameters that should be tracked by optimizer with or without weight decay"""
    # Only include the parameter if do_decay has reverse truth value of the parameter being in no_decay
    include = lambda n: do_decay != any(nd in n for nd in NO_DECAY)
    return [p for n, p in params if p.requires_grad and include(n)]

def get_optimizer(model: torch.nn.Module, weight_decay: float, lr: float) -> AdamW:
    return AdamW(
        [
            {"params": get_optimizer_params(model.named_parameters(), do_decay=True),  "weight_decay": weight_decay},
            {"params": get_optimizer_params(model.named_parameters(), do_decay=False), "weight_decay": 0}
        ],
        lr = lr,
    )

def get_lr_scheduler(
        optimizer:          Optimizer,
        num_warmup_steps:   int,
        num_training_steps: int,
        unlock_steps:       int,
        last_epoch:         int = -1,
    ) -> LambdaLR:
    """
    Adaptiation of [1] but with a triangle at `unlock_steps`.

    [1] https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup
    """
    num_unlock_warmup_steps = int(0.2 * num_warmup_steps)
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        curr_lambda = max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
        # From unlock time and num_warmup_steps ahead, we go back to low LR and rise linearly
        if 0 <= (time_after_unlock := current_step - unlock_steps) < num_unlock_warmup_steps:
            return curr_lambda * time_after_unlock / max(1, num_unlock_warmup_steps)
        return curr_lambda
    return LambdaLR(optimizer, lr_lambda, last_epoch)

if __name__ == "__main__":
    import torch
    from torch.optim import SGD
    import matplotlib.pyplot as plt

    steps = 20_000
    optimizer = SGD([torch.tensor(1)], lr=1)

    scheduler = get_lr_scheduler(optimizer, int(.06*steps), steps, steps//2)

    lrs = list()
    for _ in range(steps):
        optimizer.step()
        lrs.append(scheduler.get_last_lr())
        scheduler.step()

    plt.plot(lrs)
    plt.show()
