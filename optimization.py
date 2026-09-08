"""Detector-neutral optimizer and iteration-based LR scheduler builders."""

from bisect import bisect_right

import torch
from torch import nn

from config import TrainingConfig


class WarmupMultiStepLR(torch.optim.lr_scheduler.LRScheduler):
    """Iteration scheduler whose complete schedule is checkpointable."""

    def __init__(
        self,
        optimizer,
        milestones,
        gamma: float,
        warmup_factor: float,
        warmup_iters: int,
        last_epoch: int = -1,
    ) -> None:
        self.milestones = tuple(milestones)
        self.gamma = float(gamma)
        self.warmup_factor = float(warmup_factor)
        self.warmup_iters = int(warmup_iters)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        iteration = self.last_epoch
        if self.warmup_iters > 0 and iteration < self.warmup_iters:
            alpha = iteration / self.warmup_iters
            warmup = self.warmup_factor * (1.0 - alpha) + alpha
        else:
            warmup = 1.0
        decay = self.gamma ** bisect_right(self.milestones, iteration)
        return [base_lr * warmup * decay for base_lr in self.base_lrs]


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.SGD:
    """Build D3T-style SGD over every trainable student parameter exactly once."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("student model has no trainable parameters")
    settings = config.optimizer
    return torch.optim.SGD(
        parameters,
        lr=settings.base_lr,
        momentum=settings.momentum,
        weight_decay=settings.weight_decay,
    )


def build_lr_scheduler(optimizer, config: TrainingConfig):
    """Build the linear-warmup MultiStep schedule used by D3T."""

    settings = config.optimizer
    return WarmupMultiStepLR(
        optimizer,
        milestones=settings.milestones,
        gamma=settings.gamma,
        warmup_factor=settings.warmup_factor,
        warmup_iters=settings.warmup_iters,
    )


__all__ = ["WarmupMultiStepLR", "build_lr_scheduler", "build_optimizer"]
