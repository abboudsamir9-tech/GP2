"""Class-balanced losses for small ASL vocabularies."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Multiclass focal loss with per-true-class inverse-frequency weights."""

    def __init__(self, class_weights: torch.Tensor, *, gamma: float = 2.0) -> None:
        super().__init__()
        if class_weights.ndim != 1 or class_weights.numel() < 2:
            raise ValueError("class_weights must contain at least two classes")
        if not bool(torch.isfinite(class_weights).all()) or bool((class_weights <= 0).any()):
            raise ValueError("class_weights must be positive and finite")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.register_buffer("class_weights", class_weights.detach().clone().float())
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.class_weights.numel():
            raise ValueError("logits must have one column per class")
        if targets.shape != (logits.shape[0],):
            raise ValueError("targets must contain one label per sample")
        log_probabilities = F.log_softmax(logits, dim=1)
        log_true_probability = log_probabilities.gather(1, targets[:, None]).squeeze(1)
        true_probability = log_true_probability.exp()
        per_sample = (
            self.class_weights[targets]
            * (1.0 - true_probability).pow(self.gamma)
            * -log_true_probability
        )
        return per_sample.mean()


__all__ = ["FocalLoss"]
