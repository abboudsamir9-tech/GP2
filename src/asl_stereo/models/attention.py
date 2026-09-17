"""Learned-query temporal attention pooling."""

from __future__ import annotations

import math

import torch
from torch import nn


class TemporalQueryAttention(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        nn.init.normal_(self.query, mean=0.0, std=embedding_dim**-0.5)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 3 or sequence.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"sequence must have shape (batch, time, {self.embedding_dim})"
            )
        query = self.query.expand(sequence.shape[0], -1, -1)
        scores = torch.bmm(query, sequence.transpose(1, 2)) / math.sqrt(
            self.embedding_dim
        )
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights, sequence).squeeze(1)
        return pooled, weights.squeeze(1)


__all__ = ["TemporalQueryAttention"]
