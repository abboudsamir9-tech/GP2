"""Compact CPU-oriented temporal classifiers."""

from __future__ import annotations

import torch
from torch import nn

from .attention import TemporalQueryAttention


class SignSequenceClassifier(nn.Module):
    """Projected two-layer BiLSTM with learned-query temporal pooling."""

    def __init__(
        self,
        num_classes: int,
        *,
        input_features: int = 138,
        projection_features: int = 128,
        hidden_size: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.num_classes = num_classes
        self.input_features = input_features
        self.feature_projection = nn.Sequential(
            nn.Linear(input_features, projection_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.temporal_backbone = nn.LSTM(
            input_size=projection_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.temporal_reduction = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_pool = TemporalQueryAttention(hidden_size)
        self.classification_head = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_sequence(inputs, self.input_features)
        projected = self.feature_projection(inputs)
        temporal, _ = self.temporal_backbone(projected)
        reduced = torch.relu(self.temporal_reduction(temporal))
        pooled, _ = self.attention_pool(reduced)
        logits = self.classification_head(pooled)
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities


class CompactPreLNTransformerClassifier(nn.Module):
    """Compact pre-layer-normalized Transformer alternative."""

    def __init__(
        self,
        num_classes: int,
        *,
        input_features: int = 138,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.2,
        max_sequence_length: int = 60,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_classes = num_classes
        self.input_features = input_features
        self.max_sequence_length = max_sequence_length
        self.feature_projection = nn.Sequential(
            nn.Linear(input_features, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, max_sequence_length, model_dim)
        )
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_backbone = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.attention_pool = TemporalQueryAttention(model_dim)
        self.classification_head = nn.Linear(model_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_sequence(inputs, self.input_features)
        if inputs.shape[1] > self.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")
        projected = self.feature_projection(inputs)
        projected = projected + self.positional_embedding[:, : inputs.shape[1]]
        temporal = self.final_norm(self.temporal_backbone(projected))
        pooled, _ = self.attention_pool(temporal)
        logits = self.classification_head(pooled)
        probabilities = torch.softmax(logits, dim=-1)
        return logits, probabilities


def _validate_sequence(inputs: torch.Tensor, input_features: int) -> None:
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    # Trace inputs are validated by the export boundary. Comparing symbolic
    # dimensions here forces a Tensor-to-bool conversion under Torch 2.3.
    if torch.jit.is_tracing() or torch.jit.is_scripting():
        return
    if inputs.ndim != 3 or inputs.shape[-1] != input_features:
        raise ValueError(
            f"inputs must have shape (batch, time, {input_features})"
        )
    if not inputs.is_floating_point():
        raise TypeError("inputs must use a floating-point dtype")


__all__ = ["CompactPreLNTransformerClassifier", "SignSequenceClassifier"]
