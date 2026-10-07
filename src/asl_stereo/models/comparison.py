"""Offline capacity alternatives sharing the production 138/286 contracts."""

from __future__ import annotations

import torch
from torch import nn

from .attention import TemporalQueryAttention
from .classifier import (
    INTERNAL_FEATURE_COUNT, SignSequenceClassifier, _validate_sequence,
    temporal_hand_features,
)


class ConvGRUSequenceClassifier(nn.Module):
    """Small, time-preserving 1D CNN followed by a 96-unit GRU."""

    def __init__(self, num_classes: int, hidden_size: int = 96, dropout: float = 0.3) -> None:
        super().__init__()
        if num_classes < 2 or hidden_size <= 0:
            raise ValueError("num_classes >= 2 and positive hidden_size required")
        self.num_classes = num_classes
        self.input_features = 138
        self.internal_feature_dim = INTERNAL_FEATURE_COUNT
        self.feature_projection = nn.Sequential(
            nn.Linear(INTERNAL_FEATURE_COUNT, 128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.convolution = nn.Sequential(
            nn.Conv1d(128, hidden_size, 5, padding=2), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
        )
        self.temporal_backbone = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.attention_pool = TemporalQueryAttention(hidden_size)
        self.classification_head = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_sequence(inputs, 138)
        projected = self.feature_projection(temporal_hand_features(inputs))
        convolved = self.convolution(projected.transpose(1, 2)).transpose(1, 2)
        temporal, _ = self.temporal_backbone(convolved)
        pooled, _ = self.attention_pool(temporal)
        logits = self.classification_head(pooled)
        return logits, torch.softmax(logits, dim=-1)


MODEL_SPECS = {
    "A": {"architecture": "bilstm_attention", "hidden_size": 256, "num_layers": 2, "dropout": 0.3},
    "B": {"architecture": "bilstm_attention", "hidden_size": 128, "num_layers": 1, "dropout": 0.3},
    "C": {"architecture": "cnn_gru_attention", "hidden_size": 96, "num_layers": 1, "dropout": 0.3},
}


def build_comparison_model(name: str, num_classes: int) -> nn.Module:
    if name not in MODEL_SPECS:
        raise ValueError(f"unknown comparison model: {name}")
    spec = MODEL_SPECS[name]
    if name == "C":
        return ConvGRUSequenceClassifier(num_classes)
    return SignSequenceClassifier(
        num_classes, hidden_size=spec["hidden_size"], num_layers=spec["num_layers"],
        dropout=spec["dropout"],
        align_features=False,
    )


__all__ = ["ConvGRUSequenceClassifier", "MODEL_SPECS", "build_comparison_model"]
