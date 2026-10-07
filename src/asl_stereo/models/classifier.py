"""Compact CPU-oriented temporal classifiers."""

from __future__ import annotations

import torch
from torch import nn

from .attention import TemporalQueryAttention
from .feature_alignment import align_feature_window


RAW_FEATURE_COUNT = 138
HAND_KINEMATIC_FEATURE_COUNT = 10
INTERNAL_FEATURE_COUNT = RAW_FEATURE_COUNT * 2 + HAND_KINEMATIC_FEATURE_COUNT


def position_velocity_features(inputs: torch.Tensor) -> torch.Tensor:
    """Append first differences, with a zero velocity at the first frame."""
    velocity = torch.cat(
        (torch.zeros_like(inputs[:, :1]), inputs[:, 1:] - inputs[:, :-1]), dim=1,
    )
    return torch.cat((inputs, velocity), dim=-1)


def hand_aperture_features(inputs: torch.Tensor) -> torch.Tensor:
    """Return six tip-to-wrist/palm ratios in the checkpoint's exact order."""
    joints = inputs.reshape(inputs.shape[0], inputs.shape[1], 46, 3)
    left_wrist, right_wrist = joints[:, :, 0], joints[:, :, 21]
    left_palm = torch.linalg.vector_norm(joints[:, :, 9] - left_wrist, dim=-1)
    right_palm = torch.linalg.vector_norm(joints[:, :, 30] - right_wrist, dim=-1)
    left = torch.stack(
        (torch.linalg.vector_norm(joints[:, :, 8] - left_wrist, dim=-1),
         torch.linalg.vector_norm(joints[:, :, 12] - left_wrist, dim=-1),
         torch.linalg.vector_norm(joints[:, :, 4] - left_wrist, dim=-1)), dim=-1,
    ) / left_palm.clamp_min(1e-4).unsqueeze(-1)
    right = torch.stack(
        (torch.linalg.vector_norm(joints[:, :, 29] - right_wrist, dim=-1),
         torch.linalg.vector_norm(joints[:, :, 33] - right_wrist, dim=-1),
         torch.linalg.vector_norm(joints[:, :, 25] - right_wrist, dim=-1)), dim=-1,
    ) / right_palm.clamp_min(1e-4).unsqueeze(-1)
    return torch.cat((left, right), dim=-1).clamp(max=4.0)


def hand_closure_features(inputs: torch.Tensor) -> torch.Tensor:
    """Return four shoulder-normalized thumb-index/thumb-middle distances."""
    joints = inputs.reshape(inputs.shape[0], inputs.shape[1], 46, 3)
    return torch.stack(
        (torch.linalg.vector_norm(joints[:, :, 4] - joints[:, :, 8], dim=-1),
         torch.linalg.vector_norm(joints[:, :, 4] - joints[:, :, 12], dim=-1),
         torch.linalg.vector_norm(joints[:, :, 25] - joints[:, :, 29], dim=-1),
         torch.linalg.vector_norm(joints[:, :, 25] - joints[:, :, 33], dim=-1)),
        dim=-1,
    )


def hand_elevation_features(inputs: torch.Tensor) -> torch.Tensor:
    """Experimental height helper; excluded from the 286-feature model."""
    joints = inputs.reshape(inputs.shape[0], inputs.shape[1], 46, 3)
    shoulder_y = (joints[:, :, 42, 1] + joints[:, :, 43, 1]) * 0.5
    heights = torch.stack(
        (joints[:, :, 0, 1] - shoulder_y, joints[:, :, 21, 1] - shoulder_y), dim=-1,
    )
    present = torch.stack(
        (torch.any(joints[:, :, :21] != 0, dim=(-2, -1)),
         torch.any(joints[:, :, 21:42] != 0, dim=(-2, -1))), dim=-1,
    )
    return torch.where(present, heights, torch.zeros_like(heights))


def temporal_hand_features(
    inputs: torch.Tensor, include_wrist_height: bool = False,
) -> torch.Tensor:
    """Position, velocity, six aperture ratios, then four closure distances.

    This order matches the archived 73.91% checkpoint. Wrist height is an
    explicit opt-in for feature experiments, never used by production models.
    """
    features = torch.cat(
        (position_velocity_features(inputs), hand_aperture_features(inputs),
         hand_closure_features(inputs)), dim=-1,
    )
    if include_wrist_height:
        return torch.cat((features, hand_elevation_features(inputs)), dim=-1)
    return features


class SignSequenceClassifier(nn.Module):
    """138 external coordinates, 286 internal kinematics, BiLSTM attention."""

    def __init__(
        self,
        num_classes: int,
        *,
        input_features: int = 138,
        projection_features: int = 128,
        hidden_size: int = 256,
        dropout: float = 0.35,
        num_layers: int = 2,
        align_features: bool = True,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if input_features != RAW_FEATURE_COUNT:
            raise ValueError("the 46-joint model requires 138 input features")
        if hidden_size <= 0 or num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")
        self.num_classes = num_classes
        self.input_features = input_features
        self.internal_feature_dim = INTERNAL_FEATURE_COUNT
        self.model_type = "bilstm_attention"
        self.feature_alignment_enabled = align_features
        self.projection_features = projection_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_probability = dropout
        self.feature_projection = nn.Sequential(
            nn.Linear(self.internal_feature_dim, projection_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.temporal_backbone = nn.LSTM(
            input_size=projection_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # A single LSTM layer has no recurrent dropout in PyTorch. Explicit
        # output dropout regularizes that comparison without changing model A.
        self.temporal_dropout = nn.Dropout(dropout) if num_layers == 1 else nn.Identity()
        self.temporal_reduction = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_pool = TemporalQueryAttention(hidden_size)
        self.classification_head = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_sequence(inputs, self.input_features)
        if not torch.jit.is_tracing() and not torch.jit.is_scripting():
            if inputs.shape[1] != 45:
                raise ValueError("BiLSTM inputs must have shape (batch, 45, 138)")
        if self.feature_alignment_enabled:
            inputs = align_feature_window(inputs)
        projected = self.feature_projection(temporal_hand_features(inputs))
        temporal, _ = self.temporal_backbone(projected)
        reduced = torch.relu(self.temporal_reduction(self.temporal_dropout(temporal)))
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
        if input_features != RAW_FEATURE_COUNT:
            raise ValueError("the 46-joint model requires 138 input features")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_classes = num_classes
        self.input_features = input_features
        self.internal_feature_dim = INTERNAL_FEATURE_COUNT
        self.max_sequence_length = max_sequence_length
        self.feature_projection = nn.Sequential(
            nn.Linear(self.internal_feature_dim, model_dim),
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
        projected = self.feature_projection(temporal_hand_features(inputs))
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


__all__ = [
    "CompactPreLNTransformerClassifier", "SignSequenceClassifier",
    "INTERNAL_FEATURE_COUNT", "hand_aperture_features", "hand_closure_features",
    "hand_elevation_features", "position_velocity_features", "temporal_hand_features",
]
