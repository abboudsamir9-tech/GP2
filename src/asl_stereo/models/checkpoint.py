"""Restricted model-weight and class-map loading."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn


CHECKPOINT_KEYS = {
    "model_state_dict",
    "num_classes",
    "window_size",
    "feature_dim",
    "model_type",
    "epoch",
    "best_val_acc",
}


def load_checkpoint(checkpoint_path: str | Path) -> Mapping[str, Any]:
    payload = torch.load(
        Path(checkpoint_path), map_location=torch.device("cpu"), weights_only=True
    )
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    return payload


def load_model_weights(
    model: nn.Module, checkpoint_path: str | Path, *, strict: bool = True
) -> nn.Module:
    checkpoint = load_checkpoint(checkpoint_path)
    if "model_state_dict" in checkpoint:
        missing = CHECKPOINT_KEYS - set(checkpoint)
        if missing:
            raise ValueError(f"structured checkpoint is missing keys: {sorted(missing)}")
        _validate_checkpoint_schema(model, checkpoint)
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint must contain a state dictionary")
    model.load_state_dict(state_dict, strict=strict)
    return model


def save_training_checkpoint(
    model: nn.Module,
    output_path: str | Path,
    *,
    num_classes: int,
    window_size: int,
    feature_dim: int,
    epoch: int,
    best_val_acc: float,
    model_type: str = "bilstm_attention",
) -> Path:
    if num_classes < 2 or window_size <= 0 or feature_dim <= 0 or epoch < 0:
        raise ValueError("checkpoint dimensions and epoch are invalid")
    if not 0.0 <= best_val_acc <= 1.0:
        raise ValueError("best_val_acc must be in [0, 1]")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "window_size": window_size,
        "feature_dim": feature_dim,
        "model_type": model_type,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
    }
    torch.save(payload, output)
    return output


def save_class_map(class_map: Mapping[int, str], output_path: str | Path) -> Path:
    normalized = load_class_map(class_map)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({str(index): label for index, label in normalized.items()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return output


def _validate_checkpoint_schema(model: nn.Module, checkpoint: Mapping[str, Any]) -> None:
    expected_classes = getattr(model, "num_classes", None)
    if expected_classes is not None and int(checkpoint["num_classes"]) != expected_classes:
        raise ValueError("checkpoint num_classes does not match model")
    expected_features = getattr(model, "input_features", None)
    if expected_features is not None and int(checkpoint["feature_dim"]) != expected_features:
        raise ValueError("checkpoint feature_dim does not match model")
    if int(checkpoint["window_size"]) != 45:
        raise ValueError("checkpoint window_size must be 45")
    if str(checkpoint["model_type"]) != "bilstm_attention":
        raise ValueError("unsupported checkpoint model_type")


def load_class_map(
    source: Mapping[int | str, str] | Sequence[str] | str | Path,
) -> dict[int, str]:
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8") as stream:
            source = json.load(stream)

    if isinstance(source, Mapping):
        class_map = {int(index): str(label) for index, label in source.items()}
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        class_map = {index: str(label) for index, label in enumerate(source)}
    else:
        raise TypeError("class map must be a mapping, sequence, or JSON path")

    if not class_map or sorted(class_map) != list(range(len(class_map))):
        raise ValueError("class map indices must be contiguous and start at zero")
    if any(not label for label in class_map.values()):
        raise ValueError("class labels must be non-empty")
    if len(set(class_map.values())) != len(class_map):
        raise ValueError("class labels must be unique")
    return class_map


__all__ = [
    "CHECKPOINT_KEYS",
    "load_checkpoint",
    "load_class_map",
    "load_model_weights",
    "save_class_map",
    "save_training_checkpoint",
]
