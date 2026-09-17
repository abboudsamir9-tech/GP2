"""Timed CPU inference with restricted checkpoint and TorchScript loading."""

from __future__ import annotations

import time
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from .checkpoint import load_class_map, load_model_weights
from asl_stereo.contracts import Prediction
from asl_stereo.translation.confidence_filter import ConfidenceFilter


@dataclass(frozen=True, slots=True)
class InferenceResult:
    predicted_gloss: str
    class_index: int
    confidence_score: float
    probabilities: tuple[float, ...]
    latency_ms: float

    def __post_init__(self) -> None:
        if not self.predicted_gloss:
            raise ValueError("predicted_gloss must be non-empty")
        if self.class_index < 0:
            raise ValueError("class_index must be non-negative")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("confidence_score must be in [0, 1]")
        if self.latency_ms < 0.0:
            raise ValueError("latency_ms must be non-negative")


class InferenceEngine:
    """Own a model in evaluation mode and perform measured CPU inference."""

    MIN_CONFIDENCE = 0.65
    MIN_MARGIN = 0.15

    def __init__(
        self,
        model: nn.Module,
        class_map: Mapping[int | str, str] | Sequence[str] | str | Path,
        *,
        checkpoint_path: str | Path | None = None,
        strict_weights: bool = True,
    ) -> None:
        self.device = torch.device("cpu")
        self.class_map = load_class_map(class_map)
        self.model = model.to(self.device)
        if checkpoint_path is not None:
            self.model = self._load_artifact(
                Path(checkpoint_path), strict_weights=strict_weights
            )
        expected_classes = getattr(self.model, "num_classes", None)
        if expected_classes is not None and expected_classes != len(self.class_map):
            raise ValueError("class map size does not match model num_classes")
        self.model.eval()
        self.last_latency_ms: float | None = None

    def _load_artifact(
        self, checkpoint_path: Path, *, strict_weights: bool
    ) -> nn.Module:
        """Load a TorchScript production artifact or a restricted state checkpoint."""
        if checkpoint_path.suffix.lower() == ".pt":
            extra_files = {"metadata.json": ""}
            try:
                scripted = torch.jit.load(
                    str(checkpoint_path),
                    map_location=self.device,
                    _extra_files=extra_files,
                )
            except RuntimeError:
                # Some legacy state dictionaries use a .pt suffix.
                load_model_weights(
                    self.model, checkpoint_path, strict=strict_weights
                )
                return self.model
            self._validate_torchscript_metadata(extra_files["metadata.json"])
            return scripted.to(self.device)

        load_model_weights(self.model, checkpoint_path, strict=strict_weights)
        return self.model

    def _validate_torchscript_metadata(self, raw_metadata: str | bytes) -> None:
        if not raw_metadata:
            raise ValueError("TorchScript artifact is missing metadata.json")
        try:
            metadata = json.loads(raw_metadata)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
            raise ValueError("TorchScript artifact metadata is invalid") from exc
        if int(metadata.get("window_size", -1)) != 45:
            raise ValueError("TorchScript window_size must be 45")
        if int(metadata.get("feature_dim", -1)) != 138:
            raise ValueError("TorchScript feature_dim must be 138")
        if int(metadata.get("num_classes", -1)) != len(self.class_map):
            raise ValueError("class map size does not match TorchScript metadata")
        embedded_map = metadata.get("class_map")
        if embedded_map is not None and load_class_map(embedded_map) != self.class_map:
            raise ValueError("class map does not match TorchScript metadata")

    def predict(
        self, window: torch.Tensor | npt.NDArray[np.float32]
    ) -> InferenceResult:
        tensor = self._prepare_window(window)
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model(tensor)
            if isinstance(output, tuple):
                logits, probabilities = output
            else:
                logits = output
                probabilities = torch.softmax(logits, dim=-1)
        latency_ms = (time.perf_counter() - started) * 1_000.0
        self.last_latency_ms = latency_ms

        if logits.shape != (1, len(self.class_map)):
            raise ValueError("model logits must have shape (1, num_classes)")
        probabilities = probabilities[0].detach().to("cpu", dtype=torch.float32)
        class_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[class_index].item())
        return InferenceResult(
            predicted_gloss=self.class_map[class_index],
            class_index=class_index,
            confidence_score=confidence,
            probabilities=tuple(float(value) for value in probabilities.tolist()),
            latency_ms=latency_ms,
        )

    def gate_prediction(
        self,
        result: InferenceResult,
        *,
        confidence_threshold: float = MIN_CONFIDENCE,
        margin_threshold: float = MIN_MARGIN,
        window_end_timestamp_ns: int = 0,
    ) -> Prediction:
        """Apply immutable system floors before a result can be confirmed."""
        threshold = max(float(confidence_threshold), self.MIN_CONFIDENCE)
        margin = max(float(margin_threshold), self.MIN_MARGIN)
        labels = tuple(self.class_map[index] for index in range(len(self.class_map)))
        return ConfidenceFilter(
            confidence_threshold=threshold,
            margin_threshold=margin,
        ).apply(
            result.probabilities,
            labels,
            inference_duration_ms=result.latency_ms,
            window_end_timestamp_ns=window_end_timestamp_ns,
        )

    def _prepare_window(
        self, window: torch.Tensor | npt.NDArray[np.float32]
    ) -> torch.Tensor:
        if isinstance(window, np.ndarray):
            if window.dtype != np.float32 or not window.flags.c_contiguous:
                raise TypeError("NumPy windows must be contiguous float32 arrays")
            tensor = torch.from_numpy(window)
        elif isinstance(window, torch.Tensor):
            tensor = window
        else:
            raise TypeError("window must be a torch.Tensor or numpy.ndarray")
        if tuple(tensor.shape) != (1, 45, 138):
            raise ValueError("window must have shape (1, 45, 138)")
        if tensor.dtype != torch.float32:
            raise TypeError("window tensor must have dtype torch.float32")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("window must contain only finite values")
        return tensor.to(self.device, non_blocking=False)


__all__ = ["InferenceEngine", "InferenceResult"]
