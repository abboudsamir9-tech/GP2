"""Lazy HDF5-backed PyTorch sequence dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from asl_stereo.contracts import FEATURE_COUNT


class HDF5SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Stream one feature window at a time from a partitioned HDF5 artifact.

    Each process opens its own lazy read handle, making the dataset safe for
    ``DataLoader`` worker processes without loading all windows into RAM.
    """

    def __init__(
        self,
        h5_path: str | Path,
        partition: str,
        *,
        augment: bool = False,
        noise_std: float = 0.005,
        frame_dropout_probability: float = 0.05,
    ) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.h5_path}")
        if partition not in {"train", "val", "test"}:
            raise ValueError("partition must be 'train', 'val', or 'test'")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if not 0.0 <= frame_dropout_probability <= 1.0:
            raise ValueError("frame_dropout_probability must be in [0, 1]")
        self.partition = partition
        self.augment = augment
        self.noise_std = noise_std
        self.frame_dropout_probability = frame_dropout_probability
        self._handle: h5py.File | None = None
        self._handle_pid: int | None = None

        with h5py.File(self.h5_path, "r") as artifact:
            feature_path = f"{partition}/features"
            label_path = f"{partition}/labels"
            if feature_path not in artifact or label_path not in artifact:
                raise ValueError(f"HDF5 artifact is missing partition {partition!r}")
            features = artifact[feature_path]
            labels = artifact[label_path]
            if len(features.shape) != 3 or features.shape[1:] != (45, FEATURE_COUNT):
                raise ValueError(
                    f"{feature_path} must have shape (samples, 45, {FEATURE_COUNT})"
                )
            if np.dtype(features.dtype) != np.dtype(np.float32):
                raise TypeError(f"{feature_path} must have dtype float32")
            if len(labels.shape) != 1 or labels.shape[0] != features.shape[0]:
                raise ValueError("labels must be one-dimensional and align with features")
            if not np.issubdtype(labels.dtype, np.integer):
                raise TypeError(f"{label_path} must use an integer dtype")
            self._length = int(features.shape[0])
            self.window_size = int(features.shape[1])
            self.feature_dim = int(features.shape[2])
            raw_mapping = artifact.attrs.get("class_mapping", "{}")
            if isinstance(raw_mapping, bytes):
                raw_mapping = raw_mapping.decode("utf-8")
            parsed = json.loads(str(raw_mapping))
            self.class_mapping = {str(gloss): int(index) for gloss, index in parsed.items()}

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        normalized_index = int(index)
        if normalized_index < 0:
            normalized_index += self._length
        if normalized_index < 0 or normalized_index >= self._length:
            raise IndexError(index)

        artifact = self._get_handle()
        features = np.asarray(
            artifact[f"{self.partition}/features"][normalized_index],
            dtype=np.float32,
        )
        label = int(artifact[f"{self.partition}/labels"][normalized_index])
        tensor = torch.from_numpy(np.ascontiguousarray(features))
        if self.augment:
            tensor = self._augment(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)

    def _augment(self, features: torch.Tensor) -> torch.Tensor:
        output = features.clone()
        if self.noise_std > 0.0:
            output.add_(torch.randn_like(output), alpha=self.noise_std)
        if self.frame_dropout_probability > 0.0:
            dropped = torch.rand(output.shape[0]) < self.frame_dropout_probability
            output[dropped] = 0.0
        return output

    def _get_handle(self) -> h5py.File:
        process_id = os.getpid()
        if self._handle is None or self._handle_pid != process_id:
            self.close()
            self._handle = h5py.File(self.h5_path, "r")
            self._handle_pid = process_id
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._handle_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def __enter__(self) -> "HDF5SequenceDataset":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["HDF5SequenceDataset"]
