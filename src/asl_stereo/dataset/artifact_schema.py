"""Compressed HDF5 artifact schema for signer-independent windows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from asl_stereo.contracts import FEATURE_COUNT

PARTITIONS = ("train", "val", "test")


def write_dataset_h5(
    output_path: str | Path,
    partition_features: Mapping[str, npt.ArrayLike],
    partition_labels: Mapping[str, npt.ArrayLike],
    *,
    class_mapping: Mapping[str, int],
    signer_ids: Mapping[str, Sequence[Any]],
    window_size: int = 45,
    feature_dim: int = FEATURE_COUNT,
) -> Path:
    if feature_dim != FEATURE_COUNT:
        raise ValueError("feature_dim must be 138")
    if sorted(class_mapping.values()) != list(range(len(class_mapping))):
        raise ValueError("class mapping indices must be contiguous and start at zero")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    validated: dict[str, tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]] = {}
    for partition in PARTITIONS:
        if partition not in partition_features or partition not in partition_labels:
            raise ValueError(f"missing data for partition {partition!r}")
        features = np.asarray(partition_features[partition])
        labels = np.asarray(partition_labels[partition])
        if features.shape[1:] != (window_size, feature_dim) or features.ndim != 3:
            raise ValueError(
                f"{partition} features must have shape (samples, {window_size}, {feature_dim})"
            )
        if features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError(f"{partition} features must be finite float32 values")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(f"{partition} labels must align with feature samples")
        labels = labels.astype(np.int64, copy=False)
        if labels.size and (labels.min() < 0 or labels.max() >= len(class_mapping)):
            raise ValueError(f"{partition} labels contain an unknown class index")
        validated[partition] = (features, labels)

    with h5py.File(output, "w") as artifact:
        artifact.attrs["schema_version"] = 1
        artifact.attrs["window_size"] = window_size
        artifact.attrs["feature_dim"] = feature_dim
        artifact.attrs["class_mapping"] = json.dumps(
            dict(sorted(class_mapping.items(), key=lambda item: item[1])),
            sort_keys=True,
        )
        artifact.attrs["signer_ids"] = json.dumps(
            {partition: [str(value) for value in signer_ids[partition]] for partition in PARTITIONS},
            sort_keys=True,
        )
        for partition, (features, labels) in validated.items():
            group = artifact.create_group(partition)
            group.create_dataset(
                "features",
                data=features,
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            group.create_dataset(
                "labels",
                data=labels,
                dtype=np.int64,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
    return output


__all__ = ["PARTITIONS", "write_dataset_h5"]
