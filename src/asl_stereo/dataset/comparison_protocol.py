"""Frozen signer membership and selective loading for a one-shot comparison.

Only provenance is read globally. Test feature/label arrays are not loaded by
this module until the caller explicitly asks for the final test partition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .grouped_windows import normalize_video_id, read_signer_metadata


def _strings(values) -> np.ndarray:
    return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values])


@dataclass(frozen=True)
class FrozenPartition:
    name: str
    indices: tuple[int, ...]
    video_ids: tuple[str, ...]
    signer_ids: tuple[str, ...]


def freeze_existing_partitions(
    dataset_path: Path, membership_report: Path, metadata_csv: Path,
) -> tuple[dict[str, FrozenPartition], str]:
    """Use exact recorded train/val/test IDs, never regenerate a split."""
    recorded = json.loads(membership_report.read_text(encoding="utf-8"))["split_summary"]
    metadata = read_signer_metadata(metadata_csv)
    signer_map = dict(zip(metadata.video_id, metadata.signer_id, strict=True))
    with h5py.File(dataset_path, "r") as artifact:
        videos = np.asarray([normalize_video_id(v) for v in _strings(artifact["video_ids"][:])])
        stored_signers = _strings(artifact["signer_ids"][:])
        if artifact["features"].shape != (len(videos), 45, 138):
            raise ValueError("expected (N, 45, 138) features")
    if any(video not in signer_map for video in videos):
        raise ValueError("metadata does not cover every dataset video")
    signers = np.asarray([signer_map[video] for video in videos])
    if np.any((stored_signers != "") & (stored_signers != signers)):
        raise ValueError("HDF5/CSV signer disagreement")
    partitions = {}
    for name in ("train", "val", "test"):
        expected_signers = set(recorded[name]["signer_ids"])
        expected_videos = set(recorded[name]["video_ids"])
        indices = np.flatnonzero(np.isin(signers, sorted(expected_signers)))
        if set(videos[indices]) != expected_videos or set(signers[indices]) != expected_signers:
            raise ValueError(f"{name} provenance differs from the recorded split; refusing to resplit")
        partitions[name] = FrozenPartition(
            name, tuple(indices.tolist()), tuple(sorted(expected_videos)), tuple(sorted(expected_signers)),
        )
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(partitions[first].signer_ids) & set(partitions[second].signer_ids):
            raise ValueError("signer leakage in recorded split")
        if set(partitions[first].video_ids) & set(partitions[second].video_ids):
            raise ValueError("video leakage in recorded split")
    union = sorted(index for part in partitions.values() for index in part.indices)
    if union != list(range(len(videos))):
        raise ValueError("recorded partitions do not cover the artifact exactly once")
    manifest = {name: {"video_ids": part.video_ids, "signer_ids": part.signer_ids,
                       "indices": part.indices} for name, part in partitions.items()}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return partitions, digest


def load_partition(dataset_path: Path, partition: FrozenPartition):
    """Read only the selected rows; augmentation belongs to the train loader."""
    indices = np.asarray(partition.indices, dtype=np.int64)
    with h5py.File(dataset_path, "r") as artifact:
        features = np.asarray(artifact["features"][indices], dtype=np.float32)
        labels = np.asarray(artifact["labels"][indices], dtype=np.int64)
        videos = _strings(artifact["video_ids"][indices])
    if not len(labels) or features.shape != (len(labels), 45, 138) or not np.isfinite(features).all():
        raise ValueError("partition must contain finite float32 (N, 45, 138) features")
    return features, labels, videos


__all__ = ["FrozenPartition", "freeze_existing_partitions", "load_partition"]
