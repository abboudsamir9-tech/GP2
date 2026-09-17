"""Deterministic signer-independent dataset partitioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True, slots=True)
class SignerPartitions:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_signers: frozenset[Any]
    val_signers: frozenset[Any]
    test_signers: frozenset[Any]

    def __post_init__(self) -> None:
        assert self.train_signers.isdisjoint(self.val_signers)
        assert self.train_signers.isdisjoint(self.test_signers)
        assert self.val_signers.isdisjoint(self.test_signers)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        yield self.train
        yield self.val
        yield self.test

    def __getitem__(self, partition: str) -> pd.DataFrame:
        if partition not in {"train", "val", "test"}:
            raise KeyError(partition)
        return getattr(self, partition)

    @property
    def signer_ids(self) -> dict[str, list[Any]]:
        return {
            "train": sorted(self.train_signers, key=str),
            "val": sorted(self.val_signers, key=str),
            "test": sorted(self.test_signers, key=str),
        }


def partition_by_signer(
    metadata_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SignerPartitions:
    if "signer_id" not in metadata_df.columns:
        raise ValueError("metadata_df must contain a signer_id column")
    ratios = np.asarray((train_ratio, val_ratio, test_ratio), dtype=np.float64)
    if not np.isfinite(ratios).all() or np.any(ratios < 0):
        raise ValueError("partition ratios must be finite and non-negative")
    if not np.isclose(ratios.sum(), 1.0, atol=1e-9):
        raise ValueError("partition ratios must sum to one")
    if metadata_df["signer_id"].isna().any():
        raise ValueError("signer_id must not contain missing values")

    signers = list(pd.unique(metadata_df["signer_id"]))
    positive_partitions = int(np.count_nonzero(ratios))
    if len(signers) < positive_partitions:
        raise ValueError(
            "not enough unique signers to populate every non-empty partition"
        )
    generator = np.random.default_rng(seed)
    generator.shuffle(signers)
    counts = _partition_counts(len(signers), ratios)
    train_end = counts[0]
    val_end = train_end + counts[1]
    train_signers = frozenset(signers[:train_end])
    val_signers = frozenset(signers[train_end:val_end])
    test_signers = frozenset(signers[val_end:])

    assert train_signers.isdisjoint(val_signers)
    assert train_signers.isdisjoint(test_signers)
    assert val_signers.isdisjoint(test_signers)

    return SignerPartitions(
        train=metadata_df[
            metadata_df["signer_id"].map(train_signers.__contains__)
        ].copy(),
        val=metadata_df[
            metadata_df["signer_id"].map(val_signers.__contains__)
        ].copy(),
        test=metadata_df[
            metadata_df["signer_id"].map(test_signers.__contains__)
        ].copy(),
        train_signers=train_signers,
        val_signers=val_signers,
        test_signers=test_signers,
    )


def _partition_counts(signer_count: int, ratios: npt.NDArray[np.float64]) -> list[int]:
    minimums = (ratios > 0).astype(np.int64)
    remaining = signer_count - int(minimums.sum())
    allocation = ratios * remaining
    extras = np.floor(allocation).astype(np.int64)
    residual = remaining - int(extras.sum())
    fractional_order = np.argsort(-(allocation - extras), kind="stable")
    for index in fractional_order[:residual]:
        extras[index] += 1
    return (minimums + extras).tolist()
__all__ = ["SignerPartitions", "partition_by_signer"]
