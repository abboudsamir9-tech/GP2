"""Provenance-aware loading and leakage-free train/validation/test splits."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from asl_stereo.contracts import FEATURE_COUNT

LOGGER = logging.getLogger(__name__)
_COLUMN_ALIASES = {
    "video_id": ("video_id", "video_file", "video_filename", "video_name", "filename", "video"),
    "gloss_label": ("gloss_label", "gloss", "sign_word", "sign_words", "word", "sign", "label"),
    "signer_id": ("signer_id", "user_id", "participant_id", "subject_id", "signer", "participant", "user", "id"),
}


def normalize_video_id(value: str) -> str:
    """Align bare IDs and ASL Citizen filenames without losing leading zeros."""
    filename = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    if filename.lower().endswith(".mp4"):
        return filename[:-4].split("-", 1)[0].strip()
    return filename


def _normalize_gloss(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def read_signer_metadata(path: str | Path) -> pd.DataFrame:
    """Standardize CSV aliases to video_id, gloss_label and signer_id.

    String loading protects long numeric IDs and their leading zeros. Gloss is
    optional for older video-to-signer CSVs; supplied glosses are validated by
    the window loader against the model vocabulary.
    """
    source = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    columns = {
        re.sub(r"[^a-z0-9]+", "_", column.strip().lower()).strip("_"): column
        for column in source.columns
    }
    normalized = pd.DataFrame(index=source.index)
    for target, aliases in _COLUMN_ALIASES.items():
        column = next((columns[alias] for alias in aliases if alias in columns), None)
        if column is None:
            if target != "gloss_label":
                raise ValueError(f"metadata CSV requires a {target} column; found {list(source.columns)}")
            normalized[target] = ""
        else:
            normalized[target] = source[column].str.strip()
    normalized["video_id"] = normalized["video_id"].map(normalize_video_id)
    if normalized.empty or (normalized[["video_id", "signer_id"]] == "").any().any():
        raise ValueError("metadata CSV contains no records or an empty video_id/signer_id")
    for video_id, rows in normalized.groupby("video_id", sort=False):
        if rows["signer_id"].nunique() != 1:
            raise ValueError(f"conflicting signer IDs for video {video_id}")
        glosses = {_normalize_gloss(gloss) for gloss in rows["gloss_label"] if gloss}
        if len(glosses) > 1:
            raise ValueError(f"conflicting gloss labels for video {video_id}")
    return normalized.drop_duplicates("video_id").reset_index(drop=True)


def load_signer_metadata(path: str | Path | None) -> dict[str, str]:
    """Read the ASL Citizen video-to-signer mapping, if supplied."""
    if path is None:
        return {}
    metadata = read_signer_metadata(path)
    return dict(zip(metadata["video_id"], metadata["signer_id"], strict=True))


def _decode_ids(values: npt.ArrayLike) -> npt.NDArray[np.str_]:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=str)


@dataclass(frozen=True, slots=True)
class GroupedWindows:
    features: npt.NDArray[np.float32]
    labels: npt.NDArray[np.int64]
    video_ids: npt.NDArray[np.str_]
    signer_ids: npt.NDArray[np.str_]
    groups: npt.NDArray[np.str_]
    group_kind: str


def load_grouped_windows(
    path: str | Path, *, metadata_csv: str | Path | None = None,
    class_map: Mapping[int, str] | None = None,
) -> GroupedWindows:
    """Refuse legacy artifacts with no per-window provenance."""
    with h5py.File(path, "r") as artifact:
        required = {"features", "labels", "video_ids"}
        if not required.issubset(artifact):
            missing = sorted(required - set(artifact))
            raise ValueError(
                f"HDF5 artifact lacks {missing}; regenerate it with preprocess_dataset.py"
            )
        features = np.asarray(artifact["features"], dtype=np.float32)
        labels = np.asarray(artifact["labels"], dtype=np.int64)
        video_ids = np.asarray([
            normalize_video_id(value) for value in _decode_ids(artifact["video_ids"][:])
        ], dtype=str)
        signer_ids = (
            _decode_ids(artifact["signer_ids"][:])
            if "signer_ids" in artifact else np.full(len(labels), "", dtype=str)
        )
    count = len(labels)
    if features.shape != (count, 45, FEATURE_COUNT) or not np.isfinite(features).all():
        raise ValueError("features must be finite float32 windows of shape (N, 45, 138)")
    if labels.shape != (count,) or count == 0:
        raise ValueError("labels must contain one class ID per window")
    if video_ids.shape != (count,) or signer_ids.shape != (count,) or np.any(video_ids == ""):
        raise ValueError("video/signer IDs must align with windows; video IDs cannot be empty")

    if metadata_csv is not None:
        metadata = read_signer_metadata(metadata_csv)
        windows = pd.DataFrame({"video_id": video_ids, "existing_signer_id": signer_ids})
        aligned = windows.merge(metadata, on="video_id", how="left", sort=False,
                                validate="many_to_one", indicator=True)
        unmatched = sorted(aligned.loc[aligned["_merge"] != "both", "video_id"].unique())
        if unmatched:
            LOGGER.warning("Unmatched video IDs in signer metadata: %s", ", ".join(unmatched))
            raise ValueError(f"signer metadata does not cover {len(unmatched)} video IDs: {unmatched}")
        LOGGER.info("Signer metadata alignment: %d videos / %d windows matched; 0 unmatched",
                    len(np.unique(video_ids)), count)
        conflicts = (aligned["existing_signer_id"] != "") & (
            aligned["existing_signer_id"] != aligned["signer_id"]
        )
        if conflicts.any():
            raise ValueError(f"signer mismatch for video {aligned.loc[conflicts, 'video_id'].iloc[0]}")
        if class_map is not None:
            for index, row in aligned.iterrows():
                if row["gloss_label"]:
                    if int(labels[index]) not in class_map:
                        raise ValueError("dataset labels are incompatible with the class map")
                    if _normalize_gloss(row["gloss_label"]) != _normalize_gloss(class_map[int(labels[index])]):
                        raise ValueError(f"metadata gloss mismatch for video {row['video_id']}")
        signer_ids = aligned["signer_id"].to_numpy(dtype=str)
    # All windows from a video must share both signer and target label.
    for video_id in np.unique(video_ids):
        rows = video_ids == video_id
        if len(np.unique(signer_ids[rows])) != 1 or len(np.unique(labels[rows])) != 1:
            raise ValueError(f"inconsistent signer or label among windows for video {video_id}")
    use_signers = bool(np.all(signer_ids != ""))
    groups = signer_ids if use_signers else video_ids
    return GroupedWindows(
        features, labels, video_ids, signer_ids, groups,
        "signer_id" if use_signers else "video_id",
    )


@dataclass(frozen=True, slots=True)
class GroupedSplit:
    train: npt.NDArray[np.int64]
    val: npt.NDArray[np.int64]
    test: npt.NDArray[np.int64]
    seed: int = 42
    test_preserved: bool = False


def split_grouped_windows(
    labels: npt.ArrayLike,
    groups: npt.ArrayLike,
    *,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    seed: int = 42,
    video_ids: npt.ArrayLike | None = None,
    min_test_videos_per_class: int = 3,
    fixed_test_groups: Sequence[str] | None = None,
) -> GroupedSplit:
    """Stratify unique videos while keeping every signer and window together.

    Seeds are retried using label counts only. Supplying a valid existing test
    group list retains that holdout for comparisons between training runs.
    """
    y = np.asarray(labels, dtype=np.int64)
    group_values = _decode_ids(groups)
    if (y.ndim != 1 or group_values.shape != y.shape or len(y) == 0
            or np.any(y < 0) or np.any(group_values == "")):
        raise ValueError("labels and groups must be aligned one-dimensional arrays")
    if len(np.unique(group_values)) < 3:
        raise ValueError("at least three independent groups are needed")
    if test_ratio <= 0 or val_ratio <= 0 or test_ratio + val_ratio >= 1:
        raise ValueError("train, validation, and test ratios must be positive")
    if min_test_videos_per_class < 1:
        raise ValueError("minimum test videos per class must be positive")
    if video_ids is None:
        representative_indices = np.arange(len(y))
    else:
        videos = _decode_ids(video_ids)
        if videos.shape != y.shape or np.any(videos == ""):
            raise ValueError("video_ids must align with windows and cannot be empty")
        _, first_indices = np.unique(videos, return_index=True)
        representative_indices = np.sort(first_indices)
        for video_id in np.unique(videos):
            rows = videos == video_id
            if len(np.unique(y[rows])) != 1 or len(np.unique(group_values[rows])) != 1:
                raise ValueError(f"inconsistent label or signer for video {video_id}")
    video_labels = y[representative_indices]
    video_groups = group_values[representative_indices]
    classes = set(np.unique(y).tolist())
    for class_id in classes:
        if len(np.unique(video_groups[video_labels == class_id])) < 3:
            raise ValueError(f"class {class_id} needs at least three independent signer/video groups")
        if np.count_nonzero(video_labels == class_id) < min_test_videos_per_class + 2:
            raise ValueError(f"class {class_id} cannot supply {min_test_videos_per_class} test videos plus train/validation examples")
    overall = np.bincount(video_labels) / len(video_labels)
    group_count = len(np.unique(video_groups))
    fixed_test: npt.NDArray[np.int64] | None = None
    if fixed_test_groups is not None:
        requested = set(fixed_test_groups)
        if not requested or not requested.issubset(set(video_groups)):
            raise ValueError("fixed test signers must be nonempty and present in the dataset")
        fixed_test = np.flatnonzero(np.isin(video_groups, list(requested)))
        support = np.bincount(video_labels[fixed_test], minlength=len(overall))
        if any(support[class_id] < min_test_videos_per_class for class_id in classes):
            raise ValueError("fixed test signers do not meet minimum per-class video support")
    best: GroupedSplit | None = None
    best_score = float("inf")
    outer_folds = min(max(2, round(1.0 / test_ratio)), group_count - 1,
                      min(np.count_nonzero(video_labels == c) for c in classes))
    for attempt in range(40):
        candidate_seed = seed + attempt
        if fixed_test is not None:
            remaining = np.setdiff1d(np.arange(len(video_labels)), fixed_test)
            outer_candidates = [(remaining, fixed_test)]
        else:
            outer_candidates = StratifiedGroupKFold(
                n_splits=outer_folds, shuffle=True, random_state=candidate_seed,
            ).split(video_labels, video_labels, groups=video_groups)
        for remaining, test in outer_candidates:
            support = np.bincount(video_labels[test], minlength=len(overall))
            if any(support[c] < min_test_videos_per_class for c in classes):
                continue
            remaining_group_count = len(np.unique(video_groups[remaining]))
            minimum_class_count = min(np.count_nonzero(video_labels[remaining] == c) for c in classes)
            inner_folds = min(max(2, round((1.0 - test_ratio) / val_ratio)),
                              remaining_group_count, minimum_class_count)
            if inner_folds < 2:
                continue
            inner_candidates = StratifiedGroupKFold(
                n_splits=inner_folds, shuffle=True, random_state=candidate_seed,
            ).split(video_labels[remaining], video_labels[remaining],
                    groups=video_groups[remaining])
            for train_local, val_local in inner_candidates:
                train, val = remaining[train_local], remaining[val_local]
                if any(set(video_labels[part]) != classes for part in (train, val, test)):
                    continue
                group_sets = [set(video_groups[part]) for part in (train, val, test)]
                if any(group_sets[a] & group_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
                    raise RuntimeError("group overlap detected in train/validation/test split")
                score = sum(
                    float(np.square(np.bincount(video_labels[part], minlength=len(overall))
                                    / len(part) - overall).sum())
                    for part in (train, val, test)
                )
                score += 0.1 * sum(
                    (len(part) / len(video_labels) - ratio) ** 2
                    for part, ratio in zip((train, val, test),
                                           (1 - test_ratio - val_ratio, val_ratio, test_ratio), strict=True)
                )
                if score < best_score:
                    best_score = score
                    expanded = [np.flatnonzero(np.isin(group_values, list(groups_in_part)))
                                for groups_in_part in group_sets]
                    best = GroupedSplit(*expanded, seed=candidate_seed,
                                        test_preserved=fixed_test is not None)
        # Retry the seed only when its folds cannot meet class support. This
        # keeps the split deterministic without searching for favorable signers.
        if best is not None:
            break
    if best is not None:
        return best
    raise ValueError(
        f"no stratified signer split supplies {min_test_videos_per_class} test videos "
        "per class and full train/validation coverage; collect more independent signers"
    )


__all__ = [
    "GroupedSplit", "GroupedWindows", "load_grouped_windows",
    "load_signer_metadata", "normalize_video_id", "read_signer_metadata",
    "split_grouped_windows",
]
