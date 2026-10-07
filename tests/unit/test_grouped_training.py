"""Leakage and report regressions for the compact five-gloss workflow."""

from __future__ import annotations

import csv

import h5py
import numpy as np
import pytest

from asl_stereo.dataset import (
    load_grouped_windows, read_signer_metadata, split_grouped_windows,
)
from scripts.preprocess_dataset import write_dataset
from scripts.train_classifier import (
    build_evaluation_report, macro_f1_from_report, previous_test_signers,
)


def test_grouped_loader_uses_metadata_signers_and_rejects_legacy(tmp_path) -> None:
    features = np.zeros((6, 45, 138), dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    videos = np.array(["v1", "v1", "v2", "v2", "v3", "v4"])
    path = write_dataset(
        tmp_path / "grouped.h5", features, labels,
        class_map={0: "BYE", 1: "HELLO"}, source_video_count=4,
        video_ids=videos,
    )
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("video_id", "signer_id"))
        writer.writerows((video, signer) for video, signer in (
            ("v1", "s1"), ("v2", "s2"), ("v3", "s1"), ("v4", "s3"),
        ))
    fallback = load_grouped_windows(path)
    assert fallback.group_kind == "video_id"
    np.testing.assert_array_equal(fallback.groups, videos)

    signer_grouped = load_grouped_windows(path, metadata_csv=metadata)
    assert signer_grouped.group_kind == "signer_id"
    assert signer_grouped.groups.tolist() == ["s1", "s1", "s2", "s2", "s1", "s3"]

    legacy = tmp_path / "legacy.h5"
    with h5py.File(legacy, "w") as artifact:
        artifact["features"] = features
        artifact["labels"] = labels
    with pytest.raises(ValueError, match="regenerate"):
        load_grouped_windows(legacy)


def test_grouped_split_keeps_all_windows_of_each_group_together() -> None:
    labels = np.tile(np.arange(5, dtype=np.int64), 30)
    groups = np.repeat([f"s{index:02d}" for index in range(30)], 5)
    split = split_grouped_windows(labels, groups, seed=9)
    all_indices = np.concatenate((split.train, split.val, split.test))
    np.testing.assert_array_equal(np.sort(all_indices), np.arange(len(labels)))
    parts = [set(groups[indices]) for indices in (split.train, split.val, split.test)]
    assert parts[0].isdisjoint(parts[1])
    assert parts[0].isdisjoint(parts[2])
    assert parts[1].isdisjoint(parts[2])
    assert all(set(labels[indices]) == set(range(5))
               for indices in (split.train, split.val, split.test))
    repeated = split_grouped_windows(labels, groups, seed=9)
    for first, second in zip((split.train, split.val, split.test),
                             (repeated.train, repeated.val, repeated.test), strict=True):
        np.testing.assert_array_equal(first, second)


def test_asl_citizen_columns_and_filename_ids_align_with_hdf5(tmp_path) -> None:
    metadata = tmp_path / "data.csv"
    metadata.write_text(
        'Participant ID,Video file,Gloss,id\n'
        'P01,014439537236845323-THANK YOU.mp4,THANKYOU,P01\n'
        'P02,00042-NO.mp4,NO,P02\n', encoding="utf-8",
    )
    canonical = read_signer_metadata(metadata)
    assert set(canonical) == {"video_id", "signer_id", "gloss_label"}
    assert canonical["video_id"].tolist() == ["014439537236845323", "00042"]
    path = write_dataset(
        tmp_path / "dataset.h5", np.zeros((3, 45, 138), dtype=np.float32),
        np.array([0, 0, 1]), class_map={0: "THANK YOU", 1: "NO"},
        source_video_count=2,
        video_ids=np.array(["014439537236845323", "014439537236845323", "00042"]),
    )

    grouped = load_grouped_windows(
        path, metadata_csv=metadata, class_map={0: "THANK YOU", 1: "NO"},
    )

    assert grouped.group_kind == "signer_id"
    assert grouped.groups.tolist() == ["P01", "P01", "P02"]


def test_incomplete_metadata_logs_unmatched_ids_and_rejects_fallback(tmp_path, caplog) -> None:
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("video_id,user_id,sign_word\nv1,s1,BYE\n", encoding="utf-8")
    path = write_dataset(
        tmp_path / "dataset.h5", np.zeros((2, 45, 138), dtype=np.float32),
        np.array([0, 1]), class_map={0: "BYE", 1: "NO"}, source_video_count=2,
        video_ids=np.array(["v1", "unmatched_v2"]),
    )

    with pytest.raises(ValueError, match="does not cover"):
        load_grouped_windows(path, metadata_csv=metadata)

    assert "unmatched_v2" in caplog.text


def test_conflicting_signer_or_gloss_is_rejected(tmp_path) -> None:
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("video_id,signer_id\nv1,s1\nv1,s2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting signer"):
        read_signer_metadata(metadata)

    metadata.write_text("video_id,signer_id,gloss_label\nv1,s1,YES\n", encoding="utf-8")
    path = write_dataset(
        tmp_path / "dataset.h5", np.zeros((1, 45, 138), dtype=np.float32),
        np.array([0]), class_map={0: "NO", 1: "YES"}, source_video_count=1,
        video_ids=np.array(["v1"]),
    )
    with pytest.raises(ValueError, match="gloss mismatch"):
        load_grouped_windows(path, metadata_csv=metadata, class_map={0: "NO", 1: "YES"})


def test_report_scores_each_video_once_and_has_five_by_five_matrix() -> None:
    labels = np.array([0, 0, 1, 2, 3, 4])
    videos = np.array(["v0", "v0", "v1", "v2", "v3", "v4"])
    probabilities = np.eye(5, dtype=np.float32)[labels]
    probabilities[0] = [0.1, 0.9, 0, 0, 0]
    class_map = dict(enumerate(("BYE", "HELLO", "NO", "THANK YOU", "YES")))

    report = build_evaluation_report(labels, probabilities, videos, class_map)

    assert report["test_video_count"] == 5
    assert report["test_window_count"] == 6
    assert report["test_accuracy"] == 1.0
    assert report["window_accuracy"] == pytest.approx(5 / 6)
    assert np.asarray(report["confusion_matrix"]).shape == (5, 5)
    assert set(report["per_class"]) == set(class_map.values())
    assert macro_f1_from_report(report) == 1.0


def test_stratification_counts_videos_not_duplicate_windows() -> None:
    labels = np.tile(np.arange(5), 30)
    signers = np.repeat([f"s{index:02d}" for index in range(30)], 5)
    videos = np.array([f"v{index}" for index in range(len(labels))])
    repetitions = np.where(labels == 0, 10, 1)
    base = split_grouped_windows(labels, signers, video_ids=videos, min_test_videos_per_class=4)
    duplicated = split_grouped_windows(
        np.repeat(labels, repetitions), np.repeat(signers, repetitions),
        video_ids=np.repeat(videos, repetitions), min_test_videos_per_class=4,
    )

    assert set(signers[base.test]) == set(np.repeat(signers, repetitions)[duplicated.test])
    for class_id in range(5):
        assert len(set(videos[base.test][labels[base.test] == class_id])) >= 4


def test_fixed_test_signers_reject_inflated_window_support() -> None:
    labels = np.tile(np.arange(5), 20)
    signers = np.repeat([f"s{index}" for index in range(20)], 5)
    videos = np.array([f"v{index}" for index in range(len(labels))])
    with pytest.raises(ValueError, match="minimum per-class video support"):
        split_grouped_windows(
            np.repeat(labels, 10), np.repeat(signers, 10), video_ids=np.repeat(videos, 10),
            min_test_videos_per_class=3, fixed_test_groups=("s0", "s1"),
        )


def test_recorded_holdout_is_reused_using_provenance_only(tmp_path) -> None:
    import json

    labels = np.tile(np.arange(5), 20)
    signers = np.repeat([f"s{index}" for index in range(20)], 5)
    videos = np.array([f"v{index}" for index in range(len(labels))])
    holdout = {"s0", "s1", "s2", "s3"}
    report = {
        "evaluation_scope": "unseen_signers",
        "split_summary": {
            "train": {"video_ids": videos[~np.isin(signers, list(holdout))].tolist()},
            "test": {"signer_ids": sorted(holdout), "video_ids": videos[np.isin(signers, list(holdout))].tolist()},
        },
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    saved = previous_test_signers(path, labels, videos, signers, min_test_videos_per_class=4)
    split = split_grouped_windows(labels, signers, video_ids=videos,
                                 min_test_videos_per_class=4, fixed_test_groups=saved)

    assert split.test_preserved
    assert set(signers[split.test]) == holdout
