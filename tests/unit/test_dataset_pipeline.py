import csv
import json

import h5py
import numpy as np
import pandas as pd

from asl_stereo.dataset import (
    BatchFeatureExtractor,
    partition_by_signer,
    write_dataset_h5,
)


def test_partition_by_signer_has_zero_leakage() -> None:
    metadata = pd.DataFrame(
        {
            "video_id": [f"video_{index}" for index in range(40)],
            "gloss_label": ["HELLO", "THANKS"] * 20,
            "signer_id": [f"signer_{index // 2}" for index in range(40)],
        }
    )

    split = partition_by_signer(metadata, seed=42)

    assert split.train_signers.isdisjoint(split.val_signers)
    assert split.train_signers.isdisjoint(split.test_signers)
    assert split.val_signers.isdisjoint(split.test_signers)
    assert split.train_signers | split.val_signers | split.test_signers == set(
        metadata["signer_id"].unique()
    )
    assert set(split.train["signer_id"]) == split.train_signers
    assert set(split.val["signer_id"]) == split.val_signers
    assert set(split.test["signer_id"]) == split.test_signers


def test_csv_audit_export_has_exact_46_joint_schema(tmp_path) -> None:
    extractor = BatchFeatureExtractor(tmp_path)
    sequence = np.arange(2 * 46 * 3, dtype=np.float32).reshape(2, 46, 3)
    sequence[0, 5, 1] = np.nan

    output = extractor.export_audit_csv("sample_video", sequence)

    with output.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert output.name == "sample_video.csv"
    assert len(rows) == 3
    assert len(rows[0]) == 139
    assert rows[0][0] == "frame_idx"
    assert rows[0][1:4] == ["joint_0_x", "joint_0_y", "joint_0_z"]
    assert rows[0][-3:] == ["joint_45_x", "joint_45_y", "joint_45_z"]
    assert len(rows[1]) == len(rows[2]) == 139
    assert rows[1][0] == "0"
    assert rows[1][1 + 5 * 3 + 1].lower() == "nan"


def test_window_segmentation_returns_n_by_45_by_138() -> None:
    sequence = np.arange(61 * 138, dtype=np.float32).reshape(61, 138)

    windows = BatchFeatureExtractor.segment_sequence(sequence)

    assert windows.shape == (3, 45, 138)
    assert windows.dtype == np.float32
    assert windows.flags.c_contiguous
    np.testing.assert_array_equal(windows[0], sequence[0:45])
    np.testing.assert_array_equal(windows[1], sequence[8:53])
    np.testing.assert_array_equal(windows[2], sequence[16:61])


def test_hdf5_artifact_contains_required_partitions_and_metadata(tmp_path) -> None:
    features = {
        partition: np.zeros((1, 45, 138), dtype=np.float32)
        for partition in ("train", "val", "test")
    }
    labels = {
        partition: np.zeros(1, dtype=np.int64)
        for partition in ("train", "val", "test")
    }
    output = write_dataset_h5(
        tmp_path / "asl_dataset.h5",
        features,
        labels,
        class_mapping={"HELLO": 0},
        signer_ids={"train": ["s1"], "val": ["s2"], "test": ["s3"]},
    )

    with h5py.File(output, "r") as artifact:
        assert artifact.attrs["window_size"] == 45
        assert artifact.attrs["feature_dim"] == 138
        assert json.loads(artifact.attrs["class_mapping"]) == {"HELLO": 0}
        assert json.loads(artifact.attrs["signer_ids"]) == {
            "train": ["s1"],
            "val": ["s2"],
            "test": ["s3"],
        }
        for partition in ("train", "val", "test"):
            assert artifact[f"{partition}/features"].shape == (1, 45, 138)
            assert artifact[f"{partition}/labels"].shape == (1,)
            assert artifact[f"{partition}/features"].compression == "gzip"
