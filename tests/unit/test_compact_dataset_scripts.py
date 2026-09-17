from __future__ import annotations

import json

import h5py
import numpy as np

from scripts.preprocess_dataset import (
    parse_video_filename,
    prepare_feature_windows,
    slice_feature_windows,
    write_dataset,
)
from scripts.train_classifier import (
    SequenceDataset,
    load_feature_artifact,
    stratified_split_indices,
)


def test_filename_parser_preserves_spaces_and_splits_first_hyphen(tmp_path):
    record = parse_video_filename(tmp_path / "0636580316216-SHUT OUT.mp4")

    assert record.video_id == "0636580316216"
    assert record.gloss == "SHUT OUT"


def test_short_sequence_is_repeat_padded_to_one_window():
    features = np.arange(10 * 138, dtype=np.float32).reshape(10, 138)

    windows = slice_feature_windows(features)

    assert windows.shape == (1, 45, 138)
    np.testing.assert_array_equal(windows[0, :10], features)
    np.testing.assert_array_equal(
        windows[0, 10:], np.repeat(features[-1:], 35, axis=0)
    )


def test_long_sequence_uses_window_45_stride_8():
    features = np.zeros((61, 138), dtype=np.float32)

    windows = slice_feature_windows(features)

    assert windows.shape == (3, 45, 138)


def _valid_landmark_sequence(frame_count=45):
    sequence = np.zeros((frame_count, 46, 3), dtype=np.float32)
    sequence[:, :42, 0] = np.linspace(-0.25, 0.25, 42, dtype=np.float32)
    sequence[:, 42] = (-0.5, 0.0, 0.0)
    sequence[:, 43] = (0.5, 0.0, 0.0)
    sequence[:, 44] = (-0.5, 0.5, 0.0)
    sequence[:, 45] = (0.5, 0.5, 0.0)
    return sequence


def test_structurally_absent_hand_is_zero_after_normalization():
    trajectory = _valid_landmark_sequence()
    trajectory[:, :21] = np.nan

    windows, rejected = prepare_feature_windows(trajectory)

    assert rejected == 0
    assert windows.shape == (1, 45, 138)
    assert windows.dtype == np.float32
    np.testing.assert_array_equal(windows[:, :, :63], 0.0)
    assert np.isfinite(windows).all()


def test_active_hand_interior_gap_shorter_than_five_is_interpolated():
    trajectory = _valid_landmark_sequence()
    trajectory[10:14, :21] = np.nan

    windows, rejected = prepare_feature_windows(trajectory)

    assert rejected == 0
    assert windows.shape == (1, 45, 138)
    assert np.isfinite(windows).all()


def test_active_hand_dropout_of_five_frames_rejects_window():
    trajectory = _valid_landmark_sequence()
    trajectory[10:15, :21] = np.nan

    windows, rejected = prepare_feature_windows(trajectory)

    assert rejected == 1
    assert windows.shape == (0, 45, 138)


def test_pose_dropout_of_five_frames_rejects_window():
    trajectory = _valid_landmark_sequence()
    trajectory[10:15, 44] = np.nan

    windows, rejected = prepare_feature_windows(trajectory)

    assert rejected == 1
    assert windows.shape == (0, 45, 138)


def test_compact_hdf5_schema_round_trip(tmp_path):
    features = np.zeros((4, 45, 138), dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    output = write_dataset(
        tmp_path / "dataset.h5",
        features,
        labels,
        class_map={0: "HELLO", 1: "THANK YOU"},
        source_video_count=4,
    )

    loaded_features, loaded_labels = load_feature_artifact(output)

    assert loaded_features.shape == (4, 45, 138)
    assert loaded_features.dtype == np.float32
    np.testing.assert_array_equal(loaded_labels, labels)
    with h5py.File(output, "r") as artifact:
        assert set(artifact) == {"features", "labels"}
        assert artifact.attrs["window_size"] == 45
        assert artifact.attrs["feature_dim"] == 138
        assert json.loads(artifact.attrs["class_map"]) == {
            "0": "HELLO",
            "1": "THANK YOU",
        }


def test_stratified_split_is_disjoint_and_represents_each_class():
    labels = np.repeat(np.arange(3, dtype=np.int64), 10)

    train, validation = stratified_split_indices(labels, seed=7)

    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(range(labels.size))
    assert set(labels[train]) == {0, 1, 2}
    assert set(labels[validation]) == {0, 1, 2}
    assert validation.size == 6


def test_sequence_dataset_returns_model_ready_tensors():
    dataset = SequenceDataset(
        np.zeros((2, 45, 138), dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
    )

    features, label = dataset[1]

    assert tuple(features.shape) == (45, 138)
    assert features.dtype.is_floating_point
    assert label.item() == 1
