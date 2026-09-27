from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np

from scripts.preprocess_dataset import (
    extract_landmark_trajectory,
    parse_video_filename,
    prepare_feature_windows,
    recover_tracked_segment_windows,
    slice_feature_windows,
    write_dataset,
)
import scripts.preprocess_dataset as preprocess_module
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
    trajectory[:, 42] = (0.2, 0.0, 0.0)
    trajectory[:, 43] = (0.8, 0.0, 0.0)

    windows, rejected = prepare_feature_windows(trajectory)

    assert rejected == 0
    assert windows.shape == (1, 45, 138)
    assert windows.dtype == np.float32
    np.testing.assert_array_equal(windows[:, :, :63], 0.0)
    assert np.isfinite(windows).all()


def test_zero_padded_nondominant_hand_stays_zero_after_normalization():
    trajectory = _valid_landmark_sequence()
    trajectory[:, 21:42] = 0.0
    trajectory[:, 42] = (0.2, 0.0, 0.0)
    trajectory[:, 43] = (0.8, 0.0, 0.0)

    windows, rejected = prepare_feature_windows(trajectory)

    assert rejected == 0
    assert windows.shape == (1, 45, 138)
    np.testing.assert_array_equal(windows[:, :, 63:126], 0.0)


def test_idle_clip_edges_are_trimmed_before_windowing():
    trajectory = _valid_landmark_sequence(70)
    trajectory[:10, :42] = 0.0
    trajectory[60:, :42] = 0.0

    windows, rejected = prepare_feature_windows(trajectory)

    assert rejected == 0
    assert windows.shape == (1, 45, 138)


def test_short_active_clip_is_repeat_padded():
    trajectory = _valid_landmark_sequence(20)
    trajectory[:, 21:42] = 0.0

    windows, rejected = prepare_feature_windows(trajectory)

    assert rejected == 0
    assert windows.shape == (1, 45, 138)
    np.testing.assert_array_equal(windows[0, -1], windows[0, 19])


def test_zero_encoded_active_hand_dropout_is_interpolated_or_rejected():
    short = _valid_landmark_sequence()
    short[:, 21:42] = 0.0
    short[10:14, :21] = 0.0
    accepted, short_rejected = prepare_feature_windows(short)
    assert short_rejected == 0
    assert accepted.shape == (1, 45, 138)
    assert np.isfinite(accepted).all()

    long = _valid_landmark_sequence()
    long[:, 21:42] = 0.0
    long[10:15, :21] = 0.0
    rejected_windows, long_rejected = prepare_feature_windows(long)
    assert long_rejected == 1
    assert rejected_windows.shape == (0, 45, 138)


def test_long_gap_rejects_full_windows_but_recovers_clean_subclip():
    trajectory = _valid_landmark_sequence(63)
    trajectory[:, 21:42] = 0.0
    trajectory[36:54, :21] = np.nan

    full_windows, rejected = prepare_feature_windows(trajectory)
    recovered = recover_tracked_segment_windows(trajectory)

    assert full_windows.shape == (0, 45, 138)
    assert rejected == 3
    assert recovered.shape == (1, 45, 138)
    assert np.isfinite(recovered).all()
    np.testing.assert_array_equal(recovered[:, :, 63:126], 0.0)


def test_sparse_false_second_hand_detections_do_not_poison_one_handed_clip():
    trajectory = _valid_landmark_sequence()
    trajectory[:, 21:42] = np.nan
    for index in (4, 12, 24, 39):
        trajectory[index, 21:42] = (0.7, 0.3, 0.1)

    windows, rejected = prepare_feature_windows(trajectory)

    assert rejected == 0
    assert windows.shape == (1, 45, 138)
    np.testing.assert_array_equal(windows[:, :, 63:126], 0.0)

    shorter = _valid_landmark_sequence(31)
    shorter[:, 21:42] = np.nan
    for index in (2, 7, 15, 23, 29):
        shorter[index, 21:42] = (0.7, 0.3, 0.1)
    short_windows, short_rejected = prepare_feature_windows(shorter)
    assert short_rejected == 0
    assert short_windows.shape == (1, 45, 138)
    np.testing.assert_array_equal(short_windows[:, :, 63:126], 0.0)

    brief_flip = _valid_landmark_sequence(12)
    brief_flip[:, :21] = np.nan
    brief_flip[:3, :21] = (0.1, 0.2, 0.3)
    brief_flip[4, :21] = (0.1, 0.2, 0.3)
    brief_flip[5:11, 21:42] = (0.4, 0.5, 0.6)
    flip_windows, flip_rejected = prepare_feature_windows(brief_flip)
    assert flip_rejected == 0
    assert flip_windows.shape == (1, 45, 138)
    np.testing.assert_array_equal(flip_windows[:, :, :63], 0.0)


def test_video_extraction_trims_idle_leading_and_trailing_frames(monkeypatch):
    trajectory = _valid_landmark_sequence(7)
    trajectory[:2, :42] = 0.0
    trajectory[5:, :42] = 0.0
    state = {"index": 0, "released": False}

    class FakeCapture:
        def isOpened(self):
            return True

        def read(self):
            if state["index"] == len(trajectory):
                return False, None
            state["index"] += 1
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            state["released"] = True

    class FakeExtractor:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def process(self, _frame, *, frame_index):
            coordinates = trajectory[frame_index]
            hand_present = np.array(
                [float(np.any(coordinates[:21] != 0)), 0.0], dtype=np.float32
            )
            return SimpleNamespace(coordinates=coordinates, hand_present=hand_present)

    monkeypatch.setattr(preprocess_module.cv2, "VideoCapture", lambda _: FakeCapture())
    monkeypatch.setattr(preprocess_module, "PoseHandsExtractor", FakeExtractor)

    result = extract_landmark_trajectory("mock.mp4")

    assert result.shape == (3, 46, 3)
    assert state["released"]


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
