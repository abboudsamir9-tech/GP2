"""Training-only augmentation preserves the landmark and split contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from asl_stereo.dataset import LandmarkAugmentor
from scripts import train_classifier


def _normalized_window() -> torch.Tensor:
    joints = torch.zeros((45, 46, 3), dtype=torch.float32)
    joints[:, :21, 0] = torch.arange(45, dtype=torch.float32)[:, None] / 44
    joints[:, :21, 1] = 0.25
    joints[:, 42, 0] = -0.5
    joints[:, 43, 0] = 0.5
    joints[:, 44, :2] = torch.tensor((-0.6, 0.3))
    joints[:, 45, :2] = torch.tensor((0.6, 0.3))
    return joints.reshape(45, 138)


def test_augmentation_preserves_shape_origin_neutral_hand_and_source() -> None:
    source = _normalized_window()
    original = source.clone()

    torch.manual_seed(7)
    augmented = LandmarkAugmentor()(source)

    assert augmented.shape == (45, 138)
    assert augmented.dtype == torch.float32
    assert torch.isfinite(augmented).all()
    assert torch.equal(source, original)
    assert torch.count_nonzero(augmented[:, 63:126]) == 0
    joints = augmented.reshape(45, 46, 3)
    assert torch.allclose((joints[:, 42] + joints[:, 43]) / 2, torch.zeros(45, 3), atol=1e-6)
    assert not torch.equal(augmented[:, :63], source[:, :63])


@pytest.mark.parametrize("speed", [0.85, 1.0, 1.15])
def test_time_warp_resamples_linearly_without_other_perturbations(speed) -> None:
    source = _normalized_window()
    augmented = LandmarkAugmentor(
        noise_std=0, scale_range=(1, 1), speed_range=(speed, speed),
        max_rotation_degrees=0,
    )(source)

    length = round(45 / speed)
    offset = (length - 45) // 2 if length >= 45 else -((45 - length) // 2)
    # With align_corners, the unit ramp's value at index i is i/(length-1).
    assert augmented[10, 0].item() == pytest.approx((10 + offset) / (length - 1), abs=1e-6)
    assert augmented.shape == (45, 138)
    assert torch.count_nonzero(augmented[:, 63:126]) == 0
    assert torch.isfinite(augmented).all()
    if speed > 1:
        assert torch.equal(augmented[0], augmented[1])  # no zero-pad motion
        assert torch.equal(augmented[-1], augmented[-2])
    elif speed < 1:
        assert augmented[0, 0] > source[0, 0]
        assert augmented[-1, 0] < source[-1, 0]


def test_default_temporal_augmentation_range_is_fifteen_percent() -> None:
    assert LandmarkAugmentor().speed_range == (0.85, 1.15)


def test_only_training_dataset_applies_augmentation() -> None:
    source = _normalized_window().numpy()[None]
    labels = np.array([2], dtype=np.int64)
    training = train_classifier.SequenceDataset(source, labels, augment=True)
    evaluation = train_classifier.SequenceDataset(source, labels)

    torch.manual_seed(11)
    train_window, train_label = training[0]
    eval_window, eval_label = evaluation[0]

    assert train_label.item() == eval_label.item() == 2
    assert not torch.equal(train_window, eval_window)
    assert torch.equal(eval_window, torch.from_numpy(source[0]))


def test_metadata_discovery_finds_local_csv(monkeypatch, tmp_path) -> None:
    metadata = tmp_path / "data" / "asl_citizen_metadata.csv"
    metadata.parent.mkdir()
    metadata.write_text("video_id,signer_id\nv1,s1\n", encoding="utf-8")
    monkeypatch.setattr(train_classifier, "PROJECT_ROOT", tmp_path)

    assert train_classifier.discover_metadata_csv() == metadata


def test_metadata_discovery_prefers_provided_data_csv(monkeypatch, tmp_path) -> None:
    metadata = tmp_path / "data" / "data.csv"
    metadata.parent.mkdir()
    metadata.write_text("Participant ID,Video file,Gloss\nP1,v1-BYE.mp4,BYE\n", encoding="utf-8")
    (metadata.parent / "other_metadata.csv").write_text("unused\n", encoding="utf-8")
    monkeypatch.setattr(train_classifier, "PROJECT_ROOT", tmp_path)

    assert train_classifier.discover_metadata_csv() == metadata
