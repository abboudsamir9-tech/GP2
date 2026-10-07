"""Regression tests for motion features and the five-gloss training schedule."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from asl_stereo.models import (
    FocalLoss, SignSequenceClassifier, hand_aperture_features, hand_closure_features,
    hand_elevation_features, position_velocity_features, temporal_hand_features,
)
from scripts import train_classifier


def test_position_velocity_features_preserve_external_138_dimension() -> None:
    positions = torch.zeros((1, 45, 138), dtype=torch.float32)
    positions[0, :, 0] = torch.arange(45)

    encoded = position_velocity_features(positions)

    assert encoded.shape == (1, 45, 276)
    torch.testing.assert_close(encoded[..., :138], positions)
    assert encoded[0, 0, 138].item() == 0.0
    torch.testing.assert_close(encoded[0, 1:, 138], torch.ones(44))
    assert torch.count_nonzero(encoded[..., 139:]) == 0
    model = SignSequenceClassifier(num_classes=5, projection_features=16, hidden_size=16)
    assert model.feature_projection[0].in_features == 286
    assert model(positions)[0].shape == (1, 5)


def test_wrist_elevation_uses_both_shoulders_and_preserves_138_input() -> None:
    joints = torch.zeros((1, 45, 46, 3), dtype=torch.float32)
    joints[:, :, 42, 1] = 0.1
    joints[:, :, 43, 1] = 0.3
    joints[:, :, 0, 1] = -0.8  # elevated signing hand
    joints[:, :, 21, 1] = 0.5  # lower signing hand
    inputs = joints.reshape(1, 45, 138)
    original = inputs.clone()

    height = hand_elevation_features(inputs)
    encoded = temporal_hand_features(inputs, include_wrist_height=True)

    assert height.shape == (1, 45, 2)
    torch.testing.assert_close(height[0, 0], torch.tensor([-1.0, 0.3]))
    assert encoded.shape == (1, 45, 288)
    torch.testing.assert_close(encoded[..., -2:], height)
    torch.testing.assert_close(inputs, original)
    translated = hand_elevation_features((joints + 4).reshape(1, 45, 138))
    torch.testing.assert_close(height, translated, atol=1e-6, rtol=1e-6)


def test_wrist_elevation_keeps_structurally_absent_hand_neutral() -> None:
    joints = torch.zeros((1, 45, 46, 3), dtype=torch.float32)
    joints[:, :, 42:44, 1] = 2.0
    joints[:, :, 0, 1] = 1.0

    height = hand_elevation_features(joints.reshape(1, 45, 138))

    torch.testing.assert_close(height[0, 0], torch.tensor([-1., 0.]))
    assert torch.count_nonzero(hand_elevation_features(torch.zeros(1, 45, 138))) == 0


def test_wrist_elevation_propagates_gradients_to_wrist_and_shoulders() -> None:
    joints = torch.ones((1, 45, 46, 3), dtype=torch.float32, requires_grad=True)

    hand_elevation_features(joints.reshape(1, 45, 138)).sum().backward()

    torch.testing.assert_close(joints.grad[0, 0, [0, 21], 1], torch.ones(2))
    torch.testing.assert_close(joints.grad[0, 0, [42, 43], 1], -torch.ones(2))


def test_training_defaults_use_sixty_epochs_and_twenty_epoch_patience() -> None:
    args = train_classifier.build_parser().parse_args([])

    assert args.epochs == 60
    assert args.patience == 20
    assert args.min_epochs == 30


def test_hand_aperture_ratios_are_scale_invariant_and_absent_hand_is_zero() -> None:
    joints = torch.zeros((1, 45, 46, 3), dtype=torch.float32)
    joints[:, :, 9, 1] = 1.0  # left middle MCP sets palm scale
    joints[:, :, 8, 1] = 2.0  # index tip
    joints[:, :, 12, 1] = 3.0  # middle tip
    joints[:, :, 4, 0] = 1.0  # thumb tip

    ratios = hand_aperture_features(joints.reshape(1, 45, 138))
    doubled = hand_aperture_features((joints * 2).reshape(1, 45, 138))

    assert ratios.shape == (1, 45, 6)
    torch.testing.assert_close(ratios[0, 0], torch.tensor([2., 3., 1., 0., 0., 0.]))
    torch.testing.assert_close(ratios, doubled)


def test_right_hand_uses_its_own_wrist_and_middle_mcp() -> None:
    joints = torch.zeros((1, 45, 46, 3), dtype=torch.float32)
    joints[:, :, 21, 0] = 10.0  # right wrist is not the left-hand origin
    joints[:, :, 30, 0] = 12.0  # right middle MCP: palm length 2
    joints[:, :, 29, 0] = 14.0
    joints[:, :, 33, 0] = 16.0
    joints[:, :, 25, 0] = 10.0
    joints[:, :, 25, 1] = 2.0

    ratios = hand_aperture_features(joints.reshape(1, 45, 138))

    torch.testing.assert_close(
        ratios[0, 0], torch.tensor([0., 0., 0., 2., 3., 1.]),
    )


def test_focal_loss_focuses_hard_examples_and_backpropagates() -> None:
    criterion = FocalLoss(torch.tensor([1.0, 1.0]), gamma=2.0)
    easy = torch.tensor([[5.0, -5.0]], requires_grad=True)
    hard = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    label = torch.tensor([0])

    easy_loss = criterion(easy, label)
    hard_loss = criterion(hard, label)
    hard_loss.backward()

    assert easy_loss.item() < hard_loss.item()
    assert hard.grad is not None and torch.isfinite(hard.grad).all()


def test_thumb_finger_closure_tracks_a_snap_and_both_hand_offsets() -> None:
    joints = torch.zeros((1, 45, 46, 3), dtype=torch.float32)
    opening = torch.linspace(1.0, 0.0, 45)
    joints[0, :, 8, 0] = opening
    joints[0, :, 12, 0] = opening * 2
    joints[0, :, 25, 1] = 3
    joints[0, :, 29, 1] = 4
    joints[0, :, 33, 1] = 5

    closure = hand_closure_features(joints.reshape(1, 45, 138))

    assert closure.shape == (1, 45, 4)
    torch.testing.assert_close(closure[0, 0], torch.tensor([1., 2., 1., 2.]))
    torch.testing.assert_close(closure[0, -1], torch.tensor([0., 0., 1., 2.]))
    translated = hand_closure_features((joints + 4).reshape(1, 45, 138))
    torch.testing.assert_close(closure, translated, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(hand_closure_features(torch.zeros(1, 45, 138))) == 0


def test_class_weights_are_normalized_inverse_frequency() -> None:
    labels = np.array([0, 0, 1, 1, 1, 2, 3, 3, 4, 4], dtype=np.int64)
    class_map = dict(enumerate(("BYE", "HELLO", "NO", "THANK YOU", "YES")))

    weights = train_classifier.compute_class_weights(labels, class_map)

    expected = 10 / (5 * np.array([2, 3, 1, 2, 2]))
    expected /= expected.mean()
    expected = np.clip(expected, 0.8, 1.6)
    np.testing.assert_allclose(weights.numpy(), expected)


def test_macro_f1_checkpoint_selection_prevents_validation_class_collapse() -> None:
    full_coverage = {"per_class": {"NO": {"recall": .5, "f1": .5}, "YES": {"recall": .8, "f1": .8}}}
    collapsed = {"per_class": {"NO": {"recall": 0., "f1": 0.}, "YES": {"recall": 1., "f1": 1.}}}

    assert train_classifier.checkpoint_selection_rank(full_coverage, 2.0) > train_classifier.checkpoint_selection_rank(collapsed, .1)
    improved = {"per_class": {"NO": {"recall": .8, "f1": .8}, "YES": {"recall": .9, "f1": .9}}}
    assert train_classifier.checkpoint_selection_rank(improved, 3.0) > train_classifier.checkpoint_selection_rank(full_coverage, 1.0)


def test_warmup_then_cosine_schedule_and_minimum_epoch_gate() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = train_classifier.build_warmup_cosine_scheduler(
        optimizer, max_epochs=40, warmup_epochs=5,
    )

    rates = []
    for _ in range(40):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert rates[:5] == pytest.approx([0.0002, 0.0004, 0.0006, 0.0008, 0.001])
    assert rates[10] < rates[5]
    assert rates[-1] < rates[10]


def test_training_auto_optimization_uses_selected_checkpoint(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(train_classifier.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    checkpoint = tmp_path / "best_model.pth"
    class_map = tmp_path / "class_map.json"
    optimized = tmp_path / "optimized_model.pt"

    train_classifier.optimize_saved_checkpoint(checkpoint, class_map, optimized)

    command, kwargs = calls[0]
    assert command[command.index("--input-weights") + 1] == str(checkpoint)
    assert command[command.index("--output-model") + 1] == str(optimized)
    assert command[command.index("--class-map") + 1] == str(class_map)
    assert kwargs["check"] is True
    assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
    assert kwargs["env"]["MKL_NUM_THREADS"] == "1"
