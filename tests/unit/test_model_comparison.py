"""Leakage barriers, 138/286 contracts and validation-only selection."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from asl_stereo.dataset.comparison_protocol import freeze_existing_partitions
from asl_stereo.landmarks import PoseHandsExtractor
from asl_stereo.models.checkpoint import save_training_checkpoint
from asl_stereo.models.comparison import MODEL_SPECS, build_comparison_model
from scripts.audit_feature_alignment import alignment_checks
from scripts.compare_architectures import compare_architectures, select_validation_winner


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_comparison_models_keep_contract_and_backpropagate(name):
    torch.set_num_threads(1)
    model = build_comparison_model(name, 5)
    model.train()
    logits, probabilities = model(torch.randn(2, 45, 138))
    assert model.feature_projection[0].in_features == 286
    assert logits.shape == (2, 5)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))
    torch.nn.CrossEntropyLoss()(logits, torch.tensor([0, 1])).backward()
    assert torch.isfinite(model.feature_projection[0].weight.grad).all()
    if name == "B":
        assert model.temporal_backbone.num_layers == 1
        assert model.temporal_backbone.hidden_size == 128
        assert model.temporal_dropout.p == 0.3


def test_preprocessing_audit_checks_neutral_hand_and_smoothing_alignment():
    audit = alignment_checks()
    assert audit["default_config"]["savgol_window_length"] == 7
    assert audit["default_config"]["savgol_polyorder"] == 2
    assert audit["offline_live_45_frame_max_abs_difference"] == 0
    assert audit["temporal_buffer_live_max_abs_difference"] == 0
    assert audit["inactive_left_hand_zero_in_offline"]
    assert audit["inactive_left_hand_zero_in_live"]
    assert audit["three_frame_interior_gap_interpolated"]
    # Expose, do not silently claim, full-clip/rolling-window edge equivalence.
    assert audit["boundary_context_difference"]["max_abs_difference"] > 0


def _comparison_fixture(tmp_path):
    data = tmp_path / "dataset.h5"
    videos = [f"{side}{label}" for side in ("train", "val", "test") for label in range(5)]
    signers = [f"{side}_signer" for side in ("train", "val", "test") for _ in range(5)]
    features = np.random.default_rng(42).normal(size=(15, 45, 138)).astype(np.float32)
    labels = np.tile(np.arange(5), 3).astype(np.int64)
    with h5py.File(data, "w") as h5:
        h5["features"] = features
        h5["labels"] = labels
        h5.create_dataset("video_ids", data=videos, dtype=h5py.string_dtype())
        h5.create_dataset("signer_ids", data=signers, dtype=h5py.string_dtype())
    membership = tmp_path / "membership.json"
    membership.write_text(json.dumps({"split_summary": {
        side: {"signer_ids": [f"{side}_signer"], "video_ids": [f"{side}{label}" for label in range(5)]}
        for side in ("train", "val", "test")
    }}), encoding="utf-8")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("video_id,signer_id\n" + "\n".join(f"{v},{s}" for v, s in zip(videos, signers)), encoding="utf-8")
    vocabulary = tmp_path / "class_map.json"
    vocabulary.write_text(json.dumps({str(i): word for i, word in enumerate(("BYE", "HELLO", "NO", "THANK YOU", "YES"))}))
    return data, membership, metadata, vocabulary


def test_frozen_split_rejects_changed_membership(tmp_path):
    data, membership, metadata, _ = _comparison_fixture(tmp_path)
    parts, digest = freeze_existing_partitions(data, membership, metadata)
    assert len(digest) == 64
    assert parts["test"].indices == tuple(range(10, 15))
    report = json.loads(membership.read_text())
    report["split_summary"]["test"]["video_ids"].pop()
    membership.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="refusing to resplit"):
        freeze_existing_partitions(data, membership, metadata)


def test_selection_uses_only_macro_f1_then_capacity():
    candidates = [{"name": name, "validation": {"macro_f1": score}, "parameters": size}
                  for name, score, size in (("A", 0.7, 1000), ("B", 0.8, 500), ("C", 0.8, 100))]
    assert select_validation_winner(candidates)["name"] == "C"
    candidates[0]["test"] = {"accuracy": 1.0}
    with pytest.raises(ValueError, match="validation-only"):
        select_validation_winner(candidates)


def test_test_features_loaded_once_only_after_selection_is_saved(tmp_path, monkeypatch):
    import scripts.compare_architectures as module
    inputs = _comparison_fixture(tmp_path)
    report_path = tmp_path / "comparison.json"
    actual_load = module.load_partition
    reads = []

    def checked_load(path, part):
        if part.name == "test":
            on_disk = json.loads(report_path.read_text())
            assert on_disk["selected_model"] == "B"
            assert on_disk["status"] == "selection_locked_test_evaluation_started"
            assert len(on_disk["models"]) == 3
        reads.append(part.name)
        return actual_load(path, part)

    def fake_train(name, train, val, class_map, output_dir, **kwargs):
        model = build_comparison_model(name, 5)
        checkpoint = output_dir / f"model_{name}.pth"
        save_training_checkpoint(model, checkpoint, num_classes=5, window_size=45, feature_dim=138,
                                 epoch=1, best_val_acc=0.8, model_type=MODEL_SPECS[name]["architecture"])
        return {"name": name, "parameters": sum(p.numel() for p in model.parameters()),
                "validation": {"macro_f1": {"A": 0.6, "B": 0.8, "C": 0.7}[name], "accuracy": 0.8},
                "checkpoint": str(checkpoint), "checkpoint_sha256": module.sha256(checkpoint), "best_epoch": 1}

    monkeypatch.setattr(module, "load_partition", checked_load)
    monkeypatch.setattr(module, "train_candidate", fake_train)
    report = compare_architectures(*inputs, tmp_path / "models", report_path, epochs=1, min_epochs=1)
    assert reads == ["train", "val", "test"]
    assert report["protocol"]["test_evaluation_count"] == 1
    assert report["test"]["model"] == "B"
    with pytest.raises(FileExistsError, match="refusing another test"):
        compare_architectures(*inputs, tmp_path / "models", report_path, epochs=1, min_epochs=1)
    assert reads == ["train", "val", "test"]


@pytest.mark.parametrize("mirrored,expected", [(False, 1), (True, 0)])
def test_one_hand_assignment_changes_label_not_coordinates(mirrored, expected):
    hand = SimpleNamespace(landmark=[SimpleNamespace(x=0.6, y=0.3, z=0.01) for _ in range(21)])
    pose = SimpleNamespace(landmark=[SimpleNamespace(x=0.5, y=0.5, z=0.1) for _ in range(33)])

    class Tracker:
        def __init__(self, result):
            self.result = result
        def process(self, frame):
            return self.result
        def close(self):
            pass

    with PoseHandsExtractor(
        input_is_mirrored=mirrored,
        pose_factory=lambda **_: Tracker(SimpleNamespace(pose_landmarks=pose)),
        hands_factory=lambda **_: Tracker(SimpleNamespace(multi_hand_landmarks=[hand],
            multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(label="Left", score=0.99)])])),
    ) as extractor:
        frame = extractor.process(np.zeros((8, 8, 3), dtype=np.uint8))
    np.testing.assert_allclose(frame.coordinates[expected * 21], (0.6, 0.3, 0.01))
    np.testing.assert_array_equal(frame.coordinates[(1 - expected) * 21:(2 - expected) * 21], 0)
