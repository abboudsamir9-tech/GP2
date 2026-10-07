"""Shared hand-side/FIFO transforms and a one-shot BiLSTM training protocol."""

import json

import numpy as np
import pytest
import torch
from torch import nn

from asl_stereo.models import InferenceEngine, SignSequenceClassifier, save_training_checkpoint
from asl_stereo.models.feature_alignment import (
    align_feature_window, canonicalize_dominant_hand, fifo_neutral_context,
    hand_coordinate_variances, repeated_tail_padding_lengths,
)
from asl_stereo.preprocessing import SlidingWindowBuffer, TemporalBuffer
from scripts.optimize_model import export_torchscript, load_fp32_classifier, quantize_dynamic_model
from scripts.train_classifier import SequenceDataset


def moving_left_window(length=45):
    joints = torch.zeros(1, length, 46, 3)
    joints[:, :, :21, 0] = torch.linspace(0.1, 0.4, length)[None, :, None]
    joints[:, :, :21, 1] = -0.4
    joints[:, :, 42] = torch.tensor([0.5, 0, 0])
    joints[:, :, 43] = torch.tensor([-0.5, 0, 0])
    joints[:, :, 44] = torch.tensor([0.7, 0.3, 0.1])
    joints[:, :, 45] = torch.tensor([-0.7, 0.3, 0.1])
    return joints.reshape(1, length, 138)


def test_moving_left_only_is_mirrored_with_anatomical_pose_swaps():
    source = moving_left_window()
    original = source.clone()
    output = canonicalize_dominant_hand(source)
    assert hand_coordinate_variances(source)[0, 0] > 0
    assert hand_coordinate_variances(source)[0, 1] == 0
    assert torch.count_nonzero(output[:, :, :63]) == 0
    torch.testing.assert_close(output[:, :, 63:126:3], -source[:, :, :63:3])
    torch.testing.assert_close(output[:, :, 64:126:3], source[:, :, 1:63:3])
    torch.testing.assert_close(output[:, :, 126:], source[:, :, 126:])
    assert torch.equal(source, original)
    assert torch.equal(output, canonicalize_dominant_hand(output))


def test_visible_static_second_hand_is_not_guessed_to_be_dormant():
    source = moving_left_window()
    source[:, :, 63:126] = 0.2
    assert torch.equal(canonicalize_dominant_hand(source), source)


def test_stationary_or_empty_clip_is_not_guessed_to_be_padding_or_motion():
    source = moving_left_window()[:, :1].expand(-1, 45, -1).contiguous()
    assert torch.equal(canonicalize_dominant_hand(source), source)
    assert repeated_tail_padding_lengths(source).item() == 0
    assert torch.equal(align_feature_window(torch.zeros(1, 45, 138)), torch.zeros(1, 45, 138))


def test_repeat_tail_becomes_neutral_prefix_without_resampling_real_motion():
    observed = moving_left_window(20)
    padded = torch.cat((observed, observed[:, -1:].repeat(1, 25, 1)), dim=1)
    original = padded.clone()
    output = fifo_neutral_context(padded)
    assert repeated_tail_padding_lengths(padded).item() == 25
    assert output.shape == (1, 45, 138)
    assert torch.count_nonzero(output[:, :25, :126]) == 0
    torch.testing.assert_close(output[:, 25:], observed)
    assert torch.equal(padded, original)
    assert torch.equal(output, fifo_neutral_context(output))
    assert torch.equal(align_feature_window(padded), align_feature_window(align_feature_window(padded)))


@pytest.mark.parametrize("buffer_type", [TemporalBuffer, SlidingWindowBuffer])
def test_dataset_and_fifo_emit_identical_aligned_windows(buffer_type):
    source = moving_left_window().numpy()
    original = source.copy()
    dataset = SequenceDataset(source, np.array([0]), align_features=True)
    expected, _ = dataset[0]
    buffer = buffer_type(align_features=True)
    for index, frame in enumerate(source[0]):
        emitted = buffer.append(frame, timestamp_ns=index + 1)
    actual = emitted if isinstance(emitted, torch.Tensor) else torch.from_numpy(emitted.values.copy())
    torch.testing.assert_close(actual[0], expected)
    np.testing.assert_array_equal(source, original)


def test_primary_backbone_and_strict_external_shape():
    model = SignSequenceClassifier(5)
    assert isinstance(model.temporal_backbone, nn.LSTM)
    assert model.temporal_backbone.bidirectional
    assert model.temporal_backbone.dropout == 0.35
    assert model.feature_projection[0].in_features == 286
    with pytest.raises(ValueError, match="batch, 45, 138"):
        model(torch.zeros(1, 44, 138))


def test_checkpoint_and_export_preserve_alignment_for_left_only_inputs(tmp_path):
    torch.set_num_threads(1)
    source = SignSequenceClassifier(5, projection_features=16, hidden_size=16).eval()
    path = save_training_checkpoint(source, tmp_path / "best.pth", num_classes=5,
                                    window_size=45, feature_dim=138, epoch=1, best_val_acc=0.8)
    eager, metadata = load_fp32_classifier(path)
    assert eager.feature_alignment_enabled
    assert metadata["feature_alignment"] == "dominant_right_neutral_prefix_v1"
    destination = tmp_path / "model.pt"
    vocabulary = {i: str(i) for i in range(5)}
    quantized = quantize_dynamic_model(eager)
    export_torchscript(quantized, destination, moving_left_window(), class_map=vocabulary)
    engine = InferenceEngine(SignSequenceClassifier(5), vocabulary, checkpoint_path=destination)
    assert engine.feature_alignment_enabled
    with torch.no_grad():
        original_probs = quantized(moving_left_window())[1]
        scripted_probs = engine.model(moving_left_window())[1]
        already_aligned = engine.model(align_feature_window(moving_left_window()))[1]
    torch.testing.assert_close(original_probs, scripted_probs)
    torch.testing.assert_close(scripted_probs, already_aligned)


def test_legacy_checkpoint_does_not_silently_enable_new_alignment(tmp_path):
    model = SignSequenceClassifier(5, projection_features=16, hidden_size=16).eval()
    path = tmp_path / "legacy.pth"
    torch.save({"model_state_dict": model.state_dict(), "num_classes": 5, "feature_dim": 138,
                "window_size": 45, "model_type": "bilstm_attention", "epoch": 1, "best_val_acc": 0.8}, path)
    loaded, _ = load_fp32_classifier(path)
    assert loaded.feature_alignment_enabled is False


def test_unstructured_legacy_weights_disable_unversioned_alignment(tmp_path):
    from asl_stereo.models.checkpoint import load_model_weights
    model = SignSequenceClassifier(5, projection_features=16, hidden_size=16)
    path = tmp_path / "state.pth"
    torch.save(model.state_dict(), path)
    load_model_weights(model, path)
    assert model.feature_alignment_enabled is False


def test_training_loads_test_once_after_selection_and_export(tmp_path, monkeypatch):
    import scripts.aligned_bilstm_training as run
    import scripts.train_classifier as train
    from tests.unit.test_model_comparison import _comparison_fixture

    data, membership, metadata, vocabulary = _comparison_fixture(tmp_path)
    original_hash = run.sha256(data)
    monkeypatch.setattr(train, "PROJECT_ROOT", tmp_path)
    constructor = SignSequenceClassifier
    monkeypatch.setattr(run, "SignSequenceClassifier", lambda count, **kwargs:
                        constructor(count, projection_features=16, hidden_size=16,
                                    num_layers=2, dropout=0.35, align_features=True))
    original_load = run.load_partition
    reads = []
    exported = []

    def tracked_load(path, part):
        if part.name == "test":
            report = json.loads(membership.read_text())
            assert report["status"] == "single_test_evaluation_started"
            assert exported == [True]
            assert report["protocol"]["test_evaluation_count"] == 0
        reads.append(part.name)
        return original_load(path, part)

    def synthetic_export(path, class_map, destination):
        model, _ = load_fp32_classifier(path)
        torch.jit.save(torch.jit.script(model), str(destination))
        report_path = tmp_path / "reports/quantization_benchmark.json"
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(json.dumps({"latency_gate_passed": True}))
        exported.append(True)

    monkeypatch.setattr(run, "load_partition", tracked_load)
    monkeypatch.setattr(run, "optimize_saved_checkpoint", synthetic_export)
    run.run_aligned_bilstm(data, vocabulary, tmp_path / "weights/best.pth", epochs=1,
                           min_epochs=1, patience=1, metadata_csv=metadata, report_out=membership,
                           experiment_dir=tmp_path / "run")
    assert reads == ["train", "val", "test"]
    assert run.sha256(data) == original_hash
    report = json.loads(membership.read_text())
    assert report["protocol"]["test_evaluation_count"] == 1
    assert report["audit"]["dataset_unchanged"]
    assert report["training_config"]["model_type"] == "bilstm_attention"
    with pytest.raises(FileExistsError, match="repeated test evaluation"):
        run.run_aligned_bilstm(data, vocabulary, tmp_path / "weights/best.pth", metadata_csv=metadata,
                               report_out=membership)
