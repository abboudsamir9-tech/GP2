"""Windows-safe checkpoint path and replacement behavior."""

from __future__ import annotations

import os

import pytest
import torch

from asl_stereo.models import load_checkpoint, save_training_checkpoint


def _save(model: torch.nn.Module, path: str) -> None:
    save_training_checkpoint(
        model,
        path,
        num_classes=2,
        window_size=45,
        feature_dim=138,
        epoch=10,
        best_val_acc=0.8,
    )


def test_checkpoint_sanitizes_quoted_path_and_replaces_atomically(tmp_path) -> None:
    model = torch.nn.Linear(138, 2)
    target = tmp_path / "folder with spaces" / "best_model.pth"
    raw_path = f' \t"{target}\r\n"\x00 '

    _save(model, raw_path)
    first = load_checkpoint(target)
    assert first["epoch"] == 10
    assert not list(target.parent.glob("*.tmp"))

    with torch.no_grad():
        model.weight.add_(1.0)
    _save(model, str(target))
    second = load_checkpoint(target)
    torch.testing.assert_close(second["model_state_dict"]["weight"], model.weight)
    assert not torch.equal(
        first["model_state_dict"]["weight"],
        second["model_state_dict"]["weight"],
    )
    assert not list(target.parent.glob("*.tmp"))


def test_failed_serialization_preserves_existing_checkpoint(tmp_path, monkeypatch) -> None:
    model = torch.nn.Linear(138, 2)
    target = tmp_path / "best_model.pth"
    _save(model, str(target))
    original = target.read_bytes()

    def fail_after_partial_write(payload, stream):
        stream.write(b"partial")
        raise OSError("simulated serialization failure")

    monkeypatch.setattr("asl_stereo.models.checkpoint.torch.save", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated serialization failure"):
        _save(model, str(target))

    assert target.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_empty_checkpoint_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty after sanitization"):
        _save(torch.nn.Linear(138, 2), " \x00\r\n\t ")


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing retry")
def test_replace_retries_transient_windows_sharing_violation(
    tmp_path, monkeypatch
) -> None:
    target = tmp_path / "best_model.pth"
    original_replace = os.replace
    attempts = 0

    def delayed_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(13, "sharing violation", None, 32)
        original_replace(source, destination)

    monkeypatch.setattr("asl_stereo.models.checkpoint.os.replace", delayed_replace)
    monkeypatch.setattr("asl_stereo.models.checkpoint.time.sleep", lambda _: None)

    _save(torch.nn.Linear(138, 2), str(target))

    assert attempts == 3
    assert load_checkpoint(target)["epoch"] == 10
