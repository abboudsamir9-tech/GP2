"""Failure-path regression coverage without camera hardware or MediaPipe."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from asl_stereo.dataset import BatchFeatureExtractor


@pytest.mark.parametrize("failure_stage", ["construct", "reset"])
def test_batch_video_capture_is_released_when_extractor_setup_fails(
    tmp_path: Path, monkeypatch, failure_stage: str,
) -> None:
    video_path = tmp_path / "synthetic.mp4"
    video_path.touch()  # Existence only; no image/video bytes are persisted.
    capture = Mock()
    capture.isOpened.return_value = True
    monkeypatch.setattr("asl_stereo.dataset.batch_extractor.cv2.VideoCapture", Mock(return_value=capture))
    failure = RuntimeError("synthetic extractor setup failure")
    tracker = Mock()
    factory = Mock(return_value=tracker)
    if failure_stage == "construct":
        factory.side_effect = failure
    else:
        tracker.reset_tracking.side_effect = failure

    with BatchFeatureExtractor(tmp_path / "audits", holistic_factory=factory) as extractor:
        with pytest.raises(RuntimeError, match="synthetic extractor setup failure") as caught:
            extractor.extract_video("synthetic", video_path)
    assert caught.value is failure
    capture.release.assert_called_once_with()
    capture.read.assert_not_called()
    if failure_stage == "reset":
        tracker.close.assert_called_once_with()
