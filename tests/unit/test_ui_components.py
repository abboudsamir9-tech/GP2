import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import numpy as np
from PyQt5.QtWidgets import QApplication

from ui import PipelineWorker, RuntimeSettings, SettingsDialog, TranslationTicker
from ui.qt_messages import map_telemetry


@pytest.fixture(scope="module")
def qt_application():
    application = QApplication.instance() or QApplication([])
    yield application


def test_translation_ticker_suppresses_immediate_duplicate(qt_application) -> None:
    ticker = TranslationTicker(clock=lambda: 10.0, silence_interval_s=1.5)

    assert ticker.add_gloss("HELLO") is True
    assert ticker.add_gloss("HELLO") is False

    assert ticker.glosses == ("HELLO",)
    assert ticker.text == "HELLO"


def test_translation_ticker_clear_resets_text(qt_application) -> None:
    ticker = TranslationTicker()
    ticker.add_gloss("HELLO")
    ticker.add_gloss("WORLD")

    ticker.clear()

    assert ticker.text == ""
    assert ticker.glosses == ()


def test_settings_threshold_slider_updates_configuration(qt_application) -> None:
    settings = RuntimeSettings(confidence_threshold=0.65)
    dialog = SettingsDialog(settings)

    dialog.confidence_slider.setValue(78)
    qt_application.processEvents()

    assert settings.confidence_threshold == pytest.approx(0.78)
    assert dialog.confidence_value.text() == "0.78"


def test_frame_mailbox_coalesces_pending_updates(qt_application) -> None:
    worker = PipelineWorker(RuntimeSettings())
    delivered = []
    worker.frames_ready.connect(lambda front, side: delivered.append((front, side)))
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    latest = np.ones((2, 2, 3), dtype=np.uint8)

    worker._publish_frames(first, first)
    worker._publish_frames(latest, latest)
    qt_application.processEvents()

    assert len(delivered) == 1
    consumed = worker.consume_latest_frames()
    assert consumed is not None
    np.testing.assert_array_equal(consumed[0], latest)


def test_single_camera_telemetry_is_explicit() -> None:
    display = map_telemetry(
        {
            "single_camera": True,
            "fusion_status": "[SINGLE-CAMERA FALLBACK]",
            "fps": 20.0,
        }
    )

    assert display.sync_text == "[SINGLE-CAMERA FALLBACK]"
    assert display.fusion_text == "[SINGLE-CAMERA FALLBACK]"
