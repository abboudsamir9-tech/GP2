"""Desktop dashboard public API."""

from .dashboard import MainWindow, PipelineWorker, RuntimeSettings, SettingsDialog
from .text_ticker import TranslationTicker
from .video_widget import VideoWidget

__all__ = [
    "MainWindow",
    "PipelineWorker",
    "RuntimeSettings",
    "SettingsDialog",
    "TranslationTicker",
    "VideoWidget",
]
