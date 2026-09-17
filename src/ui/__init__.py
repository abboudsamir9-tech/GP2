"""Desktop dashboard public API."""

from .dashboard import MainWindow, PipelineWorker, RuntimeSettings, SettingsDialog
from .text_ticker import TranslationTicker

__all__ = [
    "MainWindow",
    "PipelineWorker",
    "RuntimeSettings",
    "SettingsDialog",
    "TranslationTicker",
]
