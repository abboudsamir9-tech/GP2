"""Large-format, debounced translation ticker."""

from __future__ import annotations

import time
from collections.abc import Callable

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import QTextEdit, QVBoxLayout, QWidget


class TranslationTicker(QWidget):
    """Accumulate accepted glosses while suppressing immediate duplicates."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        silence_interval_s: float = 1.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(parent)
        if silence_interval_s < 0:
            raise ValueError("silence_interval_s must be non-negative")
        self.silence_interval_s = silence_interval_s
        self._clock = clock
        self._last_gloss: str | None = None
        self._last_emission_time: float | None = None
        self._glosses: list[str] = []

        self.text_display = QTextEdit(self)
        self.text_display.setObjectName("translationText")
        self.text_display.setReadOnly(True)
        self.text_display.setAcceptRichText(False)
        self.text_display.setFont(QFont("Segoe UI", 26, QFont.DemiBold))
        self.text_display.setPlaceholderText("Recognized signs will appear here…")
        self.text_display.setStyleSheet(
            "QTextEdit { background: #0E1726; color: #FFFFFF; "
            "border: 2px solid #FFC107; border-radius: 6px; padding: 12px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.text_display)

    @property
    def glosses(self) -> tuple[str, ...]:
        return tuple(self._glosses)

    @property
    def text(self) -> str:
        return self.text_display.toPlainText()

    @pyqtSlot(str)
    def add_gloss(self, gloss: str) -> bool:
        normalized = " ".join(gloss.strip().split())
        if not normalized:
            return False
        now = self._clock()
        is_immediate_duplicate = (
            normalized == self._last_gloss
            and self._last_emission_time is not None
            and now - self._last_emission_time < self.silence_interval_s
        )
        if is_immediate_duplicate:
            return False

        if self._glosses:
            self.text_display.insertPlainText(" ")
        self.text_display.insertPlainText(normalized)
        self._glosses.append(normalized)
        self._last_gloss = normalized
        self._last_emission_time = now
        cursor = self.text_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text_display.setTextCursor(cursor)
        self.text_display.ensureCursorVisible()
        self.text_display.update()
        return True

    @pyqtSlot()
    def clear(self) -> None:
        self._glosses.clear()
        self._last_gloss = None
        self._last_emission_time = None
        self.text_display.clear()


__all__ = ["TranslationTicker"]
