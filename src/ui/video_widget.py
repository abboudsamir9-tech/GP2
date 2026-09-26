"""Coalesced, copy-free BGR frame painting on the Qt UI thread."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QRectF, Qt, pyqtSlot
from PyQt5.QtGui import QColor, QImage, QPainter
from PyQt5.QtWidgets import QWidget


class VideoWidget(QWidget):
    """Keep the NumPy frame alive while QImage wraps its memory for painting."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self._frame: np.ndarray | None = None
        self._image: QImage | None = None
        self.setObjectName("videoViewport")
        self.setMinimumSize(420, 320)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    @pyqtSlot(np.ndarray)
    def set_frame(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            return
        if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            return
        # The capture/display workers produce C-contiguous BGR. Only unusual
        # sliced arrays need a copy; never convert BGR to RGB or to a pixmap.
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        height, width = frame.shape[:2]
        self._frame = frame
        self._image = QImage(
            frame.data, width, height, frame.strides[0], QImage.Format_BGR888
        )
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#080808"))
        image = self._image
        if image is None:
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(self.rect(), Qt.AlignCenter, self.title)
            return
        scale = min(self.width() / image.width(), self.height() / image.height())
        target_width = image.width() * scale
        target_height = image.height() * scale
        destination = QRectF(
            (self.width() - target_width) / 2,
            (self.height() - target_height) / 2,
            target_width,
            target_height,
        )
        painter.drawImage(destination, image)


__all__ = ["VideoWidget"]
