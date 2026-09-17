"""Desktop launcher for the live ASL stereo dashboard."""

from __future__ import annotations

import signal
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from ui import MainWindow


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return application.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
