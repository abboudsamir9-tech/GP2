"""Desktop launcher for the live ASL stereo dashboard."""

from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication

from ui import MainWindow
from ui.dashboard import SHUTDOWN_TIMEOUT_S


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    application = QApplication(sys.argv)
    window = MainWindow()
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda signum, frame: window.close())
    # Periodically return to Python so Ctrl+C is serviced while Qt is idle.
    signal_timer = QTimer()
    signal_timer.timeout.connect(lambda: None)
    signal_timer.start(200)
    application.aboutToQuit.connect(window.stop_pipeline)
    try:
        window.show()
        QTimer.singleShot(0, window.start_pipeline)
        return application.exec_()
    finally:
        try:
            window.close()  # Cancel pending restarts, including explicit app.quit().
            window.stop_pipeline()
            worker = window.worker
            if worker is not None:
                deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
                reported = False
                # Explicit QApplication.quit() can bypass a window close event.
                # Never destroy a still-running QThread; joins remain bounded.
                while not worker.wait(50):
                    application.processEvents()
                    if time.monotonic() >= deadline and not reported:
                        logging.getLogger(__name__).warning(
                            "Waiting for native worker cleanup before exit",
                        )
                        reported = True
        finally:
            signal_timer.stop()
            signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    raise SystemExit(main())
