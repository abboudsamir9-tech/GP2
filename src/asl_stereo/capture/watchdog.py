"""Pure rolling drop-rate accounting for synchronized capture."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable
from enum import Enum


class DropReason(str, Enum):
    HARDWARE_CAPTURE = "hardware_capture"
    SYNC_REJECT = "sync_reject"
    WORKER_BACKPRESSURE = "worker_backpressure"


class SyncWatchdog:
    """Track drop rates without owning or resetting global frame identity."""

    def __init__(
        self,
        *,
        window_size: int = 150,
        drop_rate_threshold: float = 0.05,
        purge_callback: Callable[[], None] | None = None,
    ) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not 0.0 <= drop_rate_threshold <= 1.0:
            raise ValueError("drop_rate_threshold must be in [0, 1]")
        self.window_size = window_size
        self.drop_rate_threshold = drop_rate_threshold
        self._events: deque[DropReason | None] = deque(maxlen=window_size)
        self._drop_totals: Counter[DropReason] = Counter()
        self._pair_index = -1
        self._purge_callback = purge_callback
        self.purge_count = 0

    def record_frame(self, drop_reason: DropReason | None = None) -> None:
        if drop_reason is not None and not isinstance(drop_reason, DropReason):
            raise TypeError("drop_reason must be a DropReason or None")
        self._events.append(drop_reason)
        if drop_reason is not None:
            self._drop_totals[drop_reason] += 1

    def next_pair_index(self) -> int:
        self._pair_index += 1
        return self._pair_index

    @property
    def pair_index(self) -> int:
        return self._pair_index

    @property
    def drop_rate(self) -> float:
        if not self._events:
            return 0.0
        drops = sum(event is not None for event in self._events)
        return drops / len(self._events)

    @property
    def drop_totals(self) -> dict[DropReason, int]:
        return dict(self._drop_totals)

    def should_purge(self) -> bool:
        return (
            len(self._events) == self.window_size
            and self.drop_rate > self.drop_rate_threshold
        )

    def purge(self) -> int:
        """Flush synchronization state while preserving the global pair index."""
        if self._purge_callback is not None:
            self._purge_callback()
        self._events.clear()
        self.purge_count += 1
        return self._pair_index


__all__ = ["DropReason", "SyncWatchdog"]
