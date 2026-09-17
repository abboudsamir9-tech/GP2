"""Thread-safe nearest-timestamp dual-camera synchronization."""

from __future__ import annotations

from bisect import bisect_left, insort_right
from dataclasses import dataclass, field
from threading import RLock

from asl_stereo.contracts import SynchronizedFramePair, TimestampedFrame


@dataclass(order=True, slots=True)
class _QueuedFrame:
    timestamp_ns: int
    insertion_order: int
    frame: TimestampedFrame = field(compare=False)


class Synchronizer:
    """Align front and side frames using bounded, timestamp-sorted queues."""

    def __init__(
        self,
        *,
        front_camera_id: str = "front",
        side_camera_id: str = "side",
        tolerance_ns: int = 40_000_000,
        max_queue_size: int = 150,
    ) -> None:
        if not front_camera_id or not side_camera_id:
            raise ValueError("camera identifiers must be non-empty")
        if front_camera_id == side_camera_id:
            raise ValueError("front and side camera identifiers must differ")
        if isinstance(tolerance_ns, bool) or not isinstance(tolerance_ns, int):
            raise TypeError("tolerance_ns must be an integer")
        if tolerance_ns < 0:
            raise ValueError("tolerance_ns must be non-negative")
        if (
            isinstance(max_queue_size, bool)
            or not isinstance(max_queue_size, int)
            or max_queue_size <= 0
        ):
            raise ValueError("max_queue_size must be a positive integer")

        self.front_camera_id = front_camera_id
        self.side_camera_id = side_camera_id
        self.tolerance_ns = tolerance_ns
        self.max_queue_size = max_queue_size
        self._front: list[_QueuedFrame] = []
        self._side: list[_QueuedFrame] = []
        self._lock = RLock()
        self._pair_index = -1
        self._insertion_order = 0
        self._sync_rejects = 0
        self._queue_overflows = 0

    def add_frame(self, frame: TimestampedFrame) -> SynchronizedFramePair | None:
        """Insert one frame and emit its nearest valid cross-camera match."""
        with self._lock:
            own_queue, other_queue, is_front = self._route(frame.camera_id)
            queued = _QueuedFrame(
                timestamp_ns=frame.timestamp_ns,
                insertion_order=self._insertion_order,
                frame=frame,
            )
            self._insertion_order += 1
            insort_right(own_queue, queued)
            self._enforce_queue_bound(own_queue)

            match_index = self._nearest_index(other_queue, frame.timestamp_ns)
            if match_index is not None:
                other = other_queue[match_index]
                if abs(other.timestamp_ns - frame.timestamp_ns) <= self.tolerance_ns:
                    own_queue.remove(queued)
                    other_queue.pop(match_index)
                    front = frame if is_front else other.frame
                    side = other.frame if is_front else frame
                    matched_timestamp = max(front.timestamp_ns, side.timestamp_ns)
                    self._prune_older_than(self._front, matched_timestamp)
                    self._prune_older_than(self._side, matched_timestamp)
                    self._pair_index += 1
                    return SynchronizedFramePair(
                        front=front,
                        side=side,
                        delta_t_ns=side.timestamp_ns - front.timestamp_ns,
                        pair_index=self._pair_index,
                    )

            self._prune_impossible_frames()
            return None

    def add_front(self, frame: TimestampedFrame) -> SynchronizedFramePair | None:
        if frame.camera_id != self.front_camera_id:
            raise ValueError("frame does not belong to the configured front camera")
        return self.add_frame(frame)

    def add_side(self, frame: TimestampedFrame) -> SynchronizedFramePair | None:
        if frame.camera_id != self.side_camera_id:
            raise ValueError("frame does not belong to the configured side camera")
        return self.add_frame(frame)

    def purge(self) -> None:
        """Clear pending frames without resetting pair or camera indices."""
        with self._lock:
            self._front.clear()
            self._side.clear()

    @property
    def pair_index(self) -> int:
        with self._lock:
            return self._pair_index

    @property
    def queue_sizes(self) -> tuple[int, int]:
        with self._lock:
            return len(self._front), len(self._side)

    @property
    def queued_timestamps(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        with self._lock:
            return (
                tuple(item.timestamp_ns for item in self._front),
                tuple(item.timestamp_ns for item in self._side),
            )

    @property
    def sync_rejects(self) -> int:
        with self._lock:
            return self._sync_rejects

    @property
    def queue_overflows(self) -> int:
        with self._lock:
            return self._queue_overflows

    def _route(
        self, camera_id: str
    ) -> tuple[list[_QueuedFrame], list[_QueuedFrame], bool]:
        if camera_id == self.front_camera_id:
            return self._front, self._side, True
        if camera_id == self.side_camera_id:
            return self._side, self._front, False
        raise ValueError(f"unknown camera_id {camera_id!r}")

    @staticmethod
    def _nearest_index(queue: list[_QueuedFrame], timestamp_ns: int) -> int | None:
        if not queue:
            return None
        probe = _QueuedFrame(timestamp_ns, -1, queue[0].frame)
        position = bisect_left(queue, probe)
        candidates = []
        if position < len(queue):
            candidates.append(position)
        if position > 0:
            candidates.append(position - 1)
        return min(
            candidates,
            key=lambda index: (
                abs(queue[index].timestamp_ns - timestamp_ns),
                queue[index].timestamp_ns,
                queue[index].insertion_order,
            ),
        )

    def _prune_impossible_frames(self) -> None:
        """Reject only frames that no future ordered counterpart can match."""
        while self._front and self._side:
            front_timestamp = self._front[0].timestamp_ns
            side_timestamp = self._side[0].timestamp_ns
            if front_timestamp < side_timestamp - self.tolerance_ns:
                self._front.pop(0)
                self._sync_rejects += 1
            elif side_timestamp < front_timestamp - self.tolerance_ns:
                self._side.pop(0)
                self._sync_rejects += 1
            else:
                break

    def _prune_older_than(
        self, queue: list[_QueuedFrame], matched_timestamp_ns: int
    ) -> None:
        boundary = bisect_left(
            [item.timestamp_ns for item in queue], matched_timestamp_ns
        )
        if boundary:
            self._sync_rejects += boundary
            del queue[:boundary]

    def _enforce_queue_bound(self, queue: list[_QueuedFrame]) -> None:
        overflow = len(queue) - self.max_queue_size
        if overflow > 0:
            del queue[:overflow]
            self._queue_overflows += overflow
__all__ = ["Synchronizer"]
