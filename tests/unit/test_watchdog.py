from asl_stereo.capture import DropReason, SyncWatchdog


def test_watchdog_requires_a_full_window_and_rate_strictly_above_threshold() -> None:
    exact_threshold = SyncWatchdog(window_size=20, drop_rate_threshold=0.05)
    exact_threshold.record_frame(DropReason.SYNC_REJECT)
    for _ in range(18):
        exact_threshold.record_frame()
    assert exact_threshold.should_purge() is False

    exact_threshold.record_frame()
    assert exact_threshold.drop_rate == 0.05
    assert exact_threshold.should_purge() is False

    above_threshold = SyncWatchdog(window_size=20, drop_rate_threshold=0.05)
    above_threshold.record_frame(DropReason.SYNC_REJECT)
    above_threshold.record_frame(DropReason.WORKER_BACKPRESSURE)
    for _ in range(18):
        above_threshold.record_frame()
    assert above_threshold.drop_rate == 0.10
    assert above_threshold.should_purge() is True


def test_watchdog_accounts_for_drop_reasons_separately() -> None:
    watchdog = SyncWatchdog(window_size=3)
    watchdog.record_frame(DropReason.HARDWARE_CAPTURE)
    watchdog.record_frame(DropReason.SYNC_REJECT)
    watchdog.record_frame(DropReason.WORKER_BACKPRESSURE)
    assert watchdog.drop_totals == {
        DropReason.HARDWARE_CAPTURE: 1,
        DropReason.SYNC_REJECT: 1,
        DropReason.WORKER_BACKPRESSURE: 1,
    }


def test_purge_preserves_global_pair_index() -> None:
    watchdog = SyncWatchdog(window_size=2)
    assert [watchdog.next_pair_index() for _ in range(3)] == [0, 1, 2]
    watchdog.record_frame(DropReason.HARDWARE_CAPTURE)
    watchdog.record_frame(DropReason.HARDWARE_CAPTURE)
    assert watchdog.should_purge()
    assert watchdog.purge() == 2
    assert watchdog.next_pair_index() == 3
