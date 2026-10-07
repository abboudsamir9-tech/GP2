# Operational fixes: verification

Date: 2026-10-07. Follow-up to the code-quality audit.

## Result

**204 tests passed in 26.50 seconds, with zero failures and zero warnings.**
All 181 previous tests were retained unchanged; 23 regression cases were added.
Warnings were treated as errors:

```powershell
.\venv311\Scripts\python.exe -B -m pytest tests/ -v -W error -p no:cacheprovider
```

No model architecture or model source file was changed. The two-layer BiLSTM,
hidden dimensions, 286 internal features, and external `(batch, 45, 138)`
contract remain intact. No training, serving export, dataset extraction, class
map change, dependency installation, or physical camera operation was performed.
Pre-existing working-tree changes were preserved.

## Numerical preprocessing and FIFO ownership

- [smoothing.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/preprocessing/smoothing.py:40>):
  replace 138 independent Savitzky-Golay fits with one native multidimensional
  call using `axis=0`, float64 arithmetic, and the original `mode="interp"`.
  Channels containing unresolved NaNs remain entirely unchanged. Short sequences
  and empty channel dimensions remain unchanged.
- [interpolation.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/preprocessing/interpolation.py:33>):
  use vectorized preceding/following observation indices to fill only the same
  short interior gaps. Leading/trailing gaps, long gaps, and all-missing channels
  preserve NaNs; the default remains at most four missing frames.
- [normalization.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/preprocessing/normalization.py:50>)
  and [pipeline.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/preprocessing/pipeline.py:112>):
  normalize temporal frames in one batch with the same shoulder midpoint,
  Euclidean scale, epsilon guard, float64 arithmetic, and float32 encoding.
  The existing single-frame normalization function is unchanged.
- [temporal_buffer.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/preprocessing/temporal_buffer.py:62>):
  defensively copy each validated input before storing it in the feature FIFO,
  including input supplied through a FeatureVector. Emitted NumPy/Torch windows
  still share their owned snapshot; raw landmark ingestion already copied input.
  Window sizes, timestamps, validity decisions, neutral-hand rules, and cadence
  were not changed.

[Numerical/ownership regression tests](<C:/Users/desgin/Documents/ChatGPT/New project/tests/unit/test_preprocessing_optimization.py>)
compare against the original per-channel implementations with absolute tolerance
`1e-6`, no relative tolerance, and identical NaN handling. Coverage includes
mixed valid/missing channels, multiple gap limits, boundary gaps, missing and
zero-distance shoulders, empty/short sequences, neutral hands, and a producer
reusing mutable storage. The representative complete live-window comparison
had **maximum absolute difference 0.0**.

## Performance measurements

The actual raw-landmark live path was measured on this laptop using a synthetic
one-handed trajectory with short interior gaps. Full-window cleaning had 30
warmups and 200 timed calls. The raw FIFO was warmed with 90 frames and measured
for 200 more, with alignment enabled and stride eight:

| Path | Mean ms | p95 ms |
| --- | ---: | ---: |
| Full `process_live_window` | 2.310 | 5.284 |
| Raw FIFO, cleaning, and feature alignment | 2.789 | 5.922 |
| Emitting-window calls only | 4.755 | 8.919 |

There were 23 emissions; skipped tracking frames did not advance cadence.
Output remained `(1, 45, 138)` float32. The new regression gate checks **p95
below 15 ms** over 200 real raw-FIFO calls, including cleaning and alignment,
rather than only single-frame normalization. The measured results also meet
the requested below-10-ms preprocessing target.

These are local synthetic CPU measurements, not a guarantee of physical-camera
FPS or total gesture-to-text latency on every laptop. The installed Python is
3.11.3; the environment retains the previously audited Torch 2.4.1+cpu and PyQt5
5.15.11, rather than silently changing dependencies during this task.

## Graceful shutdown

- [CameraWorker](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/capture/camera_worker.py:383>)
  uses one total stop deadline, including capture-thread and release-helper joins.
  Normal reads stop cooperatively and release from the producer's `finally`.
  A stalled read can trigger one claimed release on a helper thread, so a blocked
  native `release()` does not make the caller's timeout unbounded. Restart is
  rejected while capture/release is still stopping. Stop requests also wake
  startup waiters; a late device handle is released instead of entering capture.
  Windows DirectShow selection and five-second startup defaults are unchanged.
- [PipelineWorker cleanup](<C:/Users/desgin/Documents/ChatGPT/New project/src/ui/dashboard.py:304>)
  retains camera ownership during initialization and failed cleanup. It requests
  cancellation, joins camera/vision children off the UI thread, and does not emit
  completed-parent status while children remain alive. Mailbox and overlay
  references are cleared after cleanup.
- [MainWindow shutdown](<C:/Users/desgin/Documents/ChatGPT/New project/src/ui/dashboard.py:903>)
  is asynchronous: stop returns without a six-second UI-thread wait. A Qt timer
  checks actual completion. Window close is deferred, Start remains disabled
  while stopping, and timeout messages are non-modal. Camera-setting restarts
  wait for the old worker; stale-worker signals cannot update the new session.
- [main.py](<C:/Users/desgin/Documents/ChatGPT/New project/main.py:22>) routes Ctrl+C
  through window close instead of immediate process termination, keeps Python
  signal handling responsive with a heartbeat, restores the original handler,
  and drains cleanup after an explicit QApplication quit. Pending restarts are
  cancelled before exit.

[Shutdown tests](<C:/Users/desgin/Documents/ChatGPT/New project/tests/unit/test_worker_shutdown.py>)
cover stalled reads, stalled release, single-release ownership, startup
cancellation, forbidden restart during cleanup, continued GUI timer events,
deferred close, non-modal timeout reporting, vision-child supervision, Ctrl+C,
and explicit application quit.

A native driver/tracker call cannot be forcibly killed safely within a Python
thread. On a genuine native hang, the implementation reports the delay and
retains a responsive window/owning worker until cleanup completes; it does not
pretend shutdown succeeded or destroy a live QThread. Physical-device disconnect
and long-session soak testing remain separate hardware verification tasks.

## Protected artifacts

Before/after SHA-256 hashes match both the start of this task and the pre-audit
values. Neither checkpoint was retrained, exported, or overwritten:

| File | Unchanged SHA-256 |
| --- | --- |
| `weights/best_model.pth` | `04CA4E71A8FFF8CF2EFCFADA27083C1769E82915997D80C75EB5194FE2B7F97C` |
| `weights/optimized_model.pt` | `E0B57A15326D9895FE45CBD2786F35E1C80C86DCDA69072376DD41FC22258BE4` |
| `artifacts/dataset.h5` | `44CB944C1F9AF8D2A94C05B47C22957B9F0BF7E8989850BF480F078CD964BA00` |
| `data/data.csv` | `D99BBB2F34CD7828CE167DC320E872B9031C8210337A619CC4937CA3203B2B86` |
