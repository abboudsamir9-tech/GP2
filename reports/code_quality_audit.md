# Non-destructive code-quality audit

Date: 2026-10-07. Scope: `src/`, `scripts/`, `main.py`, and `tests/` in
`C:/Users/desgin/Documents/ChatGPT/New project`.

## Executive result

The original **179 tests passed** before edits. After two narrowly scoped
cleanup/logging fixes and two new failure-path tests, **181 tests passed with
zero failures and zero warnings**, with warnings treated as errors.

The approved production model was not changed: two bidirectional LSTM layers,
hidden size 256, external `(batch, 45, 138)` input, and a 286-to-128 internal
projection. Neither serving checkpoint nor either protected data file changed.
There was no training, optimization/export, dataset extraction, held-out test
evaluation, physical camera access, or installation of dependencies.

Passing unit tests does not establish complete production readiness. The most
important outstanding findings are shutdown handling after timeouts, mutable
input ownership in the feature-buffer API, expensive live temporal cleaning,
and stereo calibration/representation limitations. These were left unchanged
because addressing them requires behavioral or policy decisions.

The workspace already contained substantial modified and untracked work when
the audit began. That work was preserved; this report describes only the edits
made during this audit and observations of the current working tree.

## 1. Safe fixes applied

### 1.1 Release a video handle when extractor setup fails

Location: [batch_extractor.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/dataset/batch_extractor.py:66>).

Previously, `VideoCapture` was opened before `_get_extractor()` and
`reset_tracking()`, but those operations were outside the `try/finally` that
released the capture. An exception during either operation leaked the handle.
Both operations now execute inside the existing release boundary:

```diff
- extractor = self._get_extractor()
- reset_tracking = getattr(extractor, "reset_tracking", None)
- if callable(reset_tracking):
-     reset_tracking()
  frame_index = 0
  try:
+     extractor = self._get_extractor()
+     reset_tracking = getattr(extractor, "reset_tracking", None)
+     if callable(reset_tracking):
+         reset_tracking()
      ... existing read loop ...
  finally:
      capture.release()
```

The successful extraction path, data schema, preprocessing, and original
exception propagation remain unchanged.

### 1.2 Make camera FPS property failures observable

Locations: [camera_worker.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/capture/camera_worker.py:211>)
and [FPS query handling](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/capture/camera_worker.py:222>).

Two previously silent `except Exception` handlers now bind the exception and
log a warning containing the camera identifier and error. The native-rate
fallback and unknown-FPS behavior are preserved. No backend, camera property,
timeout, queue, threading, or cadence setting was changed.

### 1.3 Add failure-path coverage without changing existing tests

Added [test_resource_cleanup.py](<C:/Users/desgin/Documents/ChatGPT/New project/tests/unit/test_resource_cleanup.py:12>).
Its two parameterized cases simulate extractor construction and reset failures.
They verify that the original exception survives, the capture is released
exactly once, no frame read occurs, and an already constructed extractor is
closed. The temporary `.mp4` is an empty existence placeholder, not persisted
video content. All original 179 tests were retained unchanged.

## 2. Verified invariants and working safeguards

- The current checkpoint has `feature_projection.0.weight` shape `(128, 286)`;
  its configuration has two LSTM layers, hidden size 256, and dropout 0.35.
  The current production classifier, primary training workflow, and serving
  input checks retain the required model and feature dimensions.
- Extraction uses separate Lite Pose and Hands graphs, not a Holistic or face
  mesh instance. Only 21 landmarks per hand and pose indices 11, 12, 13, and 14
  are retained in application results. Pose internally computes its native
  pose output; that is distinct from retaining lower-body joints in the
  46-joint contract.
- Interpolation defaults to interior gaps of at most four frames. Smoothing
  performs float64 arithmetic before returning float32. Shoulder normalization
  subtracts the midpoint and divides by Euclidean shoulder distance, rejecting
  missing anchors and scale below `1e-6`.
- Neutral hand aperture/closure calculations remain finite for all-zero input
  because palm denominators are clamped. Missing shoulders or zero shoulder
  separation intentionally invalidate normalization instead of dividing by
  zero. Live windows with no observed hand are suppressed rather than submitted
  for idle predictions.
- Capture has a daemon producer, a one-slot drop-oldest queue, and a bounded
  latest-frame reference. Raw landmark deques are bounded by window size.
  Emitted windows own stable snapshots, with `torch.from_numpy` sharing those
  snapshots. The separate input-ownership issue below occurs before emission.
- GUI frame delivery is coalesced through a mailbox. The BGR video widget keeps
  the NumPy buffer alive while `QImage` wraps it; ordinary contiguous input does
  not need a GUI-thread RGB conversion or deep image copy.
- Camera and vision work are separated from UI painting; inference runs at
  window cadence, not on every capture. Qt slots perform widget updates on the
  UI thread.
- Restricted checkpoint loading uses `weights_only=True`; inference uses
  `torch.no_grad()` and `time.perf_counter()`.
- Static searches found no application raw-frame `imwrite`, `VideoWriter`
  construction, or network frame-transmission path. `VideoWriter_fourcc` merely
  requests a capture format. Coordinate CSVs, HDF5 features, calibration arrays,
  model weights, and generated plots are not raw video-frame persistence.
  This source audit does not certify OS paging, crash dumps, or native-library
  behavior outside the application.
- The lazy partitioned HDF5 loader has context cleanup and process-local handle
  reopening. No bare `except:` was found. Broad handlers were reviewed rather
  than indiscriminately rewritten: cleanup/re-raise, expected empty/full queues,
  worker error forwarding, and best-effort destructors have different purposes.

## 3. Risks left for manual review

### A. Threading, lifecycle, and resource ownership

**A1 — High: window close can accept a still-running pipeline.**
[dashboard.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/ui/dashboard.py:840>)
waits up to six seconds on the UI thread, reports a timeout, then marks the
worker finished anyway. `closeEvent` accepts the close regardless. A native
camera/tracker stall can therefore freeze the GUI during the wait or leave a
live QThread associated with a closing window. A late queued `finished` or
frame signal can also affect a replacement worker because slots use
`self.worker`, not a generation-specific sender. Review asynchronous shutdown,
timeout state, and signal ownership together; changing them is a lifecycle
behavior change, not a safe local cleanup.

**A2 — High: a daemon flag does not bound tracker shutdown.**
The GUI and [live harness](<C:/Users/desgin/Documents/ChatGPT/New project/scripts/live_test_harness.py:388>)
join the vision thread for five seconds without checking whether it is still
alive. [holistic_extractor.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/landmarks/holistic_extractor.py:173>)
waits on tracker futures without a timeout, and `close()` waits for the executor.
Native tracker calls can outlive the daemon vision thread. Review cancellation
and native-call containment; a Python timeout alone cannot safely interrupt a
running native graph.

**A3 — High, driver-dependent: read/release and restart races.**
[CameraWorker.stop()](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/capture/camera_worker.py:351>)
may release a handle from the caller while the producer is blocked in `read()`;
the producer also releases it in `finally`. Backend behavior is native and
driver-specific. After a failed stop, restarting can clear the shared stop event
while the old thread is still alive. Backend construction/release can themselves
block beyond a Python join deadline. An explicit failed/stopping lifecycle and
handle-ownership policy needs review before changing this behavior.

**A4 — Medium: startup readiness is earlier than completed warmup.**
[camera_worker.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/capture/camera_worker.py:253>)
sets readiness on the first successful discarded frame, not after all ten.
Callers then allow approximately 1.5 seconds for a published frame, while
warmup retries can continue for the five-second startup period. This can reject
a slow but otherwise usable camera. Existing tests explicitly preserve early
readiness, so this was not silently changed.

**A5 — Medium: Ctrl+C bypasses normal Qt cleanup.**
[main.py](<C:/Users/desgin/Documents/ChatGPT/New project/main.py:21>) deliberately
installs `SIG_DFL`, which is process termination rather than a Qt quit/close
request. OS handle reclamation is not the same as executing application cleanup.
This was explicitly requested earlier, so it remains a policy observation.

**A6 — Medium: stereo settings changes are not always applied live.**
[open_settings()](<C:/Users/desgin/Documents/ChatGPT/New project/src/ui/dashboard.py:850>)
restarts only when camera indices change. Changing `stereo_enabled` alone leaves
the worker's copied settings unchanged. A restart/reconfiguration policy is
needed; no runtime behavior was altered in this audit.

### B. Feature correctness, inference, and temporal edge cases

**B1 — High: the feature-buffer API retains caller-owned mutable storage.**
[TemporalBuffer.append()](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/preprocessing/temporal_buffer.py:62>)
uses `np.ascontiguousarray`, which does not copy an already contiguous float32
array. A synthetic producer reusing one array for 45 frames produced a history
containing only the final value, rather than the 45 historical values.
`append_landmarks()` does copy and is safe from this specific issue. Decide
whether the public feature API transfers ownership or copies input; changing
that contract/cost was not assumed safe.

**B2 — Medium: frozen contracts are not deeply immutable.**
[readonly_view()](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/contracts/validation.py:44>)
disables writes through a view, not through its original owner. Mutating the
original array changes the supposedly immutable record. Frame buffers remain
writable, and metadata protection is shallow. Current paths generally hand off
owned allocations, but public callers can violate that assumption. Deep copying
or stricter ownership documentation requires a deliberate memory-policy choice.

**B3 — High for performance: live cleaning is recomputed before the stride gate.**
[append_landmarks()](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/preprocessing/temporal_buffer.py:179>)
runs full-window cleaning for every eligible full raw window, including
non-emission stride steps. [smoothing.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/preprocessing/smoothing.py:40>)
calls Savitzky–Golay separately for 138 channels. Synthetic profiling below
identifies that loop as the dominant cost. Batched filtering or caching may
help, but must preserve float64 arithmetic, NaN-channel skipping, boundary
behavior, and valid-frame cadence. Neither was changed speculatively.

**B4 — Medium: matching numerical parameters do not guarantee identical context.**
[preprocess_dataset.py](<C:/Users/desgin/Documents/ChatGPT/New project/scripts/preprocess_dataset.py:188>)
canonicalizes hand presence and interpolates over a whole take before slicing.
Live processing sees only the rolling window. A boundary gap can therefore be
filled offline using observations unavailable in live context. The shared
presence policy also neutralizes a second hand with fewer than half as many
detections as the dominant hand, potentially removing a real brief two-handed
segment. Context and presence changes would alter trained representations.

**B5 — Medium: repeated-tail alignment is a heuristic, not captured provenance.**
[feature_alignment.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/models/feature_alignment.py:51>)
treats four or more identical suffix frames as padding, except for fully
constant clips. A real held pose or repeated camera frame can satisfy this
heuristic. Original clip lengths are unavailable in the retained HDF5 format.
Alignment also executes in both the live buffer and enabled model, creating
duplicate, idempotent tensor work. Removing one application needs compatibility
review across old/new models and loaders.

**B6 — Medium: confidence and cadence policies are inconsistent across entrypoints.**
The GUI defaults to 0.40 with a 0.35 slider floor and calls `ConfidenceFilter`
directly; the harness calls `InferenceEngine.gate_prediction()` with its 0.65
floor. Both use the 0.15 margin. A synthetic 0.55 prediction can consequently be
accepted by the GUI and rejected by the harness. Current GUI tests explicitly
expect sub-0.65 acceptance, reflecting earlier requested behavior. Live GUI and
harness use stride six, while AGENTS.md specifies stride eight. Requested camera
rate is currently 30 FPS; 60-Hz repainting is not 60 unique captured frames.
Resolve the specification/configuration disagreement before changing defaults.

**B7 — Medium: sustained identical predictions can periodically append duplicates.**
[text_ticker.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/ui/text_ticker.py:60>)
measures time since the last emitted token, not since the last repeated detection.
Continuous HELLO arrivals at 0, 0.5, 1.0, and 1.6 seconds produce `HELLO HELLO`
despite no silence. Changing cooldown to actual silence changes text behavior.
The token list and text document also grow without a retention limit; that is
intentional accumulation, but needs a long-session retention policy.

**B8 — Medium: tracking quality and reset semantics are weaker than their names.**
Pose anchors are accepted when coordinates are finite without checking MediaPipe
visibility/presence. `reset_tracking()` clears application hand state, not native
Pose/Hands tracking state. The legacy batch extractor reuses those graphs across
clips. Quality thresholds or graph recreation would change extraction outputs
and cost, so neither was added.

**B9 — Medium: future artifacts can fail unclearly or silently relabel FP32 output.**
[inference.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/models/inference.py:94>)
assumes decoded TorchScript metadata is an object, and verifies dimensions/class
mapping but not all backbone metadata. Prediction output validation checks logit
shape but does not comprehensively validate a model-supplied probability tensor.
FP32 checkpoints do not embed the vocabulary order: a same-sized permuted class
map can silently rename classes. Current protected artifacts and maps load
successfully; stricter artifact/output validation would change failure behavior.

### C. Stereo geometry and comparative evaluation

**C1 — High when stereo is enabled: calibration pixel geometry is incompletely applied.**
[calibration.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/stereo/calibration.py:39>)
loads only projection matrices, although calibration exports distortion and image
size. Runtime multiplies normalized landmarks by current frame dimensions and
uses those raw pixels for DLT, without undistorting points or reconciling capture
resolution with calibration resolution. Geometry can be biased or spuriously
fall back. This requires calibration/coordinate-policy work, not a local
exception-handling fix.

**C2 — High relative to claimed recovery: side-only observation cannot triangulate a fully missing front joint.**
[fallback_depth.py](<C:/Users/desgin/Documents/ChatGPT/New project/src/asl_stereo/stereo/fallback_depth.py:93>)
correctly requires a finite 2D observation from both cameras for DLT. A front
joint with all XYZ missing cannot be reconstructed by that operation alone.
Whole-frame fallback also discards successful triangulations if any joint fails.
This is a limitation of the existing approved design, not evidence of guaranteed
occlusion recovery. Side-only recovery would require a different estimator.

**C3 — Medium: whole-frame consistency does not ensure temporal/training consistency.**
A window can alternate metric triangulated XYZ and front normalized-image/local-Z
fallback frames. Shoulder normalization alone does not make these representations
equivalent to one another or to the front-camera training distribution. Review
representation continuity before making dual-camera accuracy claims.

**C4 — Medium: benchmark conditions can invalidate comparison claims.**
[benchmark_occlusion.py](<C:/Users/desgin/Documents/ChatGPT/New project/scripts/benchmark_occlusion.py:129>)
uses `process_sequence()` rather than live neutral-hand processing. Recorded views
are paired by decoded ordinal rather than timestamp alignment. Jitter comparison
can involve differing coordinate systems/units; finite zero-padded joints count
as available without an observed-hand mask. Rejected-candidate fraction is not
necessarily the false rejection rate over known genuine positive signs. There
is also a handle-cleanup gap if constructing the second VideoCapture raises
before the shared `try/finally`. These evaluation semantics need explicit review.

**C5 — Medium: watchdog integration does not report every drop category.**
The GUI records sync rejection and successful-pair events but does not feed
capture failure/backpressure counters into the watchdog. Single-camera telemetry
reports zero watchdog drops. The watchdog unit implementation distinguishes these
categories; the orchestrator does not yet provide complete accounting.

**C6 — Medium, bounded rather than leaking: synchronization and DLT costs.**
The synchronizer allows 150 full-resolution frames per camera. At 1280x720 BGR,
the two queue limits together represent approximately 791 MiB of frame payload
in a stress scenario. This is bounded, not linear leakage. Degeneracy checks also
recompute both fixed camera centers by SVD for every joint. Queue-budget changes
and cached geometry require ownership/configuration review.

### D. Artifacts, environment, and test coverage

**D1 — High operational risk: export promotion is not transactional.**
[optimize_model.py](<C:/Users/desgin/Documents/ChatGPT/New project/scripts/optimize_model.py:178>)
writes the final serving file before reload/profiling and the latency gate. A
failed export or gate can leave a changed/partial artifact; training can promote
FP32 weights before optimization succeeds. Dataset/class-map writers similarly
replace outputs non-transactionally. A future staging-and-promotion policy is
recommended. No export or protected-file write was performed here.

**D2 — Medium: alternate entrypoints have materially different semantics.**
Legacy `train.py` shares the default serving-weight path but has different
selection/optimization behavior. Legacy extraction uses a partitioned schema and
different neutral-hand/short-clip handling. Several evaluation/runtime/desktop
scripts and module files remain docstring-only scaffolds. The historical feature
alignment audit script contains claims about absent canonicalization that are
stale for the enabled model. Primary runtime paths work, but alternate CLIs and
historical audit output should not be treated as interchangeable implementations.

**D3 — Medium: the tested environment differs from the declared validated pins.**
This run used Python 3.11.3, Torch **2.4.1+cpu**, NumPy 1.26.4, MediaPipe 0.10.14,
PyQt5 **5.15.11**, OpenCV contrib 4.9.0.80, and pytest **9.1.1**. AGENTS.md specifies
Torch 2.3.0+cpu and PyQt5 5.15.10; the pyproject test extra excludes pytest 9.
The pyproject Torch requirement does not itself select a CPU index, while
requirements.txt does. Matplotlib is in pyproject but not requirements.txt,
although comparative plotting requires it. One cv2 wheel is specified, avoiding
the previous dual-wheel collision. No dependencies were changed or installed.

**D4 — High verification gap: non-functional gates do not exercise the complete live path.**
[test_non_functional_gates.py](<C:/Users/desgin/Documents/ChatGPT/New project/tests/unit/test_non_functional_gates.py:17>)
times `process_frame()` plus feature-buffer append, not raw rolling-window
interpolation/smoothing. Its memory test likewise exercises that smaller path;
`tracemalloc` does not capture all native OpenCV/MediaPipe/Torch/Qt allocations.
Unit tests do not prove physical-camera recovery, long-session shutdown, unique
capture FPS, or gesture-completion-to-visible-text p95 below 500 ms. Native-memory
soak tests and end-to-end hardware measurements are needed separately.

## 4. Synthetic diagnostic evidence

These are audit-time measurements on this laptop, not portable performance
guarantees. No camera was opened and no real video or held-out sample was used.

### Live preprocessing

An isolated synthetic finite one-handed 45-frame trajectory used the actual
live pipeline. Stage timings used three warmups and 20 measurements each:

| Operation | Mean ms | p95 ms |
| --- | ---: | ---: |
| Interior-gap interpolation, 45x46x3 | 8.056 | 12.032 |
| 138 separate Savitzky-Golay channels | 165.211 | 278.689 |
| Shoulder normalization, 45 frames | 1.917 | 3.094 |
| Complete `process_live_window` | 167.430 | 226.220 |

After filling the raw buffer, 100 `append_landmarks()` calls with stride six and
feature alignment enabled averaged **117.774 ms**, p95 **187.656 ms**, with 17
emissions. Independent samples/stages have scheduling variation and their p95
values are not additive. A separate smoothing experiment averaged 132.645 ms
with default BLAS pools versus 118.474 ms with BLAS constrained to one thread;
thread limiting did not remove the dominant per-channel cost. Limits applied
only inside that disposable diagnostic process, not project configuration.

### Existing INT8 serving artifact

The loaded TorchScript graph contains `aten::quantized_lstm` and
`quantized::linear_dynamic`, confirming quantized backbone/projections in the
current artifact. CPU float32 input preparation retained the input data pointer.
Dynamic INT8 still has float32 activations and temporary kinematics/alignment
allocations; it is not allocation-free. A single profiler call recorded about
3.09 MB of aggregate `aten::empty` allocations across 380 calls, plus arithmetic
and concatenation allocations. That is allocation traffic, not retained leakage.

With Torch CPU threads set to one inside the diagnostic process, 50 warmups and
200 synthetic prediction measurements gave **16.136 ms mean / 30.178 ms p95**.
Windows process measurements after warmup and at 100/200 iterations remained
209.402 MiB working set and 314.316 MiB private bytes. No growth was observed in
this short inference-only run; this does not establish long-session native
memory stability for capture, tracking, or Qt.

## 5. Verification record

Command, executed before and after the safe fixes:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& .\venv311\Scripts\python.exe -m pytest tests/ -v -W error -p no:cacheprovider
```

```text
Baseline:      179 passed in 64.56s
After changes: 181 passed in 47.43s
Failures: 0
Warnings: 0 (warnings treated as errors)
```

AST parsing covered all 119 current Python files in scope: zero syntax errors,
zero bare `except:`, and 19 broad Exception/BaseException handlers identified
for contextual review. `git diff --check` passed for the two production files
modified during this audit.

The four protected files had identical SHA-256 hashes before and after audit:

| File | Unchanged SHA-256 |
| --- | --- |
| `weights/best_model.pth` | `04CA4E71A8FFF8CF2EFCFADA27083C1769E82915997D80C75EB5194FE2B7F97C` |
| `weights/optimized_model.pt` | `E0B57A15326D9895FE45CBD2786F35E1C80C86DCDA69072376DD41FC22258BE4` |
| `artifacts/dataset.h5` | `44CB944C1F9AF8D2A94C05B47C22957B9F0BF7E8989850BF480F078CD964BA00` |
| `data/data.csv` | `D99BBB2F34CD7828CE167DC320E872B9031C8210337A619CC4937CA3203B2B86` |

Recommended next review order: shutdown/worker ownership, feature input ownership,
live cleaning parity and performance, then stereo calibration and comparative
metric validity. Any follow-up should preserve the protected model/data assets
and add targeted tests without weakening existing assertions.
