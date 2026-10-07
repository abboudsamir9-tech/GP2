# Offline capacity comparison protocol

The membership recorded in `evaluation_metrics.json` is frozen verbatim:
26 training signers (102 videos), 6 validation signers (23 videos), and 6 test
signers (23 videos). No split generation or random-seed search is performed.
The requested 98-video training count differs from the recorded split; videos
are not dropped to force that count.

All candidates accept float32 `(batch, 45, 138)` and internally construct the
same 286 features: positions, first differences, six palm-normalized aperture
ratios, and four thumb-to-finger closure distances.

| Candidate | Backbone | Regularization |
| --- | --- | --- |
| A | 2-layer bidirectional LSTM, hidden 256 | Projection/recurrent dropout 0.3 |
| B | 1-layer bidirectional LSTM, hidden 128 | Projection/output dropout 0.3 |
| C | Two time-preserving 1D convolutions + GRU, hidden 96 | Projection/convolution dropout 0.3 |

Each starts from scratch with seed 42 and uses the same training-only
augmentation, batch size 8, bounded inverse-frequency label-smoothed CE,
AdamW (lr 0.001, weight decay 0.001), five-epoch warmup and cosine decay.
Maximum 60 epochs, minimum 30, patience 20. The first strictly highest
validation **per-video Macro-F1** checkpoint is retained for each candidate.
Average window probabilities give one vote per source video.

Across candidates, highest validation Macro-F1 wins. An exact tie favors
fewer parameters. Recall coverage, test accuracy, and validation loss are
not additional ranking criteria. The decision and checkpoint SHA256 are
saved before test features are loaded. Only the winner is evaluated on test,
once. Non-winners have no test metrics. An existing comparison report blocks
automatic reruns, even after interruption.

Serving weights, the existing evaluation report, class map, and HDF5 remain
unchanged. Candidate checkpoints are offline experiment artifacts, not
automatically promoted to serving or automatically optimized.

## Feature audit

The code and every training/validation window are audited, with real
re-extraction of ten non-test clips (two per class). Raw frames stay in RAM.
Synthetic and sampled like-for-like 45-frame contexts check agreement of
offline preprocessing and the live rolling buffer, including zero-padding
and short-gap interpolation. Simulated repeat-padding on short clips is
explicitly identified rather than claimed to be a real live capture.

Known limits are reported, not silently corrected during the capacity trial:
clip-global versus rolling-context absence/interpolation decisions can differ;
handedness flicker flags are not anatomical ground truth; no dominant-hand
canonicalization exists; HDF5 lacks preprocessing/extractor version provenance.

The existing test partition has already guided previous project experiments.
This protocol prevents further test-based model selection within this trial,
but cannot retroactively make that partition a never-used final holdout.

## Commands

Use the existing Python 3.11 environment:

```powershell
venv311\Scripts\python.exe scripts/audit_feature_alignment.py
venv311\Scripts\python.exe scripts/compare_architectures.py
venv311\Scripts\python.exe -m pytest tests/ -v
```

Review `model_comparison.json` and `feature_alignment_audit.json` before any
future data correction, new comparison, or model promotion. Do not rerun test
evaluation merely because the result is below 85%.
