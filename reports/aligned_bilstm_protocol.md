# Production BiLSTM alignment and training

The primary model and `scripts/train_classifier.py` are locked to a two-layer,
bidirectional LSTM (hidden 256) with dropout 0.35 and AdamW weight decay 0.001.
The external input remains `(batch, 45, 138)`; positions, first differences,
six aperture ratios and four closure distances form 286 internal features.
CNN/GRU and Transformer code remains available only for historical experiments.

## Shared in-memory transform

`models/feature_alignment.py` is shared by the dataset, opt-in temporal buffers,
and the model forward pass. TorchScript embeds it, and checkpoint metadata
enables it in GUI/harness queues. Legacy artifacts without alignment metadata
keep their original behavior. Applying the transform again is idempotent.

- Hand slices are left `0:63`, right `63:126`, pose `126:138`.
- Compute coordinate variance over non-neutral observations for each hand.
- Mirror only a moving left hand with the entire right-hand block zero.
- Reflect X, swap the two hands, and swap shoulders/elbows to retain anatomy.
- A visible static second hand is conservatively bilateral; label names do
  not determine handedness or activate a ground-truth-dependent transform.
- Detect at least four exactly identical suffix frames, retaining one copy.
  Move retained motion to the end of the FIFO window at the same cadence.
  Fill missing leading context with zero hands and the first normalized pose.
  Do not guess a completely constant clip to be repeat-padding.

This last step is a documented approximation, not recovery of original frame
lengths or pre-sign footage. Those fields are absent from HDF5. Existing
window-level smoothing and suppressed landmarks also cannot be undone by an
in-memory transform. No raw frames are written and the HDF5 is never modified.

## Selection and evaluation

Use the exact recorded membership: 26 training signers/102 videos, six
validation signers/23 videos, and six test signers/23 videos. No split seed
search occurs. Train/validation feature tensors load first. Select the first
strictly best per-video validation Macro-F1 checkpoint, with no test-based
selection or recall-coverage override. Maximum 60 epochs, minimum 30, patience
20; five-epoch warmup plus cosine decay; training-only augmentation.

Persist selection and its checkpoint hash before loading test features.
Compile and benchmark the optimized model on synthetic inputs, and compare
its probabilities with FP32 on validation only. Evaluate the selected optimized
serving artifact once on test. Do not score FP32 separately on the same test.
The test membership is unchanged but has been used in previous project trials;
it is not a newly untouched final holdout.

The run backs up serving artifacts and prior reports to a timestamped directory
under `artifacts/training_history/`. The final evaluation report includes
audits, before/after HDF5 SHA256, validation history, confusion matrix, per-class
metrics and CPU latency. A completed or started aligned trial blocks an
automatic repeated test evaluation.

```powershell
venv311\Scripts\python.exe -m pytest tests/ -q
venv311\Scripts\python.exe -u scripts/train_classifier.py
```
