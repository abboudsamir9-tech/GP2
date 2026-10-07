"""One validation-selected BiLSTM run on the exact recorded signer partitions."""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from asl_stereo.dataset.comparison_protocol import freeze_existing_partitions, load_partition
from asl_stereo.models import SignSequenceClassifier
from asl_stereo.models.checkpoint import load_class_map, load_model_weights, save_training_checkpoint
from asl_stereo.models.feature_alignment import (
    ALIGNMENT_VERSION, align_feature_window, dominant_left_mask, hand_coordinate_variances,
    repeated_tail_padding_lengths, fifo_neutral_context,
)
from scripts.compare_architectures import sha256
from scripts.train_classifier import (
    SequenceDataset, build_evaluation_report, build_warmup_cosine_scheduler, compute_class_weights,
    discover_metadata_csv, evaluate, macro_f1_from_report, optimize_saved_checkpoint,
    predict_test_windows, print_evaluation_report, run_epoch,
)


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def alignment_audit(features: np.ndarray) -> dict:
    inputs = torch.from_numpy(features)
    aligned = align_feature_window(inputs)
    bilateral = (inputs[:, :, :63] != 0).any(dim=-1).any(dim=1) & (inputs[:, :, 63:126] != 0).any(dim=-1).any(dim=1)
    variances = hand_coordinate_variances(inputs)
    joints = aligned.reshape(-1, 45, 46, 3)
    midpoint = (joints[:, :, 42] + joints[:, :, 43]) / 2
    return {
        "windows": len(features), "mirrored_left_only_windows": int(dominant_left_mask(inputs).sum()),
        "visible_bilateral_windows": int(bilateral.sum()),
        "bilateral_canonicalization_unchanged": bool(torch.equal(
            align_feature_window(inputs[bilateral]),
            # FIFO context may change detectable padding, not hand assignment.
            fifo_neutral_context(inputs[bilateral]),
        )) if bool(bilateral.any()) else True,
        "mean_hand_coordinate_variance": variances.mean(dim=0).tolist(),
        "windows_with_detectable_repeated_tail": int((repeated_tail_padding_lengths(inputs) > 0).sum()),
        "total_repeat_padding_rows_replaced": int(repeated_tail_padding_lengths(inputs).sum()),
        "max_shoulder_origin_error": float(midpoint.abs().max()),
        "idempotent": bool(torch.equal(aligned, align_feature_window(aligned))),
        "finite_float32_45x138": bool(aligned.dtype == torch.float32 and torch.isfinite(aligned).all()),
    }


def run_aligned_bilstm(
    dataset_h5, class_map_path, output_weights, *, batch_size=8, epochs=60,
    learning_rate=1e-3, weight_decay=1e-3, patience=20, min_epochs=30, seed=42,
    metadata_csv=None, optimized_model_out=None, report_out=None, experiment_dir=None,
) -> Path:
    from scripts.train_classifier import PROJECT_ROOT
    if (batch_size <= 0 or epochs <= 0 or learning_rate <= 0 or weight_decay != 1e-3
            or patience <= 0 or not 1 <= min_epochs <= epochs):
        raise ValueError("invalid schedule; production BiLSTM weight_decay is locked to 1e-3")
    dataset_path = Path(dataset_h5).resolve()
    report_path = Path(report_out or PROJECT_ROOT / "reports/evaluation_metrics.json").resolve()
    if not report_path.is_file():
        raise FileNotFoundError("A recorded signer membership report is required; refusing to regenerate the split")
    old_report = json.loads(report_path.read_text(encoding="utf-8"))
    if old_report.get("protocol", {}).get("experiment") == "aligned_bilstm_v1":
        raise FileExistsError("This aligned BiLSTM trial already started; refusing automatic repeated test evaluation")
    metadata_path = Path(metadata_csv) if metadata_csv is not None else discover_metadata_csv()
    if metadata_path is None:
        raise FileNotFoundError("Signer metadata is required")
    partitions, membership_hash = freeze_existing_partitions(dataset_path, report_path, metadata_path)
    class_map = load_class_map(class_map_path)
    run_dir = Path(experiment_dir) if experiment_dir is not None else (
        PROJECT_ROOT / "artifacts/training_history" /
        datetime.now(ZoneInfo("Asia/Amman")).strftime("aligned_bilstm_%Y%m%d_%H%M%S")
    )
    if run_dir.exists():
        raise FileExistsError("Experiment directory already exists; refusing to overwrite a trial")
    run_dir.mkdir(parents=True)
    output = Path(output_weights).resolve()
    optimized = Path(optimized_model_out).resolve() if optimized_model_out else output.with_name("optimized_model.pt")
    for path in (output, optimized, report_path, PROJECT_ROOT / "reports/quantization_benchmark.json"):
        if path.is_file():
            shutil.copy2(path, run_dir / path.name)
    write_report(run_dir / "frozen_membership.json", {"split_summary": old_report["split_summary"]})
    artifact_hash = sha256(dataset_path)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train_data = load_partition(dataset_path, partitions["train"])
    val_data = load_partition(dataset_path, partitions["val"])
    for _, labels, _ in (train_data, val_data):
        if set(labels.tolist()) != set(class_map):
            raise ValueError("Training and validation must cover all classes")
    train_loader = DataLoader(SequenceDataset(*train_data[:2], augment=True, align_features=True),
                              batch_size=batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(SequenceDataset(*val_data[:2], align_features=True), batch_size=batch_size)
    model = SignSequenceClassifier(len(class_map), hidden_size=256, num_layers=2,
                                   dropout=0.35, align_features=True).cpu()
    assert isinstance(model.temporal_backbone, nn.LSTM) and model.temporal_backbone.bidirectional
    class_weights = compute_class_weights(train_data[1], class_map)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = build_warmup_cosine_scheduler(optimizer, max_epochs=epochs, warmup_epochs=min(5, epochs))
    report = {
        "evaluation_scope": "unseen_signers", "group_key": "signer_id", "status": "training_validation_only",
        "protocol": {"experiment": "aligned_bilstm_v1", "selection": "validation_video_macro_f1_only",
                     "test_evaluation_count": 0, "test_feature_tensors_loaded_before_selection": False,
                     "split_membership_sha256": membership_hash, "split_regenerated": False,
                     "test_partition_previously_used_in_project": True},
        "split_summary": old_report["split_summary"],
        "training_config": {"model_type": "bilstm_attention", "hidden_size": 256, "num_layers": 2,
                            "recurrent_dropout": 0.35, "weight_decay": weight_decay, "learning_rate": learning_rate,
                            "max_epochs": epochs, "patience": patience, "min_epochs": min_epochs,
                            "seed": seed, "batch_size": batch_size, "internal_feature_dim": 286,
                            "external_input_shape": [1, 45, 138], "cpu_threads": 1,
                            "loss": "bounded_inverse_frequency_label_smoothed_CE_0.1",
                            "class_weights": class_weights.tolist(), "feature_alignment": ALIGNMENT_VERSION,
                            "checkpoint_selection": "validation_video_macro_f1_only",
                            "augmentation": "training_only_jitter_scale_rotation_temporal_resampling"},
        "audit": {"dataset_sha256_before": artifact_hash,
                  "train": alignment_audit(train_data[0]), "validation": alignment_audit(val_data[0]),
                  "hand_feature_slices": {"left": [0, 63], "right": [63, 126], "pose": [126, 138]},
                  "padding_policy": "detect >=4 exact suffix duplicates; replace copies with neutral leading context",
                  "limitations": ["Original capture lengths/idle poses are not present in HDF5; context recovery is approximate.",
                                  "Visible static second hands are not mirrored; constant clips are not guessed to be padding.",
                                  "The test membership is preserved, but previous project use prevents calling it a fresh untouched holdout."]},
        "history": [],
    }
    write_report(report_path, report)
    write_report(run_dir / "run_report.json", report)
    for name in ("train", "val", "test"):
        part = partitions[name]
        print(f"[FROZEN {name}] {len(part.signer_ids)} signers / {len(part.video_ids)} videos / {len(part.indices)} windows", flush=True)
    print("[ARCHITECTURE] BiLSTM x2, hidden=256, projection=286->128, dropout=0.35", flush=True)
    print("[AUDIT]", json.dumps(report["audit"]["train"]), flush=True)
    best_f1, bad_epochs = -1.0, 0
    started = time.perf_counter()
    candidate = run_dir / "selected_bilstm.pth"
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, _ = evaluate(model, val_loader, criterion)
        labels, scores = predict_test_windows(model, val_loader)
        validation = build_evaluation_report(labels, scores, val_data[2], class_map)
        f1 = macro_f1_from_report(validation)
        report["history"].append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                                  "validation_loss": val_loss, "validation_macro_f1": f1,
                                  "validation_accuracy": validation["test_accuracy"]})
        scheduler.step()
        improved = f1 > best_f1 + 1e-12
        if improved:
            best_f1, bad_epochs = f1, 0
            save_training_checkpoint(model, candidate, num_classes=len(class_map), window_size=45,
                                     feature_dim=138, epoch=epoch, best_val_acc=validation["test_accuracy"])
            report.update(best_validation_macro_f1=f1, best_validation_accuracy=validation["test_accuracy"],
                          best_validation_loss=val_loss, best_checkpoint_epoch=epoch)
        else:
            bad_epochs += 1
        print(f"[{epoch:02d}/{epochs}] loss={train_loss:.4f} | val loss={val_loss:.4f} "
              f"| val accuracy={validation['test_accuracy']:.2%} | val Macro-F1={f1:.4f}{' *' if improved else ''}", flush=True)
        report["epochs_completed"] = epoch
        write_report(run_dir / "run_report.json", report)
        if epoch >= min_epochs and bad_epochs >= patience:
            break
    report["training_seconds"] = time.perf_counter() - started
    load_model_weights(model, candidate)
    report["status"] = "selection_locked_before_single_test"
    report["protocol"]["selected_checkpoint_sha256"] = sha256(candidate)
    write_report(report_path, report)
    write_report(run_dir / "run_report.json", report)
    # Export/benchmark using synthetic inputs only, before using test tensors.
    output.parent.mkdir(parents=True, exist_ok=True)
    save_training_checkpoint(model, output, num_classes=len(class_map), window_size=45,
                             feature_dim=138, epoch=report["best_checkpoint_epoch"],
                             best_val_acc=report["best_validation_accuracy"])
    optimize_saved_checkpoint(output, class_map_path, optimized)
    report["latency_metrics"] = json.loads((PROJECT_ROOT / "reports/quantization_benchmark.json").read_text(encoding="utf-8"))
    scripted = torch.jit.load(str(optimized), map_location="cpu").eval()
    # Confirm serving consistency on validation, never a second test scoring.
    targets, fp32_scores = predict_test_windows(model, val_loader)
    _, int8_scores = predict_test_windows(scripted, val_loader)
    report["serving_validation_max_probability_error"] = float(np.max(np.abs(fp32_scores - int8_scores)))
    report["status"] = "single_test_evaluation_started"
    write_report(report_path, report)
    write_report(run_dir / "run_report.json", report)
    test_data = load_partition(dataset_path, partitions["test"])
    test_loader = DataLoader(SequenceDataset(*test_data[:2], align_features=True), batch_size=batch_size)
    # The single scored checkpoint is the actual optimized serving artifact.
    targets, probabilities = predict_test_windows(scripted, test_loader)
    metrics = build_evaluation_report(targets, probabilities, test_data[2], class_map)
    report.update(metrics)
    report.update(macro_f1=macro_f1_from_report(metrics), target_accuracy=0.85,
                  target_accuracy_met=metrics["test_accuracy"] >= 0.85, status="complete",
                  evaluated_artifact=str(optimized), selected_fp32_checkpoint=str(output),
                  optimized_model_test_accuracy=metrics["test_accuracy"],
                  experiment_dir=str(run_dir.resolve()), existing_test_signers_preserved=True)
    report["protocol"]["test_evaluation_count"] = 1
    report["audit"]["dataset_sha256_after"] = sha256(dataset_path)
    report["audit"]["dataset_unchanged"] = report["audit"]["dataset_sha256_after"] == artifact_hash
    if not report["audit"]["dataset_unchanged"]:
        raise RuntimeError("Dataset artifact changed during the training run")
    write_report(report_path, report)
    write_report(run_dir / "run_report.json", report)
    print_evaluation_report(report)
    print(f"Validation Macro-F1: {best_f1:.4f}; single-test Macro-F1: {report['macro_f1']:.4f}", flush=True)
    print(f"Dataset unchanged: True; backup and run artifacts: {run_dir}", flush=True)
    return output


__all__ = ["alignment_audit", "run_aligned_bilstm"]
