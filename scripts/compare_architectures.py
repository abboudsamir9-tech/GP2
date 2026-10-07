"""Validation-only capacity selection, followed by exactly one test evaluation.

This command deliberately does not overwrite serving weights, regenerate the
dataset, optimize models, or search split seeds. A durable selection record is
written before opening test features. An existing report blocks rerunning the
experiment, including after interruption: further test use needs a new decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asl_stereo.dataset.comparison_protocol import freeze_existing_partitions, load_partition
from asl_stereo.models.checkpoint import load_checkpoint, load_class_map, save_training_checkpoint
from asl_stereo.models.comparison import MODEL_SPECS, build_comparison_model
from scripts.train_classifier import (
    SequenceDataset, build_evaluation_report, build_warmup_cosine_scheduler,
    compute_class_weights, evaluate, macro_f1_from_report, predict_test_windows, run_epoch,
)


def classification_metrics(labels, probabilities, video_ids, class_map) -> dict:
    result = build_evaluation_report(labels, probabilities, video_ids, class_map)
    result["macro_f1"] = macro_f1_from_report(result)
    for old, new in (("test_accuracy", "accuracy"), ("test_video_count", "videos"),
                     ("test_window_count", "windows")):
        result[new] = result.pop(old)
    return result


def select_validation_winner(candidates: list[dict]) -> dict:
    """Macro-F1 only; a predeclared exact tie favors the smaller model."""
    if not candidates or any("test" in row for row in candidates):
        raise ValueError("selection accepts validation-only candidates")
    return max(candidates, key=lambda row: (row["validation"]["macro_f1"], -row["parameters"]))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def train_candidate(name, train_data, val_data, class_map, output_dir, *,
                    epochs=60, patience=20, min_epochs=30, seed=42, batch_size=8) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_comparison_model(name, len(class_map)).cpu()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    # Use independent, identically seeded shuffling for each candidate.
    torch.manual_seed(seed)
    train_features, train_labels, _ = train_data
    val_features, val_labels, val_videos = val_data
    train_loader = DataLoader(SequenceDataset(train_features, train_labels, augment=True),
                              batch_size=batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(SequenceDataset(val_features, val_labels), batch_size=batch_size)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_labels, class_map), label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = build_warmup_cosine_scheduler(optimizer, max_epochs=epochs, warmup_epochs=min(5, epochs))
    checkpoint = output_dir / f"model_{name}.pth"
    best_f1, best_epoch, bad_epochs, history = -1.0, 0, 0, []
    best_metrics = None
    started = time.perf_counter()
    print(f"[MODEL {name}] {MODEL_SPECS[name]} | parameters={parameters:,}", flush=True)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, _ = evaluate(model, val_loader, criterion)
        targets, probabilities = predict_test_windows(model, val_loader)
        metrics = classification_metrics(targets, probabilities, val_videos, class_map)
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "val_loss": val_loss, "val_macro_f1": metrics["macro_f1"],
                        "val_accuracy": metrics["accuracy"], "lr": learning_rate})
        improved = metrics["macro_f1"] > best_f1 + 1e-12
        if improved:
            best_f1, best_epoch, best_metrics, bad_epochs = metrics["macro_f1"], epoch, metrics, 0
            save_training_checkpoint(model, checkpoint, num_classes=len(class_map), window_size=45,
                                     feature_dim=138, epoch=epoch, best_val_acc=metrics["accuracy"],
                                     model_type=MODEL_SPECS[name]["architecture"])
        else:
            bad_epochs += 1
        print(f"[{name} {epoch:02d}/{epochs}] train loss={train_loss:.4f} | val loss={val_loss:.4f} "
              f"| val accuracy={metrics['accuracy']:.2%} | val macro-F1={metrics['macro_f1']:.4f}"
              f"{' *' if improved else ''}", flush=True)
        if epoch >= min_epochs and bad_epochs >= patience:
            break
    return {"name": name, "architecture": MODEL_SPECS[name], "parameters": parameters,
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
            "best_epoch": best_epoch, "epochs_completed": epoch, "validation": best_metrics,
            "history": history, "training_seconds": time.perf_counter() - started}


def compare_architectures(dataset_path, membership_report, metadata_csv, class_map_path,
                          output_dir, report_path, *, epochs=60, patience=20, min_epochs=30,
                          seed=42, batch_size=8, feature_audit=None) -> dict:
    if report_path.exists():
        raise FileExistsError("Comparison report already exists; refusing another test evaluation")
    if epochs < 1 or patience < 1 or not 1 <= min_epochs <= epochs:
        raise ValueError("invalid epoch/early-stopping schedule")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / f"model_{name}.pth").exists() for name in MODEL_SPECS):
        raise FileExistsError("Candidate checkpoints already exist; use a fresh experiment directory")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    partitions, digest = freeze_existing_partitions(dataset_path, membership_report, metadata_csv)
    class_map = load_class_map(class_map_path)
    production = [PROJECT_ROOT / "weights" / name for name in ("best_model.pth", "optimized_model.pt")]
    original_hashes = {str(path): sha256(path) for path in production if path.is_file()}
    train_data = load_partition(dataset_path, partitions["train"])
    val_data = load_partition(dataset_path, partitions["val"])
    if any(set(values[1].tolist()) != set(class_map) for values in (train_data, val_data)):
        raise ValueError("all glosses must be represented in training and validation")
    report = {"protocol": {
        "selection": "validation_video_macro_f1_only", "tie_break": "fewest_parameters",
        "test_feature_reads_before_selection": 0, "test_evaluation_count": 0,
        "split_membership_sha256": digest, "split_regenerated": False,
        "seed": seed, "max_epochs": epochs, "patience": patience, "min_epochs": min_epochs,
        "batch_size": batch_size, "lr": 0.001, "weight_decay": 0.001, "warmup_epochs": min(5, epochs),
        "augmentation": "training_only_jitter_scaling_rotation_temporal_resampling",
        "loss": "bounded_inverse_frequency_label_smoothed_CE_0.1", "cpu_threads": 1,
        "python_version": sys.version.split()[0], "torch_version": str(torch.__version__),
        "test_previously_used_in_project": True,
        "limitation": "This preserves an existing holdout, not a fresh untouched final test set.",
    }, "split_summary": {name: {"video_ids": list(part.video_ids), "signer_ids": list(part.signer_ids),
                               "videos": len(part.video_ids), "signers": len(part.signer_ids),
                               "windows": len(part.indices)} for name, part in partitions.items()},
        "feature_audit": feature_audit or {}, "models": [], "status": "training_validation_only"}
    for name, (_, labels, videos) in (("train", train_data), ("val", val_data)):
        report["split_summary"][name]["windows_per_gloss"] = {
            gloss: int(np.count_nonzero(labels == index)) for index, gloss in class_map.items()}
        report["split_summary"][name]["videos_per_gloss"] = {
            gloss: len(np.unique(videos[labels == index])) for index, gloss in class_map.items()}
        print(f"[SPLIT {name}] {report['split_summary'][name]['signers']} signers | "
              f"{len(np.unique(videos))} videos | {len(labels)} windows | "
              f"{report['split_summary'][name]['videos_per_gloss']}", flush=True)
    _write_json(report_path, report)
    for name in MODEL_SPECS:
        candidate = train_candidate(name, train_data, val_data, class_map, output_dir, epochs=epochs,
                                    patience=patience, min_epochs=min_epochs, seed=seed, batch_size=batch_size)
        report["models"].append(candidate)
        _write_json(report_path, report)
    winner = select_validation_winner(report["models"])
    report["selected_model"] = winner["name"]
    # Selection/checkpoint hash is durable BEFORE the first test feature read.
    report["status"] = "selection_locked_test_evaluation_started"
    report["protocol"]["selected_checkpoint_sha256"] = winner["checkpoint_sha256"]
    _write_json(report_path, report)
    model = build_comparison_model(winner["name"], len(class_map)).cpu()
    model.load_state_dict(load_checkpoint(winner["checkpoint"])["model_state_dict"], strict=True)
    test_features, test_labels, test_videos = load_partition(dataset_path, partitions["test"])
    report["split_summary"]["test"]["windows_per_gloss"] = {
        gloss: int(np.count_nonzero(test_labels == index)) for index, gloss in class_map.items()}
    report["split_summary"]["test"]["videos_per_gloss"] = {
        gloss: len(np.unique(test_videos[test_labels == index])) for index, gloss in class_map.items()}
    test_loader = DataLoader(SequenceDataset(test_features, test_labels), batch_size=batch_size)
    targets, probabilities = predict_test_windows(model, test_loader)
    report["test"] = classification_metrics(targets, probabilities, test_videos, class_map)
    report["test"]["model"] = winner["name"]
    report["test"]["target_accuracy_met"] = report["test"]["accuracy"] >= 0.85
    report["protocol"]["test_evaluation_count"] = 1
    report["production_weights_unchanged"] = all(sha256(Path(path)) == digest for path, digest in original_hashes.items())
    if not report["production_weights_unchanged"]:
        raise RuntimeError("Production weights changed during the comparison")
    report["status"] = "complete"
    _write_json(report_path, report)
    print("\nModel | Parameters | Validation accuracy | Validation Macro-F1 | Best epoch")
    for row in report["models"]:
        print(f"  {row['name']}   | {row['parameters']:>10,} | {row['validation']['accuracy']:>18.2%} "
              f"| {row['validation']['macro_f1']:>19.4f} | {row['best_epoch']:>10}")
    print(f"Selected {winner['name']} ONLY: test {report['test']['accuracy']:.2%}; macro-F1 {report['test']['macro_f1']:.4f}")
    print("Test confusion matrix (rows=true, columns=predicted):", report["test"]["class_order"])
    for row in report["test"]["confusion_matrix"]:
        print(" ".join(f"{count:3d}" for count in row))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-h5", type=Path, default=PROJECT_ROOT / "artifacts/dataset.h5")
    parser.add_argument("--membership-report", type=Path, default=PROJECT_ROOT / "reports/evaluation_metrics.json")
    parser.add_argument("--metadata-csv", type=Path, default=PROJECT_ROOT / "data/data.csv")
    parser.add_argument("--class-map", type=Path, default=PROJECT_ROOT / "configs/class_map.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/model_comparison")
    parser.add_argument("--report-out", type=Path, default=PROJECT_ROOT / "reports/model_comparison.json")
    parser.add_argument("--feature-audit", type=Path, default=PROJECT_ROOT / "reports/feature_alignment_audit.json")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=30)
    args = parser.parse_args()
    audit = json.loads(args.feature_audit.read_text(encoding="utf-8")) if args.feature_audit.is_file() else None
    compare_architectures(args.dataset_h5, args.membership_report, args.metadata_csv, args.class_map,
                          args.output_dir, args.report_out, epochs=args.epochs, patience=args.patience,
                          min_epochs=args.min_epochs, feature_audit=audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
