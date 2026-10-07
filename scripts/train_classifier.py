"""Train the production BiLSTM-attention classifier on the compact HDF5 set."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from asl_stereo.contracts import FEATURE_COUNT
from asl_stereo.dataset import LandmarkAugmentor, load_grouped_windows, split_grouped_windows
from asl_stereo.models import SignSequenceClassifier, align_feature_window
from asl_stereo.models.checkpoint import (
    load_class_map,
    load_model_weights,
    save_training_checkpoint,
)


WINDOW_SIZE = 45


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: npt.ArrayLike, labels: npt.ArrayLike, *, augment: bool = False,
                 align_features: bool = False) -> None:
        feature_array = np.asarray(features, dtype=np.float32)
        label_array = np.asarray(labels, dtype=np.int64)
        if feature_array.ndim != 3 or feature_array.shape[1:] != (
            WINDOW_SIZE,
            FEATURE_COUNT,
        ):
            raise ValueError(
                f"features must have shape (N, {WINDOW_SIZE}, {FEATURE_COUNT})"
            )
        if label_array.shape != (feature_array.shape[0],):
            raise ValueError("labels must have shape (N,) matching features")
        if not np.isfinite(feature_array).all():
            raise ValueError("features must contain only finite values")
        self.features = np.ascontiguousarray(feature_array)
        self.labels = np.ascontiguousarray(label_array)
        self.augmentor = LandmarkAugmentor() if augment else None
        self.align_features = align_features

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.from_numpy(self.features[index])
        if self.align_features:
            window = align_feature_window(window.unsqueeze(0)).squeeze(0)
        return (
            self.augmentor(window) if self.augmentor is not None else window,
            torch.tensor(self.labels[index], dtype=torch.long),
        )


def discover_metadata_csv() -> Path | None:
    provided = PROJECT_ROOT / "data" / "data.csv"
    if provided.is_file():
        return provided
    candidates = sorted(
        path for directory in (PROJECT_ROOT / "data", PROJECT_ROOT / "artifacts")
        if directory.is_dir() for path in directory.rglob("*.csv")
        if "metadata" in path.name.casefold()
    )
    if len(candidates) > 1:
        raise ValueError("multiple metadata CSVs found; pass --metadata-csv explicitly")
    return candidates[0] if candidates else None


def compute_class_weights(labels: npt.ArrayLike, class_map: dict[int, str]) -> torch.Tensor:
    values = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(values, minlength=len(class_map))
    if values.size == 0 or np.any(counts == 0):
        raise ValueError("every gloss must have training examples")
    weights = values.size / (len(class_map) * counts.astype(np.float64))
    weights /= weights.mean()
    return torch.tensor(np.clip(weights, 0.8, 1.6), dtype=torch.float32)


def previous_test_signers(
    report_path: str | Path, labels: npt.ArrayLike, video_ids: npt.ArrayLike,
    signer_ids: npt.ArrayLike, *, min_test_videos_per_class: int,
) -> tuple[str, ...] | None:
    """Reuse holdout membership/provenance, never use its measured accuracy."""
    path = Path(report_path)
    if not path.is_file():
        return None
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("evaluation_scope") != "unseen_signers":
            return None
        splits = previous["split_summary"]
        test_signers = tuple(splits["test"]["signer_ids"])
        previous_videos = {video for part in splits.values() for video in part["video_ids"]}
        values, videos, signers = (np.asarray(labels, dtype=np.int64),
                                  np.asarray(video_ids, dtype=str), np.asarray(signer_ids, dtype=str))
        selected = np.isin(signers, test_signers)
        if (previous_videos != set(videos) or not test_signers
                or not set(test_signers).issubset(set(signers))
                or set(videos[selected]) != set(splits["test"]["video_ids"])):
            return None
        _, first = np.unique(videos[selected], return_index=True)
        test_labels = values[selected][first]
        if any(np.count_nonzero(test_labels == class_id) < min_test_videos_per_class
               for class_id in np.unique(values)):
            return None
        return test_signers
    except (OSError, ValueError, KeyError, TypeError):
        logging.warning("Existing evaluation report could not supply a reusable signer holdout")
        return None


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer, *, max_epochs: int, warmup_epochs: int = 5,
) -> torch.optim.lr_scheduler.LambdaLR:
    if max_epochs <= 0 or warmup_epochs <= 0 or warmup_epochs > max_epochs:
        raise ValueError("warmup epochs must be within the training schedule")

    def multiplier(step: int) -> float:
        if step < warmup_epochs:
            return (step + 1) / warmup_epochs
        progress = min(1.0, (step - warmup_epochs) / max(1, max_epochs - warmup_epochs))
        return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def optimize_saved_checkpoint(checkpoint_path: str | Path, class_map_path: str | Path,
                              optimized_model_path: str | Path) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "optimize_model.py"),
               "--input-weights", str(checkpoint_path), "--output-model", str(optimized_model_path),
               "--class-map", str(class_map_path),
               "--report-out", str(PROJECT_ROOT / "reports" / "quantization_benchmark.json")]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=environment)


def load_feature_artifact(
    dataset_path: str | Path,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset does not exist: {path}")
    with h5py.File(path, "r") as h5_file:
        if "features" not in h5_file or "labels" not in h5_file:
            raise ValueError("dataset must contain root features and labels datasets")
        features = np.asarray(h5_file["features"], dtype=np.float32)
        labels = np.asarray(h5_file["labels"], dtype=np.int64)
    # SequenceDataset owns shape, dtype, and finiteness validation.
    dataset = SequenceDataset(features, labels)
    return dataset.features, dataset.labels


def stratified_split_indices(
    labels: npt.ArrayLike,
    *,
    validation_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return disjoint train/validation indices stratified by class.

    Every class with at least two samples is represented in both partitions.
    Singleton classes remain in training because they cannot be split without
    removing that class from the optimization set.
    """
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("labels must be a one-dimensional array with at least two samples")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between zero and one")

    generator = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    for class_id in np.unique(values):
        indices = np.flatnonzero(values == class_id)
        generator.shuffle(indices)
        if indices.size == 1:
            train.extend(indices.tolist())
            continue
        validation_count = max(1, int(round(indices.size * validation_ratio)))
        validation_count = min(validation_count, indices.size - 1)
        validation.extend(indices[:validation_count].tolist())
        train.extend(indices[validation_count:].tolist())

    if not validation:
        raise ValueError(
            "a stratified validation split requires at least one class with two samples"
        )
    generator.shuffle(train)
    generator.shuffle(validation)
    return np.asarray(train, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def run_epoch(
    model: SignSequenceClassifier,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    sample_count = 0
    for features, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        sample_count += batch_size
    return total_loss / sample_count, correct / sample_count


def evaluate(
    model: SignSequenceClassifier,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    sample_count = 0
    with torch.no_grad():
        for features, labels in loader:
            logits, _ = model(features)
            loss = criterion(logits, labels)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            sample_count += batch_size
    return total_loss / sample_count, correct / sample_count


def predict_test_windows(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets, probabilities = [], []
    with torch.no_grad():
        for features, labels in loader:
            logits, _ = model(features)
            targets.append(labels.numpy())
            probabilities.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def build_evaluation_report(
    targets: npt.ArrayLike, probabilities: npt.ArrayLike,
    video_ids: npt.ArrayLike, class_map: dict[int, str],
) -> dict[str, object]:
    """Each source video gets one vote regardless of overlapping window count."""
    y = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float32)
    videos = np.asarray(video_ids, dtype=str)
    if y.ndim != 1 or not len(y) or scores.shape != (len(y), len(class_map)) or videos.shape != y.shape:
        raise ValueError("test targets, probabilities and video IDs must align")
    clip_targets, clip_predictions = [], []
    for video_id in dict.fromkeys(videos.tolist()):
        positions = np.flatnonzero(videos == video_id)
        labels = np.unique(y[positions])
        if len(labels) != 1:
            raise ValueError(f"video {video_id} has inconsistent labels")
        clip_targets.append(int(labels[0]))
        clip_predictions.append(int(scores[positions].mean(axis=0).argmax()))
    matrix = np.zeros((len(class_map), len(class_map)), dtype=np.int64)
    for actual, predicted in zip(clip_targets, clip_predictions, strict=True):
        matrix[actual, predicted] += 1
    support, predicted_count, correct = matrix.sum(1), matrix.sum(0), np.diag(matrix)
    precision = np.divide(correct, predicted_count, out=np.zeros(len(class_map)), where=predicted_count != 0)
    recall = np.divide(correct, support, out=np.zeros(len(class_map)), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(class_map)), where=(precision + recall) != 0)
    return {
        "test_accuracy": float(np.mean(np.asarray(clip_targets) == clip_predictions)),
        "test_video_count": len(clip_targets), "test_window_count": len(y),
        "window_accuracy": float(np.mean(y == scores.argmax(1))),
        "class_order": [class_map[index] for index in sorted(class_map)],
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            class_map[index]: {"precision": float(precision[index]), "recall": float(recall[index]),
                               "f1": float(f1[index]), "support": int(support[index])}
            for index in sorted(class_map)
        },
    }


def macro_f1_from_report(report: dict[str, object]) -> float:
    return float(np.mean([values["f1"] for values in report["per_class"].values()]))


def checkpoint_selection_rank(report: dict[str, object], validation_loss: float) -> tuple[int, float, float]:
    coverage = all(values["recall"] > 0 for values in report["per_class"].values())
    return int(coverage), macro_f1_from_report(report), -validation_loss


def print_evaluation_report(report: dict[str, object]) -> None:
    print(f"Held-out test accuracy (per video): {report['test_accuracy']:.2%}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(" " * 12 + "".join(f"{name:>11}" for name in report["class_order"]))
    for name, row in zip(report["class_order"], report["confusion_matrix"], strict=True):
        print(f"{name:>12}" + "".join(f"{count:>11}" for count in row))


def train_classifier(
    dataset_h5: str | Path,
    class_map_path: str | Path,
    output_weights: str | Path,
    *,
    batch_size: int = 8,
    epochs: int = 60,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    patience: int = 20,
    min_epochs: int = 30,
    seed: int = 42,
    metadata_csv: str | Path | None = None,
    optimized_model_out: str | Path | None = None,
    report_out: str | Path = PROJECT_ROOT / "reports" / "evaluation_metrics.json",
) -> Path:
    """Production training is locked to the aligned two-layer BiLSTM."""
    from scripts.aligned_bilstm_training import run_aligned_bilstm

    return run_aligned_bilstm(
        dataset_h5, class_map_path, output_weights,
        batch_size=batch_size, epochs=epochs, learning_rate=learning_rate,
        weight_decay=weight_decay, patience=patience, min_epochs=min_epochs,
        seed=seed, metadata_csv=metadata_csv,
        optimized_model_out=optimized_model_out, report_out=report_out,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ASL sequence classifier.")
    parser.add_argument(
        "--dataset-h5",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "dataset.h5",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=PROJECT_ROOT / "configs" / "class_map.json",
    )
    parser.add_argument(
        "--output-weights",
        type=Path,
        default=PROJECT_ROOT / "weights" / "best_model.pth",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, default=PROJECT_ROOT / "reports" / "evaluation_metrics.json")
    parser.add_argument("--optimized-model-out", type=Path, default=PROJECT_ROOT / "weights" / "optimized_model.pt")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    train_classifier(
        args.dataset_h5,
        args.class_map,
        args.output_weights,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        patience=args.patience, min_epochs=args.min_epochs, metadata_csv=args.metadata_csv,
        report_out=args.report_out, optimized_model_out=args.optimized_model_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
