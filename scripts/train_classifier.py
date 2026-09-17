"""Train the production BiLSTM-attention classifier on the compact HDF5 set."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from asl_stereo.contracts import FEATURE_COUNT
from asl_stereo.models import SignSequenceClassifier
from asl_stereo.models.checkpoint import (
    load_class_map,
    save_training_checkpoint,
)


WINDOW_SIZE = 45


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: npt.ArrayLike, labels: npt.ArrayLike) -> None:
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

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[index]),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


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


def train_classifier(
    dataset_h5: str | Path,
    class_map_path: str | Path,
    output_weights: str | Path,
    *,
    batch_size: int = 8,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> Path:
    if batch_size <= 0 or epochs <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("training hyperparameters are invalid")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    features, labels = load_feature_artifact(dataset_h5)
    class_map = load_class_map(class_map_path)
    if len(class_map) < 2:
        raise ValueError("at least two classes are required for classification")
    if labels.size == 0 or labels.min() < 0 or labels.max() >= len(class_map):
        raise ValueError("dataset labels are incompatible with class_map.json")

    train_indices, validation_indices = stratified_split_indices(labels, seed=seed)
    dataset = SequenceDataset(features, labels)
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SignSequenceClassifier(
        num_classes=len(class_map), input_features=FEATURE_COUNT
    ).to(torch.device("cpu"))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    print(f"CPU training samples: {len(train_indices)}")
    print(f"Validation samples:   {len(validation_indices)}")
    print("+-------+------------+-----------+------------+-----------+")
    print("| Epoch | Train loss | Train acc | Val loss   | Val acc   |")
    print("+-------+------------+-----------+------------+-----------+")
    best_accuracy = -1.0
    best_loss = float("inf")
    output = Path(output_weights)
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer
        )
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion
        )
        print(
            f"| {epoch:>5} | {train_loss:>10.4f} | {train_accuracy:>8.2%} | "
            f"{validation_loss:>10.4f} | {validation_accuracy:>8.2%} |",
            flush=True,
        )
        improved = validation_accuracy > best_accuracy or (
            validation_accuracy == best_accuracy and validation_loss < best_loss
        )
        if improved:
            best_accuracy = validation_accuracy
            best_loss = validation_loss
            save_training_checkpoint(
                model,
                output,
                num_classes=len(class_map),
                window_size=WINDOW_SIZE,
                feature_dim=FEATURE_COUNT,
                epoch=epoch,
                best_val_acc=validation_accuracy,
            )
    print("+-------+------------+-----------+------------+-----------+")
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Checkpoint: {output}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ASL sequence classifier.")
    parser.add_argument(
        "--dataset-h5",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "asl_dataset.h5",
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
