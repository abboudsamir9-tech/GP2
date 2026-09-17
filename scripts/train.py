"""Train the signer-independent BiLSTM-attention classifier."""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from asl_stereo.dataset import HDF5SequenceDataset
from asl_stereo.models import (
    SignSequenceClassifier,
    save_class_map,
    save_training_checkpoint,
)

LOGGER = logging.getLogger("asl_stereo.training")
TARGET_ACCURACY = 0.85


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ASL sequence classifier.")
    parser.add_argument(
        "--dataset-h5",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "asl_dataset.h5",
    )
    parser.add_argument(
        "--output-weights",
        type=Path,
        default=PROJECT_ROOT / "weights" / "best_model.pth",
    )
    parser.add_argument(
        "--class-map-out",
        type=Path,
        default=PROJECT_ROOT / "configs" / "class_map.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    sample_count = 0
    for features, labels in loader:
        features = features.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("training partition contains no samples")
    return total_loss / sample_count, correct / sample_count


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    sample_count = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            logits, _ = model(features)
            loss = criterion(logits, labels)
            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            sample_count += batch_size
    if sample_count == 0:
        raise ValueError("validation partition contains no samples")
    return total_loss / sample_count, correct / sample_count


def build_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup_epochs = min(5, max(1, epochs // 10))
    if epochs <= warmup_epochs:
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / warmup_epochs,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.epochs <= 0 or args.lr <= 0 or args.patience <= 0:
        raise SystemExit("batch size, epochs, learning rate, and patience must be positive")
    if args.num_workers < 0:
        raise SystemExit("num workers must be non-negative")
    set_reproducible_seed(args.seed)
    device = resolve_device(args.device)
    LOGGER.info("Training on %s", device)

    train_dataset = HDF5SequenceDataset(args.dataset_h5, "train", augment=True)
    val_dataset = HDF5SequenceDataset(args.dataset_h5, "val", augment=False)
    if not train_dataset.class_mapping:
        raise ValueError("dataset class_mapping metadata is empty")
    if train_dataset.class_mapping != val_dataset.class_mapping:
        raise ValueError("train and validation class mappings differ")
    index_to_gloss = {
        index: gloss for gloss, index in train_dataset.class_mapping.items()
    }
    save_class_map(index_to_gloss, args.class_map_out)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    model = SignSequenceClassifier(num_classes=len(index_to_gloss)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, args.epochs)

    best_val_loss = math.inf
    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    reached_target = False
    try:
        for epoch_index in range(args.epochs):
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            LOGGER.info(
                "Epoch %03d/%03d | train loss %.5f acc %.2f%% | "
                "val loss %.5f acc %.2f%% | lr %.6g",
                epoch_index + 1,
                args.epochs,
                train_loss,
                train_accuracy * 100,
                val_loss,
                val_accuracy * 100,
                optimizer.param_groups[0]["lr"],
            )
            if val_accuracy >= TARGET_ACCURACY and not reached_target:
                reached_target = True
                LOGGER.info("Validation accuracy reached the 85%% target")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
                save_training_checkpoint(
                    model,
                    args.output_weights,
                    num_classes=len(index_to_gloss),
                    window_size=45,
                    feature_dim=138,
                    model_type="bilstm_attention",
                    epoch=epoch_index + 1,
                    best_val_acc=best_val_accuracy,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    LOGGER.info(
                        "Early stopping after %d epochs without validation-loss improvement",
                        args.patience,
                    )
                    break
    finally:
        train_dataset.close()
        val_dataset.close()

    LOGGER.info("Best validation accuracy: %.2f%%", best_val_accuracy * 100)
    if not reached_target:
        LOGGER.warning("Validation accuracy did not reach the 85%% target")
    LOGGER.info("Best checkpoint: %s", args.output_weights)
    LOGGER.info("Class map: %s", args.class_map_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
