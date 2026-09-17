import json

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from asl_stereo.dataset import HDF5SequenceDataset, write_dataset_h5
from asl_stereo.models import (
    InferenceEngine,
    SignSequenceClassifier,
    load_checkpoint,
    save_class_map,
    save_training_checkpoint,
)
from scripts.train import train_one_epoch


def _make_h5(path, sample_count: int = 8):
    generator = np.random.default_rng(42)
    features = {
        partition: generator.normal(size=(sample_count, 45, 138)).astype(np.float32)
        for partition in ("train", "val", "test")
    }
    labels = {
        partition: np.arange(sample_count, dtype=np.int64) % 3
        for partition in ("train", "val", "test")
    }
    return write_dataset_h5(
        path,
        features,
        labels,
        class_mapping={"HELLO": 0, "HELP": 1, "THANK_YOU": 2},
        signer_ids={"train": ["s1"], "val": ["s2"], "test": ["s3"]},
    )


def test_hdf5_sequence_dataset_streams_batched_tensor_slices(tmp_path) -> None:
    dataset_path = _make_h5(tmp_path / "dataset.h5")
    dataset = HDF5SequenceDataset(dataset_path, "train")
    assert dataset._handle is None

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    features, labels = next(iter(loader))

    assert features.shape == (4, 45, 138)
    assert features.dtype == torch.float32
    assert labels.shape == (4,)
    assert labels.dtype == torch.int64
    assert dataset._handle is not None
    dataset.close()


def test_one_epoch_training_step_computes_loss_and_gradients() -> None:
    torch.manual_seed(3)
    features = torch.randn(4, 45, 138)
    labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(features, labels), batch_size=4)
    model = SignSequenceClassifier(
        num_classes=3, projection_features=16, hidden_size=16, dropout=0.0
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    before = model.classification_head.weight.detach().clone()

    loss, accuracy = train_one_epoch(
        model, loader, criterion, optimizer, torch.device("cpu")
    )

    assert np.isfinite(loss) and loss > 0.0
    assert 0.0 <= accuracy <= 1.0
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert not torch.equal(before, model.classification_head.weight.detach())


def test_structured_checkpoint_loads_through_inference_engine(tmp_path) -> None:
    torch.manual_seed(11)
    source = SignSequenceClassifier(
        num_classes=3, projection_features=16, hidden_size=16
    )
    checkpoint_path = save_training_checkpoint(
        source,
        tmp_path / "best_model.pth",
        num_classes=3,
        window_size=45,
        feature_dim=138,
        model_type="bilstm_attention",
        epoch=1,
        best_val_acc=0.875,
    )
    class_map_path = save_class_map(
        {0: "HELLO", 1: "HELP", 2: "THANK_YOU"},
        tmp_path / "class_map.json",
    )
    target = SignSequenceClassifier(
        num_classes=3, projection_features=16, hidden_size=16
    )

    engine = InferenceEngine(
        target,
        class_map_path,
        checkpoint_path=checkpoint_path,
    )
    checkpoint = load_checkpoint(checkpoint_path)

    assert set(checkpoint) == {
        "model_state_dict",
        "num_classes",
        "window_size",
        "feature_dim",
        "model_type",
        "epoch",
        "best_val_acc",
    }
    assert checkpoint["epoch"] == 1
    assert checkpoint["best_val_acc"] == 0.875
    assert json.loads(class_map_path.read_text(encoding="utf-8")) == {
        "0": "HELLO",
        "1": "HELP",
        "2": "THANK_YOU",
    }
    for expected, actual in zip(source.parameters(), engine.model.parameters()):
        torch.testing.assert_close(expected, actual)
