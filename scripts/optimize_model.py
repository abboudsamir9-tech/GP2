"""Quantize, export, and benchmark the production ASL sequence model."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from asl_stereo.models import (  # noqa: E402
    SignSequenceClassifier,
    load_checkpoint,
    load_class_map,
    load_model_weights,
)

INPUT_SHAPE = (1, 45, 138)
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True, slots=True)
class BenchmarkStats:
    file_size_mb: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    peak_ram_mb: float


class _TraceableQuantizedWrapper(nn.Module):
    """Expose a traceable boundary around Torch 2.3 dynamic-quantized RNNs.

    Dynamic-quantized LSTM modules contain Python shape guards that reject
    tracing proxies in Torch 2.3. Scripting those internal operators first and
    tracing this fixed-shape production boundary retains INT8 execution while
    still producing a traced deployment artifact.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = torch.jit.script(model)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(inputs)


def validate_input_tensor(tensor: torch.Tensor) -> None:
    """Enforce the exact production input contract."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("representative input must be a torch.Tensor")
    if tuple(tensor.shape) != INPUT_SHAPE:
        raise ValueError(f"representative input must have shape {INPUT_SHAPE}")
    if tensor.dtype != torch.float32:
        raise TypeError("representative input must use torch.float32")
    if tensor.device.type != "cpu":
        raise ValueError("representative input must reside on CPU")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("representative input must contain only finite values")


def load_fp32_classifier(
    checkpoint_path: str | Path,
) -> tuple[SignSequenceClassifier, dict[str, Any]]:
    """Reconstruct the production model from a safe structured checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path)
    required = {
        "model_state_dict",
        "num_classes",
        "window_size",
        "feature_dim",
        "model_type",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"structured checkpoint is missing keys: {sorted(missing)}")
    if int(checkpoint["window_size"]) != INPUT_SHAPE[1]:
        raise ValueError("checkpoint window_size must be 45")
    if int(checkpoint["feature_dim"]) != INPUT_SHAPE[2]:
        raise ValueError("checkpoint feature_dim must be 138")
    if str(checkpoint["model_type"]) != "bilstm_attention":
        raise ValueError("unsupported checkpoint model_type")

    model = SignSequenceClassifier(num_classes=int(checkpoint["num_classes"]))
    load_model_weights(model, checkpoint_path, strict=True)
    model.cpu().eval()
    metadata = {
        "num_classes": int(checkpoint["num_classes"]),
        "window_size": int(checkpoint["window_size"]),
        "feature_dim": int(checkpoint["feature_dim"]),
        "model_type": str(checkpoint["model_type"]),
    }
    return model, metadata


def quantize_dynamic_model(model: nn.Module) -> nn.Module:
    """Return a CPU-only dynamic INT8 copy targeting Linear and LSTM layers."""
    source = copy.deepcopy(model).cpu().eval()
    quantized = torch.ao.quantization.quantize_dynamic(
        source,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8,
        inplace=False,
    )
    return quantized.eval()


def export_torchscript(
    model: nn.Module,
    output_path: str | Path,
    representative_input: torch.Tensor,
    *,
    class_map: Mapping[int | str, str],
    model_type: str = "bilstm_attention_dynamic_int8",
) -> torch.jit.ScriptModule:
    """Trace and save the optimized model with its inference contract embedded."""
    validate_input_tensor(representative_input)
    normalized_map = load_class_map(class_map)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "model_type": model_type,
        "num_classes": len(normalized_map),
        "window_size": INPUT_SHAPE[1],
        "feature_dim": INPUT_SHAPE[2],
        "class_map": {str(index): label for index, label in normalized_map.items()},
    }
    trace_target = _TraceableQuantizedWrapper(model.cpu().eval()).eval()
    with torch.no_grad():
        traced = torch.jit.trace(
            trace_target, representative_input, check_trace=True, strict=True
        )
    torch.jit.save(
        traced,
        str(output),
        _extra_files={METADATA_FILENAME: json.dumps(metadata, sort_keys=True)},
    )
    return traced


def benchmark_model(
    model: nn.Module,
    representative_input: torch.Tensor,
    artifact_path: str | Path,
    *,
    warmup: int = 50,
    iterations: int = 200,
) -> BenchmarkStats:
    """Measure CPU latency and peak Python allocation for a model artifact."""
    validate_input_tensor(representative_input)
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    model = model.cpu().eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(representative_input)

        timings_ms: list[float] = []
        tracemalloc.start()
        try:
            for _ in range(iterations):
                started = time.perf_counter()
                model(representative_input)
                timings_ms.append((time.perf_counter() - started) * 1_000.0)
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    return BenchmarkStats(
        file_size_mb=artifact.stat().st_size / (1024.0 * 1024.0),
        mean_ms=float(statistics.fmean(timings_ms)),
        median_ms=float(statistics.median(timings_ms)),
        p95_ms=float(np.percentile(np.asarray(timings_ms), 95)),
        peak_ram_mb=peak_bytes / (1024.0 * 1024.0),
    )


def format_comparison_table(
    baseline: BenchmarkStats, optimized: BenchmarkStats
) -> str:
    headers = ("Model", "Size MB", "Mean ms", "Median ms", "p95 ms", "Peak RAM MB")
    rows = (
        ("FP32", *(_format_stat(value) for value in asdict(baseline).values())),
        ("INT8 TorchScript", *(_format_stat(value) for value in asdict(optimized).values())),
    )
    widths = [
        max(len(headers[index]), *(len(str(row[index])) for row in rows))
        for index in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    format_row = lambda row: "| " + " | ".join(
        str(value).ljust(widths[index]) for index, value in enumerate(row)
    ) + " |"
    return "\n".join((separator, format_row(headers), separator, *(format_row(row) for row in rows), separator))


def _format_stat(value: float) -> str:
    return f"{value:.3f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and benchmark the dynamic-INT8 ASL model."
    )
    parser.add_argument(
        "--input-weights", type=Path, default=PROJECT_ROOT / "weights" / "best_model.pth"
    )
    parser.add_argument(
        "--output-model", type=Path, default=PROJECT_ROOT / "weights" / "optimized_model.pt"
    )
    parser.add_argument(
        "--class-map", type=Path, default=PROJECT_ROOT / "configs" / "class_map.json"
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT_ROOT / "reports" / "quantization_benchmark.json",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    class_map = load_class_map(args.class_map)
    fp32_model, checkpoint_metadata = load_fp32_classifier(args.input_weights)
    if len(class_map) != checkpoint_metadata["num_classes"]:
        raise ValueError("class map size does not match checkpoint num_classes")

    representative_input = torch.randn(*INPUT_SHAPE, dtype=torch.float32)
    int8_model = quantize_dynamic_model(fp32_model)
    export_torchscript(
        int8_model,
        args.output_model,
        representative_input,
        class_map=class_map,
    )
    extra_files = {METADATA_FILENAME: ""}
    production_model = torch.jit.load(
        str(args.output_model), map_location="cpu", _extra_files=extra_files
    ).eval()

    baseline = benchmark_model(
        fp32_model,
        representative_input,
        args.input_weights,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    optimized = benchmark_model(
        production_model,
        representative_input,
        args.output_model,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    report = {
        "input_contract": {"shape": list(INPUT_SHAPE), "dtype": "float32", "device": "cpu"},
        "benchmark": {"warmup": args.warmup, "iterations": args.iterations},
        "baseline_fp32": asdict(baseline),
        "optimized_int8_torchscript": asdict(optimized),
        "latency_gate_ms": 80.0,
        "latency_gate_passed": optimized.p95_ms < 80.0,
        "torch_version": torch.__version__,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_comparison_table(baseline, optimized))
    if optimized.p95_ms >= 80.0:
        raise AssertionError(
            f"optimized p95 latency {optimized.p95_ms:.3f} ms must be < 80 ms"
        )
    print(f"PASS: INT8 TorchScript p95 {optimized.p95_ms:.3f} ms < 80 ms")
    print(f"Report: {args.report_out}")
    print(f"Model: {args.output_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
