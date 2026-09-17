import torch

from asl_stereo.models import InferenceEngine, SignSequenceClassifier
from scripts.optimize_model import export_torchscript, quantize_dynamic_model


def _small_model() -> SignSequenceClassifier:
    torch.manual_seed(17)
    return SignSequenceClassifier(
        num_classes=4,
        projection_features=16,
        hidden_size=16,
        dropout=0.0,
    ).eval()


def test_dynamic_int8_quantization_executes() -> None:
    quantized = quantize_dynamic_model(_small_model())
    inputs = torch.randn(1, 45, 138, dtype=torch.float32)

    with torch.no_grad():
        logits, probabilities = quantized(inputs)

    assert logits.shape == probabilities.shape == (1, 4)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1))
    module_names = {type(module).__module__ for module in quantized.modules()}
    assert any("quantized.dynamic" in name for name in module_names)


def test_torchscript_export_preserves_output_contract(tmp_path) -> None:
    inputs = torch.randn(1, 45, 138, dtype=torch.float32)
    output_path = tmp_path / "optimized_model.pt"
    export_torchscript(
        quantize_dynamic_model(_small_model()),
        output_path,
        inputs,
        class_map={0: "A", 1: "B", 2: "C", 3: "D"},
    )

    loaded = torch.jit.load(str(output_path), map_location="cpu")
    with torch.no_grad():
        logits, probabilities = loaded(inputs)

    assert logits.shape == probabilities.shape == (1, 4)
    assert output_path.is_file()


def test_int8_probability_distribution_remains_close_to_fp32() -> None:
    fp32 = _small_model()
    quantized = quantize_dynamic_model(fp32)
    inputs = torch.randn(1, 45, 138, dtype=torch.float32)

    with torch.no_grad():
        _, expected = fp32(inputs)
        _, actual = quantized(inputs)

    similarity = torch.nn.functional.cosine_similarity(expected, actual).item()
    max_error = torch.max(torch.abs(expected - actual)).item()
    assert similarity >= 0.99
    assert max_error <= 0.05


def test_inference_engine_loads_torchscript_transparently(tmp_path) -> None:
    inputs = torch.randn(1, 45, 138, dtype=torch.float32)
    output_path = tmp_path / "optimized_model.pt"
    class_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    export_torchscript(
        quantize_dynamic_model(_small_model()),
        output_path,
        inputs,
        class_map=class_map,
    )

    placeholder = _small_model()
    engine = InferenceEngine(
        placeholder, class_map, checkpoint_path=output_path
    )
    result = engine.predict(inputs)

    assert result.predicted_gloss in set(class_map.values())
    assert len(result.probabilities) == 4
    assert result.latency_ms > 0.0
