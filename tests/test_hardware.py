import pytest
from colab_agent_bridge.core.hardware import (
    SUPPORTED_GPUS,
    SUPPORTED_TPUS,
    CPU_SPEC,
    AcceleratorTier,
    HardwareSpec,
    validate_accelerator,
    recommend_model_strategy,
)


def test_validate_accelerator():
    spec_t4 = validate_accelerator("T4")
    assert spec_t4.name == "T4"
    assert spec_t4.vram_gb == 16
    assert spec_t4.is_gpu is True

    spec_a100 = validate_accelerator("A100")
    assert spec_a100.name == "A100"
    assert spec_a100.vram_gb == 40

    spec_tpu = validate_accelerator("v5e1")
    assert spec_tpu.is_tpu is True

    with pytest.raises(ValueError):
        validate_accelerator("UNKNOWN_GPU")


def test_recommend_model_strategy_t4():
    # Large 32B model on T4 should recommend quantization or larger GPU
    rec = recommend_model_strategy("Qwen/Qwen2.5-Coder-32B-Instruct", "T4")
    assert rec["needs_quantization"] is True
    assert "awq" in rec["recommended_quantization"] or "gguf" in rec["recommended_quantization"]


def test_recommend_model_strategy_a100():
    # 7B model on A100 needs no quantization
    rec = recommend_model_strategy("Qwen/Qwen2.5-Coder-7B-Instruct", "A100")
    assert rec["needs_quantization"] is False


def test_validate_accelerator_cpu():
    spec_cpu = validate_accelerator("CPU")
    assert spec_cpu == CPU_SPEC
    assert spec_cpu.name == "CPU"
    assert spec_cpu.vram_gb == 0
    assert spec_cpu.is_gpu is False
    assert spec_cpu.is_tpu is False

    spec_none = validate_accelerator(None)
    assert spec_none == CPU_SPEC


def test_validate_accelerator_case_insensitive():
    spec_t4_lower = validate_accelerator("t4")
    assert spec_t4_lower.name == "T4"

    spec_l4_lower = validate_accelerator("l4")
    assert spec_l4_lower.name == "L4"

    spec_v5e_upper = validate_accelerator("V5E1")
    assert spec_v5e_upper.name == "v5e1"
    assert spec_v5e_upper.is_tpu is True

    spec_v6e = validate_accelerator("v6e1")
    assert spec_v6e.name == "v6e1"
    assert spec_v6e.vram_gb == 32


def test_recommend_model_strategy_cpu():
    rec = recommend_model_strategy("Qwen/Qwen2.5-Coder-7B-Instruct", "CPU")
    assert rec["needs_quantization"] is True
    assert rec["recommended_backend"] == "llama_cpp"
    assert "gguf" in rec["recommended_quantization"]
    assert "warning" in rec and rec["warning"] is not None


def test_recommend_model_strategy_70b_on_a100():
    rec = recommend_model_strategy("meta-llama/Llama-3.1-70B-Instruct", "A100")
    assert rec["needs_quantization"] is True
    assert rec["warning"] is not None


def test_accelerator_tier_enum():
    assert AcceleratorTier.FREE.value == "free"
    assert AcceleratorTier.PRO.value == "pro"
    assert AcceleratorTier.PRO_PLUS.value == "pro_plus"
