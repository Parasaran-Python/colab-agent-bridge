from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict


class AcceleratorTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    PRO_PLUS = "pro_plus"


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    vram_gb: int
    is_gpu: bool
    is_tpu: bool
    description: str


SUPPORTED_GPUS: Dict[str, HardwareSpec] = {
    "T4": HardwareSpec(
        name="T4",
        vram_gb=16,
        is_gpu=True,
        is_tpu=False,
        description="NVIDIA T4 16GB (Available on Free & Paid tiers)",
    ),
    "L4": HardwareSpec(
        name="L4",
        vram_gb=24,
        is_gpu=True,
        is_tpu=False,
        description="NVIDIA L4 24GB (Colab Pro / Compute Units)",
    ),
    "G4": HardwareSpec(
        name="G4",
        vram_gb=16,
        is_gpu=True,
        is_tpu=False,
        description="NVIDIA G4",
    ),
    "A100": HardwareSpec(
        name="A100",
        vram_gb=40,
        is_gpu=True,
        is_tpu=False,
        description="NVIDIA A100 40GB/80GB (Colab Pro+)",
    ),
    "H100": HardwareSpec(
        name="H100",
        vram_gb=80,
        is_gpu=True,
        is_tpu=False,
        description="NVIDIA H100 80GB (Colab Pro+ Premium)",
    ),
}

SUPPORTED_TPUS: Dict[str, HardwareSpec] = {
    "v5e1": HardwareSpec(
        name="v5e1",
        vram_gb=16,
        is_gpu=False,
        is_tpu=True,
        description="Google Cloud TPU v5e (single chip)",
    ),
    "v6e1": HardwareSpec(
        name="v6e1",
        vram_gb=32,
        is_gpu=False,
        is_tpu=True,
        description="Google Cloud TPU v6e (Trillium)",
    ),
}

CPU_SPEC = HardwareSpec(
    name="CPU",
    vram_gb=0,
    is_gpu=False,
    is_tpu=False,
    description="Standard or High-Mem CPU VM",
)


def validate_accelerator(accel: Optional[str]) -> HardwareSpec:
    if not accel or accel.upper() == "CPU":
        return CPU_SPEC
    accel_upper = accel.upper()
    for name, spec in SUPPORTED_GPUS.items():
        if name.upper() == accel_upper:
            return spec
    accel_lower = accel.lower()
    for name, spec in SUPPORTED_TPUS.items():
        if name.lower() == accel_lower:
            return spec
    raise ValueError(
        f"Unsupported accelerator '{accel}'. Supported GPUs: {list(SUPPORTED_GPUS.keys())}, TPUs: {list(SUPPORTED_TPUS.keys())}"
    )


def recommend_model_strategy(model_name: str, accel: Optional[str]) -> dict:
    spec = validate_accelerator(accel)
    name_lower = model_name.lower()

    # Heuristic parameter detection from model name
    param_size = 7
    if "70b" in name_lower or "72b" in name_lower:
        param_size = 70
    elif "32b" in name_lower or "33b" in name_lower or "34b" in name_lower:
        param_size = 32
    elif "14b" in name_lower:
        param_size = 14
    elif "1.5b" in name_lower or "1b" in name_lower or "3b" in name_lower:
        param_size = 3
    elif "7b" in name_lower or "8b" in name_lower:
        param_size = 8

    # FP16 requires roughly 2GB VRAM per billion params + 2-4GB KV cache overhead
    fp16_vram_needed = param_size * 2 + 4

    if spec.vram_gb == 0:  # CPU
        return {
            "needs_quantization": True,
            "recommended_backend": "llama_cpp",
            "recommended_quantization": ["gguf", "q4_k_m"],
            "warning": "Running on CPU; model inference will be slow without GPU acceleration.",
        }

    if spec.vram_gb < fp16_vram_needed:
        return {
            "needs_quantization": True,
            "recommended_backend": "vllm" if spec.vram_gb >= 16 and param_size <= 32 else "llama_cpp",
            "recommended_quantization": ["awq", "gptq", "gguf"],
            "warning": f"Model {model_name} (~{param_size}B) exceeds FP16 capacity of {spec.name} ({spec.vram_gb}GB VRAM). Quantization required.",
        }

    return {
        "needs_quantization": False,
        "recommended_backend": "vllm",
        "recommended_quantization": [],
        "warning": None,
    }
