import torch
import logging
from typing import Protocol
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QDType(str, Enum):
    NONE = "none"
    INT8 = "int8"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    # later: FP4, INT4, etc.


class Granularity(str, Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    # later: PER_HEAD, PER_GROUP, PER_TOKEN, ...

@dataclass
class KVQuantizationConfig:
    # what numeric "code" we use
    key_qdtype: QDType
    value_qdtype: QDType

    # how scales are applied
    key_granularity: Granularity
    value_granularity: Granularity


class TensorQuantizer(Protocol):
    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ...


class NoOpKVQuantizer(TensorQuantizer):
    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, torch.tensor(1.0, device=x.device, dtype=torch.float32)

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return q
    
class Int8PerTensorKVQuantizer(TensorQuantizer):
    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        max_abs = x.abs().max()
        if max_abs == 0.0:
            logger.warning("Input tensor is all zeros; returning zero quantized tensor.")
            scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)  # neutral; never actually used to reconstruct non-zero values
            q = x.new_zeros(x.shape, dtype=torch.int8)
            return q, scale
        # Symmetric int8 quantization: q in [-128, 127]
        # s = max_abs / 127  →  x / s ∈ [-127, 127]
        scale = max_abs / 127.0  # step size s
        q = (x / scale).round().clamp(-128, 127).to(torch.int8)
        return q, scale
    
    def dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (q.float() * scale).to(torch.bfloat16)

class Float8E4M3PerTensorKVQuantizer(TensorQuantizer):
    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        max_abs = x.abs().max()
        if max_abs == 0.0:
            logger.warning("Input tensor is all zeros; returning zero quantized tensor.")
            scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)  # neutral; never actually used to reconstruct non-zero values
            q = x.new_zeros(x.shape, dtype=torch.float8_e4m3fn)
            return q, scale
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        fp8_min = torch.finfo(torch.float8_e4m3fn).min
        scale = max_abs / fp8_max 
        y = x / scale
        y = torch.clamp(y, fp8_min, fp8_max)
        q = y.to(torch.float8_e4m3fn)
        return q, scale
    
    def dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (q.float() * scale).to(torch.bfloat16)
    
class Float8E5M2PerTensorKVQuantizer(TensorQuantizer):
    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        max_abs = x.abs().max()
        if max_abs == 0.0:
            logger.warning("Input tensor is all zeros; returning zero quantized tensor.")
            scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)  # neutral; never actually used to reconstruct non-zero values
            q = x.new_zeros(x.shape, dtype=torch.float8_e5m2)
            return q, scale
        fp8_max = torch.finfo(torch.float8_e5m2).max
        fp8_min = torch.finfo(torch.float8_e5m2).min
        scale = max_abs / fp8_max 
        y = x / scale
        y = torch.clamp(y, fp8_min, fp8_max)
        q = y.to(torch.float8_e5m2)
        return q, scale
    
    def dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (q.float() * scale).to(torch.bfloat16)


# Registry mapping (QDType, Granularity) to TensorQuantizer classes
TENSOR_QUANTIZER_REGISTRY: dict[tuple[QDType, Granularity], type[TensorQuantizer]] = {
    (QDType.NONE, Granularity.PER_TENSOR): NoOpKVQuantizer,
    (QDType.INT8, Granularity.PER_TENSOR): Int8PerTensorKVQuantizer,
    (QDType.FP8_E4M3, Granularity.PER_TENSOR): Float8E4M3PerTensorKVQuantizer,
    (QDType.FP8_E5M2, Granularity.PER_TENSOR): Float8E5M2PerTensorKVQuantizer,
    # later:
    # (QDType.INT8, Granularity.PER_CHANNEL): Int8PerChannelKVQuantizer,
    # (QDType.INT4, Granularity.PER_TENSOR): Int4PerTensorKVQuantizer,
}



class KVQuantizer:
    def __init__(self, k_quantizer: TensorQuantizer, v_quantizer: TensorQuantizer):
        self.k_quantizer = k_quantizer
        self.v_quantizer = v_quantizer

    def quantize_k(self, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_quantizer.quantize(key)

    def dequantize_k(self, key: torch.Tensor, scale_k: torch.Tensor) -> torch.Tensor:
        return self.k_quantizer.dequantize(key, scale_k)

    def quantize_v(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.v_quantizer.quantize(value)

    def dequantize_v(self, value: torch.Tensor, scale_v: torch.Tensor) -> torch.Tensor:
        return self.v_quantizer.dequantize(value, scale_v)


def build_tensor_quantizer(
    qdtype: QDType,
    granularity: Granularity,
) -> TensorQuantizer:
    # special-case NONE if you want to ignore granularity:
    if qdtype == QDType.NONE:
        return NoOpKVQuantizer()

    key = (qdtype, granularity)
    try:
        quant_cls = TENSOR_QUANTIZER_REGISTRY[key]
    except KeyError:
        if qdtype not in {k[0] for k in TENSOR_QUANTIZER_REGISTRY}:
            raise ValueError(f"Unknown qdtype {qdtype}")
        raise ValueError(f"Unsupported granularity {granularity} for {qdtype}")

    return quant_cls()


def build_kv_quantizer(cfg: KVQuantizationConfig) -> KVQuantizer:
    k_quantizer = build_tensor_quantizer(
        cfg.key_qdtype,
        cfg.key_granularity,
    )
    v_quantizer = build_tensor_quantizer(
        cfg.value_qdtype,
        cfg.value_granularity,
    )
    return KVQuantizer(k_quantizer=k_quantizer, v_quantizer=v_quantizer)



def quantization_metrics(x, x_hat):
    err = x_hat - x

    mse     = err.pow(2).mean()
    rmse    = mse.sqrt()
    mae     = err.abs().mean()
    max_err = err.abs().max()

    x_energy = x.pow(2).mean()
    rel_mse  = mse / (x_energy + 1e-12)

    x_flat     = x.reshape(-1)
    x_hat_flat = x_hat.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(x_flat, x_hat_flat, dim=0)

    return {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
        "max_err": max_err.item(),
        "rel_mse": rel_mse.item(),
        "cosine_similarity": cos_sim.item(),
    }
