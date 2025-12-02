import torch
import logging
from typing import Protocol
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# ---- enums ----
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

class QuantLocation(Enum):
    CACHE_ONLY = "cache_only"  # only on write/read from cache

@dataclass
class KVQuantizationConfig:
    key_qdtype: QDType
    key_granularity: Granularity
    key_location: QuantLocation
    
    value_qdtype: QDType
    value_granularity: Granularity
    value_location: QuantLocation


class TensorQuantizer(Protocol):
    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ...


TENSOR_QUANTIZER_REGISTRY: dict[tuple[QDType, Granularity], type[TensorQuantizer]] = {}

def register_tensor_quantizer(qdtype: QDType, granularity: Granularity):
    def deco(cls: type[TensorQuantizer]):
        TENSOR_QUANTIZER_REGISTRY[(qdtype, granularity)] = cls
        return cls
    return deco


@register_tensor_quantizer(QDType.NONE, Granularity.PER_TENSOR)
class NoOpKVQuantizer(TensorQuantizer):
    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, torch.tensor(1.0, device=x.device, dtype=torch.float32)

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return q


@register_tensor_quantizer(QDType.INT8, Granularity.PER_TENSOR)
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


@register_tensor_quantizer(QDType.FP8_E4M3, Granularity.PER_TENSOR)
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


@register_tensor_quantizer(QDType.FP8_E5M2, Granularity.PER_TENSOR)
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


@dataclass
class KVQuantizer:
    # metadata / config aspect
    key_qdtype: QDType
    key_granularity: Granularity
    key_location: QuantLocation
    value_qdtype: QDType
    value_granularity: Granularity
    value_location: QuantLocation

    # behavior aspect
    k_quantizer: TensorQuantizer
    v_quantizer: TensorQuantizer

    @classmethod
    def from_cfg(
        cls,
        key_qdtype: QDType,
        key_granularity: Granularity,
        key_location: QuantLocation,
        value_qdtype: QDType,
        value_granularity: Granularity,
        value_location: QuantLocation,
    ) -> "KVQuantizer":
        k_quantizer = build_tensor_quantizer(key_qdtype, key_granularity)
        v_quantizer = build_tensor_quantizer(value_qdtype, value_granularity)
        return cls(
            key_qdtype=key_qdtype,
            key_granularity=key_granularity,
            key_location=key_location,
            value_qdtype=value_qdtype,
            value_granularity=value_granularity,
            value_location=value_location,
            k_quantizer=k_quantizer,
            v_quantizer=v_quantizer,
        )
    
    def quantize_k(self, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_quantizer.quantize(key)
    
    def dequantize_k(self, key: torch.Tensor, scale_k: torch.Tensor) -> torch.Tensor:
        return self.k_quantizer.dequantize(key, scale_k)

    def quantize_v(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.v_quantizer.quantize(value) 

    def dequantize_v(self, value: torch.Tensor, scale_v: torch.Tensor) -> torch.Tensor:
        return self.v_quantizer.dequantize(value, scale_v)



def build_tensor_quantizer(qdtype: QDType, granularity: Granularity) -> TensorQuantizer:
    try:
        cls = TENSOR_QUANTIZER_REGISTRY[(qdtype, granularity)]
    except KeyError:
        raise ValueError(f"No TensorQuantizer registered for {qdtype} / {granularity}")
    return cls()



@dataclass
class ExperimentContext:
    name: str
    kv_quantizer: KVQuantizer


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
