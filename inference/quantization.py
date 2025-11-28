import torch
import logging

logger = logging.getLogger(__name__)

def quantize(x: torch.Tensor, qdtype: torch.dtype):
    """
    Quantize x to qdtype (torch.int8, torch.float8_e4m3fn, torch.float8_e5m2).

    Returns:
        q: quantized tensor in qdtype
        scale: float (step size s), such that x ≈ q * scale after dequantization
    """
    x = x.float()
    max_abs = x.abs().max().item()  # scalar float

    # Handle all-zero tensor: avoid division by zero.
    # x is already exactly zero everywhere, so its quantized form is also all zeros.
    if max_abs == 0.0:
        logger.warning("Input tensor is all zeros; returning zero quantized tensor.")
        scale = 1.0  # neutral; never actually used to reconstruct non-zero values
        q = x.new_zeros(x.shape, dtype=qdtype)
        return q, scale

    if qdtype == torch.int8:
        # Symmetric int8 quantization: q in [-128, 127]
        # s = max_abs / 127  →  x / s ∈ [-127, 127]
        scale = max_abs / 127.0  # step size s
        q = (x / scale).round().clamp(-128, 127).to(torch.int8)
        return q, scale

    elif qdtype == torch.float8_e4m3fn:
        # Use FP8 dynamic range with a scale factor:
        # Let max_q be the largest finite value representable in this fp8 format.
        # We choose s so that x / s fits roughly in [-max_q, max_q].
        max_q = torch.finfo(torch.float8_e4m3fn).max
        scale = max_abs / max_q   # s = max_abs / max_q
        q = (x / scale).to(torch.float8_e4m3fn)
        return q, scale

    elif qdtype == torch.float8_e5m2:
        max_q = torch.finfo(torch.float8_e5m2).max
        scale = max_abs / max_q   # s = max_abs / max_q
        q = (x / scale).to(torch.float8_e5m2)
        return q, scale

    else:
        raise ValueError(f"Unsupported quantized dtype: {qdtype}")


def dequantize(q: torch.Tensor, scale: float, dtype: torch.dtype):
    """
    Dequantize q using the step size 'scale' returned by quantize.

    x_hat = q * scale
    """
    return (q.float() * scale).to(dtype)
