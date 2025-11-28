import torch

def quantize(x, output_dtype):
    if output_dtype == torch.int8:
        x = x.float()
        scale = 127.0 / x.abs().max()
        x_quantized = (x * scale).round().clamp(-128, 127).to(torch.int8)
        return x_quantized, scale
    elif output_dtype == torch.float8_e4m3fn:
        x = x.float()
        max_fp8 = torch.finfo(torch.float8_e4m3fn).max
        scale = max_fp8 / x.abs().max()
        x_scaled = (x * scale)
        x_quantized = x_scaled.to(torch.float8_e4m3fn)
        return x_quantized, scale
    elif output_dtype == torch.float8_e5m2:
        x = x.float()
        max_fp8 = torch.finfo(torch.float8_e5m2).max
        scale = max_fp8 / x.abs().max()
        x_scaled = (x * scale)
        x_quantized = x_scaled.to(torch.float8_e5m2)
        return x_quantized, scale
    else:
        raise ValueError(f"Unsupported output dtype: {output_dtype}")

def dequantize(x_quantized, scale, output_dtype):
    x_dequantized = (x_quantized.float() / scale).to(output_dtype)
    return x_dequantized