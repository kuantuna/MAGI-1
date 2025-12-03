import datetime
import os

import numpy as np
import torch
import torch.distributed as dist
from inference.infra.distributed import parallel_state
from inference.quantizers import ExperimentContext, KVQuantizer, QDType, Granularity, QuantLocation
from inference.common import (
    EngineConfig,
    ModelConfig,
    ModelMetaArgs,
    PackedCoreAttnParams,
    PackedCrossAttnParams,
)
from inference.model.dit.dit_module import (
    FullyParallelAttention,
    FusedLayerNorm,
)

MODEL_CONFIG = ModelConfig(
    model_name="videodit_ardf",
    num_layers=34,
    hidden_size=3072,
    ffn_hidden_size=12288,
    num_attention_heads=24,
    num_query_groups=8,
    kv_channels=128,
    layernorm_epsilon=1e-06,
    apply_layernorm_1p=True,
    x_rescale_factor=1,
    half_channel_vae=False,
    params_dtype=torch.bfloat16,
    patch_size=2,
    t_patch_size=1,
    in_channels=16,
    out_channels=16,
    cond_hidden_ratio=0.25,
    caption_channels=4096,
    caption_max_length=800,
    xattn_cond_hidden_ratio=1.0,
    cond_gating_ratio=1.0,
    gated_linear_unit=False,
)

ENGINE_CONFIG = EngineConfig(
    distributed_backend="nccl",
    distributed_timeout_minutes=15,
    pp_size=1,
    cp_size=1,
    cp_strategy="cp_ulysses",
    ulysses_overlap_degree=1,
    fp8_quant=False,
    distill_nearly_clean_chunk_threshold=0.3,
    shortcut_mode="8,16,16",
    distill=True,
    kv_offload=True,
    enable_cuda_graph=False,
)


def _init_distributed():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=1,
            rank=0,
            timeout=datetime.timedelta(minutes=30),
        )

    parallel_state.initialize_model_parallel(
        tp_size=1,
        pp_size=1,
        cp_size=1,
        distributed_timeout_minutes=30,
        order="tp-cp-pp-dp",
    )

def _destroy_distributed():
    parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def init_layernorms(module):
    if isinstance(module, FusedLayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)

def test_core_attn_patch():
    _init_distributed()
    torch.random.manual_seed(42)
    device = torch.device("cuda")
    layer_number = 0
    dummy_kv_quantizer = KVQuantizer.from_cfg(
        key_qdtype=QDType.INT8,
        key_granularity=Granularity.PER_TENSOR,
        key_location=QuantLocation.POST_ROPE,
        value_qdtype=QDType.INT8,
        value_granularity=Granularity.PER_TENSOR,
        value_location=QuantLocation.POST_ROPE,
    )

    dummy_experiment_ctx = ExperimentContext(name="test_core_attn_patch", kv_quantizer=dummy_kv_quantizer)
    fpa = FullyParallelAttention(MODEL_CONFIG, ENGINE_CONFIG, dummy_experiment_ctx, layer_number)
    fpa.to(device=device).eval()
    fpa.apply(init_layernorms)
    bs = 1
    meta_args = ModelMetaArgs(
        H=45,
        W=45,
        cp_pad_size=None,
        cp_split_sizes=None,
        slice_point=0,
        denoising_range_num=1,
        range_num=1,
        extract_prefix_video_feature=False,
        fwd_extra_1st_chunk=False,
        distill_nearly_clean_chunk=False,
        clip_token_nums=12150,
        enable_cuda_graph=False,
        core_attn_params=PackedCoreAttnParams(
            q_range=torch.tensor([[0, 12150]], device=device, dtype=torch.int32),
            k_range=torch.tensor([[0, 12150]], device=device, dtype=torch.int32),
            np_q_range=np.array([[0, 12150]], dtype=np.int32),
            np_k_range=np.array([[0, 12150]], dtype=np.int32),
            max_seqlen_q=12150,
            max_seqlen_k=12150,
        ),
        cross_attn_params=PackedCrossAttnParams(
            q_ranges=torch.tensor([[0, 12150]], device=device, dtype=torch.int32),
            kv_ranges=torch.tensor([[0, 91]], device=device, dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 12150], device=device, dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, 91], device=device, dtype=torch.int32),
            max_seqlen_q=12150,
            max_seqlen_kv=800,
        ),
    )

    query = torch.randn((12150, 24, 128), dtype=torch.bfloat16).to(device)
    key = torch.randn((12150, 8, 128), dtype=torch.bfloat16).to(device)
    value = torch.randn((12150, 8, 128), dtype=torch.bfloat16).to(device)
    original_out = fpa.core_attention(query, key, value, bs, meta_args)
    patch_out = fpa.core_attention(query, key, value, bs, meta_args, return_attn_weights=True)
    torch.testing.assert_close(original_out, patch_out, atol=1e-2, rtol=1e-2)

    _destroy_distributed()


def test_core_attn_patch_n2():
    _init_distributed()
    torch.random.manual_seed(42)
    device = torch.device("cuda")
    layer_number = 0
    dummy_kv_quantizer = KVQuantizer.from_cfg(
        key_qdtype=QDType.INT8,
        key_granularity=Granularity.PER_TENSOR,
        key_location=QuantLocation.POST_ROPE,
        value_qdtype=QDType.INT8,
        value_granularity=Granularity.PER_TENSOR,
        value_location=QuantLocation.POST_ROPE,
    )

    dummy_experiment_ctx = ExperimentContext(name="test_core_attn_patch", kv_quantizer=dummy_kv_quantizer)
    fpa = FullyParallelAttention(MODEL_CONFIG, ENGINE_CONFIG, dummy_experiment_ctx, layer_number)
    fpa.to(device=device).eval()
    fpa.apply(init_layernorms)
    bs = 1
    meta_args = ModelMetaArgs(
        H=45,
        W=45,
        cp_pad_size=None,
        cp_split_sizes=None,
        slice_point=0,
        denoising_range_num=2,
        range_num=1,
        extract_prefix_video_feature=False,
        fwd_extra_1st_chunk=False,
        distill_nearly_clean_chunk=False,
        clip_token_nums=12150,
        enable_cuda_graph=False,
        core_attn_params=PackedCoreAttnParams(
            q_range=torch.tensor([[0, 12150], [12150, 24300]], device=device, dtype=torch.int32),
            k_range=torch.tensor([[0, 12150], [0, 24300]], device=device, dtype=torch.int32),
            np_q_range=np.array([[0, 12150], [12150, 24300]], dtype=np.int32),
            np_k_range=np.array([[0, 12150], [0, 24300]], dtype=np.int32),
            max_seqlen_q=12150,
            max_seqlen_k=24300,
        ),
        cross_attn_params=PackedCrossAttnParams(
            q_ranges=torch.tensor([[0, 12150]], device=device, dtype=torch.int32),
            kv_ranges=torch.tensor([[0, 91]], device=device, dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, 12150], device=device, dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, 91], device=device, dtype=torch.int32),
            max_seqlen_q=12150,
            max_seqlen_kv=800,
        ),
    )

    query = torch.randn((24300, 24, 128), dtype=torch.bfloat16).to(device)
    key = torch.randn((24300, 8, 128), dtype=torch.bfloat16).to(device)
    value = torch.randn((24300, 8, 128), dtype=torch.bfloat16).to(device)
    original_out = fpa.core_attention(query, key, value, bs, meta_args)

    patch_out = fpa.core_attention(query, key, value, bs, meta_args, return_attn_weights=True)
    torch.testing.assert_close(original_out, patch_out, atol=1e-2, rtol=1e-2)

    _destroy_distributed()


def test_core_attn_patch_rectangular_mask():
    """
    Make Lq != Lk so the mask is rectangular.
    The test is tiny (Lq=48, Lk=60) and runs on CPU or CUDA.
    """
    _init_distributed()
    torch.random.manual_seed(42)
    device = torch.device("cuda")
    layer_number = 0
    dummy_kv_quantizer = KVQuantizer.from_cfg(
        key_qdtype=QDType.INT8,
        key_granularity=Granularity.PER_TENSOR,
        key_location=QuantLocation.POST_ROPE,
        value_qdtype=QDType.INT8,
        value_granularity=Granularity.PER_TENSOR,
        value_location=QuantLocation.POST_ROPE,
    )

    dummy_experiment_ctx = ExperimentContext(name="test_core_attn_patch", kv_quantizer=dummy_kv_quantizer)
    fpa = FullyParallelAttention(MODEL_CONFIG, ENGINE_CONFIG, dummy_experiment_ctx, layer_number)
    fpa.to(device=device).eval()
    fpa.apply(init_layernorms)

    # ------------ hyper-params (tiny) ---------------------------------------
    bs = 1  # batch
    Hq = 4  # query heads
    Hkv = 2  # key/value heads  (GQA -> repeat by 2×)
    D = 16  # head dim
    Lq = 48  # query tokens
    Lk = 60  # key/value tokens (Lk > Lq triggers the old crash)

    # ------------ random Q/K/V  --------------------------------------------
    query = torch.randn((Lq * bs, Hq, D), dtype=torch.bfloat16, device=device)
    key = torch.randn((Lk * bs, Hkv, D), dtype=torch.bfloat16, device=device)
    value = torch.randn((Lk * bs, Hkv, D), dtype=torch.bfloat16, device=device)

    # ------------ meta-args -------------------------------------------------
    q_range_t = torch.tensor([[0, Lq]], device=device, dtype=torch.int32)
    k_range_t = torch.tensor([[0, Lk]], device=device, dtype=torch.int32)

    core_params = PackedCoreAttnParams(
        q_range=q_range_t,
        k_range=k_range_t,
        np_q_range=np.array([[0, Lq]], dtype=np.int32),
        np_k_range=np.array([[0, Lk]], dtype=np.int32),
        max_seqlen_q=Lq,
        max_seqlen_k=Lk,
    )
    # dummy cross-attn params (unused by _patched_core_attention)
    dummy_cross = PackedCrossAttnParams(
        q_ranges=torch.tensor([[0, 1]], device=device, dtype=torch.int32),
        kv_ranges=torch.tensor([[0, 1]], device=device, dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], device=device, dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 1], device=device, dtype=torch.int32),
        max_seqlen_q=1,
        max_seqlen_kv=1,
    )
    meta_args = ModelMetaArgs(
        H=45,
        W=45,
        cp_pad_size=None,
        cp_split_sizes=None,
        slice_point=0,
        denoising_range_num=1,
        range_num=1,
        extract_prefix_video_feature=False,
        fwd_extra_1st_chunk=False,
        distill_nearly_clean_chunk=False,
        clip_token_nums=12150,
        enable_cuda_graph=False,
        core_attn_params=core_params,
        cross_attn_params=dummy_cross,
    )

    expected = fpa.core_attention(query, key, value, bs, meta_args)
    # ------------ run patched attention ------------------------------------
    patch_out = fpa.core_attention(query, key, value, bs, meta_args, return_attn_weights=True)
    assert patch_out.shape == (Lq * bs, Hq, D)
    assert torch.isfinite(patch_out).all()

    torch.testing.assert_close(patch_out, expected, atol=1e-2, rtol=1e-2)
    _destroy_distributed()