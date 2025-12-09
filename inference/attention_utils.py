import torch
import matplotlib.pyplot as plt
from typing import Optional
import os
import csv
from enum import Enum
import math


class HeadType(str, Enum):
    S_LS = "self_local_spatial"
    S_LT = "self_local_temporal"
    S_GS = "self_global_spatial"
    S_GT = "self_global_temporal"
    P_LT = "past_local_temporal"
    P_GT = "past_global_temporal"


def vis_attn_downsample(attn_score: torch.Tensor,
                                  block_q: int = 45,
                                  block_k: int = 45,
                                  n_rows: int = 4,
                                  n_cols: int = 6,
                                  save_path: Optional[str] = None):
    """
    Visualize downsampled attention for all heads in a big grid.

    Args:
        attn_score: Attention scores tensor of shape (1, n_heads, q_len, k_len)
        per_head_size: approximate size (in inches) of each subplot's side
    """
    bs, n_heads, q_len, k_len = attn_score.shape
    assert bs == 1, "This function assumes batch size = 1"

    new_q = q_len // block_q * block_q
    new_k = k_len // block_k * block_k

    head_attn = attn_score[0].to(torch.float32)          # (n_heads, q, k)
    head_attn_cropped = head_attn[:, :new_q, :new_k]

    head_attn_down = head_attn_cropped.view(
        n_heads,
        new_q // block_q, block_q,
        new_k // block_k, block_k
    ).mean(dim=(2, 4))                                   # (n_heads, Q', K')

    ratios = [head_attn_down[h].shape[1] / head_attn_down[h].shape[0] 
            for h in range(n_heads)]
    avg_ratio = sum(ratios) / len(ratios)
    fig_width = 5.0 * n_cols * avg_ratio
    fig_height = 5.0 * n_rows
    # Big figure: 4x6 with ~4" per head -> 24" x 16" image
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=True,  # handles colorbars nicely
    )
    axes = axes.flatten()

    for h, ax in enumerate(axes):
        if h < n_heads:
            data = head_attn_down[h].cpu().numpy()
            q_blocks, k_blocks = data.shape   # Q', K'
            im = ax.imshow(data, aspect='auto')
            ax.set_title(f'Head {h} (downsampled)', fontsize=12)
            ax.set_xlabel('key blocks', fontsize=10)
            ax.set_ylabel('query blocks', fontsize=10)

            # one slim colorbar per subplot
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

            # make each axes have the same aspect ratio as the data
            ax.set_box_aspect(q_blocks / k_blocks)
        else:
            ax.axis('off')

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)



def get_video_grid_info(
    query_len: int,
    key_len: int,
    current_chunk_idx: int,
    device: torch.device,
    H: int = 45,
    W: int = 45,
):
    """
    Compute per-token metadata for autoregressive video attention.

    Returns:
        q_chunk:  [Nq]   chunk id for each query token (all = current_chunk_idx)
        q_frame:  [Nq]   frame index within chunk (0..frames_per_chunk-1)
        q_y:      [Nq]
        q_x:      [Nq]
        k_chunk:  [Nk]   chunk id for each key token
        k_frame:  [Nk]   frame index within chunk
        k_y:      [Nk]
        k_x:      [Nk]
    """
    frames_per_chunk = 6
    tokens_per_frame = H * W

    # -------- Queries: always from a single chunk --------
    Nq = query_len
    q_idx = torch.arange(Nq, device=device)  # [0..Nq-1]

    # queries cover exactly one chunk, so frame index is within chunk
    q_frame = q_idx // tokens_per_frame              # [Nq], 0..frames_per_chunk-1
    q_within_frame = q_idx % tokens_per_frame        # [Nq]
    q_y = q_within_frame // W
    q_x = q_within_frame % W

    q_chunk = torch.full_like(q_frame, fill_value=current_chunk_idx)

    # -------- Keys: concatenation of chunks 0..current_chunk_idx --------
    Nk = key_len
    k_idx = torch.arange(Nk, device=device)          # [0..Nk-1]

    # Each frame has tokens_per_frame tokens, so:
    global_frame_idx = k_idx // tokens_per_frame     # [Nk], 0..num_frames_seen-1

    # Decode global_frame_idx into (chunk, frame_in_chunk)
    k_chunk = global_frame_idx // frames_per_chunk   # [Nk], 0..current_chunk_idx
    k_frame = global_frame_idx %  frames_per_chunk   # [Nk], 0..frames_per_chunk-1

    k_within_frame = k_idx % tokens_per_frame
    k_y = k_within_frame // W
    k_x = k_within_frame % W

    return q_chunk, q_frame, q_y, q_x, k_chunk, k_frame, k_y, k_x



@torch.no_grad()
def compute_head_stats(
    attn: torch.Tensor,           # [H, Nq, Nk], softmaxed
    q_chunk: torch.Tensor,        # [Nq], int
    q_frame: torch.Tensor,        # [Nq], int (frame index within chunk)
    q_y: torch.Tensor,            # [Nq]
    q_x: torch.Tensor,            # [Nq]
    k_chunk: torch.Tensor,        # [Nk]
    k_frame: torch.Tensor,        # [Nk]
    k_y: torch.Tensor,            # [Nk]
    k_x: torch.Tensor,            # [Nk]
    frames_per_chunk: int,
    r_spatial: int = 1,
):
    """
    Compute spatiotemporal statistics per head for autoregressive video attention.

    Returns a dict of tensors of shape [H] for each statistic.
    """
    H, Nq, Nk = attn.shape
    device = attn.device

    # Global frame indices
    tau_q = q_chunk * frames_per_chunk + q_frame    # [Nq]
    tau_k = k_chunk * frames_per_chunk + k_frame    # [Nk]

    # [Nq, Nk] boolean masks
    same_chunk = (q_chunk.view(Nq, 1) == k_chunk.view(1, Nk))
    past_chunk = (k_chunk.view(1, Nk) < q_chunk.view(Nq, 1))

    same_frame = (tau_q.view(Nq, 1) == tau_k.view(1, Nk))
    diff_frame = ~same_frame

    # spatial locality
    dy = (q_y.view(Nq, 1) - k_y.view(1, Nk)).abs()
    dx = (q_x.view(Nq, 1) - k_x.view(1, Nk)).abs()
    local_space = (dy <= r_spatial) & (dx <= r_spatial)

    # expand to [H, Nq, Nk]
    same_chunk = same_chunk.unsqueeze(0)
    past_chunk = past_chunk.unsqueeze(0)
    same_frame = same_frame.unsqueeze(0)
    diff_frame = diff_frame.unsqueeze(0)
    local_space = local_space.unsqueeze(0)

    not_local_space = ~local_space

    # helper: mean over queries
    # def mean_over_queries(mask: torch.Tensor) -> torch.Tensor:
    #     # attn * mask -> [H, Nq, Nk]; sum over keys -> [H, Nq]; mean over queries -> [H]
    #     return (attn * mask).sum(dim=-1).mean(dim=-1)

    def mean_over_queries_chunked(mask: torch.Tensor, q_chunk_size: int = 1215) -> torch.Tensor:
        out = torch.empty((H, Nq), device=device)  # [H]
        for q_start in range(0, Nq, q_chunk_size):
            q_end = min(q_start + q_chunk_size, Nq)
            chunk_mask = mask[:, q_start:q_end, :]  # [H, chunk_size, Nk]
            chunk_attn = attn[:, q_start:q_end, :]  # [H, chunk_size, Nk]
            chunk_sum = (chunk_attn * chunk_mask).sum(dim=-1)
            out[:, q_start:q_end] = chunk_sum
        return out.mean(dim=-1)


    # Self vs past
    P_self = mean_over_queries_chunked(same_chunk)
    P_past = mean_over_queries_chunked(past_chunk)
    # Self-chunk decomposed
    P_self_spatial_local  = mean_over_queries_chunked(same_chunk & same_frame & local_space)
    P_self_spatial_global = mean_over_queries_chunked(same_chunk & same_frame & not_local_space)
    P_self_temporal_local  = mean_over_queries_chunked(same_chunk & diff_frame & local_space)
    P_self_temporal_global = mean_over_queries_chunked(same_chunk & diff_frame & not_local_space)

    # Past-chunk decomposed (always different frame)
    P_past_local  = mean_over_queries_chunked(past_chunk & local_space)
    P_past_global = mean_over_queries_chunked(past_chunk & not_local_space)
    # Entropy over keys per query, then mean over queries
    # eps = 1e-9
    # entropy = -(attn * (attn + eps).log()).sum(dim=-1).mean(dim=-1)  # [H]

    return {
        "P_self": P_self,
        "P_past": P_past,
        "P_self_spatial_local": P_self_spatial_local,
        "P_self_spatial_global": P_self_spatial_global,
        "P_self_temporal_local": P_self_temporal_local,
        "P_self_temporal_global": P_self_temporal_global,
        "P_past_local": P_past_local,
        "P_past_global": P_past_global,
        # "entropy": entropy,
    }

@torch.no_grad()
def classify_heads(
    attn: torch.Tensor,           # [H, Nq, Nk]
    q_chunk: torch.Tensor,        # [Nq]
    q_frame: torch.Tensor,        # [Nq]
    q_y: torch.Tensor,            # [Nq]
    q_x: torch.Tensor,            # [Nq]
    k_chunk: torch.Tensor,        # [Nk]
    k_frame: torch.Tensor,        # [Nk]
    k_y: torch.Tensor,            # [Nk]
    k_x: torch.Tensor,            # [Nk]
    frames_per_chunk: int,
    r_spatial: int = 5,
    sink_topk_thresh: float = 0.10,
    sink_k: int = 10,
):
    """
    Classify heads into spatiotemporal types and mark sink-like heads.

    Returns:
        head_types: List[HeadType] length H
        is_sink:   torch.BoolTensor [H]
        stats:     dict of tensors [H] with raw statistics (useful for analysis)
    """
    stats = compute_head_stats_autoreg(
        attn, q_chunk, q_frame, q_y, q_x,
        k_chunk, k_frame, k_y, k_x,
        frames_per_chunk, r_spatial
    )

    P_self = stats["P_self"]
    P_past = stats["P_past"]

    P_self_spatial_local  = stats["P_self_spatial_local"]
    P_self_spatial_global = stats["P_self_spatial_global"]
    P_self_temporal_local  = stats["P_self_temporal_local"]
    P_self_temporal_global = stats["P_self_temporal_global"]
    P_past_local     = stats["P_past_local"]
    P_past_global    = stats["P_past_global"]

    H_heads = P_self.shape[0]

    # Aggregate self components
    P_self_spatial = P_self_spatial_local + P_self_spatial_global
    P_self_temporal = P_self_temporal_local + P_self_temporal_global

    head_types: list[HeadType] = []

    for h in range(H_heads):
        self = P_self[h].item()
        past = P_past[h].item()
        past_local = P_past_local[h].item()
        past_global = P_past_global[h].item()
        self_spatial = P_self_spatial[h].item()
        self_temporal = P_self_temporal[h].item()
        self_spatial_local = P_self_spatial_local[h].item()
        self_spatial_global = P_self_spatial_global[h].item()
        self_temporal_local = P_self_temporal_local[h].item()
        self_temporal_global = P_self_temporal_global[h].item()
        
        # Decide region: past-dominated vs self-dominated
        if past > self:
            # Past-chunk focused head
            if past_local > past_global:
                h_type = HeadType.P_LT
            else:
                h_type = HeadType.P_GT

        else:
            # Self-chunk focused head
            if self_spatial > self_temporal:
                # Mostly spatial
                if self_spatial_local >= self_spatial_global:
                    # Local spatial
                    h_type = HeadType.S_LS
                else:
                    # Global spatial
                    h_type = HeadType.S_GS
            else:
                # Mostly temporal
                if self_temporal_local >= self_temporal_global:
                    # Local temporal
                    h_type = HeadType.S_LT
                else:
                    # Global temporal
                    h_type = HeadType.S_GT

        head_types.append(h_type)

    # Sink detection -------------------------------------------------------
    M = attn.mean(dim=1) # [H, K] mean over queries
    topk_mass = M.topk(sink_k, dim=1).values.sum(dim=1)  # [H]
    is_sink = (topk_mass > sink_topk_thresh)
    stats["topk_mass"] = topk_mass
    stats["is_sink"] = is_sink

    return head_types, is_sink, stats


@torch.no_grad()
def dump_head_stats_to_csv(
    csv_path: str,
    attn: torch.Tensor,           # [H, Nq, Nk] softmaxed
    q_chunk: torch.Tensor,        # [Nq]
    q_frame: torch.Tensor,        # [Nq]
    q_y: torch.Tensor,            # [Nq]
    q_x: torch.Tensor,            # [Nq]
    k_chunk: torch.Tensor,        # [Nk]
    k_frame: torch.Tensor,        # [Nk]
    k_y: torch.Tensor,            # [Nk]
    k_x: torch.Tensor,            # [Nk]
    chunk_idx: int,
    layer_idx: int,
    step_idx: int,
    r_spatial: int = 5,
    sink_topk_thresh: float = 0.10,
    sink_k: int = 10,
):
    """
    Classify heads and append their stats to a CSV file.

    Each row corresponds to (chunk, layer, step, head).
    """
    frames_per_chunk = 24
    head_types, is_sink, stats = classify_heads_autoreg(
        attn,
        q_chunk, q_frame, q_y, q_x,
        k_chunk, k_frame, k_y, k_x,
        frames_per_chunk=frames_per_chunk,
        r_spatial=r_spatial,
        sink_topk_thresh=sink_topk_thresh,
        sink_k=sink_k,
    )

    H_heads = len(head_types)

    # Decide which stat keys we want to save
    # (all tensors in `stats` that are 1D over heads)
    stat_keys = [
        k for k, v in stats.items()
        if isinstance(v, torch.Tensor) and v.ndim == 1 and v.shape[0] == H_heads
    ]

    # Build header
    base_columns = ["chunk", "layer", "step", "head", "head_type", "is_sink"]
    columns = base_columns + stat_keys

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)

        # Write header once
        if not file_exists:
            writer.writeheader()

        for h in range(H_heads):
            row = {
                "chunk": chunk_idx,
                "layer": layer_idx,
                "step": step_idx,
                "head": h,
                "head_type": head_types[h],
                "is_sink": bool(is_sink[h].item()),
            }

            # Add all scalar stats for this head
            for key in stat_keys:
                row[key] = float(stats[key][h].item())

            writer.writerow(row)

@torch.no_grad()
def sdpa_with_scores(query, key, value, scale=None, enable_gqa=False, q_chunk_size=2025, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    *batch_dims, L, _ = query.shape
    _, _, S, _ = key.shape
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weights_cpu = torch.empty(
        *batch_dims, L, S,
        dtype=query.dtype,
        device=device,
    )
    out = torch.empty(*batch_dims, L, value.size(-1), device=query.device, dtype=query.dtype)

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    for q_start in range(0, L, q_chunk_size):
        q_end = min(q_start + q_chunk_size, L)
        q_chunk = query[..., q_start:q_end, :]
        scores = q_chunk @ key.transpose(-2, -1) * scale_factor
        attn = torch.softmax(scores, dim=-1)
        out_chunk = attn @ value
        out[..., q_start:q_end, :] = out_chunk
        attn_weights_cpu[..., q_start:q_end, :] = attn.detach().to(device=device)
        del scores, attn, out_chunk, q_chunk
    return out, attn_weights_cpu


@torch.no_grad()
def sdpa(query, key, value, scale=None, enable_gqa=False, q_chunk_size=2025) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    *batch_dims, L, _ = query.shape
    _, _, S, _ = key.shape
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    out = torch.empty(*batch_dims, L, value.size(-1), device=query.device, dtype=query.dtype)

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    for q_start in range(0, L, q_chunk_size):
        q_end = min(q_start + q_chunk_size, L)
        q_chunk = query[..., q_start:q_end, :]
        scores = q_chunk @ key.transpose(-2, -1) * scale_factor
        attn = torch.softmax(scores, dim=-1)
        out_chunk = attn @ value
        out[..., q_start:q_end, :] = out_chunk
        del scores, attn, out_chunk, q_chunk
    return out, None