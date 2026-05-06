#!/usr/bin/env python3
"""
Training script for GENESIS using Rectified Flow Matching.
Supports CFG, validation, and early stopping.
"""

import math
import os
import sys
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# Add GENESIS to path
sys.path.insert(0, os.path.join(os.getcwd(), "GENESIS"))
from dataloader.h5 import H5Dataset
from utils.normalize import normalize, denormalize_log_minmax, denormalize_minmax, apply_minmax_geo
from utils.vis.event_show import show_event_dual_plot
import utils.vis.event_show as event_vis
from utils.device import get_default_device
from flow.rectified_flow import RectifiedFlow

# ============================================================================
# Experiment Settings
# Keep everything in this file for fast iteration and reproducibility.
# ============================================================================

# Runtime / outputs
output_dir = Path("./tasks/rectified_flow_0413_jointzero_transformer")
save_plots = True
seed = 42
compile_model = False
print_every = 50
run_final_sampling = False

# Speed profile
fast_mode = False

# Data
h5_path = "./GENESIS-data/22644_0921_time_shift.h5"
num_workers = max(8, os.cpu_count() or 8)
data_shuffle = True
data_angle_conversion = True
data_pin_memory = True  # None = auto (cuda only), otherwise use the explicit bool
train_keep_ratio = 1.0  # use only a fixed subset of the training split

# Training
batch_size = 128
num_epochs = 50
lr = 3e-4
val_ratio = 0.1
val_every = 1

# Optimizer / LR schedule
lr_scheduler_patience = 4
lr_scheduler_factor = 0.8
lr_scheduler_min = 1e-6

# Early stopping
early_stopping_patience = 22
early_stopping_min_delta = 5e-7

# CFG
use_cfg = True
cfg_dropout = 0.1
cfg_scale = 1.0

# Flow sampling
sampling_method = "dopri5"  # euler, heun, rk4, dopri5
sampling_steps = 100

# Attention
attention_type = "linformer"  # standard, linformer
linformer_k = 64

# Joint-zero loss
flow_active_weight = 1.0
flow_inactive_weight = 0.25
zero_consistency_weight = 0.1
inactive_target_value = -1.0

# Epoch-by-epoch comparison plot
epoch_compare_every = 1
epoch_compare_num_samples = 4
epoch_compare_val_indices = [0, 1, 2, 3]  # fixed indices inside val_dataset
epoch_compare_figure_size = (18, 8)
epoch_compare_marker_size = 10.0

# Rectified flow hyperparameters
flow_name = "rectified_flow"

# Normalization
npe_clip = 225.0
ftime_clip = 21000.0
log_min = 0.0
_feature_range = (-1, 1)

# Label normalization
LABEL_NAMES = ["Energy (PeV)", "ux", "uy", "X", "Y", "Z"]
_label_methods = ["log_minmax", "identity", "identity", "minmax", "minmax", "minmax"]
_label_feature_ranges = [_feature_range] * 6
_ENERGY_PEV_MINMAX = {"min": 1.0, "max": 100.0}
energy_log_min = float(np.log1p(_ENERGY_PEV_MINMAX["min"]))
energy_log_max = float(np.log1p(_ENERGY_PEV_MINMAX["max"]))
_LABEL_XYZ_MINMAX = [
    {"min": -570.9000244140625, "max": 576.3699951171875},
    {"min": -521.0800170898438, "max": 509.5},
    {"min": -509.8599853515625, "max": 506.0566711425781},
]
_GEO_XYZ_MINMAX = [
    {"min": -570.9000244140625, "max": 576.3699951171875},
    {"min": -521.0800170898438, "max": 509.5},
    {"min": -509.8599853515625, "max": 506.0566711425781},
]
geo_min = np.array([_GEO_XYZ_MINMAX[j]["min"] for j in range(3)], dtype=np.float32)
geo_max = np.array([_GEO_XYZ_MINMAX[j]["max"] for j in range(3)], dtype=np.float32)
_label_stats = [
    {"log_min": energy_log_min, "log_max": energy_log_max},
    {},
    {},
    _LABEL_XYZ_MINMAX[0],
    _LABEL_XYZ_MINMAX[1],
    _LABEL_XYZ_MINMAX[2],
]

# Model
model_d_model = 256
model_nhead = 4
model_depth = 8
model_mlp_ratio = 4.0
model_dropout = 0.0
model_label_dim = 6

if fast_mode:
    compile_model = True
    use_cfg = False
    cfg_dropout = 0.0
    cfg_scale = 1.0
    num_workers = min(8, os.cpu_count() or 8)
    sampling_method = "euler"
    sampling_steps = 16
    val_every = 2
    epoch_compare_every = 5
    epoch_compare_figure_size = (16, 7)
    epoch_compare_marker_size = 8.0
    model_d_model = 192
    model_nhead = 6
    model_depth = 6
    model_mlp_ratio = 4.0
    data_pin_memory = True

# Derived normalization values
ftime_log_max = float(np.log1p(ftime_clip))


def _print_label_normalize_config():
    """Label별 정규화 설정 출력."""
    print("label normalize (per column):")
    for j, name in enumerate(LABEL_NAMES):
        m = _label_methods[j]
        fr = _label_feature_ranges[j]
        st = _label_stats[j] if _label_stats and j < len(_label_stats) else {}
        if m == "identity":
            detail = "identity (no transform)"
        elif m == "log_minmax":
            detail = f"log_minmax -> {fr}  stats={st}"
        elif m == "minmax":
            if st and "min" in st and "max" in st:
                detail = f"minmax -> {fr}  stats={st} (dataset min/max)"
            else:
                detail = f"minmax -> {fr}  stats={st} (empty => batch min/max)"
        else:
            detail = f"{m} -> {fr}  stats={st}"
        print(f"  [{j}] {name}: {detail}")


def _print_signal_normalize_config():
    """Signal/geometry normalization summary 출력."""
    print("signal normalize (current training setup):")
    print(f"  nPE      : clamp [0, {npe_clip:g}] -> minmax to {_feature_range}")
    print(
        f"             formula: nPE_norm = 2 * clip(nPE, 0, {npe_clip:g}) / {npe_clip:g} - 1"
    )
    print(f"  FirstTime: clamp [0, {ftime_clip:g}] -> log1p -> minmax to {_feature_range}")
    print(
        f"             formula: FirstTime_norm = 2 * log1p(clip(FirstTime, 0, {ftime_clip:g})) / "
        f"log1p({ftime_clip:g}) - 1"
    )
    print("  geometry  : apply_minmax_geo(geo, geo_min, geo_max, feature_range=(0, 1))")
    print(
        f"  original rflow baseline in this repo used nPE clip 1000.0; "
        f"this 0413 script uses {npe_clip:g}."
    )


def _apply_runtime_speed_optimizations():
    """Enable backend flags that usually help training throughput."""
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True


def _build_reproducible_subset(dataset, keep_ratio: float, seed_value: int):
    """Select a fixed subset of dataset indices with a fixed seed."""
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
    if keep_ratio >= 1.0:
        return dataset, None

    total_len = len(dataset)
    keep_len = max(1, int(round(total_len * keep_ratio)))
    generator = torch.Generator().manual_seed(seed_value)
    perm = torch.randperm(total_len, generator=generator).tolist()
    selected_indices = perm[:keep_len]
    subset = Subset(dataset, selected_indices)
    return subset, selected_indices


def _clamp_sig(sig: torch.Tensor) -> torch.Tensor:
    """Clamp npe/ftime before normalize."""
    s = sig.clone()
    s[:, 0] = torch.clamp(s[:, 0], min=0.0, max=npe_clip)
    s[:, 1] = torch.clamp(s[:, 1], min=0.0, max=ftime_clip)
    return s


def get_null_label(batch_size: int, label_dim: int, device: torch.device) -> torch.Tensor:
    """Create null label for CFG."""
    return torch.zeros(batch_size, label_dim, device=device)


def _sample_flow_matching(flow_matching, method, model, x1, steps, label, device):
    method = method.lower()
    sampler_name = f"sample_ode_{method}"
    sampler = getattr(flow_matching, sampler_name, None)
    if sampler is None:
        raise ValueError(
            f"Unsupported sampling_method='{method}'. Choose from: euler, heun, rk4, dopri5"
        )
    return sampler(model, x1, steps, label, device)


def compute_masked_joint_zero_losses(
    x_t: torch.Tensor,
    t: torch.Tensor,
    v_pred: torch.Tensor,
    v_true: torch.Tensor,
    sig_clamp: torch.Tensor,
):
    """Return weighted flow loss + inactive-token zero-consistency loss."""
    if sig_clamp.dim() != 3:
        raise ValueError(f"sig_clamp must have shape (B, 2, L), got {sig_clamp.shape}")

    active_mask = ((sig_clamp[:, 0, :] > 0.0) | (sig_clamp[:, 1, :] > 0.0)).float()  # (B, L)
    inactive_mask = 1.0 - active_mask

    # Weighted flow matching loss on velocity
    flow_sq = (v_pred - v_true).pow(2)  # (B, 2, L)
    flow_weights = flow_active_weight * active_mask + flow_inactive_weight * inactive_mask  # (B, L)
    flow_loss = (flow_sq * flow_weights[:, None, :]).sum() / (flow_weights.sum() * flow_sq.shape[1] + 1e-8)

    # Reconstruct x0 from the predicted velocity and penalize inactive detectors
    x0_hat = x_t - t[:, None, None] * v_pred
    zero_target = torch.full_like(x0_hat, inactive_target_value)
    zero_sq = (x0_hat - zero_target).pow(2)
    zero_loss = (zero_sq * inactive_mask[:, None, :]).sum() / (inactive_mask.sum() * zero_sq.shape[1] + 1e-8)

    total_loss = flow_loss + zero_consistency_weight * zero_loss
    active_ratio = active_mask.mean()
    return total_loss, flow_loss, zero_loss, active_ratio


# ============================================================================
# Model Definition
# ============================================================================

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Time embedding for flow matching (t in [0, 1])."""
    if t.dim() != 1:
        t = t.view(-1)
    t = t.float()

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device, dtype=emb.dtype)], dim=-1)
    return emb


class LinformerAttention(nn.Module):
    """Linformer-style attention with sequence-length projection on keys/values."""
    def __init__(self, d: int, nhead: int, seq_len: int, k: int, dropout: float = 0.0):
        super().__init__()
        if d % nhead != 0:
            raise ValueError(f"d_model={d} must be divisible by nhead={nhead}")
        self.d = d
        self.nhead = nhead
        self.head_dim = d // nhead
        self.seq_len = seq_len
        self.k = max(1, min(int(k), int(seq_len)))
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(dropout)

        self.proj_k = nn.Parameter(torch.randn(nhead, self.k, seq_len) / math.sqrt(seq_len))
        self.proj_v = nn.Parameter(torch.randn(nhead, self.k, seq_len) / math.sqrt(seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"LinformerAttention expected seq_len={self.seq_len}, got {seq_len}")

        q = self.q_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Project sequence length before attention.
        k = torch.einsum("b h l d, h k l -> b h k d", k, self.proj_k)
        v = torch.einsum("b h l d, h k l -> b h k d", v, self.proj_v)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d)
        return self.out_proj(out)


class DiTBlock(nn.Module):
    """DiT-style Transformer block with AdaLN-Zero."""
    def __init__(
        self,
        d: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        *,
        seq_len: int,
        attention_type: str = "standard",
        linformer_k: int = 64,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attention_type = attention_type.lower()
        if self.attention_type == "standard":
            self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        elif self.attention_type == "linformer":
            self.attn = LinformerAttention(d, nhead, seq_len=seq_len, k=linformer_k, dropout=dropout)
        else:
            raise ValueError("attention_type must be 'standard' or 'linformer'")
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, 6 * d),
        )
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        params = self.ada(c).view(B, 6, d)
        shift1, scale1, gate1, shift2, scale2, gate2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]
        x1 = self.norm1(x)
        x1 = x1 * (1.0 + scale1[:, None, :]) + shift1[:, None, :]
        if self.attention_type == "standard":
            attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        else:
            attn_out = self.attn(x1)
        x = x + gate1[:, None, :] * attn_out
        x2 = self.norm2(x)
        x2 = x2 * (1.0 + scale2[:, None, :]) + shift2[:, None, :]
        mlp_out = self.mlp(x2)
        x = x + gate2[:, None, :] * mlp_out
        return x


class FlowDiTTransformer(nn.Module):
    """DiT Transformer for Flow Matching with AdaLN-Zero."""
    def __init__(
        self,
        geo: torch.Tensor,
        d_model: int = 256,
        nhead: int = 8,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        label_dim: int = 6,
        attention_type: str = "standard",
        linformer_k: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.attention_type = attention_type.lower()
        self.linformer_k = linformer_k

        if geo.dim() == 2:
            geo_tok = geo.transpose(0, 1).unsqueeze(0)
        elif geo.dim() == 3:
            geo_tok = geo.permute(0, 2, 1)
        else:
            raise ValueError(f"geo must be (3,L) or (1,3,L). got {geo.shape}")

        self.register_buffer("geo_tokens", geo_tok.contiguous(), persistent=True)
        L = self.geo_tokens.shape[1]
        self.L = L

        self.in_proj = nn.Linear(2, d_model)
        self.geo_mlp = nn.Sequential(
            nn.Linear(3, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.use_index_pos = False
        if self.use_index_pos:
            self.index_pos = nn.Parameter(torch.zeros(1, L, d_model))

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.label_mlp = nn.Sequential(
            nn.Linear(label_dim, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model,
                nhead,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                seq_len=L,
                attention_type=self.attention_type,
                linformer_k=linformer_k,
            )
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.out_proj = nn.Linear(d_model, 2)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, 2, L)
        t: (B,) in [0, 1] for flow matching
        label: (B, 6)
        return: v_hat (B, 2, L) - velocity prediction
        """
        B, C, L = x_t.shape
        assert L == self.L, f"Expected L={self.L}, got L={L}"

        tokens = x_t.permute(0, 2, 1)
        h = self.in_proj(tokens)

        pos_geo = self.geo_mlp(self.geo_tokens)
        h = h + pos_geo

        if self.use_index_pos:
            h = h + self.index_pos

        t_emb = sinusoidal_timestep_embedding(t, self.d_model)
        t_cond = self.time_mlp(t_emb)
        y_cond = self.label_mlp(label)
        c = self.cond_mlp(torch.cat([t_cond, y_cond], dim=-1))

        for blk in self.blocks:
            h = blk(h, c)

        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        h = self.final_norm(h)
        h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]
        out = self.out_proj(h)
        return out.permute(0, 2, 1)


# ============================================================================
# Normalization Functions
# ============================================================================

_channel_stats = [
    {"min": 0.0, "max": npe_clip},
    {"log_min": log_min, "log_max": ftime_log_max},
]

@normalize(
    channel_methods=["minmax", "log_minmax"],
    feature_ranges=[_feature_range, _feature_range],
    channel_stats=_channel_stats,
    arg_index=0,
    label_arg_index=1,
    label_methods=_label_methods,
    label_feature_ranges=_label_feature_ranges,
    label_stats=_label_stats,
)
def prepare_batch(
    sig: torch.Tensor, label: torch.Tensor, *, verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare batch with normalization."""
    if verbose:
        print("prepare_batch: label", label)
    return (sig, label)


def denormalize_sig(sig: torch.Tensor) -> torch.Tensor:
    """Denormalize signal."""
    out = sig.clone()
    if sig.dim() == 3:
        out[:, 0, :] = denormalize_minmax(sig[:, 0, :], 0.0, npe_clip, _feature_range)
        out[:, 1, :] = denormalize_log_minmax(sig[:, 1, :], log_min, ftime_log_max, _feature_range)
    else:
        out[0, :] = denormalize_minmax(sig[0, :], 0.0, npe_clip, _feature_range)
        out[1, :] = denormalize_log_minmax(sig[1, :], log_min, ftime_log_max, _feature_range)
    return out


def _plot_event_comparison_panel(
    ax: plt.Axes,
    sig: np.ndarray,
    geo: np.ndarray,
    label: np.ndarray,
    title: str,
    *,
    marker_size: float,
    time_norm: Normalize | None = None,
    npe_max: float | None = None,
    show_detector_hull: bool = True,
    add_colorbar: bool = False,
):
    """Draw one denormalized event panel: size=nPE, color=FirstTime."""
    x = np.asarray(geo[0], dtype=np.float32)
    y = np.asarray(geo[1], dtype=np.float32)
    z = np.asarray(geo[2], dtype=np.float32)
    npe = np.asarray(sig[0], dtype=np.float32)
    ftime = np.asarray(sig[1], dtype=np.float32)

    finite_mask = np.isfinite(npe) & np.isfinite(ftime)
    hit_mask = finite_mask & ((npe > 0) | (ftime != 0))

    if show_detector_hull:
        event_vis._draw_detector_hull(ax, x, y, z)

    ax.scatter(x, y, z, s=1, c="gray", alpha=0.25)

    if hit_mask.any():
        x_hit = x[hit_mask]
        y_hit = y[hit_mask]
        z_hit = z[hit_mask]
        npe_hit = npe[hit_mask]
        ftime_hit = ftime[hit_mask]

        if time_norm is None:
            t_min = float(np.min(ftime_hit))
            t_max = float(np.max(ftime_hit))
            if t_min == t_max:
                t_max = t_min + 1.0
            time_norm = Normalize(vmin=t_min, vmax=t_max)

        if npe_max is None:
            npe_max = float(np.max(npe_hit))
        npe_max = max(npe_max, 1.0)

        sizes = marker_size * (0.25 + 4.75 * np.sqrt(np.clip(npe_hit / npe_max, 0.0, 1.0)))
        scatter = ax.scatter(
            x_hit,
            y_hit,
            z_hit,
            c=ftime_hit,
            s=sizes,
            cmap="jet",
            norm=time_norm,
            alpha=0.85,
            edgecolors="none",
        )
        if add_colorbar:
            cbar = ax.figure.colorbar(scatter, ax=ax, shrink=0.58, aspect=20, pad=0.08)
            cbar.set_label("FirstTime (ns)", rotation=270, labelpad=18)
    else:
        ax.text2D(0.5, 0.5, "No finite hits", transform=ax.transAxes)

    event_vis._style_axes(ax)
    ax.set_title(title, fontsize=12)
    return scatter if hit_mask.any() else None


def save_epoch_comparison_plot(
    real_sig: np.ndarray,
    sampled_sig: np.ndarray,
    geo: np.ndarray,
    label: np.ndarray,
    output_path: Path,
    *,
    title_prefix: str,
    figure_size: tuple[int, int] = (18, 8),
    marker_size: float = 10.0,
):
    """Save a left-right comparison figure: real vs sampled (both denormalized)."""
    real_ftime = np.asarray(real_sig[1], dtype=np.float32)
    sampled_ftime = np.asarray(sampled_sig[1], dtype=np.float32)
    real_npe = np.asarray(real_sig[0], dtype=np.float32)
    sampled_npe = np.asarray(sampled_sig[0], dtype=np.float32)

    ftime_vals = np.concatenate([real_ftime, sampled_ftime])
    ftime_vals = ftime_vals[np.isfinite(ftime_vals) & (ftime_vals != 0)]
    if ftime_vals.size > 0:
        ftime_norm = Normalize(
            vmin=float(np.percentile(ftime_vals, 1)),
            vmax=float(np.percentile(ftime_vals, 99)),
        )
    else:
        ftime_norm = Normalize(vmin=0.0, vmax=1.0)

    npe_vals = np.concatenate([real_npe, sampled_npe])
    npe_vals = npe_vals[np.isfinite(npe_vals) & (npe_vals > 0)]
    npe_max = float(np.percentile(npe_vals, 99)) if npe_vals.size > 0 else 1.0

    fig = plt.figure(figsize=figure_size)
    ax_left = fig.add_subplot(121, projection="3d")
    ax_right = fig.add_subplot(122, projection="3d")
    fig.suptitle(title_prefix, fontsize=14, y=0.98)

    _plot_event_comparison_panel(
        ax_left,
        real_sig,
        geo,
        label,
        "Real event (denorm)",
        marker_size=marker_size,
        time_norm=ftime_norm,
        npe_max=npe_max,
    )
    _plot_event_comparison_panel(
        ax_right,
        sampled_sig,
        geo,
        label,
        "Generated event (denorm)",
        marker_size=marker_size,
        time_norm=ftime_norm,
        npe_max=npe_max,
    )

    sm = ScalarMappable(norm=ftime_norm, cmap="jet")
    sm.set_array([])
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.02, top=0.88, wspace=0.02)
    cax = fig.add_axes([0.91, 0.22, 0.018, 0.56])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("FirstTime (ns)", rotation=270, labelpad=18)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Epoch comparison plot saved to: {output_path}")


def save_epoch_histogram_plot(
    real_sig: np.ndarray,
    sampled_sig: np.ndarray,
    output_path: Path,
    *,
    title_prefix: str,
    bins: int = 80,
    log_y: bool = True,
):
    """Save raw vs denormalized histograms for nPE and FirstTime."""
    real_sig = np.asarray(real_sig, dtype=np.float32)
    sampled_sig = np.asarray(sampled_sig, dtype=np.float32)

    real_npe = real_sig[0].ravel()
    real_ftime = real_sig[1].ravel()
    sampled_npe = sampled_sig[0].ravel()
    sampled_ftime = sampled_sig[1].ravel()

    def _finite_all(arr: np.ndarray) -> np.ndarray:
        return arr[np.isfinite(arr)]

    real_npe = _finite_all(real_npe)
    real_ftime = _finite_all(real_ftime)
    sampled_npe = _finite_all(sampled_npe)
    sampled_ftime = _finite_all(sampled_ftime)

    npe_all = np.concatenate([real_npe, sampled_npe]) if real_npe.size or sampled_npe.size else np.array([0.0])
    ftime_all = np.concatenate([real_ftime, sampled_ftime]) if real_ftime.size or sampled_ftime.size else np.array([0.0])

    npe_min = float(np.min(npe_all)) if npe_all.size else 0.0
    npe_max = float(np.max(npe_all)) if npe_all.size else 1.0
    ftime_min = float(np.min(ftime_all)) if ftime_all.size else 0.0
    ftime_max = float(np.max(ftime_all)) if ftime_all.size else 1.0
    if npe_min == npe_max:
        npe_max = npe_min + 1.0
    if ftime_min == ftime_max:
        ftime_max = ftime_min + 1.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title_prefix, fontsize=14, y=0.98)

    panels = [
        (axes[0, 0], real_npe, "Raw nPE"),
        (axes[0, 1], sampled_npe, "Denorm nPE"),
        (axes[1, 0], real_ftime, "Raw FirstTime"),
        (axes[1, 1], sampled_ftime, "Denorm FirstTime"),
    ]

    for ax, arr, title in panels:
        x_range = (npe_min, npe_max) if "nPE" in title else (ftime_min, ftime_max)
        ax.hist(arr, bins=bins, range=x_range, density=True, color="steelblue", alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Density")
        ax.set_xlabel("Value")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Epoch histogram plot saved to: {output_path}")


def save_training_metrics_artifacts(
    output_dir: Path,
    epoch_train_loss_hist: list[float],
    val_epoch_hist: list[int],
    val_loss_hist: list[float],
    epoch_lr_hist: list[float],
    *,
    batch_train_loss_hist: list[float] | None = None,
    make_plot: bool = True,
):
    """Persist training curves and raw metric histories."""
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True, parents=True)

    epochs = np.arange(1, len(epoch_train_loss_hist) + 1, dtype=np.int32)
    val_epoch_arr = np.asarray(val_epoch_hist, dtype=np.int32)
    train_epoch_arr = np.asarray(epoch_train_loss_hist, dtype=np.float32)
    val_loss_arr = np.asarray(val_loss_hist, dtype=np.float32)
    lr_arr = np.asarray(epoch_lr_hist, dtype=np.float32)
    batch_train_arr = (
        np.asarray(batch_train_loss_hist, dtype=np.float32)
        if batch_train_loss_hist is not None
        else np.asarray([], dtype=np.float32)
    )

    np.savez_compressed(
        metrics_dir / "training_metrics.npz",
        epoch=np.asarray(epochs, dtype=np.int32),
        train_loss=train_epoch_arr,
        val_epoch=val_epoch_arr,
        val_loss=val_loss_arr,
        lr=lr_arr,
        batch_train_loss=batch_train_arr,
    )

    val_map = {int(e): float(v) for e, v in zip(val_epoch_arr.tolist(), val_loss_arr.tolist())}
    lr_map = {int(e): float(v) for e, v in zip(epochs.tolist(), lr_arr.tolist())}

    csv_path = metrics_dir / "training_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
        for epoch, train_loss in zip(epochs.tolist(), train_epoch_arr.tolist()):
            val_loss = val_map.get(int(epoch), "")
            lr_value = lr_map.get(int(epoch), "")
            writer.writerow([epoch, f"{train_loss:.10f}", val_loss if val_loss == "" else f"{val_loss:.10f}", lr_value if lr_value == "" else f"{lr_value:.10e}"])

    if not make_plot or epochs.size == 0:
        return

    fig, (ax_loss, ax_lr) = plt.subplots(1, 2, figsize=(14, 5))
    ax_loss.plot(epochs, train_epoch_arr, label="train", color="tab:blue", linewidth=1.8)
    if val_epoch_arr.size > 0:
        ax_loss.plot(val_epoch_arr, val_loss_arr, label="val", color="tab:orange", linewidth=1.8)
    ax_loss.set_title("Loss Curve")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_lr.plot(epochs, lr_arr, color="tab:green", linewidth=1.8)
    ax_lr.set_title("Learning Rate")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("LR")
    ax_lr.set_yscale("log")
    ax_lr.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training metrics saved to: {metrics_dir}")


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _apply_runtime_speed_optimizations()
    
    device = get_default_device()
    print("device:", device)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    model_save_dir = output_dir / "models"
    model_save_dir.mkdir(exist_ok=True, parents=True)
    
    if save_plots:
        plot_save_dir = output_dir / "plots"
        plot_save_dir.mkdir(exist_ok=True, parents=True)
    else:
        plot_save_dir = None
    
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Model save directory: {model_save_dir.absolute()}")
    if save_plots:
        print(f"Plot save directory: {plot_save_dir.absolute()}")
    else:
        print("Plot saving: disabled")
    print(f"Flow mode: {flow_name}")
    print(f"Sampling method: {sampling_method} | steps: {sampling_steps}")
    print(f"CFG: enabled={use_cfg}, dropout={cfg_dropout}, scale={cfg_scale}")
    print(f"Speed profile: fast_mode={fast_mode}, compile_model={compile_model}, run_final_sampling={run_final_sampling}")
    print(f"Training: epochs={num_epochs}, batch_size={batch_size}, lr={lr:.2e}")
    print(f"Validation: ratio={val_ratio}, every={val_every}")

    print(f"Loading dataset from: {h5_path}")
    dataset = H5Dataset(
        h5_path=h5_path,
        angle_conversion=data_angle_conversion,
        num_workers=num_workers,
        shuffle=data_shuffle,
    )
    print(f"Dataset length: {len(dataset)}")
    _print_signal_normalize_config()
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Train size: {train_size}, Val size: {val_size}")

    if train_keep_ratio < 1.0:
        print(f"Training subset: keep_ratio={train_keep_ratio:.3f}")
        print(f"Subset seed: {seed}")
        train_dataset, _ = _build_reproducible_subset(train_dataset, train_keep_ratio, seed)
        print(f"Reduced train size: {len(train_dataset)} / {train_size}")
    else:
        print("Training subset: using full train split")

    fixed_val_samples = []
    if len(val_dataset) > 0:
        compare_indices = list(range(min(epoch_compare_num_samples, len(val_dataset))))
        for fixed_val_dataset_index in compare_indices:
            fixed_val_sample = val_dataset[fixed_val_dataset_index]
            fixed_val_base_index = (
                int(val_dataset.indices[fixed_val_dataset_index])
                if hasattr(val_dataset, "indices")
                else fixed_val_dataset_index
            )
            fixed_val_samples.append(
                (fixed_val_dataset_index, fixed_val_base_index, fixed_val_sample)
            )
            print(
                f"Fixed validation sample for epoch plots: val_dataset[{fixed_val_dataset_index}]"
                f" -> dataset[{fixed_val_base_index}]"
            )
    else:
        print("Validation split is empty; per-epoch comparison plots are disabled.")
    
    _print_label_normalize_config()
    
    pin_memory = data_pin_memory if data_pin_memory is not None else (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    sig0, geo0, label0 = dataset[0]
    print("sig:", sig0.shape, sig0.dtype)
    print("geo:", geo0.shape, geo0.dtype)
    print("label:", label0.shape, label0.dtype)
    
    _geo_raw = dataset[0][1]
    _geo = apply_minmax_geo(_geo_raw, geo_min, geo_max, feature_range=(0, 1))
    
    model = FlowDiTTransformer(
        geo=_geo,
        d_model=model_d_model,
        nhead=model_nhead,
        depth=model_depth,
        mlp_ratio=model_mlp_ratio,
        dropout=model_dropout,
        label_dim=model_label_dim,
        attention_type=attention_type,
        linformer_k=linformer_k,
    ).to(device)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"[GPU] after model: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode='min',
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=lr_scheduler_min,
        verbose=False
    )
    
    try:
        scaler = GradScaler(device.type) if device.type in ("cuda", "mps") else None
    except (TypeError, ValueError):
        scaler = GradScaler() if device.type == "cuda" else None
    print("AMP enabled:", scaler is not None)
    
    if compile_model:
        try:
            if hasattr(torch, 'compile'):
                print("Compiling model with torch.compile()...")
                model = torch.compile(model, mode="reduce-overhead")
                print("Model compilation successful!")
            else:
                print("torch.compile() not available (requires PyTorch 2.0+)")
        except Exception as e:
            print(f"Model compilation failed (continuing without compile): {e}")
    
    print("params:", sum(p.numel() for p in model.parameters())/1e6, "M")
    print(f"Attention: {attention_type} (linformer_k={linformer_k})")
    print(f"CFG enabled: {use_cfg} (dropout={cfg_dropout}, scale={cfg_scale})")
    print(f"Sampling method: {sampling_method}, steps: {sampling_steps} (euler/heun/rk4/dopri5)")
    
    # Initialize Rectified Flow
    flow_matching = RectifiedFlow()
    
    model.train()
    
    train_loss_hist = []
    epoch_train_loss_hist = []
    val_loss_hist = []
    val_epoch_hist = []
    epoch_lr_hist = []
    best_val_loss = float('inf')
    best_checkpoint_path = None
    epochs_without_improvement = 0
    
    steps_per_epoch = len(train_loader)
    val_steps_per_epoch = len(val_loader)
    total_steps = num_epochs * steps_per_epoch
    
    print("\n" + "="*60)
    print("Training Configuration Summary")
    print("="*60)
    print(f"Method: Rectified Flow Matching")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Initial LR: {lr:.2e}")
    print(f"LR Scheduler: ReduceLROnPlateau (patience={lr_scheduler_patience}, factor={lr_scheduler_factor})")
    print(f"Early Stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    print(f"Train batches per epoch: {steps_per_epoch}")
    print(f"Val batches per epoch: {val_steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    print("="*60)
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*60 + "\n")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_losses = []
        
        pbar = tqdm(enumerate(train_loader, 1), total=steps_per_epoch, desc=f"Epoch {epoch}/{num_epochs} [Train]", file=sys.stdout)
        
        for batch_idx, (sig, geo, label) in pbar:
            sig = sig.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            if use_cfg:
                mask = torch.rand(label.shape[0], device=device) < cfg_dropout
                label_cfg = label.clone()
                label_cfg[mask] = get_null_label(mask.sum().item(), model_label_dim, device)
            else:
                label_cfg = label
            
            sig_clamp = _clamp_sig(sig)
            x0, label_norm = prepare_batch(sig_clamp, label_cfg)
            
            B = x0.shape[0]
            
            # Flow Matching: sample time t ~ U(0, 1)
            t = torch.rand(B, device=device, dtype=torch.float32)
            
            # Sample noise x_1 ~ N(0, I)
            x1 = torch.randn_like(x0)
            
            # Compute path x_t
            x_t = flow_matching.compute_path(x0, x1, t)
            
            # Compute ground truth velocity
            v_true = flow_matching.compute_velocity(x0, x1, x_t, t)
            
            # Forward pass: predict velocity
            with autocast(device.type, enabled=(scaler is not None)):
                v_pred = model(x_t, t, label_norm)
                loss, _, _, _ = compute_masked_joint_zero_losses(
                    x_t=x_t,
                    t=t,
                    v_pred=v_pred,
                    v_true=v_true,
                    sig_clamp=sig_clamp,
                )
            
            # Backward pass
            optim.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            
            loss_val = float(loss.item())
            epoch_train_losses.append(loss_val)
            train_loss_hist.append(loss_val)
            
            current_step = (epoch - 1) * steps_per_epoch + batch_idx
            avg_loss_so_far = np.mean(epoch_train_losses)
            pbar.set_postfix({"loss": f"{avg_loss_so_far:.6f}", "lr": f"{optim.param_groups[0]['lr']:.2e}"})
            pbar.refresh()
            sys.stdout.flush()
        
        epoch_train_loss = np.mean(epoch_train_losses)
        epoch_train_loss_hist.append(epoch_train_loss)
        
        # Validation phase
        if epoch % val_every == 0:
            model.eval()
            epoch_val_losses = []
            should_stop = False
            
            with torch.inference_mode():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]", file=sys.stdout)
                for sig, geo, label in val_pbar:
                    sig = sig.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)
                    
                    sig_clamp = _clamp_sig(sig)
                    x0, label_norm = prepare_batch(sig_clamp, label)
                    
                    B = x0.shape[0]
                    t = torch.rand(B, device=device, dtype=torch.float32)
                    x1 = torch.randn_like(x0)
                    x_t = flow_matching.compute_path(x0, x1, t)
                    v_true = flow_matching.compute_velocity(x0, x1, x_t, t)
                    
                    with autocast(device.type, enabled=(scaler is not None)):
                        v_pred = model(x_t, t, label_norm)
                        loss, _, _, _ = compute_masked_joint_zero_losses(
                            x_t=x_t,
                            t=t,
                            v_pred=v_pred,
                            v_true=v_true,
                            sig_clamp=sig_clamp,
                        )
                    
                    loss_val = float(loss.item())
                    epoch_val_losses.append(loss_val)
                    val_pbar.set_postfix({"loss": f"{loss_val:.6f}"})
                    val_pbar.refresh()
                    sys.stdout.flush()
            
            epoch_val_loss = np.mean(epoch_val_losses)
            val_loss_hist.append(epoch_val_loss)
            val_epoch_hist.append(epoch)
            
            old_lr = optim.param_groups[0]['lr']
            lr_scheduler.step(epoch_val_loss)
            new_lr = optim.param_groups[0]['lr']
            lr_reduced = (old_lr != new_lr)
            
            if lr_reduced:
                print(f"\n[LR Scheduler] Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
            
            improvement = best_val_loss - epoch_val_loss
            if improvement > early_stopping_min_delta:
                best_val_loss = epoch_val_loss
                epochs_without_improvement = 0
                
                if best_checkpoint_path is not None and best_checkpoint_path.exists():
                    best_checkpoint_path.unlink()
                
                best_checkpoint_path = model_save_dir / f"best_checkpoint_epoch_{epoch:03d}_val_loss_{best_val_loss:.6f}.pt"
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "best_val_loss": best_val_loss,
                    "d_model": model_d_model,
                    "nhead": model_nhead,
                    "depth": model_depth,
                    "mlp_ratio": model_mlp_ratio,
                    "dropout": model_dropout,
                    "label_dim": model_label_dim,
                    "use_cfg": use_cfg,
                    "cfg_dropout": cfg_dropout,
                    "cfg_scale": cfg_scale,
                    "sampling_method": sampling_method,
                    "sampling_steps": sampling_steps,
                    "attention_type": attention_type,
                    "linformer_k": linformer_k,
                    "flow_active_weight": flow_active_weight,
                    "flow_inactive_weight": flow_inactive_weight,
                    "zero_consistency_weight": zero_consistency_weight,
                }
                torch.save(checkpoint, best_checkpoint_path)
                print(f"\n✓ New best model saved: {best_checkpoint_path.name}")
            else:
                epochs_without_improvement += 1
            
            print(f"\nEpoch {epoch:3d}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | Best Val: {best_val_loss:.6f}")
            lr_msg = f"LR: {optim.param_groups[0]['lr']:.2e}"
            if lr_reduced:
                lr_msg += f" (reduced from {old_lr:.2e})"
            print(f"{lr_msg} | Epochs without improvement: {epochs_without_improvement}/{early_stopping_patience}")
            print("-"*60)
            should_stop = epochs_without_improvement >= early_stopping_patience
        else:
            print(f"\nEpoch {epoch:3d}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | LR: {optim.param_groups[0]['lr']:.2e}")
            print("-"*60)

        epoch_lr_hist.append(float(optim.param_groups[0]["lr"]))
        save_training_metrics_artifacts(
            output_dir,
            epoch_train_loss_hist,
            val_epoch_hist,
            val_loss_hist,
            epoch_lr_hist,
            batch_train_loss_hist=train_loss_hist,
            make_plot=save_plots,
        )

        if save_plots and plot_save_dir is not None and fixed_val_samples and epoch_compare_every > 0 and (epoch % epoch_compare_every == 0):
            model.eval()
            with torch.inference_mode():
                for sample_idx, (fixed_val_dataset_index, fixed_val_base_index, fixed_val_sample) in enumerate(fixed_val_samples):
                    sig_ref_raw, geo_ref_raw, label_ref_raw = fixed_val_sample

                    sig_ref = sig_ref_raw.unsqueeze(0).to(device)
                    geo_ref = geo_ref_raw.detach().cpu().numpy()
                    label_ref = label_ref_raw.unsqueeze(0).to(device)

                    sig_ref_clamp = _clamp_sig(sig_ref)
                    _, label_ref_norm = prepare_batch(sig_ref_clamp, label_ref, verbose=False)

                    num_samples = 1
                    x1 = torch.randn(num_samples, 2, model.L, device=device)
                    print(
                        f"Running epoch comparison sampling on val_dataset[{fixed_val_dataset_index}] "
                        f"(dataset[{fixed_val_base_index}]) with {sampling_method} ({sampling_steps} steps)..."
                    )

                    if use_cfg:
                        x_uncond = _sample_flow_matching(flow_matching, sampling_method, model, x1, sampling_steps, None, device)
                        x_cond = _sample_flow_matching(flow_matching, sampling_method, model, x1, sampling_steps, label_ref_norm, device)
                        x_gen = x_uncond + cfg_scale * (x_cond - x_uncond)
                    else:
                        x_gen = _sample_flow_matching(flow_matching, sampling_method, model, x1, sampling_steps, label_ref_norm, device)

                    sampled_sig = denormalize_sig(x_gen)[0].detach().cpu().numpy()
                    real_sig = sig_ref_raw.detach().cpu().numpy()
                    label_ref_np = label_ref_raw.detach().cpu().numpy()

                    comparison_output_path = plot_save_dir / f"epoch_{epoch:03d}_sample_{sample_idx:02d}_valref_{fixed_val_dataset_index:04d}.png"
                    save_epoch_comparison_plot(
                        real_sig=real_sig,
                        sampled_sig=sampled_sig,
                        geo=geo_ref,
                        label=label_ref_np,
                        output_path=comparison_output_path,
                        title_prefix=(
                            f"Epoch {epoch:03d} | Fixed val sample val_dataset[{fixed_val_dataset_index}] "
                            f"-> dataset[{fixed_val_base_index}] | Real vs Generated"
                        ),
                        figure_size=epoch_compare_figure_size,
                        marker_size=epoch_compare_marker_size,
                    )

                    histogram_output_path = plot_save_dir / f"epoch_{epoch:03d}_sample_{sample_idx:02d}_valref_{fixed_val_dataset_index:04d}_hist.png"
                    save_epoch_histogram_plot(
                        real_sig=real_sig,
                        sampled_sig=sampled_sig,
                        output_path=histogram_output_path,
                        title_prefix=(
                            f"Epoch {epoch:03d} | Fixed val sample val_dataset[{fixed_val_dataset_index}] "
                            f"-> dataset[{fixed_val_base_index}] | Raw vs Denorm Histograms"
                        ),
                        bins=80,
                        log_y=True,
                    )

        if epoch % val_every == 0 and should_stop:
            print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {early_stopping_patience} epochs)")
            break

    print("\nTraining done!")
    
    final_checkpoint_path = model_save_dir / "model_checkpoint_final.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "epoch": epoch,
        "final_train_loss": epoch_train_loss_hist[-1] if epoch_train_loss_hist else None,
        "final_val_loss": val_loss_hist[-1] if val_loss_hist else None,
        "best_val_loss": best_val_loss,
        "d_model": model_d_model,
        "nhead": model_nhead,
        "depth": model_depth,
        "mlp_ratio": model_mlp_ratio,
        "dropout": model_dropout,
        "label_dim": model_label_dim,
        "use_cfg": use_cfg,
        "cfg_dropout": cfg_dropout,
        "cfg_scale": cfg_scale,
        "sampling_method": sampling_method,
        "sampling_steps": sampling_steps,
        "attention_type": attention_type,
        "linformer_k": linformer_k,
        "flow_active_weight": flow_active_weight,
        "flow_inactive_weight": flow_inactive_weight,
        "zero_consistency_weight": zero_consistency_weight,
    }
    torch.save(checkpoint, final_checkpoint_path)
    print(f"Final model saved to: {final_checkpoint_path}")
    if best_checkpoint_path:
        print(f"Best model (val_loss={best_val_loss:.6f}) saved to: {best_checkpoint_path.name}")
    
    # Sampling
    print("\n" + "="*60)
    print("Sampling from trained model...")
    print("="*60)
    
    if run_final_sampling:
        model.eval()
        with torch.inference_mode():
            ref_idx = 0
            sig_ref_raw, geo_ref_raw, label_ref_raw = dataset[ref_idx]
            
            sig_ref_clamp = _clamp_sig(sig_ref_raw.unsqueeze(0).to(device))
            label_ref = label_ref_raw.unsqueeze(0).to(device)
            _, label_ref_norm = prepare_batch(sig_ref_clamp, label_ref, verbose=False)
            
            num_samples = 1
            B = num_samples
            
            # Start from noise
            x1 = torch.randn(B, 2, model.L, device=device)
            
            print(f"Running ODE solver ({sampling_method}, {sampling_steps} steps)...")
            
            # CFG for sampling
            if use_cfg:
                x_uncond = _sample_flow_matching(flow_matching, sampling_method, model, x1, sampling_steps, None, device)
                x_cond = _sample_flow_matching(flow_matching, sampling_method, model, x1, sampling_steps, label_ref_norm, device)
                # CFG combination
                x = x_uncond + cfg_scale * (x_cond - x_uncond)
            else:
                x = _sample_flow_matching(flow_matching, sampling_method, model, x1, sampling_steps, label_ref_norm, device)
            
            samples_denorm = denormalize_sig(x)
            sample_np = samples_denorm[0].detach().cpu().numpy()
            
            print("\nSampling completed!")
            print(f"Sample shape: {sample_np.shape}")
            print(f"Sample nPE range: [{sample_np[0].min():.2f}, {sample_np[0].max():.2f}]")
            print(f"Sample FirstTime range: [{sample_np[1].min():.2f}, {sample_np[1].max():.2f}]")
            
            if plot_save_dir is not None:
                geo_ref_np = geo_ref_raw.detach().cpu().numpy()
                label_ref_np = label_ref_raw.detach().cpu().numpy()
                
                sampled_output_path = plot_save_dir / f"sampled_event_{ref_idx}.png"
                fig_sampled, _ = show_event_dual_plot(
                    sig=sample_np,
                    geo=geo_ref_np,
                    label=label_ref_np,
                    output_path=str(sampled_output_path),
                    figure_size=(18, 8),
                    marker_size=8.0,
                    show_detector_hull=True,
                    show=False,
                    title_prefix=f"Rectified Flow | Sampled data | using label from event {ref_idx}",
                    firsttime_title="FirstTime (sampled)",
                    npe_title="nPE (sampled)",
                )
                print(f"Sampled event plot saved to: {sampled_output_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()
