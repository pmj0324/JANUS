#!/usr/bin/env python3
"""
Transformer autoencoder for GENESIS signals.

Stage 1 for a later latent-flow experiment:
  - normalize signal with the same log1p active-p95 style as 0413_loglog_ft
  - encode detector tokens with learnable CLS latent tokens included in self-attention
  - decode detector outputs by cross-attending geometry queries to the CLS latents
  - condition encoder/decoder blocks with label through AdaLN-Zero
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from matplotlib.colors import Normalize
from torch.amp import autocast

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm


script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from dataloader.h5 import H5Dataset
from utils.device import get_default_device
from utils.normalize import apply_minmax, apply_minmax_geo
import utils.vis.event_show as event_vis


# ============================================================================
# Experiment Settings
# ============================================================================

output_dir = Path("./tasks/autoencoder_0413_loglog_cls")
save_plots = True
seed = 42
compile_model = True
print_every = 200
fast_mode = False

h5_path = "./GENESIS-data/22644_0921_time_shift.h5"
num_workers = max(8, os.cpu_count() or 8)
data_shuffle = True
data_angle_conversion = True
data_pin_memory = True
train_keep_ratio = 1.0
epoch_data_keep_ratio = 0.5

batch_size = 128
num_epochs = 50
lr = 3e-4
val_ratio = 0.1
val_every = 1
resume_checkpoint = None
resume_load_optimizer = True

lr_scheduler_patience = 4
lr_scheduler_factor = 0.8
lr_scheduler_min = 1e-7
early_stopping_patience = 18
early_stopping_min_delta = 5e-7

active_value_weight = 1.0
hit_bce_weight = 0.1
inactive_recon_weight = 0.01
active_threshold = 0.0

epoch_compare_every = 1
epoch_compare_num_samples = 1
epoch_compare_val_indices = [0, 1, 2, 3]
epoch_compare_figure_size = (18, 8)
epoch_compare_marker_size = 10.0

npe_clip = 225.0
ftime_clip = 21000.0
_feature_range = (-1, 1)

LABEL_NAMES = ["Energy (PeV)", "ux", "uy", "X", "Y", "Z"]
_label_methods = ["log_minmax", "identity", "identity", "minmax", "minmax", "minmax"]
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

model_d_model = 128
model_nhead = 4
model_encoder_depth = 6
model_decoder_depth = 4
model_num_latents = 16
model_mlp_ratio = 4.0
model_dropout = 0.0
model_label_dim = 6

if fast_mode:
    compile_model = True
    num_workers = min(8, os.cpu_count() or 8)
    val_every = 2
    epoch_compare_every = 5
    model_d_model = 128
    model_nhead = 6
    model_encoder_depth = 4
    model_decoder_depth = 3
    model_num_latents = 16


# ============================================================================
# Normalization
# ============================================================================

def _compute_signal_p95_stats(h5_path: str, chunk_size: int = 4096) -> dict:
    stats = {
        "p95_log_npe": np.nan,
        "p95_log_ftime": np.nan,
        "active_npe_count": 0,
        "active_ftime_count": 0,
    }
    log_npe_vals = []
    log_ftime_vals = []

    with h5py.File(h5_path, "r") as f:
        sig_ds = f["input"]
        total = sig_ds.shape[0]
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            sig = np.asarray(sig_ds[start:end], dtype=np.float32)
            npe = np.clip(sig[:, 0, :].ravel(), 0.0, npe_clip)
            ftime = np.clip(sig[:, 1, :].ravel(), 0.0, ftime_clip)

            npe_active = npe[np.isfinite(npe) & (npe > 0.0)]
            ftime_active = ftime[np.isfinite(ftime) & (ftime > 0.0)]
            if npe_active.size > 0:
                log_npe = np.log1p(npe_active)
                log_npe_vals.append(log_npe)
                stats["active_npe_count"] += int(log_npe.size)
            if ftime_active.size > 0:
                log_ftime = np.log1p(ftime_active)
                log_ftime_vals.append(log_ftime)
                stats["active_ftime_count"] += int(log_ftime.size)

    if log_npe_vals:
        stats["p95_log_npe"] = float(np.percentile(np.concatenate(log_npe_vals), 95))
    if log_ftime_vals:
        stats["p95_log_ftime"] = float(np.percentile(np.concatenate(log_ftime_vals), 95))

    if not np.isfinite(stats["p95_log_npe"]) or stats["p95_log_npe"] <= 0:
        stats["p95_log_npe"] = 1.0
    if not np.isfinite(stats["p95_log_ftime"]) or stats["p95_log_ftime"] <= 0:
        stats["p95_log_ftime"] = 1.0
    return stats


signal_norm_stats = _compute_signal_p95_stats(h5_path)


def _clamp_sig(sig: torch.Tensor) -> torch.Tensor:
    out = sig.clone()
    out[:, 0] = torch.clamp(out[:, 0], min=0.0, max=npe_clip)
    out[:, 1] = torch.clamp(out[:, 1], min=0.0, max=ftime_clip)
    return out


def _normalize_signal(sig: torch.Tensor) -> torch.Tensor:
    out = sig.clone()
    out[:, 0, :] = torch.log1p(out[:, 0, :]) / float(signal_norm_stats["p95_log_npe"])
    out[:, 1, :] = torch.log1p(out[:, 1, :]) / float(signal_norm_stats["p95_log_ftime"])
    return out


def _denormalize_signal(sig: torch.Tensor) -> torch.Tensor:
    out = sig.clone()
    out[:, 0, :] = torch.expm1(torch.clamp(out[:, 0, :], min=0.0) * float(signal_norm_stats["p95_log_npe"]))
    out[:, 1, :] = torch.expm1(torch.clamp(out[:, 1, :], min=0.0) * float(signal_norm_stats["p95_log_ftime"]))
    return out


def _normalize_label(label: torch.Tensor) -> torch.Tensor:
    out = label.clone()
    out[:, 0] = apply_minmax(
        torch.log1p(out[:, 0]),
        feature_range=_feature_range,
        data_min=energy_log_min,
        data_max=energy_log_max,
    )
    out[:, 3] = apply_minmax(
        out[:, 3],
        feature_range=_feature_range,
        data_min=_LABEL_XYZ_MINMAX[0]["min"],
        data_max=_LABEL_XYZ_MINMAX[0]["max"],
    )
    out[:, 4] = apply_minmax(
        out[:, 4],
        feature_range=_feature_range,
        data_min=_LABEL_XYZ_MINMAX[1]["min"],
        data_max=_LABEL_XYZ_MINMAX[1]["max"],
    )
    out[:, 5] = apply_minmax(
        out[:, 5],
        feature_range=_feature_range,
        data_min=_LABEL_XYZ_MINMAX[2]["min"],
        data_max=_LABEL_XYZ_MINMAX[2]["max"],
    )
    return out


def _build_model_input(sig_norm: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    hit = active_mask.float()
    logq = sig_norm[:, 0, :] * hit
    ftime = sig_norm[:, 1, :] * hit
    return torch.stack([hit, logq, ftime], dim=1)


def prepare_batch(
    sig: torch.Tensor, label: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sig_clamp = _clamp_sig(sig)
    sig_norm = _normalize_signal(sig_clamp)
    label_norm = _normalize_label(label)
    active_mask = sig_clamp[:, 0, :] > active_threshold
    model_input = _build_model_input(sig_norm, active_mask)
    return model_input, sig_norm, label_norm, active_mask


# ============================================================================
# Model
# ============================================================================

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class AdaSelfAttentionBlock(nn.Module):
    def __init__(self, d: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        bsz, _, d = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).view(bsz, 6, d).unbind(dim=1)
        x1 = _modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + gate1[:, None, :] * attn_out
        x2 = _modulate(self.norm2(x), shift2, scale2)
        x = x + gate2[:, None, :] * self.mlp(x2)
        return x


class AdaCrossAttentionBlock(nn.Module):
    def __init__(self, d: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_self = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.norm_cross_q = nn.LayerNorm(d, elementwise_affine=False)
        self.norm_cross_kv = nn.LayerNorm(d, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d, elementwise_affine=False)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        bsz, _, d = x.shape
        params = self.ada(c).view(bsz, 9, d)
        shift1, scale1, gate1 = params[:, 0], params[:, 1], params[:, 2]
        shift2, scale2, gate2 = params[:, 3], params[:, 4], params[:, 5]
        shift3, scale3, gate3 = params[:, 6], params[:, 7], params[:, 8]

        x1 = _modulate(self.norm_self(x), shift1, scale1)
        self_out, _ = self.self_attn(x1, x1, x1, need_weights=False)
        x = x + gate1[:, None, :] * self_out

        q = _modulate(self.norm_cross_q(x), shift2, scale2)
        kv = self.norm_cross_kv(memory)
        cross_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x + gate2[:, None, :] * cross_out

        x3 = _modulate(self.norm_mlp(x), shift3, scale3)
        x = x + gate3[:, None, :] * self.mlp(x3)
        return x


class CLSConditionedTransformerAutoEncoder(nn.Module):
    def __init__(
        self,
        geo: torch.Tensor,
        d_model: int = 256,
        nhead: int = 4,
        encoder_depth: int = 6,
        decoder_depth: int = 4,
        num_latents: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        label_dim: int = 6,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_latents = num_latents

        if geo.dim() == 2:
            geo_tok = geo.transpose(0, 1).unsqueeze(0)
        elif geo.dim() == 3:
            geo_tok = geo.permute(0, 2, 1)
        else:
            raise ValueError(f"geo must be (3,L) or (1,3,L), got {geo.shape}")

        self.register_buffer("geo_tokens", geo_tok.contiguous(), persistent=True)
        self.L = self.geo_tokens.shape[1]

        self.cls_tokens = nn.Parameter(torch.randn(1, num_latents, d_model) * 0.02)
        self.in_proj = nn.Linear(3, d_model)
        self.geo_mlp = nn.Sequential(nn.Linear(3, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model))
        self.decoder_query = nn.Parameter(torch.zeros(1, self.L, d_model))

        self.label_mlp = nn.Sequential(
            nn.Linear(label_dim, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.encoder_blocks = nn.ModuleList(
            [AdaSelfAttentionBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(encoder_depth)]
        )
        self.decoder_blocks = nn.ModuleList(
            [AdaCrossAttentionBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(decoder_depth)]
        )

        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(d_model)
        self.pool_proj = nn.Linear(d_model, d_model)

        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.value_proj = nn.Linear(d_model, 2)
        self.hit_head = nn.Linear(d_model, 1)

        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.hit_head.weight)
        nn.init.zeros_(self.hit_head.bias)

    def pool_latents(self, latents: torch.Tensor) -> torch.Tensor:
        query = self.pool_query.expand(latents.shape[0], -1, -1)
        pooled, _ = self.pool_attn(query, latents, latents, need_weights=False)
        pooled = self.pool_norm(pooled[:, 0, :])
        return self.pool_proj(pooled)

    def encode(self, model_input: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, seq_len = model_input.shape
        if seq_len != self.L:
            raise ValueError(f"Expected L={self.L}, got {seq_len}")
        c = self.label_mlp(label)
        geo = self.geo_mlp(self.geo_tokens)
        detector_tokens = self.in_proj(model_input.permute(0, 2, 1)) + geo
        cls = self.cls_tokens.expand(bsz, -1, -1)
        h = torch.cat([cls, detector_tokens], dim=1)
        for block in self.encoder_blocks:
            h = block(h, c)
        return h[:, : self.num_latents], c

    def decode(self, latents: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        geo = self.geo_mlp(self.geo_tokens)
        h = self.decoder_query.expand(latents.shape[0], -1, -1) + geo
        for block in self.decoder_blocks:
            h = block(h, latents, c)
        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        h = _modulate(self.final_norm(h), shift, scale)
        active_values = F.softplus(self.value_proj(h)).permute(0, 2, 1)
        hit_logits = self.hit_head(h).squeeze(-1)
        return active_values, hit_logits

    def forward(
        self, model_input: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latents, c = self.encode(model_input, label)
        active_values, hit_logits = self.decode(latents, c)
        pooled_latent = self.pool_latents(latents)
        return active_values, hit_logits, latents, pooled_latent


# ============================================================================
# Loss / plots / checkpoints
# ============================================================================

def compose_reconstruction(active_values: torch.Tensor, hit_logits: torch.Tensor) -> torch.Tensor:
    hit_prob = torch.sigmoid(hit_logits).unsqueeze(1)
    return hit_prob * active_values


def reconstruction_loss(
    active_values: torch.Tensor,
    target: torch.Tensor,
    hit_logits: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    value_mse_per_token = (active_values - target).pow(2).mean(dim=1)
    if active_mask.any():
        value_loss = value_mse_per_token[active_mask].mean()
    else:
        value_loss = torch.zeros((), device=target.device, dtype=target.dtype)
    hit_loss = F.binary_cross_entropy_with_logits(hit_logits, active_mask.float())
    recon = compose_reconstruction(active_values, hit_logits)
    recon_mse_per_token = (recon - target).pow(2).mean(dim=1)
    if (~active_mask).any():
        inactive_recon_loss = recon_mse_per_token[~active_mask].mean()
    else:
        inactive_recon_loss = torch.zeros((), device=target.device, dtype=target.dtype)
    loss = (
        active_value_weight * value_loss
        + hit_bce_weight * hit_loss
        + inactive_recon_weight * inactive_recon_loss
    )
    with torch.no_grad():
        active_loss = (
            value_mse_per_token[active_mask].mean()
            if active_mask.any()
            else torch.zeros((), device=target.device)
        )
    return loss, {
        "active_value_mse": float(value_loss.detach().cpu()),
        "hit_bce": float(hit_loss.detach().cpu()),
        "active_mse": float(active_loss.detach().cpu()),
        "inactive_recon_mse": float(inactive_recon_loss.detach().cpu()),
    }


def _plot_event_panel(ax, sig: np.ndarray, geo: np.ndarray, title: str, marker_size: float):
    x, y, z = np.asarray(geo[0]), np.asarray(geo[1]), np.asarray(geo[2])
    npe = np.asarray(sig[0])
    ftime = np.asarray(sig[1])
    finite = np.isfinite(npe) & np.isfinite(ftime)
    hit = finite & ((npe > 0.0) | (ftime > 0.0))
    event_vis._draw_detector_hull(ax, x, y, z)
    ax.scatter(x, y, z, s=1, c="gray", alpha=0.2)
    if hit.any():
        vals = ftime[hit]
        t_min = float(np.min(vals))
        t_max = float(np.max(vals)) if float(np.max(vals)) > t_min else t_min + 1.0
        npe_max = max(float(np.max(npe[hit])), 1.0)
        sizes = marker_size * (0.35 + 1.65 * np.clip(npe[hit] / npe_max, 0.0, 1.0))
        sc = ax.scatter(
            x[hit],
            y[hit],
            z[hit],
            c=ftime[hit],
            s=sizes,
            cmap="jet",
            norm=Normalize(vmin=t_min, vmax=t_max),
            alpha=0.85,
            edgecolors="none",
        )
        cbar = ax.figure.colorbar(sc, ax=ax, shrink=0.58, aspect=20, pad=0.08)
        cbar.set_label("FirstTime (ns)", rotation=270, labelpad=18)
    event_vis._style_axes(ax)
    ax.set_title(title, fontsize=12)


def save_reconstruction_plot(
    real_sig: np.ndarray,
    recon_sig: np.ndarray,
    geo: np.ndarray,
    output_path: Path,
    *,
    title_prefix: str,
):
    fig = plt.figure(figsize=epoch_compare_figure_size)
    fig.suptitle(title_prefix, fontsize=14, y=0.98)
    ax_left = fig.add_subplot(121, projection="3d")
    ax_right = fig.add_subplot(122, projection="3d")
    _plot_event_panel(ax_left, real_sig, geo, "Real event", epoch_compare_marker_size)
    _plot_event_panel(ax_right, recon_sig, geo, "AE reconstruction", epoch_compare_marker_size)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_histogram_plot(real_sig: np.ndarray, recon_sig: np.ndarray, output_path: Path, *, title_prefix: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title_prefix, fontsize=14, y=0.98)
    panels = [
        (axes[0, 0], real_sig[0].ravel(), "Real nPE"),
        (axes[0, 1], recon_sig[0].ravel(), "Recon nPE"),
        (axes[1, 0], real_sig[1].ravel(), "Real FirstTime"),
        (axes[1, 1], recon_sig[1].ravel(), "Recon FirstTime"),
    ]
    for ax, arr, title in panels:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            arr = np.array([0.0])
        ax.hist(arr, bins=80, density=True, color="steelblue", alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _to_serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def build_run_config(device: torch.device, dataset_len: int, train_size: int, val_size: int) -> dict:
    return {
        "output_dir": output_dir,
        "device": device,
        "seed": seed,
        "h5_path": h5_path,
        "dataset_len": dataset_len,
        "train_size": train_size,
        "val_size": val_size,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "train_keep_ratio": train_keep_ratio,
        "epoch_data_keep_ratio": epoch_data_keep_ratio,
        "val_ratio": val_ratio,
        "active_value_weight": active_value_weight,
        "hit_bce_weight": hit_bce_weight,
        "inactive_recon_weight": inactive_recon_weight,
        "signal_normalization": {
            "method": "log1p_active_p95",
            "npe_clip": npe_clip,
            "ftime_clip": ftime_clip,
            "p95_log_npe": float(signal_norm_stats["p95_log_npe"]),
            "p95_log_ftime": float(signal_norm_stats["p95_log_ftime"]),
        },
        "model": {
            "type": "CLSConditionedTransformerAutoEncoder",
            "d_model": model_d_model,
            "nhead": model_nhead,
            "encoder_depth": model_encoder_depth,
            "decoder_depth": model_decoder_depth,
            "num_latents": model_num_latents,
            "fm_latent_mode": "attention_pool",
            "fm_global_latent_dim": model_d_model,
            "mlp_ratio": model_mlp_ratio,
            "dropout": model_dropout,
            "label_dim": model_label_dim,
        },
    }


def save_run_config_yaml(config: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(_to_serializable(config), f, sort_keys=False, default_flow_style=False)


def build_checkpoint_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    scaler,
    epoch: int,
    train_loss: float | None,
    val_loss: float | None,
    best_val_loss: float,
    run_config: dict,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "train_loss": None if train_loss is None else float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        "best_val_loss": float(best_val_loss),
        "run_config": _to_serializable(run_config),
        "signal_norm_stats": _to_serializable(signal_norm_stats),
    }


def save_training_metrics(output_dir: Path, train_hist: list[float], val_epoch_hist: list[int], val_hist: list[float], lr_hist: list[float]):
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True, parents=True)
    epochs = np.arange(1, len(train_hist) + 1, dtype=np.int32)
    np.savez_compressed(
        metrics_dir / "ae_training_metrics.npz",
        epoch=epochs,
        train_loss=np.asarray(train_hist, dtype=np.float32),
        val_epoch=np.asarray(val_epoch_hist, dtype=np.int32),
        val_loss=np.asarray(val_hist, dtype=np.float32),
        lr=np.asarray(lr_hist, dtype=np.float32),
    )
    with (metrics_dir / "ae_training_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
        val_map = {int(e): float(v) for e, v in zip(val_epoch_hist, val_hist)}
        for epoch, train_loss, lr_value in zip(epochs.tolist(), train_hist, lr_hist):
            val_loss = val_map.get(int(epoch), "")
            writer.writerow([epoch, f"{train_loss:.10f}", val_loss if val_loss == "" else f"{val_loss:.10f}", f"{lr_value:.10e}"])


def _apply_runtime_speed_optimizations():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        cpu_threads = os.cpu_count() or 1
        torch.set_num_threads(max(1, min(cpu_threads, 16)))
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
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass


def _get_amp_dtype(device: torch.device):
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return None


def _build_reproducible_subset(dataset, keep_ratio: float, seed_value: int):
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
    if keep_ratio >= 1.0:
        return dataset, None
    total_len = len(dataset)
    keep_len = max(1, int(round(total_len * keep_ratio)))
    generator = torch.Generator().manual_seed(seed_value)
    selected_indices = torch.randperm(total_len, generator=generator).tolist()[:keep_len]
    return Subset(dataset, selected_indices), selected_indices


def _sample_epoch_subset(dataset, keep_ratio: float):
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
    if keep_ratio >= 1.0:
        return dataset, None
    total_len = len(dataset)
    keep_len = max(1, int(round(total_len * keep_ratio)))
    selected_indices = torch.randperm(total_len).tolist()[:keep_len]
    return Subset(dataset, selected_indices), selected_indices


# ============================================================================
# Main
# ============================================================================

def main():
    _apply_runtime_speed_optimizations()
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_default_device()
    amp_dtype = _get_amp_dtype(device)
    output_dir.mkdir(exist_ok=True, parents=True)
    model_save_dir = output_dir / "models"
    plot_save_dir = output_dir / "plots"
    model_save_dir.mkdir(exist_ok=True, parents=True)
    plot_save_dir.mkdir(exist_ok=True, parents=True)

    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print("Current setup:")
    print(
        f"  model: d_model={model_d_model}, nhead={model_nhead}, "
        f"enc={model_encoder_depth}, dec={model_decoder_depth}, cls_latents={model_num_latents}"
    )
    print(
        f"  training: batch_size={batch_size}, epochs={num_epochs}, lr={lr}, "
        f"val_ratio={val_ratio}, num_workers={num_workers}"
    )
    print(f"  epoch data keep ratio: {epoch_data_keep_ratio}")
    print(
        f"  losses: active_value_weight={active_value_weight}, "
        f"hit_bce_weight={hit_bce_weight}, inactive_recon_weight={inactive_recon_weight}"
    )
    print(f"  FM latent: attention_pool -> dim={model_d_model}")
    print(f"  runtime: compile_model={compile_model}, save_plots={save_plots}, val_every={val_every}, amp_dtype={amp_dtype}")
    print("Signal normalization:")
    print(f"  nPE p95(log1p active): {signal_norm_stats['p95_log_npe']:.6g}")
    print(f"  FirstTime p95(log1p active): {signal_norm_stats['p95_log_ftime']:.6g}")

    dataset = H5Dataset(
        h5_path=h5_path,
        preload_geometry=True,
        angle_conversion=data_angle_conversion,
    )
    train_size = int((1.0 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_dataset, selected_train_indices = _build_reproducible_subset(train_dataset, train_keep_ratio, seed)

    run_config = build_run_config(device, len(dataset), len(train_dataset), len(val_dataset))
    if selected_train_indices is not None:
        run_config["selected_train_indices"] = selected_train_indices
    save_run_config_yaml(run_config, output_dir / "config.yaml")

    pin_memory = data_pin_memory if data_pin_memory is not None else (device.type == "cuda")
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

    geo_norm = apply_minmax_geo(geo0, geo_min, geo_max, feature_range=(0, 1)).to(device)
    model = CLSConditionedTransformerAutoEncoder(
        geo=geo_norm,
        d_model=model_d_model,
        nhead=model_nhead,
        encoder_depth=model_encoder_depth,
        decoder_depth=model_decoder_depth,
        num_latents=model_num_latents,
        mlp_ratio=model_mlp_ratio,
        dropout=model_dropout,
        label_dim=model_label_dim,
    ).to(device)

    adamw_kwargs = {"lr": lr}
    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(model.parameters(), fused=True, **adamw_kwargs)
        except TypeError:
            optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=lr_scheduler_min,
        verbose=False,
    )

    try:
        scaler = GradScaler(device.type) if device.type in ("cuda", "mps") else None
    except (TypeError, ValueError):
        scaler = GradScaler() if device.type == "cuda" else None
    print("AMP enabled:", scaler is not None)

    if compile_model:
        try:
            if hasattr(torch, "compile"):
                model = torch.compile(model, mode="reduce-overhead")
                print("Model compilation successful.")
        except Exception as e:
            print(f"Model compilation failed; continuing without compile: {e}")

    print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    print(f"CLS latents: {model_num_latents}, encoder_depth={model_encoder_depth}, decoder_depth={model_decoder_depth}")
    print(f"FM global latent dim: {model_d_model} (attention pooled from {model_num_latents} CLS latents)")

    train_hist = []
    val_hist = []
    val_epoch_hist = []
    lr_hist = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1
    best_checkpoint_path = model_save_dir / "best.pt"
    last_checkpoint_path = model_save_dir / "last.pt"

    if resume_checkpoint is not None:
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if resume_load_optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if resume_load_optimizer and ckpt.get("lr_scheduler_state_dict") is not None:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        if scaler is not None and resume_load_optimizer and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        print(f"Resumed from: {resume_checkpoint} at epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_train_dataset, _epoch_indices = _sample_epoch_subset(train_dataset, epoch_data_keep_ratio)
        train_loader = DataLoader(
            epoch_train_dataset,
            batch_size=batch_size,
            shuffle=data_shuffle,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        model.train()
        running_loss = 0.0
        running_count = 0
        print(
            f"Epoch {epoch}: using {len(epoch_train_dataset):,}/{len(train_dataset):,} "
            f"train samples ({epoch_data_keep_ratio:.0%})"
        )
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, (sig, _geo, label) in enumerate(pbar, start=1):
            sig = sig.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            model_input, x0, label_norm, active_mask = prepare_batch(sig, label)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast(device_type=device.type, dtype=amp_dtype):
                    active_values, hit_logits, _latents, _pooled_latent = model(model_input, label_norm)
                    loss, loss_parts = reconstruction_loss(active_values, x0, hit_logits, active_mask)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                active_values, hit_logits, _latents, _pooled_latent = model(model_input, label_norm)
                loss, loss_parts = reconstruction_loss(active_values, x0, hit_logits, active_mask)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running_loss += float(loss.detach().cpu()) * sig.shape[0]
            running_count += sig.shape[0]
            if batch_idx % print_every == 0:
                pbar.set_postfix(
                    loss=f"{running_loss / max(running_count, 1):.6f}",
                    active=f"{loss_parts['active_mse']:.6f}",
                    inactive=f"{loss_parts['inactive_recon_mse']:.6f}",
                )

        epoch_train_loss = running_loss / max(running_count, 1)
        train_hist.append(epoch_train_loss)
        lr_hist.append(float(optimizer.param_groups[0]["lr"]))

        val_loss = None
        if epoch % val_every == 0:
            model.eval()
            val_running = 0.0
            val_count = 0
            with torch.no_grad():
                for sig, _geo, label in tqdm(val_loader, desc="Validation"):
                    sig = sig.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)
                    model_input, x0, label_norm, active_mask = prepare_batch(sig, label)
                    with autocast(device_type=device.type, dtype=amp_dtype, enabled=(scaler is not None)):
                        active_values, hit_logits, _latents, _pooled_latent = model(model_input, label_norm)
                        loss, _loss_parts = reconstruction_loss(active_values, x0, hit_logits, active_mask)
                    val_running += float(loss.detach().cpu()) * sig.shape[0]
                    val_count += sig.shape[0]
            val_loss = val_running / max(val_count, 1)
            val_hist.append(val_loss)
            val_epoch_hist.append(epoch)
            lr_scheduler.step(val_loss)
            print(f"Epoch {epoch}: train={epoch_train_loss:.6f}, val={val_loss:.6f}")

            if val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(
                    build_checkpoint_state(model, optimizer, lr_scheduler, scaler, epoch, epoch_train_loss, val_loss, best_val_loss, run_config),
                    best_checkpoint_path,
                )
                print(f"Best checkpoint saved: {best_checkpoint_path}")
            else:
                epochs_without_improvement += 1
        else:
            print(f"Epoch {epoch}: train={epoch_train_loss:.6f}")

        torch.save(
            build_checkpoint_state(model, optimizer, lr_scheduler, scaler, epoch, epoch_train_loss, val_loss, best_val_loss, run_config),
            last_checkpoint_path,
        )
        save_training_metrics(output_dir, train_hist, val_epoch_hist, val_hist, lr_hist)

        if save_plots and epoch % epoch_compare_every == 0:
            model.eval()
            with torch.no_grad():
                for plot_idx, val_idx in enumerate(epoch_compare_val_indices[:epoch_compare_num_samples]):
                    sig_ref, geo_ref, label_ref = val_dataset[val_idx]
                    sig_ref_b = sig_ref.unsqueeze(0).to(device)
                    label_ref_b = label_ref.unsqueeze(0).to(device)
                    model_input, x0, label_norm, _active_mask = prepare_batch(sig_ref_b, label_ref_b)
                    active_values, hit_logits, _latents, _pooled_latent = model(model_input, label_norm)
                    recon_norm = compose_reconstruction(active_values, hit_logits)
                    real_denorm = _denormalize_signal(x0).squeeze(0).detach().cpu().numpy()
                    recon_denorm = _denormalize_signal(recon_norm).squeeze(0).detach().cpu().numpy()
                    save_reconstruction_plot(
                        real_denorm,
                        recon_denorm,
                        geo_ref.numpy(),
                        plot_save_dir / f"epoch_{epoch:03d}_recon_{plot_idx}.png",
                        title_prefix=f"AE reconstruction epoch {epoch} sample {plot_idx}",
                    )
                    save_histogram_plot(
                        real_denorm,
                        recon_denorm,
                        plot_save_dir / f"epoch_{epoch:03d}_hist_{plot_idx}.png",
                        title_prefix=f"AE histograms epoch {epoch} sample {plot_idx}",
                    )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}; best_val_loss={best_val_loss:.6f}")
            break

    final_path = model_save_dir / "model_checkpoint_final.pt"
    torch.save(
        build_checkpoint_state(model, optimizer, lr_scheduler, scaler, epoch, train_hist[-1], val_hist[-1] if val_hist else None, best_val_loss, run_config),
        final_path,
    )
    print(f"Final checkpoint saved: {final_path}")
    print(f"Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
