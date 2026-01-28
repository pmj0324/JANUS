#!/usr/bin/env python3
"""
Training script for GENESIS diffusion model.
Converted from train.ipynb
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add GENESIS to path
sys.path.insert(0, os.path.join(os.getcwd(), "GENESIS"))

from dataloader.h5 import H5Dataset
from diffusion.schedules import sigmoid_beta_schedule
from diffusion.forward import apply_forward_diffusion
from utils.normalize import normalize, denormalize_log_minmax
from utils.vis.event_show import show_event_dual_plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---- config (원하는대로 바꿔도 됨) ----
T = 1000
beta_start, beta_end = 1e-4, 2e-2  # sigmoid schedule params

batch_size = 1
num_workers = 0  # 노트북에서는 0 추천
lr = 3e-4
steps = 500
print_every = 25

# 데이터/정규화 (normalize_sig 방식): clamp 후 log1p + minmax to [-1, 1]
npe_clip = 1000.0
ftime_clip = 8.0
log_min = 0.0
npe_log_max = float(np.log1p(npe_clip))
ftime_log_max = float(np.log1p(ftime_clip))
_feature_range = (-1, 1)

# label 정규화: [Energy, ux, uy, X, Y, Z] -> Energy log_minmax, ux/uy identity, X/Y/Z minmax (dataset min/max)
energy_clip_pev = 100.0
energy_log_max = float(np.log1p(energy_clip_pev))
_label_methods = ["log_minmax", "identity", "identity", "minmax", "minmax", "minmax"]
_label_feature_ranges = [_feature_range] * 6
# X,Y,Z min/max: 22644_0921_time_shift.h5 전체 데이터셋 기준 (하드코딩)
_LABEL_XYZ_MINMAX = [
    {"min": -570.9000244140625, "max": 576.3699951171875},   # X
    {"min": -521.0800170898438, "max": 509.5},               # Y
    {"min": -509.8599853515625, "max": 506.0566711425781},   # Z
]
_label_stats = [
    {"log_min": log_min, "log_max": energy_log_max},
    {},
    {},
    _LABEL_XYZ_MINMAX[0],
    _LABEL_XYZ_MINMAX[1],
    _LABEL_XYZ_MINMAX[2],
]

LABEL_NAMES = ["Energy (PeV)", "ux", "uy", "X", "Y", "Z"]


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


# h5 자동 탐색
h5_path = "/Users/monocerotis/0121/git0121/GENESIS/GENESIS-data/22644_0921_time_shift.h5"

# ---- dataset / dataloader ----
dataset = H5Dataset(h5_path=h5_path)
_print_label_normalize_config()

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

print("dataset length:", len(dataset))
sig0, geo0, label0 = dataset[0]
print("sig:", sig0.shape, sig0.dtype)
print("geo:", geo0.shape, geo0.dtype)
print("label:", label0.shape, label0.dtype, label0)

# ---- normalization: decorator-wrapped prepare_batch (normalize_sig 동일 방식) ----
# normalize_sig: clamp npe [0, npe_clip], ftime [0, ftime_clip]; then log_minmax per channel
#   data_min=log_min, data_max=npe_log_max / ftime_log_max, feature_range=(-1, 1)
_channel_stats = [
    {"log_min": log_min, "log_max": npe_log_max},
    {"log_min": log_min, "log_max": ftime_log_max},
]

@normalize(
    channel_methods=["log_minmax", "log_minmax"],
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
    """입력: (sig, label). 데코레이터가 sig는 [-1,1] log_minmax, label은 Energy log_minmax / ux,uy identity / X,Y,Z minmax.
    출력: (sig_norm, label_norm). verbose=True면 print."""
    if verbose:
        print("prepare_batch: label", label)
    return (sig, label)


def _clamp_sig(sig: torch.Tensor) -> torch.Tensor:
    """Clamp npe/ftime before normalize (normalize_sig와 동일). Returns clamped copy."""
    s = sig.clone()
    s[:, 0] = torch.clamp(s[:, 0], min=0.0, max=npe_clip)
    s[:, 1] = torch.clamp(s[:, 1], min=0.0, max=ftime_clip)
    return s


def denormalize_sig(sig: torch.Tensor) -> torch.Tensor:
    """정규화된 sig ([-1, 1] log_minmax)를 원 스케일로 역정규화. prepare_batch 출력용."""
    out = sig.clone()
    if sig.dim() == 3:
        out[:, 0, :] = denormalize_log_minmax(sig[:, 0, :], log_min, npe_log_max, _feature_range)
        out[:, 1, :] = denormalize_log_minmax(sig[:, 1, :], log_min, ftime_log_max, _feature_range)
    else:
        out[0, :] = denormalize_log_minmax(sig[0, :], log_min, npe_log_max, _feature_range)
        out[1, :] = denormalize_log_minmax(sig[1, :], log_min, ftime_log_max, _feature_range)
    return out


def sample_timesteps(batch: int, T: int, device: torch.device) -> torch.Tensor:
    """t in [0, T] (note: forward code treats t=0 as clean)"""
    return torch.randint(low=0, high=T + 1, size=(batch,), device=device, dtype=torch.long)

# ---- sigmoid noise schedule ----
betas = sigmoid_beta_schedule(timesteps=T, beta_start=beta_start, beta_end=beta_end).to(device)
print("betas:", betas.shape, betas.min().item(), betas.max().item())

# ---- Transformer (DiT-style): token=DOM, conditional on t + label ----

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    t: (B,) int/float
    return: (B, dim)
    """
    if t.dim() != 1:
        t = t.view(-1)
    t = t.float()

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )  # (half,)
    args = t[:, None] * freqs[None, :]  # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device, dtype=emb.dtype)], dim=-1)
    return emb


class DiTBlock(nn.Module):
    """
    DiT-style Transformer block with AdaLN modulation from conditioning vector c.
    x: (B, L, d)
    c: (B, d)
    """
    def __init__(self, d: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

        # cond -> (shift1, scale1, gate1, shift2, scale2, gate2)
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, 6 * d),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        params = self.ada(c).view(B, 6, d)
        shift1, scale1, gate1, shift2, scale2, gate2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]

        # Attention
        x1 = self.norm1(x)
        x1 = x1 * (1.0 + scale1[:, None, :]) + shift1[:, None, :]
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + gate1[:, None, :] * attn_out

        # MLP
        x2 = self.norm2(x)
        x2 = x2 * (1.0 + scale2[:, None, :]) + shift2[:, None, :]
        mlp_out = self.mlp(x2)
        x = x + gate2[:, None, :] * mlp_out

        return x


class DiffusionDiTTransformer(nn.Module):
    def __init__(
        self,
        geo: torch.Tensor,          # (3, 5160) or (1, 3, 5160)
        d_model: int = 256,
        nhead: int = 8,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        label_dim: int = 6,
    ):
        super().__init__()
        self.d_model = d_model

        # ---- geo buffer (고정) ----
        # geo: (3, L) -> (1, L, 3)
        if geo.dim() == 2:
            geo_tok = geo.transpose(0, 1).unsqueeze(0)  # (1, L, 3)
        elif geo.dim() == 3:
            # (1, 3, L) -> (1, L, 3)
            geo_tok = geo.permute(0, 2, 1)
        else:
            raise ValueError(f"geo must be (3,L) or (1,3,L). got {geo.shape}")

        self.register_buffer("geo_tokens", geo_tok.contiguous(), persistent=True)
        L = self.geo_tokens.shape[1]
        self.L = L

        # ---- input embedding: (B, L, 2) -> (B, L, d) ----
        self.in_proj = nn.Linear(2, d_model)

        # ---- geo positional embedding: (1, L, 3) -> (1, L, d) ----
        self.geo_mlp = nn.Sequential(
            nn.Linear(3, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.use_index_pos = False
        if self.use_index_pos:
            self.index_pos = nn.Parameter(torch.zeros(1, L, d_model))

        # ---- conditioning: time + label -> (B, d) ----
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

        # ---- DiT blocks ----
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        # ---- final layer (DiT 스타일로 마지막에도 cond로 LN modulation) ----
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),  # shift, scale
        )
        self.out_proj = nn.Linear(d_model, 2)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, 2, L)
        t: (B,)
        label: (B, 6)
        return: eps_hat (B, 2, L)
        """
        B, C, L = x_t.shape
        assert L == self.L, f"Expected L={self.L}, got L={L}"

        # (B, 2, L) -> (B, L, 2)
        tokens = x_t.permute(0, 2, 1)
        h = self.in_proj(tokens)  # (B, L, d)

        # geo positional embedding
        pos_geo = self.geo_mlp(self.geo_tokens)  # (1, L, d)
        h = h + pos_geo

        if self.use_index_pos:
            h = h + self.index_pos

        # conditioning
        t_emb = sinusoidal_timestep_embedding(t, self.d_model)  # (B, d)
        c = self.time_mlp(t_emb) + self.label_mlp(label)        # (B, d)

        # DiT blocks
        for blk in self.blocks:
            h = blk(h, c)

        # final AdaLN + output
        shift, scale = self.final_ada(c).chunk(2, dim=-1)       # (B,d), (B,d)
        h = self.final_norm(h)
        h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]
        out = self.out_proj(h)                                   # (B, L, 2)
        return out.permute(0, 2, 1)                             # (B, 2, L)


# Fixed geo from first sample (model uses single geometry for all samples)
_geo = dataset[0][1]  # (3, L)
model = DiffusionDiTTransformer(geo=_geo, d_model=256, nhead=8, depth=6, mlp_ratio=4.0, dropout=0.0, label_dim=6).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=lr)

print("params:", sum(p.numel() for p in model.parameters())/1e6, "M")

# ---- training loop (objective: eps) ----
model.train()

it = iter(loader)
loss_hist = []

for step in range(1, steps + 1):
    try:
        sig, geo, label = next(it)
    except StopIteration:
        it = iter(loader)
        sig, geo, label = next(it)

    sig = sig.to(device)         # (B, 2, L)
    label = label.to(device)     # (B, 6)

    sig_clamp = _clamp_sig(sig)
    if step == 1:
        _label_raw_before = label.clone()
    x0, label = prepare_batch(sig_clamp, label, verbose=(step == 1))  # (B, 2, L) in [-1, 1]
    if step == 1:
        print("prepare_batch: label (raw, same batch)", _label_raw_before)
        for j in [1, 2]:
            ok = torch.allclose(_label_raw_before[:, j], label[:, j])
            print(f"  identity col {j} ({LABEL_NAMES[j]}): {'OK' if ok else 'MISMATCH'} raw={_label_raw_before[0, j].item():.6f} norm={label[0, j].item():.6f}")

    B = x0.shape[0]
    t = sample_timesteps(B, T, device)

    noise = torch.randn_like(x0)
    x_t = apply_forward_diffusion(x0=x0, betas=betas, timesteps=t, noise=noise)

    eps_hat = model(x_t, t, label)
    loss = F.mse_loss(eps_hat, noise)

    optim.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

    loss_hist.append(float(loss.item()))

    if step % print_every == 0:
        print(f"step {step:5d} | loss {np.mean(loss_hist[-print_every:]):.6f}")

print("done")

# ---- quick visualization: one event, different t ----
model.eval()

event_idx = 0
sig_raw, geo_raw, label_raw = dataset[event_idx]

geo_np = geo_raw.detach().cpu().numpy()
label_np = label_raw.detach().cpu().numpy()

sig = sig_raw.unsqueeze(0).to(device)  # (1,2,L)
label = label_raw.unsqueeze(0).to(device)  # (1,6)

sig_clamp = _clamp_sig(sig)
x0, label = prepare_batch(sig_clamp, label, verbose=True)

for t_val in [0, 250, 500, 750, 1000]:
    t = torch.tensor([t_val], device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = apply_forward_diffusion(x0=x0, betas=betas, timesteps=t, noise=noise)
    # 출력 역정규화 후 시각화 (prepare_batch 출력용 denormalize_sig)
    x_t_denorm = denormalize_sig(x_t)[0].detach().cpu().numpy()

    fig, _ = show_event_dual_plot(
        sig=x_t_denorm,
        geo=geo_np,
        label=label_np,
        figure_size=(18, 8),
        marker_size=8.0,
        show_detector_hull=True,
        show=False,
        title_prefix=f"train_exp.py | Sigmoid schedule | event {event_idx} | t={t_val}",
        firsttime_title="FirstTime (x_t, denorm)",
        npe_title="nPE (x_t, denorm)",
    )
    plt.show()
