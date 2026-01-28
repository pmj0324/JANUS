#!/usr/bin/env python3
"""
Training script for GENESIS diffusion model.
Converted from train.ipynb
"""

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
from utils.normalize import apply_log_minmax
from utils.vis.event_show import show_event_dual_plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---- config (원하는대로 바꿔도 됨) ----
T = 1000
beta_start, beta_end = 1e-4, 2e-2  # sigmoid schedule params

batch_size = 16
num_workers = 6  # 노트북에서는 0 추천
lr = 3e-4
steps = 500
print_every = 25

# 데이터/정규화: log1p 후 [-1, 1] minmax
npe_clip = 1000.0
ftime_clip = 8.0
log_min = 0.0
npe_log_max = float(np.log1p(npe_clip))
ftime_log_max = float(np.log1p(ftime_clip))

# h5 자동 탐색
h5_path = None
for p in [
    "GENESIS/GENESIS-data/22644_0921_time_shift.h5",
    "./GENESIS-data/22644_0921_time_shift.h5",
    "../GENESIS-data/22644_0921_time_shift.h5",
]:
    if os.path.exists(p):
        h5_path = p
        break

if h5_path is None:
    raise FileNotFoundError("H5 파일을 찾지 못했습니다. h5_path 후보를 추가하거나 직접 지정해주세요.")

print("h5_path:", h5_path)

# ---- dataset / dataloader ----
dataset = H5Dataset(h5_path=h5_path)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

print("dataset length:", len(dataset))
sig0, geo0, label0 = dataset[0]
print("sig:", sig0.shape, sig0.dtype)
print("geo:", geo0.shape, geo0.dtype)
print("label:", label0.shape, label0.dtype)

# ---- normalization helpers ----

def normalize_sig(sig: torch.Tensor) -> torch.Tensor:
    """sig: (B, 2, L) -> normalized to [-1, 1]"""
    sig = sig.clone()
    # clamp to avoid weird values
    sig[:, 0] = torch.clamp(sig[:, 0], min=0.0, max=npe_clip)
    sig[:, 1] = torch.clamp(sig[:, 1], min=0.0, max=ftime_clip)

    # apply_log_minmax expects (log space) data_min/max
    sig[:, 0] = apply_log_minmax(sig[:, 0], feature_range=(-1, 1), data_min=log_min, data_max=npe_log_max)
    sig[:, 1] = apply_log_minmax(sig[:, 1], feature_range=(-1, 1), data_min=log_min, data_max=ftime_log_max)
    return sig


def sample_timesteps(batch: int, T: int, device: torch.device) -> torch.Tensor:
    """t in [0, T] (note: forward code treats t=0 as clean)"""
    return torch.randint(low=0, high=T + 1, size=(batch,), device=device, dtype=torch.long)

# ---- sigmoid noise schedule ----
betas = sigmoid_beta_schedule(timesteps=T, beta_start=beta_start, beta_end=beta_end).to(device)
print("betas:", betas.shape, betas.min().item(), betas.max().item())

# ---- Transformer (DiT-like baseline): token=DOM, conditional on t + label ----

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) -> (B, dim)"""
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        label_dim: int = 6,
    ):
        super().__init__()
        self.d_model = d_model

        # (B, 2, L) -> tokens (B, L, d)
        self.in_proj = nn.Linear(2, d_model)

        # time + label conditioning -> (B, d)
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

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(d_model, 2)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, 2, L)
        t: (B,)
        label: (B, 6)
        return eps_hat: (B, 2, L)
        """
        B, C, L = x_t.shape
        # tokens: (B, L, 2)
        tokens = x_t.permute(0, 2, 1)
        h = self.in_proj(tokens)  # (B, L, d)

        t_emb = sinusoidal_timestep_embedding(t, self.d_model)
        t_cond = self.time_mlp(t_emb)  # (B, d)
        y_cond = self.label_mlp(label)  # (B, d)
        cond = (t_cond + y_cond).unsqueeze(1)  # (B, 1, d)

        h = h + cond
        h = self.encoder(h)
        out = self.out_proj(h)  # (B, L, 2)
        return out.permute(0, 2, 1)  # (B, 2, L)


model = DiffusionTransformer().to(device)
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

    x0 = normalize_sig(sig)      # (B, 2, L) in [-1, 1]

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

x0 = normalize_sig(sig)

for t_val in [0, 250, 500, 750, 999]:
    t = torch.tensor([t_val], device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = apply_forward_diffusion(x0=x0, betas=betas, timesteps=t, noise=noise)
    sig_t = x_t[0].detach().cpu().numpy()

    fig, _ = show_event_dual_plot(
        sig=sig_t,
        geo=geo_np,
        label=label_np,
        figure_size=(18, 8),
        marker_size=8.0,
        show_detector_hull=True,
        show=False,
        title_prefix=f"train_exp.py | Sigmoid schedule | event {event_idx} | t={t_val}",
        firsttime_title="FirstTime (x_t)",
        npe_title="nPE (x_t)",
    )
    plt.show()
