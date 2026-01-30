#!/usr/bin/env python3
"""4개 스케줄러: β(t) 2×2 한 장, ᾱ(t) 2×2 한 장."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from diffusion.schedules import (
    linear_beta_schedule,
    cosine_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
    compute_alpha_schedule,
)

T = 1000
schedules = [
    ("Linear", linear_beta_schedule, {"timesteps": T}),
    ("Cosine", cosine_beta_schedule, {"timesteps": T, "s": 0.008}),
    ("Quadratic", quadratic_beta_schedule, {"timesteps": T}),
    ("Sigmoid", sigmoid_beta_schedule, {"timesteps": T}),
]

t = np.arange(1, T + 1, dtype=np.float32)
data = []
for name, fn, kwargs in schedules:
    betas = fn(**kwargs)
    alpha_schedule = compute_alpha_schedule(betas)
    alpha_bar = alpha_schedule["alphas_cumprod"].numpy()
    data.append((name, betas.numpy(), alpha_bar))

base = Path(__file__).parent

# 1) Beta 그림 (2×2)
fig_beta, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
for idx, (name, betas, _) in enumerate(data):
    ax = axes[idx]
    ax.plot(t, betas, color="C0", lw=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\beta_t$")
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, T)
plt.suptitle(r"$\beta_t$ schedule (T=1000)", fontsize=12, y=1.02)
plt.tight_layout()
out_beta = base / "schedule_beta_2x2.png"
plt.savefig(out_beta, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_beta}")

# 2) Alpha_bar 그림 (2×2)
fig_alpha, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
for idx, (name, _, alpha_bar) in enumerate(data):
    ax = axes[idx]
    ax.plot(t, alpha_bar, color="C1", lw=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\bar{\alpha}_t$")
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, T)
plt.suptitle(r"$\bar{\alpha}_t$ schedule (T=1000)", fontsize=12, y=1.02)
plt.tight_layout()
out_alpha = base / "schedule_alpha_2x2.png"
plt.savefig(out_alpha, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_alpha}")
