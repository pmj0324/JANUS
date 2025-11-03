#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmt-dit.py

Conditional diffusion for p(x|c) where:
  - x = PMT signals (npe, time) over L PMTs
  - c = event-level label (Energy, Zenith, Azimuth, X, Y, Z)
  - geom = per-PMT geometry (xpmt, ypmt, zpmt) used as non-noisy conditioning

Model: DiT-style Transformer with
  - Separate signal/geom emb (2->H, 3->H)
  - Fusion: SUM or FiLM(label) on signal emb, then + geom emb
  - adaLN-Zero conditioning with (timestep + label) per block
Output: epsilon prediction for signals only → shape (B, 2, L)

Author: Minje Park 
        - SKKU 
        - pmj032400@naver.com
        - github ID : pmj0324 
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Sinusoidal timestep embedding
# -------------------------
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: (B,) int/float in [0, T)
    return: (B, dim)
    """
    device = t.device
    half = dim // 2
    if half == 0:
        return torch.zeros(t.shape[0], dim, device=device)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device) / half)
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# -------------------------
# adaLN-Zero (as in DiT)
# -------------------------
class AdaLNZero(nn.Module):
    """
    LayerNorm (no affine) + scale/shift from condition.
    Zero-init so residual starts as identity.
    """
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_scale = nn.Linear(cond_dim, hidden_dim)
        self.to_shift = nn.Linear(cond_dim, hidden_dim)
        nn.init.zeros_(self.to_scale.weight); nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight); nn.init.zeros_(self.to_shift.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B,L,H), c: (B,cond_dim)
        h = self.ln(x)
        s = self.to_scale(c)[:, None, :]  # (B,1,H)
        b = self.to_shift(c)[:, None, :]
        return h * (1 + s) + b


class MLP(nn.Module):
    def __init__(self, hidden: int, ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        inner = int(hidden * ratio)
        self.fc1 = nn.Linear(hidden, inner)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, cond_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ada_attn = AdaLNZero(hidden, cond_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, dropout=dropout, batch_first=True)
        self.ada_mlp = AdaLNZero(hidden, cond_dim)
        self.mlp = MLP(hidden, mlp_ratio, dropout)
        # gated residuals (zero-init)
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_mlp  = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.ada_attn(x, c)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.gate_attn * a
        h = self.ada_mlp(x, c)
        x = x + self.gate_mlp * self.mlp(h)
        return x


# -------------------------
# Signal/Geom embedding + SUM / FiLM(label)
# -------------------------
class PMTEmbedding(nn.Module):
    """
    signal: (npe,time) -> H
    geom:   (x,y,z)    -> H

    Fusion:
      - "SUM" : tokens = h_sig + h_pos
      - "FiLM": tokens = (h_sig * (1+gamma(label)) + beta(label)) + h_pos
                (label is event-level → (B,1,H) and broadcast across L)
    """
    def __init__(
        self,
        hidden: int,
        dropout: float,
        fusion: str = "SUM",
        label_dim: int = 6,
        film_zero_init: bool = True,
    ):
        super().__init__()
        self.hidden = hidden
        self.fusion = fusion.upper()
        assert self.fusion in {"SUM", "FILM"}, "fusion must be 'SUM' or 'FiLM'"

        self.signal_net = nn.Sequential(
            nn.Linear(2, hidden), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.geom_net = nn.Sequential(
            nn.Linear(3, hidden), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        if self.fusion == "FILM":
            self.label_to_film = nn.Sequential(
                nn.Linear(label_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, 2 * hidden),
            )
            if film_zero_init:
                last: nn.Linear = self.label_to_film[-1]  # type: ignore
                nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, signal: torch.Tensor, geom: torch.Tensor, label: Optional[torch.Tensor]) -> torch.Tensor:
        """
        signal: (B,L,2)  = [npe,time]
        geom:   (B,L,3)  = [xpmt,ypmt,zpmt]
        label:  (B,6)    = event-level condition (FiLM only)
        return: tokens (B,L,H)
        """
        h_sig = self.signal_net(signal)  # (B,L,H)
        h_pos = self.geom_net(geom)      # (B,L,H)

        if self.fusion == "SUM":
            return h_sig + h_pos

        assert label is not None, "fusion='FiLM' requires label (B,6)"
        film_vec = self.label_to_film(label)           # (B,2H)
        gamma, beta = torch.chunk(film_vec, 2, dim=-1) # (B,H),(B,H)
        gamma = gamma.unsqueeze(1)  # (B,1,H)
        beta  = beta.unsqueeze(1)   # (B,1,H)
        tokens = h_sig * (1.0 + gamma) + beta
        return tokens + h_pos


# -------------------------
# Conditional DiT (ε-prediction) for signals only
# -------------------------

class PMTDit(nn.Module):
    """
    ε̂(x_sig_t, t, c) predictor for signals (npe,time) given geom & label.

    Inputs:
      - x_sig_t: (B,2,L)   noisy signals at t      [npe, time]
      - geom:    (B,3,L)   geometry (non-noisy)    [xpmt, ypmt, zpmt]
      - t:       (B,)      integer timesteps
      - label:   (B,6)     event-level condition c

    Output:
      - eps_hat: (B,2,L)   noise prediction for signals

    ⚠️ NORMALIZATION POLICY:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. Dataloader에서 정규화 수행 (ln + affine)
    2. Model.forward()는 이미 정규화된 데이터를 받음
    3. Model.forward()는 정규화를 하지 않음! (성능 최적화)
    4. affine_* / label_* 파라미터는 METADATA만 저장
    
    ❓ 자주 묻는 질문:
    
    Q: 왜 Model에 affine 파라미터가 있나요?
    A: Checkpoint 파일에 정규화 정보를 포함하기 위한 metadata입니다.
       Forward pass에서는 전혀 사용하지 않습니다!
       
    Q: Reverse diffusion 후 denormalization은 자동?
    A: 아니요! 수동으로 해야 합니다:
       ```
       norm_params = model.get_normalization_params()
       samples_raw = denormalize_signal(samples_norm, 
                                        norm_params['affine_offsets'],
                                        norm_params['affine_scales'],
                                        norm_params['time_transform'])
       ```
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    channels order = [npe, time, xpmt, ypmt, zpmt]
    """
    def __init__(
        self,
        seq_len: int = 5160,
        hidden: int = 512,
        depth: int = 8,
        heads: int = 8,
        dropout: float = 0.1,
        fusion: str = "SUM",        # "SUM" or "FiLM"
        label_dim: int = 6,
        t_embed_dim: int = 128,
        mlp_ratio: float = 4.0,
        # Normalization metadata (NOT used in forward, only for sampling/denormalization)
        affine_offsets = (0.0, 0.0, 0.0, 0.0, 0.0),  # [npe,time,xpmt,ypmt,zpmt]
        affine_scales  = (1.0, 1.0, 1.0, 1.0, 1.0),   # Default: no normalization
        label_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # [Energy, Zenith, Azimuth, X, Y, Z]
        label_scales = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),    # Default: no normalization
        # Data preprocessing settings (metadata only)
        time_transform = "ln",  # "ln" or "log10" - always use log(1+x)
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.t_embed_dim = t_embed_dim
        self.time_transform = time_transform

        # --- token embedder (기존과 동일) ---
        self.embedder = PMTEmbedding(
            hidden=hidden, dropout=dropout, fusion=fusion, label_dim=label_dim
        )

        # NOTE: No learnable position embedding - geometry (x,y,z) already provides position info

        # --- condition encoders ---
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
        )
        self.y_mlp = nn.Sequential(
            nn.Linear(label_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
        )
        self.cond_dim = (hidden // 2) * 2

        # --- transformer blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(hidden, heads, cond_dim=self.cond_dim, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.ln_out = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, 2)  # (npe,time)만 예측

        # =================================================================
        # NORMALIZATION METADATA
        # =================================================================
        # These parameters are stored for denormalization during sampling.
        # They are NOT used in forward() - normalization happens in Dataloader.
        # =================================================================
        off = torch.tensor(affine_offsets, dtype=torch.float32).reshape(5, 1)
        scl = torch.tensor(affine_scales,  dtype=torch.float32).reshape(5, 1)
        self.register_buffer("affine_offset", off)  # metadata (not used in forward)
        self.register_buffer("affine_scale",  scl)  # metadata (not used in forward)
        
        # Label normalization parameters (metadata)
        label_off = torch.tensor(label_offsets, dtype=torch.float32)
        label_scl = torch.tensor(label_scales, dtype=torch.float32)
        self.register_buffer("label_offset", label_off)  # metadata
        self.register_buffer("label_scale", label_scl)   # metadata

    # =========================================================================
    # METADATA MANAGEMENT
    # =========================================================================
    
    @torch.no_grad()
    def set_affine(self, offsets, scales):
        """
        Update affine normalization metadata (does NOT affect forward).
        
        Args:
            offsets: Length 5 [npe, time, xpmt, ypmt, zpmt]
            scales:  Length 5 [npe, time, xpmt, ypmt, zpmt]
        """
        off = torch.as_tensor(offsets, dtype=torch.float32, device=self.affine_offset.device).reshape(5, 1)
        scl = torch.as_tensor(scales,  dtype=torch.float32, device=self.affine_scale.device).reshape(5, 1)
        self.affine_offset.copy_(off)
        self.affine_scale.copy_(scl)

    @torch.no_grad()
    def reset_affine(self):
        """Reset affine parameters to identity (offset=0, scale=1)."""
        self.affine_offset.zero_()
        self.affine_scale.fill_(1.0)
    
    def get_normalization_params(self):
        """
        Retrieve normalization parameters for denormalization.
        
        Returns:
            dict with keys:
                - affine_offsets: (5,) numpy array [charge, time, x, y, z]
                - affine_scales:  (5,) numpy array
                - label_offsets:  (6,) numpy array [Energy, Zenith, Azimuth, X, Y, Z]
                - label_scales:   (6,) numpy array
                - time_transform: str ("ln" or "log10")
        """
        return {
            "affine_offsets": self.affine_offset.squeeze(-1).cpu().numpy(),  # (5,)
            "affine_scales": self.affine_scale.squeeze(-1).cpu().numpy(),    # (5,)
            "label_offsets": self.label_offset.cpu().numpy(),                # (6,)
            "label_scales": self.label_scale.cpu().numpy(),                  # (6,)
            "time_transform": self.time_transform,
        }

    # ------------------------------------------
    # forward (No normalization here!)
    # ------------------------------------------
    def forward(self, x_sig_t: torch.Tensor, geom: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - expects PRE-NORMALIZED data from dataloader.
        
        Args:
            x_sig_t: (B,2,L) - NORMALIZED signals [charge, time]
            geom:    (B,3,L) - NORMALIZED geometry [x, y, z]
            t:       (B,)    - Diffusion timestep
            label:   (B,6)   - NORMALIZED labels [Energy, Zenith, Azimuth, X, Y, Z]
        
        Returns:
            eps: (B,2,L) - Predicted noise for signals
        
        Note: All inputs are already normalized by the dataloader.
              No normalization is performed here.
        """
        B, Csig, L = x_sig_t.shape
        assert Csig == 2 and L == self.seq_len, f"expected x_sig_t (B,2,{self.seq_len}), got {x_sig_t.shape}"
        assert geom.shape == (B, 3, L), f"expected geom (B,3,{L}), got {geom.shape}"
        
        # =================================================================
        # DATA IS ALREADY NORMALIZED - NO NORMALIZATION HERE
        # =================================================================
        # x_sig_t, geom, label are all pre-normalized by the dataloader
        # We directly use them for model computation
        # =================================================================

        # --------- 1) 토큰화/컨디션 (No position embedding - geo already contains position) ---------
        sig = x_sig_t.transpose(1, 2)   # (B,L,2)
        geo = geom.transpose(1, 2)      # (B,L,3)

        tokens = self.embedder(sig, geo, label)  # (B,L,H) - label already normalized
        # No learned position embedding - geometry (x,y,z) provides position info

        te = timestep_embedding(t, self.t_embed_dim)  # (B,t_dim)
        te = self.t_mlp(te)                           # (B,H/2)
        ye = self.y_mlp(label)                        # (B,H/2) - label already normalized
        cond = torch.cat([te, ye], dim=-1)            # (B,H)

        # --------- 2) Transformer blocks ---------
        h = tokens
        for blk in self.blocks:
            h = blk(h, cond)
        h = self.ln_out(h)
        
        # --------- 3) Output projection ---------
        eps = self.out_proj(h).transpose(1, 2)        # (B,2,L)
        return eps

# -------------------------
# NOTE: GaussianDiffusion has been moved to diffusion/gaussian_diffusion.py
# Import it from there:
#   from diffusion import GaussianDiffusion, DiffusionConfig
# -------------------------

# For backward compatibility, import from diffusion module
# Users should update their code to:
#   from diffusion import GaussianDiffusion, DiffusionConfig
from diffusion import GaussianDiffusion, DiffusionConfig


# -------------------------
# OLD CODE BELOW (KEPT FOR REFERENCE, NOT USED)
# -------------------------
if False:  # This code is not executed, kept for reference only
    # q(x_t | x_0) for signals only
    def q_sample_old(self, x0_sig: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x0_sig: (B,2,L)
        t:      (B,)
        """
        if noise is None:
            noise = torch.randn_like(x0_sig)
        fac1 = self.sqrt_alphas_cumprod[t][:, None, None]
        fac2 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return fac1 * x0_sig + fac2 * noise

    # training objective
    def loss(self, x0_sig: torch.Tensor, geom: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        x0_sig: clean signals (B,2,L)
        geom:   geometry      (B,3,L)
        label:  condition c   (B,6)
        """
        B = x0_sig.size(0)
        device = x0_sig.device
        t = torch.randint(0, self.cfg.timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0_sig)
        x_sig_t = self.q_sample(x0_sig, t, noise=noise)        # add noise to signals
        pred = self.model(x_sig_t, geom, t, label)             # (B,2,L)

        if self.cfg.objective == "eps":
            target = noise
        else:  # x0 prediction
            target = x0_sig
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def sample(self, label: torch.Tensor, geom: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        label: (B,6)
        geom:  (B,3,L)
        shape: (B,2,L)  -> output samples in signal space
        returns x0_sig samples ~ p(x|c)
        """
        B, C, L = shape
        assert C == 2 and geom.shape == (B, 3, L), "shape/geom mismatch"
        device = next(self.parameters()).device
        x = torch.randn(B, C, L, device=device)

        for t in reversed(range(self.cfg.timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            eps_hat = self.model(x, geom, t_batch, label)  # (B,2,L)

            alpha = self.alphas[t]
            alpha_bar = self.alphas_cumprod[t]
            beta = self.betas[t]

            # DDPM mean update (eps-pred)
            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * eps_hat)

            if t > 0:
                noise = torch.randn_like(x)
                var = torch.sqrt(self.posterior_variance[t])
                x = mean + var * noise
            else:
                x = mean
        return x


# -------------------------
# Minimal smoke test
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, L = 2, 5160
    x_sig = torch.randn(B, 2, L, device=device)      # (npe,time)
    geom  = torch.randn(B, 3, L, device=device)      # (x,y,z)  (실사용시 고정/정규화 권장)
    label = torch.randn(B, 6, device=device)         # (E, zenith, azimuth, X, Y, Z)

    print(x_sig)
    print(geom)
    print(label)


    model = PMTDit(
        seq_len=L, hidden=512, depth=8, heads=8,
        dropout=0.1,  # "SUM" or "FiLM"
        label_dim=6, t_embed_dim=128
    ).to(device)

    diff = GaussianDiffusion(model, DiffusionConfig(timesteps=1000, objective="eps")).to(device)

    # one training step
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    loss = diff.loss(x_sig, geom, label)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    print(f"[train] loss = {loss.item():.6f}")

    # sampling (from p(x|c))
    with torch.no_grad():
        samples = diff.sample(label[:1], geom[:1], shape=(1, 2, L))
        print("samples:", samples.shape)  # (1,2,L)
