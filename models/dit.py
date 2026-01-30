#!/usr/bin/env python3
"""
DiT-style Transformer for diffusion (conditional on timestep + label).
"""

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def sinusoidal_timestep_embedding(
    t: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """
    t: (B,) int/float
    return: (B, dim)
    """
    if t.dim() != 1:
        t = t.view(-1)
    t = t.float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat(
            [emb, torch.zeros((emb.shape[0], 1), device=t.device, dtype=emb.dtype)],
            dim=-1,
        )
    return emb


class DiTBlock(nn.Module):
    """
    DiT-style Transformer block with AdaLN modulation from conditioning vector c.
    x: (B, L, d), c: (B, d)
    """

    def __init__(
        self, d: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0
    ):
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
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, 6 * d),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        params = self.ada(c).view(B, 6, d)
        (
            shift1,
            scale1,
            gate1,
            shift2,
            scale2,
            gate2,
        ) = (
            params[:, 0],
            params[:, 1],
            params[:, 2],
            params[:, 3],
            params[:, 4],
            params[:, 5],
        )
        x1 = self.norm1(x)
        x1 = x1 * (1.0 + scale1[:, None, :]) + shift1[:, None, :]
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + gate1[:, None, :] * attn_out
        x2 = self.norm2(x)
        x2 = x2 * (1.0 + scale2[:, None, :]) + shift2[:, None, :]
        mlp_out = self.mlp(x2)
        x = x + gate2[:, None, :] * mlp_out
        return x


class DiffusionDiTTransformer(nn.Module):
    """DiT-style diffusion model conditioned on time and label; uses fixed geo."""

    def __init__(
        self,
        geo: torch.Tensor,
        d_model: int = 256,
        nhead: int = 8,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        label_dim: int = 6,
        use_checkpointing: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_checkpointing = use_checkpointing
        if geo.dim() == 2:
            geo_tok = geo.transpose(0, 1).unsqueeze(0)
        elif geo.dim() == 3:
            geo_tok = geo.permute(0, 2, 1)
        else:
            raise ValueError(f"geo must be (3,L) or (1,3,L). got {geo.shape}")
        self.register_buffer("geo_tokens", geo_tok.contiguous(), persistent=True)
        self.L = self.geo_tokens.shape[1]

        self.in_proj = nn.Linear(2, d_model)
        self.geo_mlp = nn.Sequential(
            nn.Linear(3, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.use_index_pos = False
        if self.use_index_pos:
            self.index_pos = nn.Parameter(torch.zeros(1, self.L, d_model))

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
        self.blocks = nn.ModuleList(
            [
                DiTBlock(d_model, nhead, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.out_proj = nn.Linear(d_model, 2)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        B, C, L = x_t.shape
        assert L == self.L, f"Expected L={self.L}, got L={L}"
        tokens = x_t.permute(0, 2, 1)
        h = self.in_proj(tokens)
        pos_geo = self.geo_mlp(self.geo_tokens)
        h = h + pos_geo
        if self.use_index_pos:
            h = h + self.index_pos
        t_emb = sinusoidal_timestep_embedding(t, self.d_model)
        c = self.time_mlp(t_emb) + self.label_mlp(label)
        for blk in self.blocks:
            if self.use_checkpointing and self.training:
                h = checkpoint(blk, h, c, use_reentrant=False)
            else:
                h = blk(h, c)
        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        h = self.final_norm(h)
        h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]
        out = self.out_proj(h)
        return out.permute(0, 2, 1)
