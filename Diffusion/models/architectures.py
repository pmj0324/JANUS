#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiple model architectures for IceCube neutrino event generation.

This module provides various architectures for the diffusion model:
1. PMTDit - DiT-style transformer (original)
2. PMTCNN - CNN-based architecture
3. PMTMLP - MLP-based architecture  
4. PMTHybrid - Hybrid CNN-Transformer architecture
5. PMTResNet - ResNet-based architecture
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import model classes (alias to avoid shadowing by local legacy class)
from .pmt_dit import PMTDit as ExternalPMTDit
from .pmtc_dit import PMTCDit


# =============================================================================
# Common Components
# =============================================================================

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
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


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with zero initialization."""
    
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_scale = nn.Linear(cond_dim, hidden_dim)
        self.to_shift = nn.Linear(cond_dim, hidden_dim)
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        s = self.to_scale(c)[:, None, :]
        b = self.to_shift(c)[:, None, :]
        return h * (1 + s) + b


class FiLM(nn.Module):
    """Feature-wise Linear Modulation."""
    
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta = nn.Linear(cond_dim, feature_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(c)  # (B, feature_dim)
        beta = self.beta(c)    # (B, feature_dim)
        # x is (B, feature_dim, L), need to add dimension for broadcasting
        gamma = gamma.unsqueeze(-1)  # (B, feature_dim, 1)
        beta = beta.unsqueeze(-1)    # (B, feature_dim, 1)
        return x * (1 + gamma) + beta


# =============================================================================
# Architecture 1: PMTDit (Original DiT-style)
# =============================================================================

class DiTBlock(nn.Module):
    """DiT-style transformer block."""
    
    def __init__(self, hidden: int, heads: int, cond_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ada_attn = AdaLNZero(hidden, cond_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, dropout=dropout, batch_first=True)
        self.ada_mlp = AdaLNZero(hidden, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, int(hidden * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden * mlp_ratio), hidden),
            nn.Dropout(dropout)
        )
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_mlp = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.ada_attn(x, c)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.gate_attn * a
        
        h = self.ada_mlp(x, c)
        x = x + self.gate_mlp * self.mlp(h)
        return x


class PMTEmbedding(nn.Module):
    """PMT signal and geometry embedding."""
    
    def __init__(self, hidden: int, dropout: float, fusion: str = "SUM", label_dim: int = 6):
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
            self.film = FiLM(label_dim, hidden)

    def forward(self, signal: torch.Tensor, geom: torch.Tensor, label: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_combined: Combined embedding (B, L, hidden)
            h_geo: Geometry embedding to be added after transformer (B, L, hidden)
        """
        h_sig = self.signal_net(signal)  # (B, L, hidden)
        h_geo = self.geom_net(geom)      # (B, L, hidden)

        if self.fusion == "SUM":
            h_combined = h_sig + h_geo
        else:
            assert label is not None, "fusion='FiLM' requires label"
            h_sig = self.film(h_sig, label)
            h_combined = h_sig + h_geo
        
        return h_combined, h_geo


class PMTDitLegacy(nn.Module):
    """DEPRECATED: Legacy DiT-style transformer for PMT signals.
    
    This is an old implementation kept for backward compatibility.
    Use ExternalPMTDit from models/pmt_dit.py for new code.
    """
    
    def __init__(
        self,
        seq_len: int = 5160,
        hidden: int = 512,
        depth: int = 8,
        heads: int = 8,
        dropout: float = 0.1,
        fusion: str = "SUM",
        label_dim: int = 6,
        t_embed_dim: int = 128,
        mlp_ratio: float = 4.0,
        affine_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0),
        affine_scales: Tuple[float, ...] = (1.0, 100000.0, 1.0, 1.0, 1.0),
        label_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        label_scales: Tuple[float, ...] = (1e-7, 1.0, 1.0, 0.01, 0.01, 0.01),
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.t_embed_dim = t_embed_dim

        # Embedding
        self.embedder = PMTEmbedding(hidden, dropout, fusion, label_dim)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # Conditioning
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
        )
        self.y_mlp = nn.Sequential(
            nn.Linear(label_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
        )
        self.cond_dim = (hidden // 2) * 2

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden, heads, self.cond_dim, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.ln_out = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, 2)

        # Affine normalization for signals and geometry
        off = torch.tensor(affine_offsets, dtype=torch.float32).reshape(5, 1)
        scl = torch.tensor(affine_scales, dtype=torch.float32).reshape(5, 1)
        self.register_buffer("affine_offset", off)
        self.register_buffer("affine_scale", scl)
        
        # Affine normalization for labels
        label_off = torch.tensor(label_offsets, dtype=torch.float32).reshape(6)
        label_scl = torch.tensor(label_scales, dtype=torch.float32).reshape(6)
        self.register_buffer("label_offset", label_off)
        self.register_buffer("label_scale", label_scl)

    def forward(self, x_sig_t: torch.Tensor, geom: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        B, Csig, L = x_sig_t.shape
        assert Csig == 2 and L == self.seq_len

        # Affine normalization for signals and geometry
        x5 = torch.cat([x_sig_t, geom], dim=1)  # (B, 5, L)
        off = self.affine_offset.view(1, 5, 1)
        scl = self.affine_scale.view(1, 5, 1)
        x5 = (x5 - off) / scl  # Subtract offset, then divide by scale
        x_sig_t = x5[:, 0:2, :]  # (B, 2, L)
        geom = x5[:, 2:5, :]      # (B, 3, L)
        
        # Affine normalization for labels
        label_off = self.label_offset.view(1, 6)
        label_scl = self.label_scale.view(1, 6)
        label = (label - label_off) / label_scl  # Subtract offset, then divide by scale

        # Tokenization
        sig = x_sig_t.transpose(1, 2)  # (B, L, 2)
        geo = geom.transpose(1, 2)      # (B, L, 3)
        tokens, h_geo = self.embedder(sig, geo, label)  # Get both combined and geo embeddings
        tokens = tokens + self.pos

        # Conditioning
        te = timestep_embedding(t, self.t_embed_dim)
        te = self.t_mlp(te)
        ye = self.y_mlp(label)
        cond = torch.cat([te, ye], dim=-1)

        # Transformer
        h = tokens
        for blk in self.blocks:
            h = blk(h, cond)
        
        # Add geometry embedding again after transformer
        h = h + h_geo  # (B, L, hidden) + (B, L, hidden)
        
        h = self.ln_out(h)
        eps = self.out_proj(h).transpose(1, 2)  # (B, L, 2) -> (B, 2, L)
        return eps


# =============================================================================
# Architecture 2: PMTCNN (CNN-based)
# =============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with conditioning."""
    
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.film = FiLM(cond_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film(x, c)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Residual connection if dimensions match
        if residual.shape[1] == x.shape[1]:
            x = x + residual
        
        return x


class PMTCNN(nn.Module):
    """CNN-based architecture for PMT signals."""
    
    def __init__(
        self,
        seq_len: int = 5160,
        hidden: int = 512,
        depth: int = 8,
        dropout: float = 0.1,
        label_dim: int = 6,
        t_embed_dim: int = 128,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7, 9),
        affine_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0),
        affine_scales: Tuple[float, ...] = (1.0, 100000.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.t_embed_dim = t_embed_dim

        # Input projection
        self.input_proj = nn.Conv1d(5, hidden, 1)  # 5 channels: 2 signals + 3 geometry
        
        # Multi-scale convolutional blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            kernel_size = kernel_sizes[i % len(kernel_sizes)]
            self.blocks.append(ConvBlock(hidden, hidden, hidden, kernel_size, dropout))
        
        # Conditioning
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.y_mlp = nn.Sequential(
            nn.Linear(label_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        
        # Output projection
        self.output_proj = nn.Conv1d(hidden, 2, 1)
        
        # Affine normalization
        off = torch.tensor(affine_offsets, dtype=torch.float32).reshape(5, 1)
        scl = torch.tensor(affine_scales, dtype=torch.float32).reshape(5, 1)
        self.register_buffer("affine_offset", off)
        self.register_buffer("affine_scale", scl)

    def forward(self, x_sig_t: torch.Tensor, geom: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        B, Csig, L = x_sig_t.shape
        assert Csig == 2 and L == self.seq_len

        # Affine normalization
        x5 = torch.cat([x_sig_t, geom], dim=1)
        off = self.affine_offset.view(1, 5, 1)
        scl = self.affine_scale.view(1, 5, 1)
        x5 = (x5 + off) * scl

        # Input projection
        x = self.input_proj(x5)

        # Conditioning
        te = timestep_embedding(t, self.t_embed_dim)
        te = self.t_mlp(te)
        ye = self.y_mlp(label)
        cond = te + ye

        # Convolutional blocks
        for block in self.blocks:
            x = block(x, cond)

        # Output projection
        eps = self.output_proj(x)
        return eps


# =============================================================================
# Architecture 3: PMTMLP (MLP-based)
# =============================================================================

class MLPBlock(nn.Module):
    """MLP block with conditioning."""
    
    def __init__(self, hidden: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout)
        )
        self.film = FiLM(cond_dim, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        x = self.film(x, c)
        return x + residual


class PMTMLP(nn.Module):
    """MLP-based architecture for PMT signals."""
    
    def __init__(
        self,
        seq_len: int = 5160,
        hidden: int = 512,
        depth: int = 8,
        dropout: float = 0.1,
        label_dim: int = 6,
        t_embed_dim: int = 128,
        affine_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0),
        affine_scales: Tuple[float, ...] = (1.0, 100000.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.t_embed_dim = t_embed_dim

        # Input projection
        self.input_proj = nn.Linear(5, hidden)  # 5 channels: 2 signals + 3 geometry
        
        # MLP blocks
        self.blocks = nn.ModuleList([
            MLPBlock(hidden, hidden, dropout) for _ in range(depth)
        ])
        
        # Conditioning
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.y_mlp = nn.Sequential(
            nn.Linear(label_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden, 2)
        
        # Affine normalization
        off = torch.tensor(affine_offsets, dtype=torch.float32).reshape(5, 1)
        scl = torch.tensor(affine_scales, dtype=torch.float32).reshape(5, 1)
        self.register_buffer("affine_offset", off)
        self.register_buffer("affine_scale", scl)

    def forward(self, x_sig_t: torch.Tensor, geom: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        B, Csig, L = x_sig_t.shape
        assert Csig == 2 and L == self.seq_len

        # Affine normalization
        x5 = torch.cat([x_sig_t, geom], dim=1)
        off = self.affine_offset.view(1, 5, 1)
        scl = self.affine_scale.view(1, 5, 1)
        x5 = (x5 + off) * scl

        # Reshape for MLP: (B, 5, L) -> (B*L, 5)
        x = x5.transpose(1, 2).reshape(B * L, 5)
        x = self.input_proj(x)

        # Conditioning
        te = timestep_embedding(t, self.t_embed_dim)
        te = self.t_mlp(te)
        ye = self.y_mlp(label)
        cond = te + ye
        
        # Expand conditioning to all positions
        cond = cond.unsqueeze(1).expand(B, L, self.hidden).reshape(B * L, self.hidden)

        # MLP blocks
        for block in self.blocks:
            x = block(x, cond)

        # Output projection
        eps = self.output_proj(x)
        eps = eps.reshape(B, L, 2).transpose(1, 2)
        return eps


# =============================================================================
# Architecture 4: PMTHybrid (CNN + Transformer)
# =============================================================================

class HybridBlock(nn.Module):
    """Hybrid block combining CNN and attention."""
    
    def __init__(self, hidden: int, heads: int, cond_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        # CNN part
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Attention part
        self.ada_attn = AdaLNZero(hidden, cond_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, dropout=dropout, batch_first=True)
        self.gate_attn = nn.Parameter(torch.zeros(1))
        
        # MLP part
        self.ada_mlp = AdaLNZero(hidden, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout)
        )
        self.gate_mlp = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        
        # CNN processing
        x_conv = self.conv(x)
        
        # Convert to sequence for attention
        x_seq = x_conv.transpose(1, 2)  # (B, L, C)
        
        # Attention
        h = self.ada_attn(x_seq, c)
        a, _ = self.attn(h, h, h, need_weights=False)
        x_seq = x_seq + self.gate_attn * a
        
        # MLP
        h = self.ada_mlp(x_seq, c)
        x_seq = x_seq + self.gate_mlp * self.mlp(h)
        
        # Convert back to conv format
        x = x_seq.transpose(1, 2)  # (B, C, L)
        
        return x


class PMTHybrid(nn.Module):
    """Hybrid CNN-Transformer architecture."""
    
    def __init__(
        self,
        seq_len: int = 5160,
        hidden: int = 512,
        depth: int = 8,
        heads: int = 8,
        dropout: float = 0.1,
        label_dim: int = 6,
        t_embed_dim: int = 128,
        kernel_size: int = 3,
        affine_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0),
        affine_scales: Tuple[float, ...] = (1.0, 100000.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.t_embed_dim = t_embed_dim

        # Input projection
        self.input_proj = nn.Conv1d(5, hidden, 1)
        
        # Hybrid blocks
        self.blocks = nn.ModuleList([
            HybridBlock(hidden, heads, hidden, kernel_size, dropout)
            for _ in range(depth)
        ])
        
        # Conditioning
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.y_mlp = nn.Sequential(
            nn.Linear(label_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        
        # Output projection
        self.output_proj = nn.Conv1d(hidden, 2, 1)
        
        # Affine normalization
        off = torch.tensor(affine_offsets, dtype=torch.float32).reshape(5, 1)
        scl = torch.tensor(affine_scales, dtype=torch.float32).reshape(5, 1)
        self.register_buffer("affine_offset", off)
        self.register_buffer("affine_scale", scl)

    def forward(self, x_sig_t: torch.Tensor, geom: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        B, Csig, L = x_sig_t.shape
        assert Csig == 2 and L == self.seq_len

        # Affine normalization
        x5 = torch.cat([x_sig_t, geom], dim=1)
        off = self.affine_offset.view(1, 5, 1)
        scl = self.affine_scale.view(1, 5, 1)
        x5 = (x5 + off) * scl

        # Input projection
        x = self.input_proj(x5)

        # Conditioning
        te = timestep_embedding(t, self.t_embed_dim)
        te = self.t_mlp(te)
        ye = self.y_mlp(label)
        cond = te + ye

        # Hybrid blocks
        for block in self.blocks:
            x = block(x, cond)

        # Output projection
        eps = self.output_proj(x)
        return eps


# =============================================================================
# Architecture 5: PMTResNet (ResNet-based)
# =============================================================================

class ResNetBlock(nn.Module):
    """ResNet block with conditioning."""
    
    def __init__(self, channels: int, cond_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.film = FiLM(cond_dim, channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film(x, c)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x + residual


class PMTResNet(nn.Module):
    """ResNet-based architecture for PMT signals."""
    
    def __init__(
        self,
        seq_len: int = 5160,
        hidden: int = 512,
        depth: int = 8,
        dropout: float = 0.1,
        label_dim: int = 6,
        t_embed_dim: int = 128,
        kernel_size: int = 3,
        affine_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0),
        affine_scales: Tuple[float, ...] = (1.0, 100000.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.t_embed_dim = t_embed_dim

        # Input projection
        self.input_proj = nn.Conv1d(5, hidden, 1)
        
        # ResNet blocks
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden, hidden, kernel_size, dropout)
            for _ in range(depth)
        ])
        
        # Conditioning
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.y_mlp = nn.Sequential(
            nn.Linear(label_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        
        # Output projection
        self.output_proj = nn.Conv1d(hidden, 2, 1)
        
        # Affine normalization
        off = torch.tensor(affine_offsets, dtype=torch.float32).reshape(5, 1)
        scl = torch.tensor(affine_scales, dtype=torch.float32).reshape(5, 1)
        self.register_buffer("affine_offset", off)
        self.register_buffer("affine_scale", scl)

    def forward(self, x_sig_t: torch.Tensor, geom: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        B, Csig, L = x_sig_t.shape
        assert Csig == 2 and L == self.seq_len

        # Affine normalization
        x5 = torch.cat([x_sig_t, geom], dim=1)
        off = self.affine_offset.view(1, 5, 1)
        scl = self.affine_scale.view(1, 5, 1)
        x5 = (x5 + off) * scl

        # Input projection
        x = self.input_proj(x5)

        # Conditioning
        te = timestep_embedding(t, self.t_embed_dim)
        te = self.t_mlp(te)
        ye = self.y_mlp(label)
        cond = te + ye

        # ResNet blocks
        for block in self.blocks:
            x = block(x, cond)

        # Output projection
        eps = self.output_proj(x)
        return eps


# =============================================================================
# Model Factory
# =============================================================================

@dataclass
class ArchitectureConfig:
    """Configuration for model architectures."""
    name: str = "dit"
    seq_len: int = 5160
    hidden: int = 512
    depth: int = 8
    heads: int = 8
    dropout: float = 0.1
    fusion: str = "SUM"
    label_dim: int = 6
    t_embed_dim: int = 128
    mlp_ratio: float = 4.0
    kernel_size: int = 3
    kernel_sizes: Tuple[int, ...] = (3, 5, 7, 9)
    affine_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    affine_scales: Tuple[float, ...] = (100.0, 10.0, 600.0, 550.0, 550.0)
    label_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    label_scales: Tuple[float, ...] = (50000000.0, 1.0, 1.0, 600.0, 550.0, 550.0)
    time_transform: str = "ln"  # "ln" or "log10" - always use log(1+x)


def create_model(config: ArchitectureConfig) -> nn.Module:
    """Create model based on architecture configuration."""
    arch_name = config.name.lower()
    
    if arch_name == "dit":
        return ExternalPMTDit(
            seq_len=config.seq_len,
            hidden=config.hidden,
            depth=config.depth,
            heads=config.heads,
            dropout=config.dropout,
            fusion=config.fusion,
            label_dim=config.label_dim,
            t_embed_dim=config.t_embed_dim,
            mlp_ratio=config.mlp_ratio,
            affine_offsets=config.affine_offsets,
            affine_scales=config.affine_scales,
            label_offsets=config.label_offsets,
            label_scales=config.label_scales,
            time_transform=config.time_transform,
        )
    elif arch_name == "c-dit":
        return PMTCDit(
            seq_len=config.seq_len,
            hidden=config.hidden,
            classifier_size=getattr(config, 'classifier_size', 128),
            depth=config.depth,
            heads=config.heads,
            dropout=config.dropout,
            factor=getattr(config, 'factor', 2),
            label_dim=config.label_dim,
            t_embed_dim=config.t_embed_dim,
            combine=getattr(config, 'combine', 'add'),
            update_with_classifier=getattr(config, 'update_with_classifier', False),
            n_cls_tokens=getattr(config, 'n_cls_tokens', 1),
            affine_offsets=config.affine_offsets,
            affine_scales=config.affine_scales,
            label_offsets=config.label_offsets,
            label_scales=config.label_scales,
            time_transform=config.time_transform,
        )
    elif arch_name == "cnn":
        return PMTCNN(
            seq_len=config.seq_len,
            hidden=config.hidden,
            depth=config.depth,
            dropout=config.dropout,
            label_dim=config.label_dim,
            t_embed_dim=config.t_embed_dim,
            kernel_sizes=config.kernel_sizes,
            affine_offsets=config.affine_offsets,
            affine_scales=config.affine_scales,
        )
    elif arch_name == "mlp":
        return PMTMLP(
            seq_len=config.seq_len,
            hidden=config.hidden,
            depth=config.depth,
            dropout=config.dropout,
            label_dim=config.label_dim,
            t_embed_dim=config.t_embed_dim,
            affine_offsets=config.affine_offsets,
            affine_scales=config.affine_scales,
        )
    elif arch_name == "hybrid":
        return PMTHybrid(
            seq_len=config.seq_len,
            hidden=config.hidden,
            depth=config.depth,
            heads=config.heads,
            dropout=config.dropout,
            label_dim=config.label_dim,
            t_embed_dim=config.t_embed_dim,
            kernel_size=config.kernel_size,
            affine_offsets=config.affine_offsets,
            affine_scales=config.affine_scales,
        )
    elif arch_name == "resnet":
        return PMTResNet(
            seq_len=config.seq_len,
            hidden=config.hidden,
            depth=config.depth,
            dropout=config.dropout,
            label_dim=config.label_dim,
            t_embed_dim=config.t_embed_dim,
            kernel_size=config.kernel_size,
            affine_offsets=config.affine_offsets,
            affine_scales=config.affine_scales,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")


def get_architecture_info() -> Dict[str, Dict[str, Any]]:
    """Get information about available architectures."""
    return {
        "dit": {
            "name": "PMTDit",
            "description": "DiT-style transformer with attention mechanisms",
            "strengths": ["Long-range dependencies", "Flexible conditioning", "State-of-the-art performance"],
            "weaknesses": ["High memory usage", "Slow for very long sequences"],
            "best_for": "High-quality generation, research applications"
        },
        "cnn": {
            "name": "PMTCNN", 
            "description": "CNN-based architecture with multi-scale convolutions",
            "strengths": ["Fast inference", "Local feature extraction", "Memory efficient"],
            "weaknesses": ["Limited long-range dependencies", "Fixed receptive field"],
            "best_for": "Fast generation, production deployment"
        },
        "mlp": {
            "name": "PMTMLP",
            "description": "MLP-based architecture with position-wise processing",
            "strengths": ["Simple architecture", "Fast training", "Easy to understand"],
            "weaknesses": ["Limited expressiveness", "No spatial structure"],
            "best_for": "Baseline models, quick prototyping"
        },
        "hybrid": {
            "name": "PMTHybrid",
            "description": "Combines CNN and transformer for balanced performance",
            "strengths": ["Local and global features", "Balanced speed/quality", "Flexible"],
            "weaknesses": ["Complex architecture", "More hyperparameters"],
            "best_for": "Balanced applications, production use"
        },
        "resnet": {
            "name": "PMTResNet",
            "description": "ResNet-based architecture with residual connections",
            "strengths": ["Stable training", "Deep networks", "Good gradients"],
            "weaknesses": ["Limited to local features", "Fixed architecture"],
            "best_for": "Stable training, deep models"
        }
    }


if __name__ == "__main__":
    # Test all architectures
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, L = 2, 5160
    
    x_sig = torch.randn(B, 2, L, device=device)
    geom = torch.randn(B, 3, L, device=device)
    label = torch.randn(B, 6, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    
    configs = [
        ArchitectureConfig(name="dit", hidden=256, depth=4),
        ArchitectureConfig(name="cnn", hidden=256, depth=4),
        ArchitectureConfig(name="mlp", hidden=256, depth=4),
        ArchitectureConfig(name="hybrid", hidden=256, depth=4),
        ArchitectureConfig(name="resnet", hidden=256, depth=4),
    ]
    
    for config in configs:
        model = create_model(config).to(device)
        with torch.no_grad():
            output = model(x_sig, geom, t, label)
        print(f"{config.name.upper()}: {output.shape}, {sum(p.numel() for p in model.parameters()):,} params")
