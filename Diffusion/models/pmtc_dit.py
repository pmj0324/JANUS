#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmtc_dit.py

Classifier-Attention based Diffusion Transformer for PMT signals.
Based on PmtCModel architecture but adapted for diffusion reconstruction.

Architecture:
  - Separate signal/geometry embeddings
  - Classifier token (latent) attention mechanism
  - Cross-attention between PMTs and latent
  - Decode latent back to signal space

Key differences from PMTDit:
  - Uses cross-attention with a learned classifier token
  - Bidirectional information flow (PMT ↔ Latent)
  - More compact latent representation

Author: Minje Park
"""

from __future__ import annotations
import math
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
# Cross-Attention Block (PMT → Classifier)
# -------------------------
class CBlock(nn.Module):
    """
    Cross-attention block where classifier token attends to PMT tokens.
    
    Args:
        xdim: PMT token dimension
        cdim: Classifier token dimension
        factor: MLP expansion factor
        dropout: Dropout rate
        n_heads: Number of attention heads
    """
    def __init__(self, xdim: int, cdim: int, factor: int, dropout: float, n_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(xdim)
        self.mha = nn.MultiheadAttention(
            embed_dim=cdim, 
            num_heads=n_heads, 
            batch_first=True, 
            dropout=dropout, 
            kdim=xdim, 
            vdim=xdim
        )
        self.combine = nn.Sequential(
            nn.LayerNorm(cdim),
            nn.Linear(cdim, cdim * factor), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(cdim * factor, cdim), 
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: PMT tokens (B, L, xdim)
        c: Classifier tokens (B, n_cls, cdim)
        mask: Optional mask for PMTs
        return: Updated classifier tokens (B, n_cls, cdim)
        """
        x = self.norm(x)
        c = c + self.mha(c, x, x, key_padding_mask=mask)[0]
        c = c + self.combine(c)
        return c


# -------------------------
# PMT Signal & Geometry Embedding
# -------------------------
class PMTEmbedding(nn.Module):
    """
    Embed signal (charge, time) and geometry (x, y, z) separately,
    then combine them.
    
    Args:
        hidden: Hidden dimension
        dropout: Dropout rate
        combine: How to combine signal and geometry ("add" or "concat")
    """
    def __init__(self, hidden: int, dropout: float, combine: str = "add"):
        super().__init__()
        self.hidden = hidden
        self.combine = combine.lower()
        assert self.combine in {"add", "concat"}, "combine must be 'add' or 'concat'"
        
        # Signal embedding (charge, time) -> hidden
        self.signal_net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        
        # Geometry embedding (x, y, z) -> hidden
        self.geom_net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        
        # If concat, output is 2*hidden
        self.output_dim = 2 * hidden if self.combine == "concat" else hidden
    
    def forward(self, signal: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        """
        signal: (B, L, 2) [charge, time]
        geom: (B, L, 3) [x, y, z]
        return: (B, L, output_dim)
        """
        h_sig = self.signal_net(signal)  # (B, L, hidden)
        h_geom = self.geom_net(geom)     # (B, L, hidden)
        
        if self.combine == "add":
            return h_sig + h_geom
        else:  # concat
            return torch.cat([h_sig, h_geom], dim=-1)


# -------------------------
# Main Model: PMTCDit
# -------------------------
class PMTCDit(nn.Module):
    """
    Classifier-Attention Diffusion Transformer for PMT signals.
    
    Architecture:
      1. Embed signal and geometry
      2. Iteratively update:
         - PMT tokens via self-update (conditioned on classifier)
         - Classifier token via cross-attention to PMT tokens
      3. Decode classifier token back to signal space
    
    Args:
        seq_len: Number of PMTs (default: 5160)
        hidden: Hidden dimension for PMT tokens
        classifier_size: Dimension of classifier (latent) token
        depth: Number of transformer blocks
        heads: Number of attention heads
        dropout: Dropout rate
        factor: MLP expansion factor
        label_dim: Dimension of event-level labels
        t_embed_dim: Dimension of timestep embedding
        combine: How to combine signal/geom ("add" or "concat")
        update_with_classifier: If True, PMT update uses classifier info
        n_cls_tokens: Number of classifier tokens (default: 1)
        
        # Normalization metadata (same as PMTDit)
        affine_offsets: Normalization offsets for [charge, time, x, y, z]
        affine_scales: Normalization scales
        label_offsets: Label normalization offsets
        label_scales: Label normalization scales
        time_transform: Time transformation type ("ln" or "log10")
    """
    def __init__(
        self,
        seq_len: int = 5160,
        hidden: int = 16,
        classifier_size: int = 128,
        depth: int = 2,
        heads: int = 2,
        dropout: float = 0.1,
        factor: int = 2,
        label_dim: int = 6,
        t_embed_dim: int = 128,
        combine: str = "add",
        update_with_classifier: bool = False,
        n_cls_tokens: int = 1,
        # Normalization metadata
        affine_offsets = (0.0, 0.0, 0.0, 0.0, 0.0),
        affine_scales = (1.0, 1.0, 1.0, 1.0, 1.0),
        label_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        label_scales = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        time_transform = "ln",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.classifier_size = classifier_size
        self.depth = depth
        self.combine = combine
        self.update_with_classifier = update_with_classifier
        self.n_cls_tokens = n_cls_tokens
        self.time_transform = time_transform
        
        # --- Embedding ---
        self.embedder = PMTEmbedding(hidden, dropout, combine)
        self.hidden_size_attention = self.embedder.output_dim
        
        # --- Classifier token (learnable latent) ---
        self.cls_token = nn.Parameter(torch.randn(1, n_cls_tokens, classifier_size))
        
        # --- Timestep embedding ---
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        
        # --- Label embedding ---
        self.label_mlp = nn.Sequential(
            nn.Linear(label_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        
        # --- Condition projector (t + label -> condition) ---
        self.cond_proj = nn.Linear(hidden * 2, classifier_size)
        
        # --- Transformer blocks ---
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            # PMT token update (self-update, optionally conditioned on classifier)
            pmt_update_dim = self.hidden_size_attention + classifier_size if update_with_classifier else self.hidden_size_attention
            pmt_block = nn.Sequential(
                nn.Linear(pmt_update_dim, factor * self.hidden_size_attention),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(factor * self.hidden_size_attention, self.hidden_size_attention)
            )
            
            # Classifier token update (cross-attention to PMT tokens)
            cls_block = CBlock(self.hidden_size_attention, classifier_size, factor, dropout, heads)
            
            self.blocks.append(nn.ModuleList([pmt_block, cls_block]))
        
        # --- Decoder: classifier token -> PMT signals ---
        # Expand classifier token back to per-PMT predictions
        self.decoder = nn.Sequential(
            nn.Linear(classifier_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.hidden_size_attention),
        )
        
        # Final projection to signal space
        self.final_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_size_attention),
            nn.Linear(self.hidden_size_attention, 2)  # [charge, time]
        )
        
        # --- Normalization metadata (not used in forward, only for denormalization) ---
        self.register_buffer('affine_offset', torch.tensor(affine_offsets, dtype=torch.float32).view(1, 5, 1))
        self.register_buffer('affine_scale', torch.tensor(affine_scales, dtype=torch.float32).view(1, 5, 1))
        self.register_buffer('label_offset', torch.tensor(label_offsets, dtype=torch.float32).view(1, 6))
        self.register_buffer('label_scale', torch.tensor(label_scales, dtype=torch.float32).view(1, 6))
    
    def forward(
        self, 
        x_sig: torch.Tensor, 
        t: torch.Tensor, 
        label: torch.Tensor, 
        geom: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass (ε-prediction for diffusion).
        
        Args:
            x_sig: Noisy signal (B, 2, L) [charge, time] - NORMALIZED
            t: Timestep (B,)
            label: Event-level condition (B, 6) - NORMALIZED
            geom: PMT geometry (B, 3, L) [x, y, z] - NORMALIZED
        
        Returns:
            eps_pred: Predicted noise (B, 2, L)
        """
        B, _, L = x_sig.shape
        
        # Transpose for embedding: (B, 2, L) -> (B, L, 2)
        x_sig = x_sig.transpose(1, 2)  # (B, L, 2)
        geom = geom.transpose(1, 2)    # (B, L, 3)
        
        # --- Embedding ---
        x = self.embedder(x_sig, geom)  # (B, L, hidden_size_attention)
        
        # --- Timestep & Label conditioning ---
        t_emb = timestep_embedding(t, self.t_mlp[0].in_features)  # (B, t_embed_dim)
        t_emb = self.t_mlp(t_emb)  # (B, hidden)
        
        label_emb = self.label_mlp(label)  # (B, hidden)
        
        # Combine timestep and label
        cond = torch.cat([t_emb, label_emb], dim=-1)  # (B, 2*hidden)
        cond = self.cond_proj(cond)  # (B, classifier_size)
        
        # Initialize classifier token
        c = self.cls_token.expand(B, -1, -1)  # (B, n_cls_tokens, classifier_size)
        
        # Add conditioning to classifier token
        c = c + cond.unsqueeze(1)  # (B, n_cls_tokens, classifier_size)
        
        # --- Transformer blocks ---
        for pmt_block, cls_block in self.blocks:
            # Update PMT tokens
            if self.update_with_classifier:
                # Expand classifier and concatenate with PMT tokens
                c_expanded = c.expand(-1, L, -1)  # (B, L, classifier_size)
                x_with_c = torch.cat([x, c_expanded], dim=-1)  # (B, L, hidden+classifier_size)
                x = x + pmt_block(x_with_c)
            else:
                x = x + pmt_block(x)
            
            # Update classifier token via cross-attention to PMT tokens
            c = cls_block(x, c)
        
        # --- Decode classifier token back to signal space ---
        # Option 1: Use classifier to modulate PMT tokens
        c_decoded = self.decoder(c)  # (B, n_cls_tokens, hidden_size_attention)
        
        # If multiple classifier tokens, average them
        if self.n_cls_tokens > 1:
            c_decoded = c_decoded.mean(dim=1, keepdim=True)  # (B, 1, hidden_size_attention)
        
        # Broadcast and add to PMT tokens
        x = x + c_decoded.expand(-1, L, -1)  # (B, L, hidden_size_attention)
        
        # Final projection to signal space
        eps_pred = self.final_proj(x)  # (B, L, 2)
        
        # Transpose back: (B, L, 2) -> (B, 2, L)
        eps_pred = eps_pred.transpose(1, 2)
        
        return eps_pred
    
    def get_normalization_params(self):
        """Get normalization parameters for denormalization."""
        return {
            'affine_offsets': tuple(self.affine_offset.squeeze().cpu().numpy()),
            'affine_scales': tuple(self.affine_scale.squeeze().cpu().numpy()),
            'label_offsets': tuple(self.label_offset.squeeze().cpu().numpy()),
            'label_scales': tuple(self.label_scale.squeeze().cpu().numpy()),
            'time_transform': self.time_transform,
        }


# -------------------------
# Test code
# -------------------------
if __name__ == "__main__":
    # Test the model
    B, L = 4, 5160
    
    model = PMTCDit(
        seq_len=L,
        hidden=16,
        classifier_size=128,
        depth=2,
        heads=2,
        dropout=0.1,
        factor=2,
        combine="add",
        update_with_classifier=True,
        n_cls_tokens=1
    )
    
    # Dummy inputs (normalized)
    x_sig = torch.randn(B, 2, L)
    t = torch.randint(0, 1000, (B,))
    label = torch.randn(B, 6)
    geom = torch.randn(B, 3, L)
    
    # Forward pass
    eps_pred = model(x_sig, t, label, geom)
    
    print(f"Input shape: {x_sig.shape}")
    print(f"Output shape: {eps_pred.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n✅ PMTCDit model test passed!")

