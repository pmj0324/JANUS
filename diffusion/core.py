#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian Diffusion Module
==========================

DDPM-style diffusion for conditional generation of PMT signals.

Forward process: q(x_t | x_0) = N(x_t; sqrt(α̅_t)x_0, (1-α̅_t)I)
Reverse process: p_θ(x_{t-1} | x_t, c) where θ = model parameters, c = conditions

Supports:
- ε-prediction (predict noise)
- x0-prediction (predict clean signal)
- Forward diffusion (q_sample)
- DDPM/DDIM sampling (see diffusion.sampling module)
- Training loss (see training.loss module)

Author: Minje Park
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .schedules import (
    linear_beta_schedule,
    cosine_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
)


@dataclass
class DiffusionConfig:
    """Configuration for Gaussian Diffusion model."""
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    objective: str = "eps"  # "eps" or "x0"
    schedule: str = "linear"  # "linear", "cosine", "quadratic", "sigmoid"
    cosine_s: float = 0.008  # Only used for cosine schedule
    use_cfg: bool = False  # Classifier-free guidance
    cfg_scale: float = 1.0  # CFG scale
    cfg_dropout: float = 0.1  # CFG dropout rate


class GaussianDiffusion(nn.Module):
    """
    DDPM-style trainer/sampler for p(x|c) with geometry.
    
    Model predicts ε̂(x_sig_t, t, label, geom) → (B, 2, L)
    - x_sig: PMT signals (charge, time) - noised during diffusion
    - geom: PMT geometry (x, y, z) - kept clean as conditioning
    - label: Event properties (Energy, Zenith, Azimuth, X, Y, Z)
    
    Args:
        model: Neural network model (e.g., PMTDiT)
        cfg: DiffusionConfig object
    """
    
    def __init__(self, model: nn.Module, cfg: DiffusionConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        
        T = cfg.timesteps
        
        # Create noise schedule using schedules module
        if cfg.schedule == "linear":
            betas = linear_beta_schedule(T, cfg.beta_start, cfg.beta_end)
        elif cfg.schedule == "cosine":
            betas = cosine_beta_schedule(T, cfg.cosine_s)
        elif cfg.schedule == "quadratic":
            betas = quadratic_beta_schedule(T, cfg.beta_start, cfg.beta_end)
        elif cfg.schedule == "sigmoid":
            betas = sigmoid_beta_schedule(T, cfg.beta_start, cfg.beta_end)
        else:
            raise ValueError(
                f"Unknown schedule: {cfg.schedule}. "
                f"Choose from: linear, cosine, quadratic, sigmoid"
            )
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register all schedule-related tensors as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Posterior variance for DDPM sampling
        # q(x_{t-1} | x_t, x_0) variance
        posterior_variance = betas * (1 - alphas_cumprod.roll(1, 0)) / (1 - alphas_cumprod)
        posterior_variance[0] = betas[0]
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
    
    def get_normalization_params(self):
        """
        Get normalization parameters from the model.
        Returns affine_offset, affine_scale, label_offset, label_scale, time_transform.
        """
        if hasattr(self.model, 'affine_offset'):
            affine_offset = self.model.affine_offset.squeeze().cpu()
            affine_scale = self.model.affine_scale.squeeze().cpu()
            label_offset = self.model.label_offset.cpu() if hasattr(self.model, 'label_offset') else None
            label_scale = self.model.label_scale.cpu() if hasattr(self.model, 'label_scale') else None
            time_transform = self.model.time_transform if hasattr(self.model, 'time_transform') else "ln"
            return affine_offset, affine_scale, label_offset, label_scale, time_transform
        else:
            return None, None, None, None, "ln"
    
    def q_sample(
        self, 
        x0_sig: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: sample from q(x_t | x_0) for signals only.
        
        Args:
            x0_sig: Clean signals (B, 2, L) - [charge, time]
            t: Timesteps (B,) - integer values in [0, T]
                • t=0: Original data (no noise, no parameter used)
                • t=1: First noise step (uses betas[0])
                • t=2: Second noise step (uses betas[1])
                • ...
                • t=T: Final timestep (uses betas[T-1], maximum noise)
            noise: Optional pre-generated noise (B, 2, L)
        
        Returns:
            Noised signals x_t (B, 2, L)
        
        Note:
            Parameter indexing: betas[t-1] for t > 0
            This ensures all T noise parameters are used (betas[0] through betas[T-1])
        """
        # Special case: t=0 returns original data (no noise)
        mask_t0 = (t == 0)
        if mask_t0.all():
            return x0_sig.clone()
        
        if noise is None:
            noise = torch.randn_like(x0_sig)
        
        # For t > 0, use parameter at index t-1
        # t=1 → index 0, t=2 → index 1, ..., t=T → index T-1
        t_idx = torch.where(t > 0, t - 1, torch.zeros_like(t))
        
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t_idx][:, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t_idx][:, None, None]
        
        # Compute noised version
        x_t = sqrt_alpha_bar * x0_sig + sqrt_one_minus_alpha_bar * noise
        
        # Replace t=0 samples with original (if any in batch)
        if mask_t0.any():
            x_t[mask_t0] = x0_sig[mask_t0].clone()
        
        return x_t
    
    def _denormalize_samples(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize samples using model's metadata.
        
        Process:
        1. Get normalization params from model
        2. Affine inverse: x = (x_norm * scale) + offset
        3. Time transform inverse: time = exp(time_norm) - 1 or 10^time_norm - 1
        4. Clamp to prevent overflow
        
        Args:
            x_norm: (B, 2, L) normalized samples
        
        Returns:
            x_raw: (B, 2, L) physical units (NPE, ns)
        """
        # Get normalization parameters from model metadata
        # Returns: (affine_offset, affine_scale, label_offset, label_scale, time_transform)
        affine_offset, affine_scale, _, _, time_transform = self.get_normalization_params()
        
        # Extract signal channels only (first 2 channels: charge, time)
        if affine_offset is not None and affine_scale is not None:
            offsets = torch.tensor(
                affine_offset[:2].tolist() if hasattr(affine_offset, 'tolist') else affine_offset[:2],
                device=x_norm.device,
                dtype=x_norm.dtype
            ).view(1, 2, 1)
            
            scales = torch.tensor(
                affine_scale[:2].tolist() if hasattr(affine_scale, 'tolist') else affine_scale[:2],
                device=x_norm.device,
                dtype=x_norm.dtype
            ).view(1, 2, 1)
        else:
            # Default values if normalization params not available
            offsets = torch.zeros(1, 2, 1, device=x_norm.device, dtype=x_norm.dtype)
            scales = torch.ones(1, 2, 1, device=x_norm.device, dtype=x_norm.dtype)
        
        # Step 1: Affine inverse
        # x_physical = (x_norm * scale) + offset
        x = (x_norm * scales) + offsets
        
        # Step 2: Time transform inverse (only for time channel)
        if time_transform == "ln":
            # Inverse of ln(1+x) is exp(x) - 1
            x[:, 1, :] = torch.exp(x[:, 1, :]) - 1.0
        elif time_transform == "log10":
            # Inverse of log10(1+x) is 10^x - 1
            x[:, 1, :] = torch.pow(10.0, x[:, 1, :]) - 1.0
        else:
            raise ValueError(f"Unknown time_transform: {time_transform}")
        
        # Step 3: Clamp time to prevent overflow (exp can produce huge values)
        x[:, 1, :] = torch.clamp(x[:, 1, :], min=0.0, max=1e8)
        
        return x


def create_gaussian_diffusion(
    model: nn.Module,
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    objective: str = "eps",
    schedule: str = "linear"
) -> GaussianDiffusion:
    """
    Factory function to create GaussianDiffusion instance.
    
    Args:
        model: Neural network model
        timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        objective: "eps" or "x0"
        schedule: "linear", "cosine", "quadratic", or "sigmoid"
    
    Returns:
        GaussianDiffusion instance
    """
    cfg = DiffusionConfig(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        objective=objective,
        schedule=schedule
    )
    return GaussianDiffusion(model, cfg)
