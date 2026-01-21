#!/usr/bin/env python3
"""
Forward Diffusion Process
=========================

Functions for applying forward diffusion (adding noise) to data.
"""

import torch
from typing import Optional
from ..schedules import compute_alpha_schedule


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    Extract values from a at indices t and reshape for broadcasting.
    
    Args:
        a: Source tensor (T,)
        t: Index tensor (B,)
        x_shape: Target shape (B, C, L)
    
    Returns:
        Extracted values reshaped to (B, 1, 1) for broadcasting
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def q_sample_batch(
    x0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    noise: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Sample from q(x_t | x_0) for a batch.
    
    Forward diffusion: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
    
    Args:
        x0: Clean samples (B, C, L)
        t: Timesteps (B,) - integer values in [0, T]
            • t=0: Original data (no noise, no parameter used)
            • t=1: First noise step (uses parameter[0])
            • t=2: Second noise step (uses parameter[1])
            • ...
            • t=T: Final timestep (uses parameter[T-1], maximum noise)
        sqrt_alphas_cumprod: sqrt(ᾱ_t) values (size T)
        sqrt_one_minus_alphas_cumprod: sqrt(1-ᾱ_t) values (size T)
        noise: Optional pre-generated noise
    
    Returns:
        Noised samples x_t (B, C, L)
        
    Note:
        Parameter indexing: use t-1 as index for t > 0
        This ensures all T noise parameters are used
    """
    # Special case: t=0 returns original data
    mask_t0 = (t == 0)
    if mask_t0.all():
        return x0.clone()
    
    if noise is None:
        noise = torch.randn_like(x0)
    
    # For t > 0, use parameter at index t-1
    t_idx = torch.where(t > 0, t - 1, torch.zeros_like(t))
    
    sqrt_alpha_bar = extract(sqrt_alphas_cumprod, t_idx, x0.shape)
    sqrt_one_minus_alpha_bar = extract(sqrt_one_minus_alphas_cumprod, t_idx, x0.shape)
    
    # Compute noised version
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    
    # Replace t=0 samples with original (if any in batch)
    if mask_t0.any():
        x_t[mask_t0] = x0[mask_t0].clone()
    
    return x_t


def apply_forward_diffusion(
    x0: torch.Tensor,
    betas: torch.Tensor,
    timesteps: torch.Tensor,
    noise: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply forward diffusion to data using a noise schedule.
    
    Args:
        x0: Clean samples (B, C, L)
        betas: Beta schedule (T,)
        timesteps: Timesteps to apply (B,) - integer values in [0, T]
        noise: Optional pre-generated noise
    
    Returns:
        Noised samples x_t (B, C, L)
    """
    # Compute alpha schedule from betas
    alpha_schedule = compute_alpha_schedule(betas)
    
    return q_sample_batch(
        x0=x0,
        t=timesteps,
        sqrt_alphas_cumprod=alpha_schedule["sqrt_alphas_cumprod"],
        sqrt_one_minus_alphas_cumprod=alpha_schedule["sqrt_one_minus_alphas_cumprod"],
        noise=noise
    )
