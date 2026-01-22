#!/usr/bin/env python3
"""
DDIM Sampling
=============

DDIM (Denoising Diffusion Implicit Models) sampling for faster generation.
"""

import torch
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from diffusion.core import GaussianDiffusion
    from .ddpm import predict_start_from_noise


def sample_ddim(
    diffusion: "GaussianDiffusion",
    label: torch.Tensor,
    geom: torch.Tensor,
    shape: Tuple[int, int, int],
    eta: float = 0.0,
    ddim_steps: int = 50
) -> torch.Tensor:
    """
    DDIM sampling: faster sampling with fewer steps.
    
    Args:
        diffusion: GaussianDiffusion instance
        label: Conditions (B, 6)
        geom: Geometry (B, 3, L)
        shape: Output shape (B, 2, L)
        eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
        ddim_steps: Number of sampling steps (< timesteps for speedup)
    
    Returns:
        Samples x_0 (B, 2, L)
    """
    from .ddpm import predict_start_from_noise
    
    B, C, L = shape
    # Get device from model or buffers
    device = diffusion.betas.device
    
    # Create subsequence of timesteps (exclude t=0 which is original)
    # Generate t values: [step_size, 2*step_size, ..., T]
    step_size = diffusion.cfg.timesteps // ddim_steps
    timesteps = list(range(step_size, diffusion.cfg.timesteps + 1, step_size))
    if timesteps[-1] != diffusion.cfg.timesteps:
        timesteps.append(diffusion.cfg.timesteps)  # Ensure final timestep T is included
    timesteps.reverse()  # [T, ..., step_size]
    
    # Start from pure noise
    x = torch.randn(B, C, L, device=device)
    
    for i, t_val in enumerate(timesteps):
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
        
        # Predict noise
        eps_hat = diffusion.model(x, geom, t_batch, label)
        
        # Get next timestep (stop at t=1, not t=0)
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
        else:
            t_prev = 1  # Stop at t=1, not t=0
        
        # Get alphas (use t-1 as index for t > 0)
        idx = t_val - 1
        idx_prev = t_prev - 1
        
        # DDIM update (simplified)
        alpha_bar = diffusion.alphas_cumprod[idx]
        alpha_bar_prev = diffusion.alphas_cumprod[idx_prev]
        
        # Predict x_0
        x0_pred = predict_start_from_noise(diffusion, x, t_batch, eps_hat)
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_bar_prev - eta ** 2) * eps_hat
        
        # DDIM update
        x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
        
        if eta > 0 and idx_prev > 0:
            noise = torch.randn_like(x)
            x = x + eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * \
                torch.sqrt(1 - alpha_bar / alpha_bar_prev) * noise
    
    return x

