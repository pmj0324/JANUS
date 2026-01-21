#!/usr/bin/env python3
"""
Noise Schedules for Diffusion
==============================

Different noise schedules for diffusion models:
- Linear
- Cosine
- Quadratic
- Sigmoid

Each schedule defines β_t which controls how noise is added at each timestep.
"""

import torch
import math
from typing import Dict


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """
    Linear beta schedule from DDPM paper.
    
    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
    
    Returns:
        Beta values (timesteps,)
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    
    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent β_t from being too small near t=0
    
    Returns:
        Beta values (timesteps,)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """
    Quadratic beta schedule.
    
    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
    
    Returns:
        Beta values (timesteps,)
    """
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """
    Sigmoid beta schedule.
    
    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
    
    Returns:
        Beta values (timesteps,)
    """
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def get_noise_schedule(
    schedule_name: str,
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    **kwargs
) -> torch.Tensor:
    """
    Get noise schedule by name.
    
    Args:
        schedule_name: "linear", "cosine", "quadratic", or "sigmoid"
        timesteps: Number of diffusion steps
        beta_start: Starting beta value
        beta_end: Ending beta value
        **kwargs: Additional schedule-specific parameters
    
    Returns:
        Beta values (timesteps,)
    """
    schedule_name = schedule_name.lower()
    
    if schedule_name == "linear":
        return linear_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule_name == "cosine":
        s = kwargs.get("s", 0.008)
        return cosine_beta_schedule(timesteps, s)
    elif schedule_name == "quadratic":
        return quadratic_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule_name == "sigmoid":
        return sigmoid_beta_schedule(timesteps, beta_start, beta_end)
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}. Choose from: linear, cosine, quadratic, sigmoid")


def compute_alpha_schedule(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute all alpha-related values from betas.
    
    Args:
        betas: Beta schedule (T,)
    
    Returns:
        Dictionary with all schedule values
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, device=betas.device, dtype=betas.dtype), alphas_cumprod[:-1]])
    
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "log_one_minus_alphas_cumprod": torch.log(1.0 - alphas_cumprod),
        "sqrt_recip_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod),
        "sqrt_recipm1_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod - 1),
    }
