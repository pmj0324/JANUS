#!/usr/bin/env python3
"""
Common Utilities for Noise Schedules
=====================================

Common functions used across different noise schedules.
"""

import torch
from typing import Dict


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
        from .linear import linear_beta_schedule
        return linear_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule_name == "cosine":
        from .cosine import cosine_beta_schedule
        s = kwargs.get("s", 0.008)
        return cosine_beta_schedule(timesteps, s)
    elif schedule_name == "quadratic":
        from .quadratic import quadratic_beta_schedule
        return quadratic_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule_name == "sigmoid":
        from .sigmoid import sigmoid_beta_schedule
        return sigmoid_beta_schedule(timesteps, beta_start, beta_end)
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}. Choose from: linear, cosine, quadratic, sigmoid")

