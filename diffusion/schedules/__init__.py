"""
Noise Schedules for Diffusion Models
=====================================

Different noise schedules for diffusion models:
- Linear: Linear beta schedule from DDPM paper
- Cosine: Cosine schedule for smoother noise addition
- Quadratic: Quadratic beta schedule
- Sigmoid: Sigmoid beta schedule

Each schedule defines β_t which controls how noise is added at each timestep.

Usage:
    # Import schedule functions directly
    from diffusion.schedules import linear_beta_schedule, cosine_beta_schedule
    betas = linear_beta_schedule(timesteps=1000)
    betas = cosine_beta_schedule(timesteps=1000, s=0.008)
    
    # Or use the factory function
    from diffusion.schedules import get_noise_schedule
    betas = get_noise_schedule("cosine", timesteps=1000, s=0.008)
    
    # Common utilities
    from diffusion.schedules import compute_alpha_schedule
    alpha_schedule = compute_alpha_schedule(betas)
"""

# Import common utilities
from .common import (
    get_noise_schedule,
    compute_alpha_schedule,
)

# Import individual schedule functions for backward compatibility
from .linear import linear_beta_schedule
from .cosine import cosine_beta_schedule
from .quadratic import quadratic_beta_schedule
from .sigmoid import sigmoid_beta_schedule

__all__ = [
    # Individual schedule functions
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "quadratic_beta_schedule",
    "sigmoid_beta_schedule",
    
    # Common utilities
    "get_noise_schedule",
    "compute_alpha_schedule",
]
