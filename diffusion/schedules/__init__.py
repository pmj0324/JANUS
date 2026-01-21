"""
Noise Schedules for Diffusion Models
=====================================

Different noise schedules for diffusion models:
- Linear: Linear beta schedule from DDPM paper
- Cosine: Cosine schedule for smoother noise addition
- Quadratic: Quadratic beta schedule
- Sigmoid: Sigmoid beta schedule

Each schedule defines β_t which controls how noise is added at each timestep.
"""

from .schedules import (
    linear_beta_schedule,
    cosine_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
    get_noise_schedule,
    compute_alpha_schedule,
)

__all__ = [
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "quadratic_beta_schedule",
    "sigmoid_beta_schedule",
    "get_noise_schedule",
    "compute_alpha_schedule",
]
