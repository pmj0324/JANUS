"""
Diffusion Module for GENESIS
=============================

This module contains all diffusion-related components organized by functionality:
- schedules/: Noise schedulers (linear, cosine, quadratic, sigmoid)
- forward/: Forward diffusion process (adding noise)
- core.py: Main diffusion model (DDPM, DDIM)

Visualization tools are in utils/vis/visualize_forward_diffusion.py

Usage:
    # Use noise schedules
    from diffusion.schedules import get_noise_schedule
    betas = get_noise_schedule("cosine", timesteps=1000, s=0.008)
    
    # Apply forward diffusion
    from diffusion.forward import apply_forward_diffusion
    x_t = apply_forward_diffusion(x0, betas, timesteps)
    
    # Visualize forward diffusion
    from utils.vis.visualize_forward_diffusion import visualize_forward_diffusion
    visualize_forward_diffusion(x0, geom, label, schedules, timesteps)
"""

from .core import (
    GaussianDiffusion,
    create_gaussian_diffusion
)

# Import from new organized structure
from .schedules import (
    linear_beta_schedule,
    cosine_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
    get_noise_schedule,
    compute_alpha_schedule
)

from .forward import (
    q_sample_batch,
    apply_forward_diffusion
)

__all__ = [
    # Gaussian diffusion
    "GaussianDiffusion",
    "create_gaussian_diffusion",
    
    # Noise schedules
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "quadratic_beta_schedule",
    "sigmoid_beta_schedule",
    "get_noise_schedule",
    "compute_alpha_schedule",
    
    # Forward diffusion
    "q_sample_batch",
    "apply_forward_diffusion",
]
