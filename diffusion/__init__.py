"""
Diffusion Module for GENESIS
=============================

This module contains all diffusion-related components:
- Gaussian diffusion process (DDPM, DDIM)
- Noise schedulers (linear, cosine, etc.)
- Diffusion visualization and analysis tools
- Distribution analysis (convergence to Gaussian)

Future: Will also support Flow Matching in a separate 'flow' module.
"""

from .gaussian_diffusion import (
    GaussianDiffusion,
    create_gaussian_diffusion
)
from .noise_schedules import (
    linear_beta_schedule,
    cosine_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
    get_noise_schedule,
    compute_alpha_schedule
)
from .diffusion_utils import (
    extract,
    q_sample_batch,
    visualize_noise_schedule,
    compare_noise_schedules
)
# Analysis functions moved to forward_data_stats_analysis.py
# Import them directly if needed:
# from .forward_data_stats_analysis import analyze_forward_diffusion, batch_analysis

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
    
    # Utilities
    "extract",
    "q_sample_batch",
    "visualize_noise_schedule",
    "compare_noise_schedules",
    
    # Analysis functions moved to forward_data_stats_analysis.py
]