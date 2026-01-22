#!/usr/bin/env python3
"""
Linear Beta Schedule
====================

Linear beta schedule from DDPM paper.
"""

import torch


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

