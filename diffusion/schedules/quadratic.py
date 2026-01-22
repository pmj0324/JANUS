#!/usr/bin/env python3
"""
Quadratic Beta Schedule
=======================

Quadratic beta schedule for diffusion models.
"""

import torch


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

