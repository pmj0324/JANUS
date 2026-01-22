#!/usr/bin/env python3
"""
Sigmoid Beta Schedule
=====================

Sigmoid beta schedule for diffusion models.
"""

import torch


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

