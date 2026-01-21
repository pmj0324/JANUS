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

Also includes visualization utilities for comparing schedules.
"""

import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Tuple


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


def compute_alpha_schedule(betas: torch.Tensor) -> dict:
    """
    Compute all alpha-related values from betas.
    
    Args:
        betas: Beta schedule (T,)
    
    Returns:
        Dictionary with all schedule values
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
    
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


def get_noise_schedule(schedule_type: str, timesteps: int, **kwargs) -> torch.Tensor:
    """
    Get noise schedule by name.
    
    Args:
        schedule_type: Type of schedule ("linear", "cosine", "quadratic", "sigmoid")
        timesteps: Number of timesteps
        **kwargs: Additional parameters for specific schedules
        
    Returns:
        Beta values tensor
    """
    if schedule_type.lower() == "linear":
        return linear_beta_schedule(timesteps, 
                                  kwargs.get('beta_start', 1e-4), 
                                  kwargs.get('beta_end', 2e-2))
    elif schedule_type.lower() == "cosine":
        return cosine_beta_schedule(timesteps, 
                                  kwargs.get('s', 0.008))
    elif schedule_type.lower() == "quadratic":
        return quadratic_beta_schedule(timesteps, 
                                     kwargs.get('beta_start', 1e-4), 
                                     kwargs.get('beta_end', 2e-2))
    elif schedule_type.lower() == "sigmoid":
        return sigmoid_beta_schedule(timesteps, 
                                   kwargs.get('beta_start', 1e-4), 
                                   kwargs.get('beta_end', 2e-2))
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def plot_noise_schedules_comparison(timesteps: int = 1000, 
                                  beta_start: float = 1e-4, 
                                  beta_end: float = 2e-2,
                                  cosine_s: float = 0.008,
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot comparison of all noise schedules.
    
    Args:
        timesteps: Number of timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        cosine_s: Cosine schedule offset
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Generate all schedules
    schedules = {
        'Linear': linear_beta_schedule(timesteps, beta_start, beta_end),
        'Cosine': cosine_beta_schedule(timesteps, cosine_s),
        'Quadratic': quadratic_beta_schedule(timesteps, beta_start, beta_end),
        'Sigmoid': sigmoid_beta_schedule(timesteps, beta_start, beta_end)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    colors = ['blue', 'red', 'green', 'orange']
    
    # Main beta plot
    ax = axes[0, 0]
    for i, (name, betas) in enumerate(schedules.items()):
        ax.plot(range(timesteps), betas, label=name, color=colors[i], linewidth=2)
    ax.set_title('Beta Schedules Comparison', fontsize=14)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('β_t')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Zoomed beta plot (first 100 timesteps)
    ax = axes[0, 1]
    for i, (name, betas) in enumerate(schedules.items()):
        ax.plot(range(100), betas[:100], label=name, color=colors[i], linewidth=2)
    ax.set_title('Beta Schedules (First 100 Steps)', fontsize=14)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('β_t')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Beta statistics
    ax = axes[1, 0]
    schedule_stats = []
    for name, betas in schedules.items():
        stats = {
            'name': name,
            'min': betas.min().item(),
            'max': betas.max().item(),
            'mean': betas.mean().item(),
            'std': betas.std().item()
        }
        schedule_stats.append(stats)
    
    names = [s['name'] for s in schedule_stats]
    means = [s['mean'] for s in schedule_stats]
    stds = [s['std'] for s in schedule_stats]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax.bar(x - width/2, means, width, label='Mean', color='skyblue')
    ax.bar(x + width/2, stds, width, label='Std', color='lightcoral')
    
    ax.set_title('Beta Statistics Comparison', fontsize=14)
    ax.set_xlabel('Schedule')
    ax.set_ylabel('Value')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Alpha cumprod comparison
    ax = axes[1, 1]
    for i, (name, betas) in enumerate(schedules.items()):
        alphas = compute_alpha_schedule(betas)
        ax.plot(range(timesteps), alphas['alphas_cumprod'], 
               label=name, color=colors[i], linewidth=2)
    ax.set_title('ᾱ_t (Alpha Cumprod)', fontsize=14)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('ᾱ_t')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Noise schedules comparison saved to: {save_path}")
    
    return fig


def plot_schedule_effects_on_sample(sample_data: torch.Tensor,
                                  timesteps: int = 1000,
                                  test_timesteps: List[int] = [0, 1, 100, 250, 500, 750, 1000],
                                  beta_start: float = 1e-4,
                                  beta_end: float = 2e-2,
                                  cosine_s: float = 0.008,
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot forward diffusion effects on sample data for different schedules.
    
    Args:
        sample_data: Sample data tensor [batch_size, channels, seq_len]
        timesteps: Number of diffusion timesteps
        test_timesteps: Timesteps to test
        beta_start: Starting beta value
        beta_end: Ending beta value
        cosine_s: Cosine schedule offset
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Generate all schedules
    schedules = {
        'Linear': linear_beta_schedule(timesteps, beta_start, beta_end),
        'Cosine': cosine_beta_schedule(timesteps, cosine_s),
        'Quadratic': quadratic_beta_schedule(timesteps, beta_start, beta_end),
        'Sigmoid': sigmoid_beta_schedule(timesteps, beta_start, beta_end)
    }
    
    # Use first sample for visualization
    sample = sample_data[0:1]  # Keep batch dimension
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    colors = ['blue', 'red', 'green', 'orange']
    
    # Apply forward diffusion for each schedule
    results = {}
    for name, betas in schedules.items():
        alphas = compute_alpha_schedule(betas)
        schedule_results = []
        for t in test_timesteps:
            # Apply forward diffusion
            sqrt_alphas_cumprod_t = alphas['sqrt_alphas_cumprod'][t]
            sqrt_one_minus_alphas_cumprod_t = alphas['sqrt_one_minus_alphas_cumprod'][t]
            
            noise = torch.randn_like(sample)
            noisy_sample = sqrt_alphas_cumprod_t * sample + sqrt_one_minus_alphas_cumprod_t * noise
            
            schedule_results.append(noisy_sample)
        results[name] = schedule_results
    
    # Plot charge distributions (final timestep)
    ax = axes[0, 0]
    for i, (name, noisy_samples) in enumerate(results.items()):
        final_sample = noisy_samples[-1]  # Use final timestep
        charge_data = final_sample[0, 0].cpu().numpy()  # First channel (charge)
        ax.hist(charge_data, bins=50, alpha=0.6, label=f'{name} (t={test_timesteps[-1]})', 
               color=colors[i])
    ax.set_title('Charge Distributions (Final Timestep)', fontsize=14)
    ax.set_xlabel('Charge Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot time distributions (final timestep)
    ax = axes[0, 1]
    for i, (name, noisy_samples) in enumerate(results.items()):
        final_sample = noisy_samples[-1]  # Use final timestep
        time_data = final_sample[0, 1].cpu().numpy()  # Second channel (time)
        ax.hist(time_data, bins=50, alpha=0.6, label=f'{name} (t={test_timesteps[-1]})', 
               color=colors[i])
    ax.set_title('Time Distributions (Final Timestep)', fontsize=14)
    ax.set_xlabel('Time Value')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot signal coefficient evolution
    ax = axes[1, 0]
    for i, (name, betas) in enumerate(schedules.items()):
        alphas = compute_alpha_schedule(betas)
        signal_coeffs = [alphas['sqrt_alphas_cumprod'][t].item() for t in test_timesteps]
        ax.plot(test_timesteps, signal_coeffs, 'o-', label=name, 
               color=colors[i], linewidth=2, markersize=6)
    ax.set_title('Signal Coefficient Evolution', fontsize=14)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('√ᾱ_t (Signal Coefficient)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot noise coefficient evolution
    ax = axes[1, 1]
    for i, (name, betas) in enumerate(schedules.items()):
        alphas = compute_alpha_schedule(betas)
        noise_coeffs = [alphas['sqrt_one_minus_alphas_cumprod'][t].item() for t in test_timesteps]
        ax.plot(test_timesteps, noise_coeffs, 'o-', label=name, 
               color=colors[i], linewidth=2, markersize=6)
    ax.set_title('Noise Coefficient Evolution', fontsize=14)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('√(1-ᾱ_t) (Noise Coefficient)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Schedule effects plot saved to: {save_path}")
    
    return fig


def quick_schedule_comparison(timesteps: int = 1000, save_path: Optional[str] = None) -> plt.Figure:
    """
    Quick comparison of all schedules with default parameters.
    
    Args:
        timesteps: Number of timesteps
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    return plot_noise_schedules_comparison(
        timesteps=timesteps,
        beta_start=1e-4,
        beta_end=2e-2,
        cosine_s=0.008,
        save_path=save_path
    )

