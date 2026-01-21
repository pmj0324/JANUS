#!/usr/bin/env python3
"""
Diffusion Utilities (Legacy Compatibility)
==========================================

This file is kept for backward compatibility.
New code should use: from diffusion.forward import ...
"""

# Re-export from new structure
from .forward.forward_diffusion import (
    extract,
    q_sample_batch,
)

# Keep visualization functions here for now (they're not critical)
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple


def visualize_noise_schedule(
    betas: torch.Tensor,
    title: str = "Noise Schedule",
    save_path: Optional[str] = None
):
    """
    Visualize noise schedule and related values.
    
    Args:
        betas: Beta values (T,)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    T = len(betas)
    timesteps = np.arange(T)
    
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Plot 1: Beta schedule
    ax = axes[0, 0]
    ax.plot(timesteps, betas.numpy(), 'b-', linewidth=2)
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('β_t')
    ax.set_title('Beta Schedule')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Alpha and Alpha_bar
    ax = axes[0, 1]
    ax.plot(timesteps, alphas.numpy(), 'g-', linewidth=2, label='α_t = 1 - β_t')
    ax.plot(timesteps, alphas_cumprod.numpy(), 'r-', linewidth=2, label='ᾱ_t = ∏α_t')
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Value')
    ax.set_title('Alpha Schedules')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Signal-to-Noise Ratio
    ax = axes[1, 0]
    snr = alphas_cumprod / (1 - alphas_cumprod)
    ax.plot(timesteps, snr.numpy(), 'purple', linewidth=2)
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('SNR = ᾱ_t / (1 - ᾱ_t)')
    ax.set_title('Signal-to-Noise Ratio')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Noise level
    ax = axes[1, 1]
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alphas_cumprod)
    ax.plot(timesteps, sqrt_one_minus_alpha_bar.numpy(), 'orange', linewidth=2)
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('sqrt(1 - ᾱ_t)')
    ax.set_title('Noise Level')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved noise schedule visualization to: {save_path}")
    
    plt.show()


def compare_noise_schedules(
    schedules: List[Tuple[str, torch.Tensor]],
    save_path: Optional[str] = None
):
    """
    Compare multiple noise schedules.
    
    Args:
        schedules: List of (name, betas) tuples
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Noise Schedule Comparison', fontsize=14, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for idx, (name, betas) in enumerate(schedules):
        color = colors[idx % len(colors)]
        T = len(betas)
        timesteps = np.arange(T)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        snr = alphas_cumprod / (1 - alphas_cumprod)
        
        # Plot betas
        axes[0, 0].plot(timesteps, betas.numpy(), color=color, linewidth=2, label=name)
        
        # Plot alpha_bar
        axes[0, 1].plot(timesteps, alphas_cumprod.numpy(), color=color, linewidth=2, label=name)
        
        # Plot SNR
        axes[1, 0].plot(timesteps, snr.numpy(), color=color, linewidth=2, label=name)
        
        # Plot noise level
        axes[1, 1].plot(timesteps, torch.sqrt(1.0 - alphas_cumprod).numpy(), 
                       color=color, linewidth=2, label=name)
    
    # Configure axes
    axes[0, 0].set_xlabel('Timestep t')
    axes[0, 0].set_ylabel('β_t')
    axes[0, 0].set_title('Beta Schedules')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Timestep t')
    axes[0, 1].set_ylabel('ᾱ_t')
    axes[0, 1].set_title('Cumulative Alpha')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Timestep t')
    axes[1, 0].set_ylabel('SNR')
    axes[1, 0].set_title('Signal-to-Noise Ratio')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Timestep t')
    axes[1, 1].set_ylabel('Noise Level')
    axes[1, 1].set_title('Noise Level sqrt(1 - ᾱ_t)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    plt.show()


__all__ = [
    "extract",
    "q_sample_batch",
    "visualize_noise_schedule",
    "compare_noise_schedules",
]
