#!/usr/bin/env python3
"""
Diffusion Process Analysis
===========================

Tools to analyze and validate diffusion processes:
- Check convergence to Gaussian distribution
- Visualize forward diffusion at different timesteps
- Compare signal vs noise distributions
- Statistical tests for normality
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional, List, Tuple
from pathlib import Path


def analyze_forward_diffusion(
    x0: torch.Tensor,
    diffusion,
    timesteps_to_check: Optional[List[int]] = None,
    num_samples: int = 10000,
    save_dir: Optional[str] = None
) -> dict:
    """
    Analyze forward diffusion process and check convergence to Gaussian.
    
    Args:
        x0: Clean samples (N, C, L) - will sample from these
        diffusion: GaussianDiffusion instance
        timesteps_to_check: List of timesteps to analyze (default: [0, T//4, T//2, 3T//4, T-1])
        num_samples: Number of samples to use for analysis
        save_dir: Directory to save plots (optional)
    
    Returns:
        Dictionary with analysis results
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    T = diffusion.cfg.timesteps
    # Use diffusion model's device
    device = next(diffusion.parameters()).device
    
    # Default timesteps to check
    if timesteps_to_check is None:
        timesteps_to_check = [0, T//4, T//2, 3*T//4, T-1]
    
    # Sample from x0 and move to device
    N = min(num_samples, x0.size(0))
    x0_samples = x0[:N].to(device)  # (N, C, L) - move to device
    
    results = {}
    
    print("\n" + "="*70)
    print("Forward Diffusion Analysis")
    print("="*70)
    
    for t_idx in timesteps_to_check:
        print(f"\n📊 Analyzing timestep t={t_idx}")
        
        # Create noisy samples at timestep t
        t_batch = torch.full((N,), t_idx, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x0_samples, t_batch)
        
        # Flatten for analysis
        x_t_flat = x_t.reshape(-1).cpu().numpy()
        
        # Compute statistics
        mean = np.mean(x_t_flat)
        std = np.std(x_t_flat)
        skewness = stats.skew(x_t_flat)
        kurtosis = stats.kurtosis(x_t_flat)
        
        # Normality tests
        _, ks_pval = stats.kstest(x_t_flat, 'norm', args=(mean, std))
        _, shapiro_pval = stats.shapiro(x_t_flat[:min(5000, len(x_t_flat))])  # Shapiro limited to 5000
        
        # Store results
        results[t_idx] = {
            'mean': mean,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'ks_test_pval': ks_pval,
            'shapiro_test_pval': shapiro_pval,
            'is_normal': ks_pval > 0.05 and shapiro_pval > 0.05
        }
        
        # Print statistics
        print(f"  Mean: {mean:.6f}")
        print(f"  Std: {std:.6f}")
        print(f"  Skewness: {skewness:.6f} (normal ≈ 0)")
        print(f"  Kurtosis: {kurtosis:.6f} (normal ≈ 0)")
        print(f"  KS test p-value: {ks_pval:.4f} (>0.05 = normal)")
        print(f"  Shapiro test p-value: {shapiro_pval:.4f} (>0.05 = normal)")
        print(f"  {'✅ Passes normality tests' if results[t_idx]['is_normal'] else '❌ Fails normality tests'}")
    
    # Visualize
    _plot_diffusion_convergence(x0_samples, diffusion, timesteps_to_check, save_dir)
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    # Check final timestep
    final_t = timesteps_to_check[-1]
    final_results = results[final_t]
    
    print(f"\n🎯 Final timestep (t={final_t}) check:")
    print(f"   Mean ≈ 0: {abs(final_results['mean']) < 0.1} (|mean|={abs(final_results['mean']):.4f})")
    print(f"   Std ≈ 1: {abs(final_results['std'] - 1.0) < 0.2} (std={final_results['std']:.4f})")
    print(f"   Skewness ≈ 0: {abs(final_results['skewness']) < 0.5} (skew={abs(final_results['skewness']):.4f})")
    print(f"   Is Normal: {final_results['is_normal']}")
    
    if final_results['is_normal']:
        print("\n✅ Forward diffusion successfully converges to Gaussian!")
    else:
        print("\n⚠️ Forward diffusion may not fully converge to Gaussian.")
        print("   Consider: increasing timesteps, adjusting beta schedule, or checking data preprocessing.")
    
    print("="*70)
    
    return results


def _plot_diffusion_convergence(
    x0: torch.Tensor,
    diffusion,
    timesteps: List[int],
    save_dir: Optional[Path]
):
    """Plot histograms showing convergence to Gaussian."""
    N = x0.size(0)
    # x0 is already on the correct device from analyze_forward_diffusion
    device = x0.device
    
    fig, axes = plt.subplots(2, len(timesteps), figsize=(4*len(timesteps), 8))
    if len(timesteps) == 1:
        axes = axes.reshape(2, 1)
    
    for idx, t in enumerate(timesteps):
        # Sample at timestep t
        t_batch = torch.full((N,), t, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x0, t_batch)
        x_t_flat = x_t.reshape(-1).cpu().numpy()
        
        # Histogram
        ax = axes[0, idx]
        counts, bins, _ = ax.hist(x_t_flat, bins=100, density=True, alpha=0.7, color='blue', edgecolor='black')
        
        # Overlay theoretical Gaussian
        mean, std = x_t_flat.mean(), x_t_flat.std()
        x_range = np.linspace(x_t_flat.min(), x_t_flat.max(), 1000)
        gaussian = stats.norm.pdf(x_range, mean, std)
        ax.plot(x_range, gaussian, 'r-', linewidth=2, label='N(μ, σ²)')
        
        ax.set_title(f't={t}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax = axes[1, idx]
        stats.probplot(x_t_flat, dist="norm", plot=ax)
        ax.set_title(f't={t} Q-Q Plot')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Forward Diffusion Convergence to Gaussian', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / 'diffusion_convergence.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved convergence plot to: {save_path}")
    
    plt.show()


def visualize_diffusion_process(
    x0_sample: torch.Tensor,
    diffusion,
    num_steps: int = 10,
    save_path: Optional[str] = None
):
    """
    Visualize forward diffusion process for a single sample.
    
    Args:
        x0_sample: Single clean sample (1, C, L) or (C, L)
        diffusion: GaussianDiffusion instance
        num_steps: Number of intermediate steps to show
        save_path: Path to save figure (optional)
    """
    if x0_sample.ndim == 2:
        x0_sample = x0_sample.unsqueeze(0)  # (1, C, L)
    
    T = diffusion.cfg.timesteps
    # Use diffusion model's device
    device = next(diffusion.parameters()).device
    x0_sample = x0_sample.to(device)
    
    # Select timesteps to visualize
    timesteps = np.linspace(0, T-1, num_steps, dtype=int)
    
    fig, axes = plt.subplots(2, num_steps, figsize=(2*num_steps, 4))
    if num_steps == 1:
        axes = axes.reshape(2, 1)
    
    for idx, t in enumerate(timesteps):
        t_batch = torch.tensor([t], device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x0_sample, t_batch)
        
        # Plot charge (channel 0)
        ax = axes[0, idx]
        charge = x_t[0, 0, :].cpu().numpy()
        ax.plot(charge, linewidth=0.5)
        ax.set_title(f't={t}')
        ax.set_ylabel('Charge')
        if idx == 0:
            ax.set_ylabel('Charge\n(channel 0)', fontweight='bold')
        
        # Plot time (channel 1)
        ax = axes[1, idx]
        time = x_t[0, 1, :].cpu().numpy()
        ax.plot(time, linewidth=0.5)
        ax.set_xlabel('PMT index')
        if idx == 0:
            ax.set_ylabel('Time\n(channel 1)', fontweight='bold')
    
    fig.suptitle('Forward Diffusion Process', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diffusion process to: {save_path}")
    
    plt.show()


def batch_analysis(
    dataloader,
    diffusion,
    num_batches: int = 10,
    save_dir: Optional[str] = None
) -> dict:
    """
    Analyze forward diffusion over multiple batches from dataloader.
    
    Args:
        dataloader: DataLoader instance
        diffusion: GaussianDiffusion instance
        num_batches: Number of batches to analyze
        save_dir: Directory to save results
    
    Returns:
        Analysis results dictionary
    """
    print(f"\n🔍 Analyzing {num_batches} batches from dataloader...")
    
    all_x0 = []
    all_geom = []
    
    for idx, (x_sig, geom, label, _) in enumerate(dataloader):
        if idx >= num_batches:
            break
        all_x0.append(x_sig)
        all_geom.append(geom)
    
    # Concatenate all batches
    x0 = torch.cat(all_x0, dim=0)  # (N, 2, L)
    
    print(f"✅ Collected {x0.size(0)} samples")
    
    # Run analysis
    results = analyze_forward_diffusion(
        x0,
        diffusion,
        num_samples=min(10000, x0.size(0)),
        save_dir=save_dir
    )
    
    return results

