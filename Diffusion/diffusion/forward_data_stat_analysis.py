#!/usr/bin/env python3
"""
forward_analyze.py

Batch Forward Diffusion Statistical Analysis
============================================

Analyze forward diffusion process for large batches of events.
Focus on statistical analysis, Gaussian convergence, and per-channel
statistics for both charge and time channels.

Usage:
    python diffusion/forward_analyze.py \
        --config configs/default.yaml \
        --batch-size 100 \
        --timesteps 0 100 200 500 999

Features:
    - Batch statistics (mean, std, percentiles)
    - Gaussian convergence testing
    - Per-channel analysis (charge, time)
    - Q-Q plots for Gaussian verification
    - SNR analysis across timesteps
"""

from __future__ import annotations
import argparse
import sys
import os
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config_from_file
from dataloader.pmt_dataloader import PMTSignalsH5
from models.factory import ModelFactory


def load_batch_and_diffusion(config_path: str, batch_size: int):
    """Load batch of events and create diffusion for forward process."""
    print(f"\n📂 Loading configuration from: {config_path}")
    config = load_config_from_file(config_path)
    
    # Create dataset
    print(f"📂 Loading dataset from: {config.data.h5_path}")
    dataset = PMTSignalsH5(
        h5_path=config.data.h5_path,
        replace_time_inf_with=0.0,
        channel_first=True,
        time_transform=config.data.time_transform,
        affine_offsets=tuple(config.data.affine_offsets),
        affine_scales=tuple(config.data.affine_scales),
        label_offsets=tuple(config.data.label_offsets),
        label_scales=tuple(config.data.label_scales),
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} events total")
    
    # Create data loader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    x_sig = batch[0]    # (B, 2, 5160)
    geom = batch[1]     # (B, 3, 5160)
    labels = batch[2]   # (B, 6)
    
    print(f"✅ Batch loaded: {x_sig.shape[0]} events")
    print(f"   Signal shape: {x_sig.shape}")
    print(f"   Geometry shape: {geom.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    # Create diffusion
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model, config.diffusion
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = diffusion.to(device)
    
    return x_sig, geom, labels, diffusion, config, device


def analyze_gaussian_convergence(
    x_sig: torch.Tensor,
    diffusion,
    device: torch.device,
    timesteps: list,
    output_dir: str = "./forward_analysis",
):
    """Analyze Gaussian convergence for forward diffusion."""
    print(f"\n📊 Analyzing Gaussian convergence...")
    
    # Move to device
    x_sig = x_sig.to(device)
    batch_size = x_sig.shape[0]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Statistics table
    print(f"\n{'='*100}")
    print(f"📈 Forward Diffusion Batch Statistics")
    print(f"{'='*100}")
    print(f"{'Timestep':<10} {'Channel':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'P25':<12} {'P75':<12}")
    print(f"{'-'*100}")
    
    results = {}
    
    for t_val in timesteps:
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
        
        # Forward diffusion
        start_time = time.perf_counter()
        x_t = diffusion.q_sample(x_sig, t)
        forward_time = time.perf_counter() - start_time
        
        # Analyze each channel
        for ch_idx, channel_name in enumerate(['charge', 'time']):
            channel_data = x_t[:, ch_idx, :].cpu().numpy()  # (B, 5160)
            flat_data = channel_data.flatten()
            
            # Calculate statistics
            mean_val = np.mean(flat_data)
            std_val = np.std(flat_data)
            min_val = np.min(flat_data)
            max_val = np.max(flat_data)
            p25 = np.percentile(flat_data, 25)
            p75 = np.percentile(flat_data, 75)
            
            print(f"{t_val:<10} {channel_name:<8} {mean_val:<12.4f} {std_val:<12.4f} "
                  f"{min_val:<12.4f} {max_val:<12.4f} {p25:<12.4f} {p75:<12.4f}")
            
            # Store for later analysis
            if t_val not in results:
                results[t_val] = {}
            results[t_val][channel_name] = {
                'data': flat_data,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'p25': p25,
                'p75': p75,
            }
        
        print(f"   ⏱️  Forward time: {forward_time*1000:.1f}ms")
    
    # Gaussian convergence analysis
    print(f"\n{'='*80}")
    print(f"🔬 Gaussian Convergence Analysis")
    print(f"{'='*80}")
    
    for t_val in timesteps:
        print(f"\nTimestep {t_val}:")
        
        for channel_name in ['charge', 'time']:
            data = results[t_val][channel_name]['data']
            
            # Shapiro-Wilk test (for normality)
            # Use subset for large datasets
            if len(data) > 5000:
                sample_indices = np.random.choice(len(data), 5000, replace=False)
                sample_data = data[sample_indices]
            else:
                sample_data = data
            
            try:
                shapiro_stat, shapiro_p = stats.shapiro(sample_data)
                is_normal = shapiro_p > 0.05
                print(f"   {channel_name}: Shapiro-Wilk p={shapiro_p:.6f} ({'Normal' if is_normal else 'Non-normal'})")
            except:
                print(f"   {channel_name}: Shapiro-Wilk test failed (too many ties)")
            
            # Kurtosis and skewness
            kurtosis = stats.kurtosis(data)
            skewness = stats.skew(data)
            print(f"         Kurtosis: {kurtosis:.4f}, Skewness: {skewness:.4f}")
            
            # Theoretical vs empirical std
            theoretical_std = diffusion.sqrt_one_minus_alphas_cumprod[t_val].item()
            empirical_std = results[t_val][channel_name]['std']
            std_ratio = empirical_std / theoretical_std if theoretical_std > 0 else float('inf')
            print(f"         Std ratio (empirical/theoretical): {std_ratio:.4f}")
    
    # SNR analysis
    print(f"\n{'='*80}")
    print(f"📡 SNR Analysis")
    print(f"{'='*80}")
    print(f"{'Timestep':<10} {'SNR':<15} {'Signal Power':<15} {'Noise Power':<15}")
    print(f"{'-'*80}")
    
    for t_val in timesteps:
        sqrt_alpha_bar = diffusion.sqrt_alphas_cumprod[t_val].item()
        sqrt_one_minus = diffusion.sqrt_one_minus_alphas_cumprod[t_val].item()
        snr = sqrt_alpha_bar / sqrt_one_minus if sqrt_one_minus > 0 else float('inf')
        
        signal_power = sqrt_alpha_bar ** 2
        noise_power = sqrt_one_minus ** 2
        
        print(f"{t_val:<10} {snr:<15.4f} {signal_power:<15.4f} {noise_power:<15.4f}")
    
    # Create Q-Q plots
    create_qq_plots(results, timesteps, output_path)
    
    return results


def create_qq_plots(results: dict, timesteps: list, output_path: Path):
    """Create Q-Q plots for Gaussian verification."""
    print(f"\n📊 Creating Q-Q plots...")
    
    for t_val in timesteps:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Q-Q Plots for Forward Diffusion (t={t_val})', fontsize=16)
        
        for ch_idx, (channel_name, ax) in enumerate(zip(['charge', 'time'], axes)):
            data = results[t_val][channel_name]['data']
            
            # Sample data for plotting (Q-Q plots are expensive for large datasets)
            if len(data) > 10000:
                sample_indices = np.random.choice(len(data), 10000, replace=False)
                sample_data = data[sample_indices]
            else:
                sample_data = data
            
            # Create Q-Q plot
            stats.probplot(sample_data, dist="norm", plot=ax)
            ax.set_title(f'{channel_name.title()} Channel')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        qq_path = output_path / f'qq_plot_t{t_val}.png'
        plt.savefig(qq_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Q-Q plot saved: {qq_path}")


def create_statistics_plots(results: dict, timesteps: list, output_path: Path):
    """Create statistics plots across timesteps."""
    print(f"\n📈 Creating statistics plots...")
    
    # Extract statistics
    timesteps_list = sorted(results.keys())
    channels = ['charge', 'time']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Forward Diffusion Statistics Across Timesteps', fontsize=16)
    
    # Mean and std plots
    for ch_idx, channel in enumerate(channels):
        means = [results[t][channel]['mean'] for t in timesteps_list]
        stds = [results[t][channel]['std'] for t in timesteps_list]
        
        # Mean plot
        axes[0, ch_idx].plot(timesteps_list, means, 'o-', label='Mean')
        axes[0, ch_idx].set_title(f'{channel.title()} Channel Mean')
        axes[0, ch_idx].set_xlabel('Timestep')
        axes[0, ch_idx].set_ylabel('Mean Value')
        axes[0, ch_idx].grid(True, alpha=0.3)
        
        # Std plot
        axes[1, ch_idx].plot(timesteps_list, stds, 's-', label='Std', color='orange')
        axes[1, ch_idx].set_title(f'{channel.title()} Channel Std')
        axes[1, ch_idx].set_xlabel('Timestep')
        axes[1, ch_idx].set_ylabel('Standard Deviation')
        axes[1, ch_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    stats_path = output_path / 'statistics_plots.png'
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Statistics plots saved: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze forward diffusion process for batch of events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for analysis"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[0, 100, 200, 500, 999],
        help="Timesteps to analyze"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./forward_analysis",
        help="Output directory for analysis results"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only t=0, t=T/2, t=T-1"
    )
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        T = 1000  # Default timesteps
        args.timesteps = [0, T//2, T-1]
        print(f"🚀 Quick mode: timesteps = {args.timesteps}")
    
    print("\n" + "="*80)
    print("📊 Batch Forward Diffusion Statistical Analysis")
    print("="*80)
    
    # Load batch and diffusion
    x_sig, geom, labels, diffusion, config, device = load_batch_and_diffusion(
        args.config, args.batch_size
    )
    
    # Analyze Gaussian convergence
    results = analyze_gaussian_convergence(
        x_sig=x_sig,
        diffusion=diffusion,
        device=device,
        timesteps=args.timesteps,
        output_dir=args.output_dir,
    )
    
    # Create additional plots
    output_path = Path(args.output_dir)
    create_statistics_plots(results, args.timesteps, output_path)
    
    print(f"\n📁 All analysis results saved to: {output_path}")
    print("\n" + "="*80)
    print("✅ Batch analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
