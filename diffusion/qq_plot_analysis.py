#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-Q Plot Analysis for Forward Diffusion Gaussian Convergence

This module provides Q-Q plot visualization to verify that forward diffusion
converges to Gaussian distribution as expected.

Usage:
    python diffusion/qq_plot_analysis.py \
        --config configs/default.yaml \
        --batch-size 100 \
        --timesteps 0 100 250 500 750 1000
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from io import StringIO

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config_from_file
from dataloader.pmt_dataloader import PMTSignalsH5
from models.factory import ModelFactory


def create_qq_plots(
    x0: torch.Tensor,
    diffusion,
    timesteps: List[int],
    save_dir: Optional[str] = None,
    per_channel: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Create Q-Q plots for Gaussian convergence verification.
    
    Args:
        x0: Clean samples (N, C, L)
        diffusion: GaussianDiffusion instance
        timesteps: List of timesteps to analyze
        save_dir: Directory to save plots (optional)
        per_channel: Whether to analyze each channel separately
    
    Returns:
        Dictionary with Q-Q plot results
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    T = diffusion.cfg.timesteps
    device = next(diffusion.parameters()).device
    
    # Move to device
    x0 = x0.to(device)
    N = x0.size(0)
    
    results = {}
    
    print("\n" + "="*80)
    print("📊 Q-Q Plot Analysis for Gaussian Convergence")
    print("="*80)
    print(f"Note: t=0 is original data, t=1 is first noise step, t={T} is final timestep")
    
    for t_idx in timesteps:
        print(f"\n📊 Creating Q-Q plots for timestep t={t_idx}")
        
        # Create noisy samples at timestep t
        t_batch = torch.full((N,), t_idx, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x0, t_batch)
        
        if per_channel:
            # Per-channel analysis (charge, time separately)
            results[t_idx] = {}
            
            # Create subplot for each channel
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'Q-Q Plots for Timestep t={t_idx}', fontsize=16, fontweight='bold')
            
            for ch_idx, (ch_name, ax) in enumerate(zip(['charge', 'time'], axes)):
                ch_data = x_t[:, ch_idx, :].flatten().cpu().numpy()
                
                # Remove NaN values
                ch_data = ch_data[~np.isnan(ch_data)]
                
                if len(ch_data) > 0:
                    # Create Q-Q plot
                    stats.probplot(ch_data, dist="norm", plot=ax)
                    ax.set_title(f'{ch_name.title()} Channel (t={t_idx})')
                    ax.set_xlabel('Theoretical Quantiles')
                    ax.set_ylabel('Sample Quantiles')
                    ax.grid(True, alpha=0.3)
                    
                    # Calculate correlation coefficient
                    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(ch_data)))
                    sample_quantiles = np.sort(ch_data)
                    correlation = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]
                    
                    # Add correlation info to plot
                    ax.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                           transform=ax.transAxes, fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Store results
                    results[t_idx][ch_name] = {
                        'correlation': correlation,
                        'is_gaussian': correlation > 0.99,  # High correlation indicates Gaussian
                        'mean': np.mean(ch_data),
                        'std': np.std(ch_data),
                        'count': len(ch_data)
                    }
                    
                    print(f"   {ch_name}: Correlation={correlation:.4f}, Gaussian={correlation > 0.99}")
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    results[t_idx][ch_name] = {
                        'correlation': 0.0,
                        'is_gaussian': False,
                        'mean': 0.0,
                        'std': 0.0,
                        'count': 0
                    }
            
            plt.tight_layout()
            
            # Save plot
            if save_dir:
                qq_path = save_dir / f'qq_plot_t{t_idx}.png'
                plt.savefig(qq_path, dpi=150, bbox_inches='tight')
                print(f"   ✅ Q-Q plot saved: {qq_path}")
            
            plt.close()
            
        else:
            # Overall analysis (all channels combined)
            x_t_flat = x_t.reshape(-1).cpu().numpy()
            x_t_flat = x_t_flat[~np.isnan(x_t_flat)]
            
            if len(x_t_flat) > 0:
                # Create single Q-Q plot
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                fig.suptitle(f'Q-Q Plot for Timestep t={t_idx} (All Channels)', fontsize=16, fontweight='bold')
                
                stats.probplot(x_t_flat, dist="norm", plot=ax)
                ax.set_title(f'Overall Distribution (t={t_idx})')
                ax.set_xlabel('Theoretical Quantiles')
                ax.set_ylabel('Sample Quantiles')
                ax.grid(True, alpha=0.3)
                
                # Calculate correlation coefficient
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(x_t_flat)))
                sample_quantiles = np.sort(x_t_flat)
                correlation = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]
                
                # Add correlation info to plot
                ax.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                       transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                
                # Save plot
                if save_dir:
                    qq_path = save_dir / f'qq_plot_t{t_idx}.png'
                    plt.savefig(qq_path, dpi=150, bbox_inches='tight')
                    print(f"   ✅ Q-Q plot saved: {qq_path}")
                
                plt.close()
                
                results[t_idx] = {
                    'correlation': correlation,
                    'is_gaussian': correlation > 0.99,
                    'mean': np.mean(x_t_flat),
                    'std': np.std(x_t_flat),
                    'count': len(x_t_flat)
                }
                
                print(f"   Overall: Correlation={correlation:.4f}, Gaussian={correlation > 0.99}")
            else:
                results[t_idx] = {
                    'correlation': 0.0,
                    'is_gaussian': False,
                    'mean': 0.0,
                    'std': 0.0,
                    'count': 0
                }
    
    # Create summary plot
    if save_dir and per_channel:
        create_correlation_summary_plot(results, timesteps, save_dir)
    
    return results


def create_correlation_summary_plot(results: Dict, timesteps: List[int], save_dir: Path):
    """Create summary plot showing correlation coefficients across timesteps."""
    print(f"\n📈 Creating correlation summary plot...")
    
    timesteps_list = sorted(results.keys())
    channels = ['charge', 'time']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Q-Q Plot Correlation Coefficients Across Timesteps', fontsize=16, fontweight='bold')
    
    for ch_idx, channel in enumerate(channels):
        correlations = []
        gaussian_flags = []
        
        for t in timesteps_list:
            if t in results and channel in results[t]:
                correlations.append(results[t][channel]['correlation'])
                gaussian_flags.append(results[t][channel]['is_gaussian'])
            else:
                correlations.append(0.0)
                gaussian_flags.append(False)
        
        # Plot correlation coefficients
        ax = axes[ch_idx]
        line = ax.plot(timesteps_list, correlations, 'o-', linewidth=2, markersize=8, label='Correlation')
        
        # Color points based on Gaussian status
        for i, (t, corr, is_gauss) in enumerate(zip(timesteps_list, correlations, gaussian_flags)):
            color = 'green' if is_gauss else 'red'
            ax.scatter(t, corr, c=color, s=100, alpha=0.7, zorder=5)
        
        # Add threshold line
        ax.axhline(y=0.99, color='green', linestyle='--', alpha=0.7, label='Gaussian Threshold (0.99)')
        ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='Good Fit Threshold (0.95)')
        
        ax.set_title(f'{channel.title()} Channel')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Correlation Coefficient')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0.8, 1.0)
    
    plt.tight_layout()
    
    summary_path = save_dir / 'qq_correlation_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Correlation summary saved: {summary_path}")


def print_qq_summary(results: Dict, timesteps: List[int]):
    """Print Q-Q plot analysis summary."""
    print("\n" + "="*80)
    print("📋 Q-Q Plot Analysis Summary")
    print("="*80)
    
    final_t = timesteps[-1]
    if final_t in results:
        if isinstance(results[final_t], dict) and 'charge' in results[final_t]:
            # Per-channel summary
            print(f"\n🎯 Final timestep (t={final_t}) Q-Q analysis:")
            for channel in ['charge', 'time']:
                if channel in results[final_t]:
                    ch_results = results[final_t][channel]
                    print(f"   {channel}:")
                    print(f"     Correlation: {ch_results['correlation']:.6f}")
                    print(f"     Is Gaussian: {ch_results['is_gaussian']} {'✅' if ch_results['is_gaussian'] else '❌'}")
                    print(f"     Mean: {ch_results['mean']:.6f}")
                    print(f"     Std: {ch_results['std']:.6f}")
                    print(f"     Count: {ch_results['count']:,}")
        else:
            # Overall summary
            final_results = results[final_t]
            print(f"\n🎯 Final timestep (t={final_t}) Q-Q analysis:")
            print(f"   Correlation: {final_results['correlation']:.6f}")
            print(f"   Is Gaussian: {final_results['is_gaussian']} {'✅' if final_results['is_gaussian'] else '❌'}")
            print(f"   Mean: {final_results['mean']:.6f}")
            print(f"   Std: {final_results['std']:.6f}")
            print(f"   Count: {final_results['count']:,}")
    
    # Convergence check
    print(f"\n🔍 Q-Q Convergence Check (Last 3 timesteps: {timesteps[-3:]}):")
    failed_tests = []
    
    for t in timesteps[-3:]:
        if t in results:
            if isinstance(results[t], dict) and 'charge' in results[t]:
                # Per-channel analysis
                for channel in ['charge', 'time']:
                    if channel in results[t]:
                        ch_results = results[t][channel]
                        is_gaussian = ch_results['is_gaussian']
                        correlation = ch_results['correlation']
                        
                        status = "✅ PASS" if is_gaussian else "❌ FAIL"
                        print(f"   t={t}, {channel}: {status} (correlation={correlation:.4f})")
                        
                        if not is_gaussian:
                            failed_tests.append(f"t={t}, {channel} (correlation={correlation:.4f})")
            else:
                # Overall analysis
                overall_results = results[t]
                is_gaussian = overall_results['is_gaussian']
                correlation = overall_results['correlation']
                
                status = "✅ PASS" if is_gaussian else "❌ FAIL"
                print(f"   t={t}, overall: {status} (correlation={correlation:.4f})")
                
                if not is_gaussian:
                    failed_tests.append(f"t={t}, overall (correlation={correlation:.4f})")
    
    # Final convergence result
    all_gaussian = all(
        results[t].get('is_gaussian', False) if isinstance(results[t], dict) and 'charge' not in results[t]
        else all(results[t][ch]['is_gaussian'] for ch in ['charge', 'time'] if ch in results[t])
        for t in timesteps[-3:]  # Check last 3 timesteps
    )
    
    print(f"\n📊 Q-Q Convergence Result:")
    if all_gaussian:
        print("✅ Forward diffusion successfully converges to Gaussian!")
        print("   All Q-Q plots show high correlation (>0.99) with theoretical Gaussian")
    else:
        print("⚠️ Forward diffusion may not fully converge to Gaussian.")
        print("   Failed Q-Q tests:")
        for failed_test in failed_tests:
            print(f"     • {failed_test}")
        print("   Consider: increasing timesteps, adjusting beta schedule, or checking data preprocessing.")
    
    print("="*80)


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


def main():
    parser = argparse.ArgumentParser(
        description="Q-Q plot analysis for forward diffusion Gaussian convergence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=100,
        help="Batch size for analysis"
    )
    
    parser.add_argument(
        "-t", "--timesteps",
        type=int,
        nargs="+",
        default=[0, 1, 250, 500, 750, 1000],
        help="Timesteps to analyze"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./qq_analysis",
        help="Output directory for Q-Q plots"
    )
    
    parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Quick mode: only t=0, t=1, t=T/2, t=T (final timestep)"
    )
    
    parser.add_argument(
        "-p", "--per-channel",
        action="store_true",
        default=True,
        help="Analyze each channel separately (charge, time)"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Quick mode
        if args.quick:
            T = 1000  # Default timesteps
            args.timesteps = [0, 1, T//2, T]
            print(f"🚀 Quick mode: timesteps = {args.timesteps}")
        
        print("\n" + "="*80)
        print("📊 Q-Q Plot Analysis for Forward Diffusion Gaussian Convergence")
        print("="*80)
        
        # Load batch and diffusion
        x_sig, geom, labels, diffusion, config, device = load_batch_and_diffusion(
            args.config, args.batch_size
        )
        
        # Run Q-Q plot analysis
        results = create_qq_plots(
            x_sig,
            diffusion,
            timesteps=args.timesteps,
            save_dir=args.output_dir,
            per_channel=args.per_channel
        )
        
        # Print summary
        print_qq_summary(results, args.timesteps)
        
        print(f"\n📁 All Q-Q plots saved to: {output_path}")
        print("\n" + "="*80)
        print("✅ Q-Q plot analysis complete!")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error during Q-Q plot analysis: {e}")
        raise


if __name__ == "__main__":
    main()
