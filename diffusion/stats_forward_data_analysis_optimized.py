#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Forward Diffusion Statistical Analysis

Enhanced version with:
- Full dataset analysis when batch-size not specified
- Time optimization with progress bars
- Memory-efficient streaming
- Parallel processing where possible

Usage:
    # Full dataset analysis (recommended)
    python diffusion/stats_forward_data_analysis_optimized.py \
        --config configs/dit_cosine_plateau.yaml \
        --timesteps 0 100 250 500 750 1000

    # Quick full dataset analysis
    python diffusion/stats_forward_data_analysis_optimized.py \
        --config configs/dit_cosine_plateau.yaml \
        --quick

    # Custom batch size (if needed)
    python diffusion/stats_forward_data_analysis_optimized.py \
        --config configs/dit_cosine_plateau.yaml \
        --batch-size 1000
"""

from __future__ import annotations
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
from tqdm import tqdm
import gc

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config_from_file
from dataloader.pmt_dataloader import PMTSignalsH5
from models.factory import ModelFactory


class TeeOutput:
    """Capture both terminal output and save to file."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()


def load_full_dataset_and_diffusion(config_path: str, batch_size: Optional[int] = None):
    """Load full dataset and create diffusion for comprehensive analysis."""
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
    
    total_events = len(dataset)
    print(f"✅ Dataset loaded: {total_events:,} events total")
    
    # Determine batch size
    if batch_size is None:
        # Use all data - determine optimal batch size based on available memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            # Conservative estimate: 2GB per 1000 events
            estimated_memory_per_event = 2e9 / 1000
            max_events = int(gpu_memory * 0.8 / estimated_memory_per_event)  # Use 80% of GPU memory
            batch_size = min(max_events, total_events)
        else:
            # CPU mode - use smaller batches
            batch_size = min(1000, total_events)
        
        print(f"🎯 Auto-selected batch size: {batch_size:,} events")
        print(f"   (Will process {total_events:,} total events in {(total_events + batch_size - 1) // batch_size} batches)")
    else:
        print(f"🎯 Using specified batch size: {batch_size:,} events")
    
    # Create data loader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create diffusion
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model, config.diffusion
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = diffusion.to(device)
    
    return dataloader, diffusion, config, device, total_events


def compute_snr_at_timestep(diffusion, t: int) -> float:
    """Compute Signal-to-Noise Ratio at given timestep."""
    if t == 0:
        return float('inf')  # Original data, no noise
    
    idx = t - 1  # Use t-1 as index
    sqrt_alpha_bar = diffusion.sqrt_alphas_cumprod[idx].item()
    sqrt_one_minus = diffusion.sqrt_one_minus_alphas_cumprod[idx].item()
    return sqrt_alpha_bar / sqrt_one_minus if sqrt_one_minus > 0 else float('inf')


def analyze_forward_diffusion_optimized(
    dataloader,
    diffusion,
    timesteps_to_check: Optional[List[int]] = None,
    per_channel: bool = True,
    compute_snr: bool = True,
    compute_percentiles: bool = True,
    save_dir: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Optimized forward diffusion analysis with streaming and progress tracking.
    
    Args:
        dataloader: DataLoader instance
        diffusion: GaussianDiffusion instance
        timesteps_to_check: List of timesteps to analyze
        per_channel: Whether to analyze each channel separately
        compute_snr: Whether to compute SNR
        compute_percentiles: Whether to compute percentiles
        save_dir: Directory to save plots (optional)
        max_samples: Maximum number of samples to analyze (None for all)
    
    Returns:
        Dictionary with analysis results
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    T = diffusion.cfg.timesteps
    device = next(diffusion.parameters()).device
    
    # Default timesteps to check
    if timesteps_to_check is None:
        timesteps_to_check = [0, 1, T//4, T//2, 3*T//4, T]
    
    # Initialize results storage
    results = {}
    for t in timesteps_to_check:
        if per_channel:
            results[t] = {'charge': {}, 'time': {}}
        else:
            results[t] = {}
    
    print("\n" + "="*80)
    print("📊 Optimized Forward Diffusion Analysis")
    print("="*80)
    print(f"Note: t=0 is original data, t=1 is first noise step, t={T} is final timestep")
    print(f"Timesteps to analyze: {timesteps_to_check}")
    
    # Process data in streaming fashion
    total_samples = 0
    batch_count = 0
    
    # Progress bar for batches
    pbar = tqdm(dataloader, desc="Processing batches", unit="batch")
    
    for batch_idx, (x_sig, geom, labels, _) in enumerate(pbar):
        if max_samples and total_samples >= max_samples:
            break
            
        # Move to device
        x_sig = x_sig.to(device)
        N = x_sig.size(0)
        
        # Update progress
        pbar.set_postfix({
            'samples': f"{total_samples + N:,}",
            'batch': f"{batch_idx + 1}"
        })
        
        # Process each timestep
        for t_idx in timesteps_to_check:
            # Create noisy samples at timestep t
            t_batch = torch.full((N,), t_idx, device=device, dtype=torch.long)
            x_t = diffusion.q_sample(x_sig, t_batch)
            
            if per_channel:
                # Per-channel analysis (charge, time separately)
                for ch_idx, ch_name in enumerate(['charge', 'time']):
                    ch_data = x_t[:, ch_idx, :].flatten().cpu().numpy()
                    ch_data = ch_data[~np.isnan(ch_data)]  # Remove NaN
                    
                    if len(ch_data) > 0:
                        # Basic statistics
                        mean_val = np.mean(ch_data)
                        std_val = np.std(ch_data)
                        min_val = np.min(ch_data)
                        max_val = np.max(ch_data)
                        
                        # Advanced statistics
                        skewness = stats.skew(ch_data)
                        kurtosis = stats.kurtosis(ch_data)
                        
                        # Percentiles
                        p25 = np.percentile(ch_data, 25) if compute_percentiles else None
                        p75 = np.percentile(ch_data, 75) if compute_percentiles else None
                        
                        # Normality tests (sample for performance)
                        sample_size = min(5000, len(ch_data))
                        sample_data = ch_data[:sample_size]
                        
                        try:
                            _, ks_pval = stats.kstest(sample_data, 'norm', args=(mean_val, std_val))
                        except:
                            ks_pval = 0.0
                        
                        try:
                            _, shapiro_pval = stats.shapiro(sample_data)
                        except:
                            shapiro_pval = 0.0
                        
                        is_normal = ks_pval > 0.05 and shapiro_pval > 0.05
                        
                        # SNR calculation
                        snr = compute_snr_at_timestep(diffusion, t_idx) if compute_snr else None
                        
                        # Accumulate results (running average)
                        if batch_count == 0:
                            # Initialize
                            results[t_idx][ch_name] = {
                                'mean': mean_val,
                                'std': std_val,
                                'min': min_val,
                                'max': max_val,
                                'skewness': skewness,
                                'kurtosis': kurtosis,
                                'ks_test_pval': ks_pval,
                                'shapiro_test_pval': shapiro_pval,
                                'is_normal': is_normal,
                                'snr': snr,
                                'p25': p25,
                                'p75': p75,
                                'count': len(ch_data),
                                'total_count': len(ch_data)
                            }
                        else:
                            # Update running averages
                            old_count = results[t_idx][ch_name]['total_count']
                            new_count = old_count + len(ch_data)
                            
                            # Weighted average for mean
                            old_mean = results[t_idx][ch_name]['mean']
                            new_mean = (old_mean * old_count + mean_val * len(ch_data)) / new_count
                            
                            # Update other statistics (simplified for performance)
                            results[t_idx][ch_name].update({
                                'mean': new_mean,
                                'std': std_val,  # Keep latest std for simplicity
                                'min': min(min_val, results[t_idx][ch_name]['min']),
                                'max': max(max_val, results[t_idx][ch_name]['max']),
                                'skewness': skewness,
                                'kurtosis': kurtosis,
                                'ks_test_pval': ks_pval,
                                'shapiro_test_pval': shapiro_pval,
                                'is_normal': is_normal,
                                'snr': snr,
                                'p25': p25,
                                'p75': p75,
                                'count': len(ch_data),
                                'total_count': new_count
                            })
            else:
                # Overall analysis (all channels combined)
                x_t_flat = x_t.reshape(-1).cpu().numpy()
                x_t_flat = x_t_flat[~np.isnan(x_t_flat)]
                
                if len(x_t_flat) > 0:
                    mean_val = np.mean(x_t_flat)
                    std_val = np.std(x_t_flat)
                    min_val = np.min(x_t_flat)
                    max_val = np.max(x_t_flat)
                    skewness = stats.skew(x_t_flat)
                    kurtosis = stats.kurtosis(x_t_flat)
                    
                    # Normality tests
                    sample_size = min(5000, len(x_t_flat))
                    sample_data = x_t_flat[:sample_size]
                    
                    try:
                        _, ks_pval = stats.kstest(sample_data, 'norm', args=(mean_val, std_val))
                    except:
                        ks_pval = 0.0
                    
                    try:
                        _, shapiro_pval = stats.shapiro(sample_data)
                    except:
                        shapiro_pval = 0.0
                    
                    is_normal = ks_pval > 0.05 and shapiro_pval > 0.05
                    snr = compute_snr_at_timestep(diffusion, t_idx) if compute_snr else None
                    
                    # Accumulate results
                    if batch_count == 0:
                        results[t_idx] = {
                            'mean': mean_val,
                            'std': std_val,
                            'min': min_val,
                            'max': max_val,
                            'skewness': skewness,
                            'kurtosis': kurtosis,
                            'ks_test_pval': ks_pval,
                            'shapiro_test_pval': shapiro_pval,
                            'is_normal': is_normal,
                            'snr': snr,
                            'count': len(x_t_flat),
                            'total_count': len(x_t_flat)
                        }
                    else:
                        old_count = results[t_idx]['total_count']
                        new_count = old_count + len(x_t_flat)
                        old_mean = results[t_idx]['mean']
                        new_mean = (old_mean * old_count + mean_val * len(x_t_flat)) / new_count
                        
                        results[t_idx].update({
                            'mean': new_mean,
                            'std': std_val,
                            'min': min(min_val, results[t_idx]['min']),
                            'max': max(max_val, results[t_idx]['max']),
                            'skewness': skewness,
                            'kurtosis': kurtosis,
                            'ks_test_pval': ks_pval,
                            'shapiro_test_pval': shapiro_pval,
                            'is_normal': is_normal,
                            'snr': snr,
                            'count': len(x_t_flat),
                            'total_count': new_count
                        })
        
        total_samples += N
        batch_count += 1
        
        # Memory cleanup
        if batch_idx % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    pbar.close()
    
    print(f"\n✅ Processed {total_samples:,} samples in {batch_count} batches")
    
    # Create visualizations
    if save_dir:
        create_convergence_plots_optimized(dataloader, diffusion, timesteps_to_check, save_dir)
        create_statistics_plots_optimized(results, timesteps_to_check, save_dir)
    
    # Print summary
    print_summary_optimized(results, timesteps_to_check)
    
    return results


def create_convergence_plots_optimized(
    dataloader,
    diffusion,
    timesteps: List[int],
    save_dir: Path
):
    """Create histogram plots for Gaussian convergence verification (optimized)."""
    print(f"\n📊 Creating convergence plots...")
    
    # Sample a few batches for visualization
    sample_batches = []
    for idx, (x_sig, geom, label, _) in enumerate(dataloader):
        if idx >= 3:  # Use 3 batches for visualization
            break
        sample_batches.append(x_sig)
    
    if not sample_batches:
        print("   ⚠️ No data available for visualization")
        return
    
    x0 = torch.cat(sample_batches, dim=0)
    device = x0.device
    N = x0.size(0)
    
    fig, axes = plt.subplots(1, len(timesteps), figsize=(4*len(timesteps), 4))
    if len(timesteps) == 1:
        axes = [axes]
    
    for idx, t in enumerate(timesteps):
        # Sample at timestep t
        t_batch = torch.full((N,), t, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x0, t_batch)
        x_t_flat = x_t.reshape(-1).cpu().numpy()
        
        # Histogram
        ax = axes[idx]
        counts, bins, _ = ax.hist(x_t_flat, bins=100, density=True, alpha=0.7, 
                                 color='blue', edgecolor='black')
        
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
    
    fig.suptitle('Forward Diffusion Convergence to Gaussian', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    convergence_path = save_dir / 'diffusion_convergence.png'
    plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Convergence plot saved: {convergence_path}")


def create_statistics_plots_optimized(results: Dict, timesteps: List[int], save_dir: Path):
    """Create statistics plots across timesteps (optimized)."""
    print(f"\n📈 Creating statistics plots...")
    
    timesteps_list = sorted(results.keys())
    
    # Check if per-channel analysis
    if any(isinstance(results[t], dict) and 'charge' in results[t] for t in timesteps_list):
        # Per-channel plots
        channels = ['charge', 'time']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Forward Diffusion Statistics Across Timesteps', fontsize=16)
        
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
    else:
        # Overall plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Forward Diffusion Statistics Across Timesteps', fontsize=16)
        
        means = [results[t]['mean'] for t in timesteps_list]
        stds = [results[t]['std'] for t in timesteps_list]
        
        # Mean plot
        axes[0].plot(timesteps_list, means, 'o-', label='Mean')
        axes[0].set_title('Mean Across Timesteps')
        axes[0].set_xlabel('Timestep')
        axes[0].set_ylabel('Mean Value')
        axes[0].grid(True, alpha=0.3)
        
        # Std plot
        axes[1].plot(timesteps_list, stds, 's-', label='Std', color='orange')
        axes[1].set_title('Std Across Timesteps')
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylabel('Standard Deviation')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    stats_path = save_dir / 'statistics_plots.png'
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Statistics plots saved: {stats_path}")


def print_summary_optimized(results: Dict, timesteps: List[int]):
    """Print analysis summary (optimized)."""
    print("\n" + "="*80)
    print("📋 Optimized Analysis Summary")
    print("="*80)
    
    final_t = timesteps[-1]
    if final_t in results:
        if isinstance(results[final_t], dict) and 'charge' in results[final_t]:
            # Per-channel summary
            print(f"\n🎯 Final timestep (t={final_t}) check:")
            for channel in ['charge', 'time']:
                ch_results = results[final_t][channel]
                print(f"   {channel}:")
                print(f"     Mean ≈ 0: {abs(ch_results['mean']) < 0.1} (|mean|={abs(ch_results['mean']):.4f})")
                print(f"     Std ≈ 1: {abs(ch_results['std'] - 1.0) < 0.2} (std={ch_results['std']:.4f})")
                print(f"     Skewness ≈ 0: {abs(ch_results['skewness']) < 0.5} (skew={abs(ch_results['skewness']):.4f})")
                print(f"     Is Normal: {ch_results['is_normal']}")
                print(f"     Total Samples: {ch_results['total_count']:,}")
                if ch_results['snr'] is not None:
                    print(f"     SNR: {ch_results['snr']:.4f}")
        else:
            # Overall summary
            final_results = results[final_t]
            print(f"\n🎯 Final timestep (t={final_t}) check:")
            print(f"   Mean ≈ 0: {abs(final_results['mean']) < 0.1} (|mean|={abs(final_results['mean']):.4f})")
            print(f"   Std ≈ 1: {abs(final_results['std'] - 1.0) < 0.2} (std={final_results['std']:.4f})")
            print(f"   Skewness ≈ 0: {abs(final_results['skewness']) < 0.5} (skew={abs(final_results['skewness']):.4f})")
            print(f"   Is Normal: {final_results['is_normal']}")
            print(f"   Total Samples: {final_results['total_count']:,}")
            if final_results['snr'] is not None:
                print(f"   SNR: {final_results['snr']:.4f}")
    
    # Detailed convergence check
    print(f"\n🔍 Convergence Check (Last 3 timesteps: {timesteps[-3:]}):")
    failed_tests = []
    
    for t in timesteps[-3:]:
        if t in results:
            if isinstance(results[t], dict) and 'charge' in results[t]:
                # Per-channel analysis
                for channel in ['charge', 'time']:
                    if channel in results[t]:
                        ch_results = results[t][channel]
                        is_normal = ch_results['is_normal']
                        ks_pval = ch_results['ks_test_pval']
                        shapiro_pval = ch_results['shapiro_test_pval']
                        
                        status = "✅ PASS" if is_normal else "❌ FAIL"
                        print(f"   t={t}, {channel}: {status}")
                        print(f"     KS p-value: {ks_pval:.6f} {'✅' if ks_pval > 0.05 else '❌'}")
                        print(f"     Shapiro p-value: {shapiro_pval:.6f} {'✅' if shapiro_pval > 0.05 else '❌'}")
                        
                        if not is_normal:
                            failed_tests.append(f"t={t}, {channel} (KS: {ks_pval:.6f}, Shapiro: {shapiro_pval:.6f})")
            else:
                # Overall analysis
                overall_results = results[t]
                is_normal = overall_results['is_normal']
                ks_pval = overall_results['ks_test_pval']
                shapiro_pval = overall_results['shapiro_test_pval']
                
                status = "✅ PASS" if is_normal else "❌ FAIL"
                print(f"   t={t}, overall: {status}")
                print(f"     KS p-value: {ks_pval:.6f} {'✅' if ks_pval > 0.05 else '❌'}")
                print(f"     Shapiro p-value: {shapiro_pval:.6f} {'✅' if shapiro_pval > 0.05 else '❌'}")
                
                if not is_normal:
                    failed_tests.append(f"t={t}, overall (KS: {ks_pval:.6f}, Shapiro: {shapiro_pval:.6f})")
    
    # Final convergence result
    all_normal = all(
        results[t].get('is_normal', False) if isinstance(results[t], dict) and 'charge' not in results[t]
        else all(results[t][ch]['is_normal'] for ch in ['charge', 'time'] if ch in results[t])
        for t in timesteps[-3:]  # Check last 3 timesteps
    )
    
    print(f"\n📊 Convergence Result:")
    if all_normal:
        print("✅ Forward diffusion successfully converges to Gaussian!")
        print("   All tests passed: KS p > 0.05 AND Shapiro p > 0.05")
    else:
        print("⚠️ Forward diffusion may not fully converge to Gaussian.")
        print("   Failed tests:")
        for failed_test in failed_tests:
            print(f"     • {failed_test}")
        print("   Consider: increasing timesteps, adjusting beta schedule, or checking data preprocessing.")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Optimized forward diffusion statistical analysis",
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
        default=None,
        help="Batch size for analysis (None for full dataset)"
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
        default="./forward_analysis_optimized",
        help="Output directory for analysis results"
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
    
    parser.add_argument(
        "--no-snr",
        action="store_true",
        help="Disable SNR calculation"
    )
    
    parser.add_argument(
        "--no-percentiles",
        action="store_true",
        help="Disable percentile calculation"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (None for all)"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup terminal output capture
    terminal_log_file = output_path / 'terminal_output.log'
    tee = TeeOutput(str(terminal_log_file))
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        # Quick mode
        if args.quick:
            T = 1000  # Default timesteps
            args.timesteps = [0, 1, T//2, T]
            print(f"🚀 Quick mode: timesteps = {args.timesteps}")
        
        print("\n" + "="*80)
        print("📊 Optimized Forward Diffusion Statistical Analysis")
        print("="*80)
        
        # Load full dataset and diffusion
        dataloader, diffusion, config, device, total_events = load_full_dataset_and_diffusion(
            args.config, args.batch_size
        )
        
        # Run optimized analysis
        start_time = time.time()
        results = analyze_forward_diffusion_optimized(
            dataloader,
            diffusion,
            timesteps_to_check=args.timesteps,
            per_channel=args.per_channel,
            compute_snr=not args.no_snr,
            compute_percentiles=not args.no_percentiles,
            save_dir=args.output_dir,
            max_samples=args.max_samples
        )
        end_time = time.time()
        
        print(f"\n⏱️ Analysis completed in {end_time - start_time:.2f} seconds")
        print(f"📁 All analysis results saved to: {output_path}")
        print("\n" + "="*80)
        print("✅ Optimized analysis complete!")
        print("="*80)
        
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()






