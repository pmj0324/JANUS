#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Forward Diffusion Statistical Analysis
==========================================

A simplified, faster version of forward diffusion analysis.
Processes data in smaller batches with clear progress reporting.

Author: Minje Park
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import load_config_from_file
from dataloader.pmt_dataloader import make_dataloader
from models.factory import ModelFactory


def analyze_forward_diffusion_fast(
    diffusion,
    dataloader,
    timesteps_to_check: List[int] = [0, 1, 250, 500, 750, 1000],
    max_samples: Optional[int] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Fast forward diffusion analysis with clear progress reporting.
    
    Args:
        diffusion: Diffusion model
        dataloader: Data loader
        timesteps_to_check: List of timesteps to analyze
        max_samples: Maximum number of samples to process (None for all)
        device: Device to use
        
    Returns:
        Dictionary containing analysis results
    """
    print("="*80)
    print("📊 Fast Forward Diffusion Analysis")
    print("="*80)
    
    T = diffusion.cfg.timesteps
    print(f"Note: t=0 is x0 (original data), t=1 is x1 (first noise step), t={T} is xT (final timestep, completely noisy)")
    print(f"Timesteps to analyze: {timesteps_to_check}")
    
    # Initialize results storage
    results = {}
    for t in timesteps_to_check:
        results[f't_{t}'] = {
            'charge_data': [],
            'time_data': [],
            'snr_values': []
        }
    
    # Process data
    total_samples = 0
    batch_count = 0
    total_batches = len(dataloader)
    
    print(f"\n🚀 Starting analysis...")
    print(f"📊 Total batches: {total_batches}")
    print(f"📊 Batch size: {dataloader.batch_size}")
    print(f"📊 Estimated total samples: {total_batches * dataloader.batch_size:,}")
    
    start_time = time.time()
    
    for batch_idx, (x_sig, geom, labels, _) in enumerate(dataloader):
        if max_samples and total_samples >= max_samples:
            print(f"\n⏹️  Stopping at {total_samples:,} samples (max_samples limit)")
            break
            
        # Move to device
        x_sig = x_sig.to(device)
        N = x_sig.size(0)
        total_samples += N
        batch_count += 1
        
        # Progress update every 5 batches or at the end
        if batch_count % 5 == 0 or batch_count == total_batches:
            elapsed = time.time() - start_time
            progress = (batch_count / total_batches) * 100
            samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
            print(f"📈 Progress: {batch_count}/{total_batches} batches ({progress:.1f}%) - {total_samples:,} samples - {samples_per_sec:.0f} samples/sec")
        
        # Process each timestep
        for t_idx in timesteps_to_check:
            if t_idx == 0:
                # Original data (no noise)
                x_t = x_sig
            else:
                # Add noise
                t_batch = torch.full((N,), t_idx, device=device, dtype=torch.long)
                x_t = diffusion.q_sample(x_sig, t_batch)
            
            # Extract charge and time data
            charge_data = x_t[:, 0, :].cpu().numpy().flatten()  # (N*L,)
            time_data = x_t[:, 1, :].cpu().numpy().flatten()    # (N*L,)
            
            # Store data
            results[f't_{t_idx}']['charge_data'].extend(charge_data)
            results[f't_{t_idx}']['time_data'].extend(time_data)
            
            # Calculate SNR for this batch
            if t_idx > 0:
                # Calculate noise added
                noise = x_t - x_sig
                signal_power = torch.mean(x_sig**2)
                noise_power = torch.mean(noise**2)
                snr = signal_power / (noise_power + 1e-8)
                results[f't_{t_idx}']['snr_values'].append(snr.item())
    
    # Convert lists to numpy arrays and compute statistics
    print(f"\n📊 Computing statistics...")
    final_results = {}
    
    for t in timesteps_to_check:
        key = f't_{t}'
        charge_data = np.array(results[key]['charge_data'])
        time_data = np.array(results[key]['time_data'])
        
        print(f"  t={t} (x{t}): {len(charge_data):,} samples")
        
        # Basic statistics
        charge_stats = {
            'mean': np.mean(charge_data),
            'std': np.std(charge_data),
            'min': np.min(charge_data),
            'max': np.max(charge_data),
            'median': np.median(charge_data),
            'q25': np.percentile(charge_data, 25),
            'q75': np.percentile(charge_data, 75)
        }
        
        time_stats = {
            'mean': np.mean(time_data),
            'std': np.std(time_data),
            'min': np.min(time_data),
            'max': np.max(time_data),
            'median': np.median(time_data),
            'q25': np.percentile(time_data, 25),
            'q75': np.percentile(time_data, 75)
        }
        
        # SNR statistics
        snr_values = results[key]['snr_values']
        snr_stats = {
            'mean': np.mean(snr_values) if snr_values else 0,
            'std': np.std(snr_values) if snr_values else 0,
            'min': np.min(snr_values) if snr_values else 0,
            'max': np.max(snr_values) if snr_values else 0
        }
        
        final_results[t] = {
            'charge_stats': charge_stats,
            'time_stats': time_stats,
            'snr_stats': snr_stats,
            'sample_count': len(charge_data)
        }
    
    # Print summary
    print(f"\n📊 ANALYSIS SUMMARY:")
    print("="*80)
    print(f"Total samples processed: {total_samples:,}")
    print(f"Total time: {time.time() - start_time:.1f} seconds")
    print(f"Processing rate: {total_samples / (time.time() - start_time):.0f} samples/sec")
    
    print(f"\n📈 TIMESTEP STATISTICS:")
    for t in timesteps_to_check:
        result = final_results[t]
        print(f"\n  t={t} (x{t}):")
        print(f"    Charge: μ={result['charge_stats']['mean']:.4f}, σ={result['charge_stats']['std']:.4f}")
        print(f"    Time:   μ={result['time_stats']['mean']:.4f}, σ={result['time_stats']['std']:.4f}")
        if result['snr_stats']['mean'] > 0:
            print(f"    SNR:    μ={result['snr_stats']['mean']:.4f}, σ={result['snr_stats']['std']:.4f}")
        print(f"    Samples: {result['sample_count']:,}")
    
    return final_results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast forward diffusion statistical analysis")
    parser.add_argument("-c", "--config", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (if not specified, uses config default)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[0, 1, 250, 500, 750, 1000],
                       help="Timesteps to analyze")
    parser.add_argument("--output-dir", type=str, default="./fast_analysis_output",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load config
    print(f"📂 Loading configuration from: {args.config}")
    config = load_config_from_file(args.config)
    
    # Override batch size if specified
    if args.batch_size:
        config.data.batch_size = args.batch_size
        print(f"📊 Using batch size: {args.batch_size}")
    
    # Create dataloader
    print(f"📂 Creating dataloader...")
    dataloader = make_dataloader(
        config.data.h5_path,
        batch_size=config.data.batch_size,
        shuffle=False,  # Don't shuffle for consistent results
        num_workers=2,  # Reduced workers for stability
        pin_memory=True,
        replace_time_inf_with=config.data.replace_time_inf_with,
        channel_first=config.data.channel_first,
        time_transform=config.model.time_transform,
        affine_offsets=tuple(config.model.affine_offsets),
        affine_scales=tuple(config.model.affine_scales),
        label_offsets=tuple(config.model.label_offsets),
        label_scales=tuple(config.model.label_scales)
    )
    
    # Create diffusion model
    print(f"🧠 Creating diffusion model...")
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model, config.diffusion
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = diffusion.to(device)
    
    print(f"🔧 Device: {device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Run analysis
    print(f"\n🚀 Starting fast analysis...")
    results = analyze_forward_diffusion_fast(
        diffusion=diffusion,
        dataloader=dataloader,
        timesteps_to_check=args.timesteps,
        max_samples=args.max_samples,
        device=device
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary_file = output_dir / "analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Fast Forward Diffusion Analysis Summary\n")
        f.write("="*50 + "\n\n")
        
        for t in args.timesteps:
            result = results[t]
            f.write(f"t={t} (x{t}):\n")
            f.write(f"  Charge: μ={result['charge_stats']['mean']:.6f}, σ={result['charge_stats']['std']:.6f}\n")
            f.write(f"  Time:   μ={result['time_stats']['mean']:.6f}, σ={result['time_stats']['std']:.6f}\n")
            if result['snr_stats']['mean'] > 0:
                f.write(f"  SNR:    μ={result['snr_stats']['mean']:.6f}, σ={result['snr_stats']['std']:.6f}\n")
            f.write(f"  Samples: {result['sample_count']:,}\n\n")
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📄 Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()





