#!/usr/bin/env python3
"""
forward_visualize.py

Single Event Forward Diffusion Visualization
============================================

Visualize how a single event progresses through forward diffusion.
Focus on both statistics and 3D visualization to see how the event
appears at different noise levels.

Usage:
    python diffusion/forward_visualize.py \
        --config configs/default.yaml \
        --event-index 0 \
        --timesteps 0 250 500 750 999 \
        --save-images

Features:
    - Per-timestep statistics (mean, std, range)
    - 3D visualization at each timestep
    - NPZ files for each timestep
    - Quick mode for fast testing
"""

from __future__ import annotations
import argparse
import sys
import os
import time
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config_from_file
from dataloader.pmt_dataloader import PMTSignalsH5
from models.factory import ModelFactory
from utils.event_visualization.event_fast import plot_event_fast
import matplotlib.pyplot as plt


def load_event_and_diffusion(config_path: str, event_index: int):
    """Load event from dataloader and create diffusion for forward process."""
    print(f"\n📂 Loading configuration from: {config_path}")
    config = load_config_from_file(config_path)
    
    # Extract normalization parameters
    affine_offsets = list(config.data.affine_offsets)
    affine_scales = list(config.data.affine_scales)
    label_offsets = list(config.data.label_offsets)
    label_scales = list(config.data.label_scales)
    
    # Create dataset
    print(f"📂 Loading dataset from: {config.data.h5_path}")
    dataset = PMTSignalsH5(
        h5_path=config.data.h5_path,
        replace_time_inf_with=0.0,
        channel_first=True,
        time_transform=config.data.time_transform,
        affine_offsets=tuple(affine_offsets),
        affine_scales=tuple(affine_scales),
        label_offsets=tuple(label_offsets),
        label_scales=tuple(label_scales),
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} events total")
    
    # Check if index is valid
    if event_index < 0 or event_index >= len(dataset):
        print(f"❌ Invalid event index: {event_index} (dataset size: {len(dataset)})")
        sys.exit(1)
    
    # Load event
    sample = dataset[event_index]
    x_sig = sample[0].unsqueeze(0)    # (1, 2, 5160)
    geom = sample[1].unsqueeze(0)     # (1, 3, 5160)
    labels = sample[2].unsqueeze(0)   # (1, 6)
    
    print(f"✅ Event {event_index} loaded:")
    print(f"   Signal shape: {x_sig.shape}")
    print(f"   Geometry shape: {geom.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    # Create diffusion (we only need it for forward process)
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model, config.diffusion
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = diffusion.to(device)
    
    return x_sig, geom, labels, diffusion, config, device


def denormalize_signal(
    x_sig_norm: np.ndarray,
    charge_offset: float,
    charge_scale: float,
    time_offset: float,
    time_scale: float,
    time_transform: str,
) -> np.ndarray:
    """Denormalize signal to original scale."""
    result = x_sig_norm.copy()
    
    # Step 1: Reverse affine normalization
    result[0] = (result[0] * charge_scale) + charge_offset
    result[1] = (result[1] * time_scale) + time_offset
    
    # Step 2: Reverse log transformation for time
    if time_transform == 'ln':
        result[1] = np.exp(result[1]) - 1.0
    elif time_transform == 'log10':
        result[1] = np.power(10.0, result[1]) - 1.0
    
    # Clamp to prevent overflow
    result[0] = np.clip(result[0], 0.0, 1e10)
    result[1] = np.clip(result[1], 0.0, 1e10)
    
    return result


def denormalize_labels(
    labels_norm: np.ndarray,
    label_offsets: np.ndarray,
    label_scales: np.ndarray,
) -> np.ndarray:
    """Denormalize labels to original scale."""
    return (labels_norm * label_scales) + label_offsets


def plot_histograms(
    charge_data: np.ndarray,
    time_data: np.ndarray,
    output_path: Path,
    t_val: int,
    title_suffix: str = ""
):
    """Create histogram plots for charge and time data."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Charge histogram
    charge_valid = charge_data[charge_data > 0]
    if len(charge_valid) > 0:
        ax1.hist(charge_valid, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('NPE')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Charge Distribution t={t_val}{title_suffix}')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_charge = np.mean(charge_valid)
        std_charge = np.std(charge_valid)
        ax1.axvline(mean_charge, color='red', linestyle='--', label=f'Mean: {mean_charge:.3f}')
        ax1.axvline(mean_charge + std_charge, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_charge:.3f}')
        ax1.axvline(mean_charge - std_charge, color='orange', linestyle='--', alpha=0.7)
        ax1.legend()
    
    # Time histogram
    time_valid = time_data[(time_data > 0) & np.isfinite(time_data)]
    if len(time_valid) > 0:
        ax2.hist(time_valid, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Time Distribution t={t_val}{title_suffix}')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_time = np.mean(time_valid)
        std_time = np.std(time_valid)
        ax2.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.1f}')
        ax2.axvline(mean_time + std_time, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_time:.1f}')
        ax2.axvline(mean_time - std_time, color='orange', linestyle='--', alpha=0.7)
        ax2.legend()
    
    plt.tight_layout()
    
    # Save histogram
    hist_path = output_path / f"histogram_t{t_val}{title_suffix.lower().replace(' ', '_')}.png"
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"   📊 Histogram saved: {hist_path}")
    return hist_path


def visualize_forward_process(
    x_sig: torch.Tensor,
    geom: torch.Tensor,
    labels: torch.Tensor,
    diffusion,
    config,
    device: torch.device,
    timesteps: list,
    save_images: bool = False,
    save_npz: bool = False,
    save_histograms: bool = False,
    show_normalized: bool = False,
    show_denormalized: bool = True,
    output_dir: str = "./forward_visualization",
    detector_csv: str = "./configs/detector_geometry.csv",
):
    """Visualize forward diffusion process for a single event."""
    print(f"\n🎨 Visualizing forward diffusion process...")
    print(f"   Timesteps: {timesteps}")
    print(f"   Save images: {save_images}")
    print(f"   Save NPZ: {save_npz}")
    print(f"   Save histograms: {save_histograms}")
    print(f"   Show normalized: {show_normalized}")
    print(f"   Show denormalized: {show_denormalized}")
    
    # Start overall timing
    overall_start = time.perf_counter()
    
    # Move to device
    x_sig = x_sig.to(device)
    
    # Create output directory
    if save_images:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"   Output directory: {output_path}")
    
    # Denormalize original labels for display
    labels_np = labels.cpu().numpy()[0]  # Store normalized labels for plotting
    labels_denorm = denormalize_labels(
        labels_np,
        label_offsets=np.array(config.data.label_offsets),
        label_scales=np.array(config.data.label_scales),
    )
    
    # Print event information
    label_names = ['energy', 'zenith', 'azimuth', 'x', 'y', 'z']
    print(f"\n📊 Event Information:")
    for i, name in enumerate(label_names):
        print(f"   {name}: {labels_denorm[i]:.3f}")
    
    # Statistics table header
    print(f"\n{'='*80}")
    print(f"📈 Forward Diffusion Statistics")
    print(f"{'='*80}")
    print(f"Note: t=0 is original data, t=1 is first noise step, t={diffusion.cfg.timesteps} is final timestep")
    print(f"{'Timestep':<10} {'Charge Range':<30} {'Time Range':<30} {'SNR':<10}")
    print(f"{'-'*80}")
    
    # Process each timestep
    total_forward_time = 0
    total_denorm_time = 0
    total_npz_time = 0
    total_viz_time = 0
    
    for t_val in timesteps:
        timestep_start = time.perf_counter()
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        
        # Forward diffusion
        forward_start = time.perf_counter()
        x_t = diffusion.q_sample(x_sig, t)
        forward_time = time.perf_counter() - forward_start
        total_forward_time += forward_time
        
        # Statistics
        charge = x_t[0, 0].cpu().numpy()  # (5160,)
        time_vals = x_t[0, 1].cpu().numpy()  # (5160,)
        
        charge_range = f"[{charge.min():.3f}, {charge.max():.3f}]"
        time_range = f"[{time_vals.min():.3f}, {time_vals.max():.3f}]"
        
        # SNR calculation (use t-1 as index for t > 0)
        if t_val == 0:
            snr = float('inf')  # Original data, no noise
        else:
            idx = t_val - 1
            sqrt_alpha_bar = diffusion.sqrt_alphas_cumprod[idx].item()
            sqrt_one_minus = diffusion.sqrt_one_minus_alphas_cumprod[idx].item()
            snr = sqrt_alpha_bar / sqrt_one_minus if sqrt_one_minus > 0 else float('inf')
        
        print(f"{t_val:<10} {charge_range:<30} {time_range:<30} {snr:<10.4f}")
        
        # Save NPZ and images if requested
        if save_images:
            # Step 1: Prepare normalized data (for normalized plots/histograms)
            x_t_np = x_t[0].cpu().numpy()  # (2, 5160)
            
            # Step 2: Denormalize for visualization
            denorm_start = time.perf_counter()
            x_t_denorm = denormalize_signal(
                x_t_np,
                charge_offset=config.data.affine_offsets[0],
                charge_scale=config.data.affine_scales[0],
                time_offset=config.data.affine_offsets[1],
                time_scale=config.data.affine_scales[1],
                time_transform=config.data.time_transform,
            )
            denorm_time = time.perf_counter() - denorm_start
            total_denorm_time += denorm_time
            
            # Step 2: Save NPZ (if requested)
            npz_time = 0
            if save_npz:
                npz_start = time.perf_counter()
                npz_path = output_path / f"forward_t{t_val}.npz"
                np.savez(
                    npz_path,
                    input=x_t_denorm,
                    label=labels_denorm,
                    info=labels_denorm,
                )
                npz_time = time.perf_counter() - npz_start
                total_npz_time += npz_time
                print(f"   💾 NPZ saved: {npz_path}")
            
            # Step 3: Create visualizations
            viz_time = 0
            if save_images:
                try:
                    viz_start = time.perf_counter()
                    
                    # Load geometry from CSV
                    geometry = np.loadtxt(detector_csv, delimiter=',', skiprows=1, usecols=(1,2,3))
                    
                    # Denormalized plots (default)
                    if show_denormalized:
                        png_path = output_path / f"forward_t{t_val}_denorm.png"
                        plot_event_fast(
                            charge_data=x_t_denorm[0],
                            time_data=x_t_denorm[1],
                            geometry=geometry,
                            labels=labels_denorm,
                            output_path=str(png_path),
                            plot_type="both",
                            figure_size=(16, 8),
                            show_detector_hull=True,
                            show_background=True,
                            sphere_size=2.0,
                            alpha=0.8
                        )
                        print(f"   🎨 Denormalized plot saved: {png_path}")
                    
                    # Normalized plots (if requested)
                    if show_normalized:
                        png_path = output_path / f"forward_t{t_val}_norm.png"
                        plot_event_fast(
                            charge_data=x_t_np[0],
                            time_data=x_t_np[1],
                            geometry=geometry,
                            labels=labels_np,
                            output_path=str(png_path),
                            plot_type="both",
                            figure_size=(16, 8),
                            show_detector_hull=True,
                            show_background=True,
                            sphere_size=2.0,
                            alpha=0.8
                        )
                        print(f"   🎨 Normalized plot saved: {png_path}")
                    
                    viz_time = time.perf_counter() - viz_start
                    total_viz_time += viz_time
                    
                except Exception as e:
                    print(f"   ⚠️  t={t_val}: Visualization failed: {e}")
            
            # Step 4: Create histograms (if requested)
            hist_time = 0
            if save_histograms:
                try:
                    hist_start = time.perf_counter()
                    
                    # Denormalized histograms (default)
                    if show_denormalized:
                        plot_histograms(
                            x_t_denorm[0], x_t_denorm[1], 
                            output_path, t_val, " (Denormalized)"
                        )
                    
                    # Normalized histograms (if requested)
                    if show_normalized:
                        plot_histograms(
                            x_t_np[0], x_t_np[1], 
                            output_path, t_val, " (Normalized)"
                        )
                    
                    hist_time = time.perf_counter() - hist_start
                    
                except Exception as e:
                    print(f"   ⚠️  t={t_val}: Histogram failed: {e}")
            
            # Detailed timing breakdown
            timestep_total = forward_time + denorm_time + npz_time + viz_time + hist_time
            print(f"   ✅ t={t_val}: Forward={forward_time*1000:.1f}ms, "
                  f"Denorm={denorm_time*1000:.1f}ms, "
                  f"NPZ={npz_time*1000:.1f}ms, "
                  f"Viz={viz_time*1000:.1f}ms, "
                  f"Hist={hist_time*1000:.1f}ms, "
                  f"Total={timestep_total*1000:.1f}ms")
        
        timestep_total = time.perf_counter() - timestep_start
        print(f"   📊 t={t_val} total time: {timestep_total*1000:.1f}ms")
    
    # Overall statistics
    print(f"\n{'='*80}")
    print(f"📊 Overall Statistics")
    print(f"{'='*80}")
    
    # Original statistics
    orig_charge = x_sig[0, 0].cpu().numpy()
    orig_time = x_sig[0, 1].cpu().numpy()
    
    nonzero_orig_charge = orig_charge[orig_charge > 0]
    nonzero_orig_time = orig_time[orig_charge > 0]
    
    print(f"Original Event:")
    print(f"   Charge: {len(nonzero_orig_charge)}/{len(orig_charge)} non-zero ({100*len(nonzero_orig_charge)/len(orig_charge):.1f}%)")
    if len(nonzero_orig_charge) > 0:
        print(f"      Mean: {nonzero_orig_charge.mean():.2f}, Std: {nonzero_orig_charge.std():.2f}")
    
    if len(nonzero_orig_time) > 0:
        finite_time = nonzero_orig_time[np.isfinite(nonzero_orig_time)]
        if len(finite_time) > 0:
            print(f"   Time: Mean: {finite_time.mean():.2f}, Std: {finite_time.std():.2f}")
    
    # End overall timing
    overall_time = time.perf_counter() - overall_start
    
    # Print detailed timing summary
    print(f"\n{'='*80}")
    print(f"⏱️  Detailed Timing Summary")
    print(f"{'='*80}")
    print(f"Forward diffusion: {total_forward_time*1000:.1f}ms ({total_forward_time/overall_time*100:.1f}%)")
    if save_images:
        print(f"Denormalization:  {total_denorm_time*1000:.1f}ms ({total_denorm_time/overall_time*100:.1f}%)")
        print(f"NPZ saving:       {total_npz_time*1000:.1f}ms ({total_npz_time/overall_time*100:.1f}%)")
        print(f"Visualization:    {total_viz_time*1000:.1f}ms ({total_viz_time/overall_time*100:.1f}%)")
        print(f"Other overhead:   {(overall_time-total_forward_time-total_denorm_time-total_npz_time-total_viz_time)*1000:.1f}ms")
    print(f"Total time:       {overall_time*1000:.1f}ms (100.0%)")
    
    print(f"\n✅ Forward diffusion visualization complete!")
    if save_images:
        print(f"📁 Files saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize forward diffusion process for a single event",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "-e", "--event-index",
        type=int,
        default=0,
        help="Event index to visualize"
    )
    
    parser.add_argument(
        "-t", "--timesteps",
        type=int,
        nargs="+",
        default=[0, 250, 500, 750, 1000],
        help="Timesteps to visualize"
    )
    
    parser.add_argument(
        "-i", "--save-images",
        action="store_true",
        help="Save 3D visualization images"
    )
    
    parser.add_argument(
        "-n", "--save-npz",
        action="store_true",
        help="Save NPZ files for each timestep (default: False)"
    )
    
    parser.add_argument(
        "-g", "--save-histograms",
        action="store_true",
        help="Save histogram plots for each timestep"
    )
    
    parser.add_argument(
        "--show-normalized",
        action="store_true",
        help="Show normalized data plots"
    )
    
    parser.add_argument(
        "--show-denormalized",
        action="store_true",
        default=True,
        help="Show denormalized data plots (default: True)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./forward_visualization",
        help="Output directory for saved files"
    )
    
    parser.add_argument(
        "-d", "--detector-csv",
        type=str,
        default="./configs/detector_geometry.csv",
        help="Path to detector geometry CSV file"
    )
    
    parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Quick mode: only t=0 and t=T-1 (final timestep)"
    )
    
    args = parser.parse_args()
    
    # Quick mode: show t=0 (original), t=1 (first noise), and t=T (final)
    if args.quick:
        T = 1000  # Default timesteps
        args.timesteps = [0, 1, T]
        print(f"🚀 Quick mode: timesteps = {args.timesteps}")
    
    print("\n" + "="*80)
    print("🎨 Single Event Forward Diffusion Visualization")
    print("="*80)
    
    # Load event and diffusion
    x_sig, geom, labels, diffusion, config, device = load_event_and_diffusion(
        args.config, args.event_index
    )
    
    # Visualize forward process
    visualize_forward_process(
        x_sig=x_sig,
        geom=geom,
        labels=labels,
        diffusion=diffusion,
        config=config,
        device=device,
        timesteps=args.timesteps,
        save_images=args.save_images,
        save_npz=args.save_npz,
        save_histograms=args.save_histograms,
        show_normalized=args.show_normalized,
        show_denormalized=args.show_denormalized,
        output_dir=args.output_dir,
        detector_csv=args.detector_csv,
    )
    
    print("\n" + "="*80)
    print("✅ Visualization complete!")
    print("="*80)


if __name__ == "__main__":
    main()
