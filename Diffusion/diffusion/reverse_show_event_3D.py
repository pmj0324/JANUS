#!/usr/bin/env python3
"""
reverse_compare.py

Reverse Diffusion Comparison Tool
=================================

Compare reverse diffusion generation with real data.
Focus on visual comparison of detector images at different timesteps
during the reverse process.

Usage:
    python diffusion/reverse_compare.py \
        --pth-path outputs/best_model.pth \
        --config configs/default.yaml \
        --event-index 0 \
        --timesteps 0 100 500 999 \
        --compare-real

Features:
    - Generate sample using trained model
    - Visualize reverse process at specified timesteps
    - Optional comparison with real data at same timesteps
    - NPZ and 3D visualization for each timestep
    - Performance timing
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
from utils.event_visualization.event_fast import plot_event_fast, plot_event_comparison_fast


def load_model_and_event(config_path: str, pth_path: str, event_index: int):
    """Load trained model and event from dataloader."""
    print(f"\n📂 Loading configuration from: {config_path}")
    config = load_config_from_file(config_path)
    
    # Load model
    print(f"🧠 Loading model from: {pth_path}")
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model, config.diffusion
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(pth_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    
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
    
    # Check if index is valid
    if event_index < 0 or event_index >= len(dataset):
        print(f"❌ Invalid event index: {event_index} (dataset size: {len(dataset)})")
        sys.exit(1)
    
    # Load event
    sample = dataset[event_index]
    x_sig_orig = sample[0].unsqueeze(0)    # (1, 2, 5160)
    geom = sample[1].unsqueeze(0)          # (1, 3, 5160)
    labels = sample[2].unsqueeze(0)        # (1, 6)
    
    print(f"✅ Event {event_index} loaded:")
    print(f"   Signal shape: {x_sig_orig.shape}")
    print(f"   Geometry shape: {geom.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    return x_sig_orig, geom, labels, model, diffusion, config, device


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


def generate_with_timesteps(
    model,
    diffusion,
    labels: torch.Tensor,
    geom: torch.Tensor,
    shape: tuple,
    timesteps: list,
    device: torch.device,
) -> dict:
    """Generate sample with intermediate timesteps saved."""
    print(f"\n🎨 Generating sample with timesteps: {timesteps}")
    
    # Move inputs to device
    labels = labels.to(device)
    geom = geom.to(device)
    
    # Generate using DDIM with specified timesteps
    # We'll use a custom sampling function that saves intermediate steps
    T = diffusion.cfg.timesteps
    ddim_steps = max(timesteps) + 1  # Ensure we go to max timestep
    
    print(f"   Using DDIM with {ddim_steps} steps")
    
    # Generate sample with return_all_timesteps=True
    generated_samples = diffusion.sample(
        labels,
        geom,
        shape=shape,
        return_all_timesteps=True,
        ddim_steps=ddim_steps,
    )
    
    # Extract requested timesteps
    results = {}
    for t_val in timesteps:
        if t_val < len(generated_samples):
            results[t_val] = generated_samples[t_val]
        else:
            print(f"   ⚠️  Timestep {t_val} not available (max: {len(generated_samples)-1})")
    
    return results


def apply_forward_to_timestep(
    x_sig_orig: torch.Tensor,
    diffusion,
    target_t: int,
    device: torch.device,
) -> torch.Tensor:
    """Apply forward diffusion to original signal up to target timestep."""
    t = torch.full((x_sig_orig.shape[0],), target_t, device=device, dtype=torch.long)
    return diffusion.q_sample(x_sig_orig, t)


def compare_reverse_process(
    x_sig_orig: torch.Tensor,
    geom: torch.Tensor,
    labels: torch.Tensor,
    model,
    diffusion,
    config,
    device: torch.device,
    timesteps: list,
    compare_real: bool = False,
    output_dir: str = "./reverse_comparison",
    detector_csv: str = "./configs/detector_geometry.csv",
):
    """Compare reverse diffusion process with optional real data comparison."""
    print(f"\n🔄 Comparing reverse diffusion process...")
    print(f"   Timesteps: {timesteps}")
    print(f"   Compare with real: {compare_real}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"   Output directory: {output_path}")
    
    # Denormalize labels for display
    labels_denorm = denormalize_labels(
        labels.cpu().numpy()[0],
        label_offsets=np.array(config.data.label_offsets),
        label_scales=np.array(config.data.label_scales),
    )
    
    # Print event information
    label_names = ['energy', 'zenith', 'azimuth', 'x', 'y', 'z']
    print(f"\n📊 Event Information:")
    for i, name in enumerate(label_names):
        print(f"   {name}: {labels_denorm[i]:.3f}")
    
    # Generate sample with intermediate timesteps
    start_time = time.perf_counter()
    generated_samples = generate_with_timesteps(
        model=model,
        diffusion=diffusion,
        labels=labels,
        geom=geom,
        shape=x_sig_orig.shape,
        timesteps=timesteps,
        device=device,
    )
    generation_time = time.perf_counter() - start_time
    print(f"✅ Sample generation complete in {generation_time:.2f} seconds")
    
    # Process each timestep
    for t_val in timesteps:
        print(f"\n📊 Processing timestep {t_val}...")
        
        # Generated sample at this timestep
        if t_val in generated_samples:
            generated = generated_samples[t_val][0].cpu().numpy()  # (2, 5160)
            
            # Denormalize generated sample
            generated_denorm = denormalize_signal(
                generated,
                charge_offset=config.data.affine_offsets[0],
                charge_scale=config.data.affine_scales[0],
                time_offset=config.data.affine_offsets[1],
                time_scale=config.data.affine_scales[1],
                time_transform=config.data.time_transform,
            )
            
            # Save generated NPZ
            gen_npz_path = output_path / f"generated_t{t_val}.npz"
            np.savez(
                gen_npz_path,
                input=generated_denorm,
                label=labels_denorm,
                info=labels_denorm,
            )
            
            # Create generated visualization
            gen_png_path = output_path / f"generated_t{t_val}.png"
            try:
                viz_start = time.perf_counter()
                
                # Load geometry from CSV
                geometry = np.loadtxt(detector_csv, delimiter=',', skiprows=1, usecols=(1,2,3))
                
                # Direct fast plotting
                plot_event_fast(
                    charge_data=generated_denorm[0],
                    time_data=generated_denorm[1],
                    geometry=geometry,
                    labels=labels_denorm,
                    output_path=str(gen_png_path),
                    plot_type="both",
                    figure_size=(16, 8),
                    show_detector_hull=True,
                    show_background=True,
                    sphere_size=2.0,
                    alpha=0.8
                )
                
                viz_time = time.perf_counter() - viz_start
                print(f"   ✅ Generated t={t_val}: Viz={viz_time*1000:.1f}ms")
            except Exception as e:
                print(f"   ⚠️  Generated t={t_val}: Visualization failed: {e}")
        
        # Real data at this timestep (if requested)
        if compare_real:
            real_at_t = apply_forward_to_timestep(x_sig_orig, diffusion, t_val, device)
            real_denorm = denormalize_signal(
                real_at_t[0].cpu().numpy(),
                charge_offset=config.data.affine_offsets[0],
                charge_scale=config.data.affine_scales[0],
                time_offset=config.data.affine_offsets[1],
                time_scale=config.data.affine_scales[1],
                time_transform=config.data.time_transform,
            )
            
            # Save real NPZ
            real_npz_path = output_path / f"real_t{t_val}.npz"
            np.savez(
                real_npz_path,
                input=real_denorm,
                label=labels_denorm,
                info=labels_denorm,
            )
            
            # Create real visualization
            real_png_path = output_path / f"real_t{t_val}.png"
            try:
                viz_start = time.perf_counter()
                
                # Direct fast plotting
                plot_event_fast(
                    charge_data=real_denorm[0],
                    time_data=real_denorm[1],
                    geometry=geometry,
                    labels=labels_denorm,
                    output_path=str(real_png_path),
                    plot_type="both",
                    figure_size=(16, 8),
                    show_detector_hull=True,
                    show_background=True,
                    sphere_size=2.0,
                    alpha=0.8
                )
                
                viz_time = time.perf_counter() - viz_start
                print(f"   ✅ Real t={t_val}: Viz={viz_time*1000:.1f}ms")
            except Exception as e:
                print(f"   ⚠️  Real t={t_val}: Visualization failed: {e}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"📊 Summary Statistics")
    print(f"{'='*80}")
    
    # Original event statistics
    orig_denorm = denormalize_signal(
        x_sig_orig[0].cpu().numpy(),
        charge_offset=config.data.affine_offsets[0],
        charge_scale=config.data.affine_scales[0],
        time_offset=config.data.affine_offsets[1],
        time_scale=config.data.affine_scales[1],
        time_transform=config.data.time_transform,
    )
    
    orig_charge = orig_denorm[0]
    orig_time = orig_denorm[1]
    
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
    
    print(f"\n✅ Reverse comparison complete!")
    print(f"📁 Files saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare reverse diffusion generation with real data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--pth-path",
        type=str,
        required=True,
        help="Path to trained model .pth file"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "--event-index",
        type=int,
        default=0,
        help="Event index to use from dataset"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[0, 100, 500, 999],
        help="Timesteps to visualize during reverse process"
    )
    
    parser.add_argument(
        "--compare-real",
        action="store_true",
        help="Also show real data at same timesteps"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reverse_comparison",
        help="Output directory for saved files"
    )
    
    parser.add_argument(
        "--detector-csv",
        type=str,
        default="./configs/detector_geometry.csv",
        help="Path to detector geometry CSV file"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only t=0 and t=T-1"
    )
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        T = 1000  # Default timesteps
        args.timesteps = [0, T-1]
        print(f"🚀 Quick mode: timesteps = {args.timesteps}")
    
    print("\n" + "="*80)
    print("🔄 Reverse Diffusion Comparison Tool")
    print("="*80)
    
    # Load model and event
    x_sig_orig, geom, labels, model, diffusion, config, device = load_model_and_event(
        args.config, args.pth_path, args.event_index
    )
    
    # Compare reverse process
    compare_reverse_process(
        x_sig_orig=x_sig_orig,
        geom=geom,
        labels=labels,
        model=model,
        diffusion=diffusion,
        config=config,
        device=device,
        timesteps=args.timesteps,
        compare_real=args.compare_real,
        output_dir=args.output_dir,
        detector_csv=args.detector_csv,
    )
    
    print("\n" + "="*80)
    print("✅ Comparison complete!")
    print("="*80)


if __name__ == "__main__":
    main()
