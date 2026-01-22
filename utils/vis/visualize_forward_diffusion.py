#!/usr/bin/env python3
"""
Forward Diffusion Visualization
================================

Visualize forward diffusion process with:
- 3D plots showing event geometry with firstTime and nPE
- Histograms for firstTime and nPE at each timestep
- Support for multiple noise schedules
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colormaps

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from diffusion.schedules import get_noise_schedule, compute_alpha_schedule
from diffusion.forward import apply_forward_diffusion
from utils.vis.event_show import show_event_dual_plot
from utils.normalize import normalize, denormalize_minmax


def plot_histograms(
    npe_data: np.ndarray,
    ftime_data: np.ndarray,
    output_path: Path,
    t_val: int,
    schedule_name: str = "",
    title_suffix: str = ""
):
    """
    Create histogram plots for nPE and firstTime.
    
    X-axis range is set to be centered at 0 for better visualization of Gaussian distribution at final timestep.
    Data remains in [0, 1] range (no transformation).
    
    Args:
        npe_data: NPE values (L,) - normalized [0, 1]
        ftime_data: firstTime values (L,) - normalized [0, 1]
        output_path: Output directory path
        t_val: Timestep value
        schedule_name: Name of the noise schedule
        title_suffix: Additional title suffix
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # NPE histogram - use data as is [0, 1]
    npe_valid = npe_data[np.isfinite(npe_data)]
    if len(npe_valid) > 0:
        # Calculate x-axis range to fit all data with padding, centered at 0
        data_min = npe_valid.min()
        data_max = npe_valid.max()
        # Find the maximum absolute distance from 0 to ensure symmetric range
        max_dist_from_zero = max(abs(data_min), abs(data_max))
        # Add 15% padding for better visibility
        x_range = max_dist_from_zero * 1.15
        # Ensure minimum range for visibility (at least show [0, 1] range)
        x_range = max(x_range, 0.5)
        
        ax1.hist(npe_valid, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('NPE (normalized)')
        ax1.set_ylabel('Count')
        title = f'NPE Distribution t={t_val}'
        if schedule_name:
            title += f' ({schedule_name})'
        if title_suffix:
            title += title_suffix
        ax1.set_title(title)
        ax1.set_yscale('linear')  # Use linear scale for y-axis
        ax1.set_xlim(-x_range, x_range)  # Set x-axis range centered at 0
        ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='0')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_npe = np.mean(npe_valid)
        std_npe = np.std(npe_valid)
        ax1.axvline(mean_npe, color='red', linestyle='--', label=f'Mean: {mean_npe:.3f}')
        ax1.axvline(mean_npe + std_npe, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_npe:.3f}')
        ax1.axvline(mean_npe - std_npe, color='orange', linestyle='--', alpha=0.7)
        ax1.legend()
    
    # firstTime histogram - use data as is [0, 1]
    ftime_valid = ftime_data[np.isfinite(ftime_data)]
    if len(ftime_valid) > 0:
        # Calculate x-axis range to fit all data with padding, centered at 0
        data_min = ftime_valid.min()
        data_max = ftime_valid.max()
        # Find the maximum absolute distance from 0 to ensure symmetric range
        max_dist_from_zero = max(abs(data_min), abs(data_max))
        # Add 15% padding for better visibility
        x_range = max_dist_from_zero * 1.15
        # Ensure minimum range for visibility (at least show [0, 1] range)
        x_range = max(x_range, 0.5)
        
        ax2.hist(ftime_valid, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('FirstTime (normalized)')
        ax2.set_ylabel('Count')
        title = f'FirstTime Distribution t={t_val}'
        if schedule_name:
            title += f' ({schedule_name})'
        if title_suffix:
            title += title_suffix
        ax2.set_title(title)
        ax2.set_yscale('linear')  # Use linear scale for y-axis
        ax2.set_xlim(-x_range, x_range)  # Set x-axis range centered at 0
        ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='0')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_ftime = np.mean(ftime_valid)
        std_ftime = np.std(ftime_valid)
        ax2.axvline(mean_ftime, color='red', linestyle='--', label=f'Mean: {mean_ftime:.3f}')
        ax2.axvline(mean_ftime + std_ftime, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_ftime:.3f}')
        ax2.axvline(mean_ftime - std_ftime, color='orange', linestyle='--', alpha=0.7)
        ax2.legend()
    
    plt.tight_layout()
    
    # Save histogram
    hist_filename = f"histogram_t{t_val}"
    if schedule_name:
        hist_filename += f"_{schedule_name}"
    hist_filename += title_suffix.lower().replace(' ', '_')
    hist_path = output_path / f"{hist_filename}.png"
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return hist_path


@normalize(
    channel_methods=['log_minmax', 'log_minmax'],
    arg_index=0,  # Normalize first argument (x0_sig)
    denormalize=False,  # We'll denormalize manually inside the function
    channel_stats=[
        {'log_min': 0.0, 'log_max': np.log1p(225.0)},      # npe: log(1+0)=0, log(1+225)
        {'log_min': 0.0, 'log_max': np.log1p(20676.0)}     # firstTime: log(1+0)=0, log(1+20676)
    ]
)
def visualize_forward_diffusion(
    x0_sig: torch.Tensor,
    geom: torch.Tensor,
    label: torch.Tensor,
    schedules: List[Tuple[str, dict]],
    timesteps: List[int],
    output_dir: str = "./forward_visualization",
    detector_csv: Optional[str] = None,
    save_3d: bool = True,
    save_histograms: bool = True,
    denormalize_fn: Optional[callable] = None,
):
    """
    Visualize forward diffusion process with multiple noise schedules.
    
    This function automatically normalizes input signals using fixed dataset-wide min/max with log_minmax:
    - npe channel (x0_sig[:, 0, :]): log_minmax normalization [0, 1] using fixed min=0, max=225
    - firstTime channel (x0_sig[:, 1, :]): log_minmax normalization [0, 1] using fixed min=0, max=20676
    
    Forward diffusion uses normalized data [0, 1], visualization denormalizes back to original scale.
    
    Args:
        x0_sig: Clean signals (B, 2, L) - automatically normalized by decorator
            Normalized using log_minmax: npe [0, 225] -> log(1+x) -> [0, 1], firstTime [0, 20676] -> log(1+x) -> [0, 1]
        geom: Geometry (B, 3, L)
        label: Labels (B, 6)
        schedules: List of (schedule_name, schedule_kwargs) tuples
            Example: [("linear", {}), ("cosine", {"s": 0.008})]
        timesteps: List of timesteps to visualize (e.g., [0, 100, 500, 999])
        output_dir: Output directory for saved files
        detector_csv: Path to detector geometry CSV
        save_3d: Whether to save 3D plots
        save_histograms: Whether to save histograms
        denormalize_fn: Optional function to denormalize signals (deprecated, auto-denormalize is used)
            Should take (sig_norm, ...) and return sig_raw
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = x0_sig.device
    B, C, L = x0_sig.shape
    assert C == 2, "Signal must have 2 channels (nPE, firstTime)"
    
    # x0_sig is already normalized by decorator to [0, 1]
    # Get normalization statistics for denormalization during visualization
    if hasattr(visualize_forward_diffusion, '_normalization_stats'):
        channel_stats = visualize_forward_diffusion._normalization_stats
        npe_stats = channel_stats[0]
        firstTime_stats = channel_stats[1]
        
        # Print normalization parameters used
        print(f"\nNormalization parameters:")
        print(f"  npe channel (log_minmax):")
        if 'log_min' in npe_stats and 'log_max' in npe_stats:
            print(f"    log_min: {npe_stats.get('log_min', 'N/A'):.6f}")
            print(f"    log_max: {npe_stats.get('log_max', 'N/A'):.6f}")
            print(f"    (original max: {np.expm1(npe_stats.get('log_max', 0)):.6f})")
        else:
            print(f"    min: {npe_stats.get('min', 'N/A'):.6f}")
            print(f"    max: {npe_stats.get('max', 'N/A'):.6f}")
        print(f"  firstTime channel (log_minmax):")
        if 'log_min' in firstTime_stats and 'log_max' in firstTime_stats:
            print(f"    log_min: {firstTime_stats.get('log_min', 'N/A'):.6f}")
            print(f"    log_max: {firstTime_stats.get('log_max', 'N/A'):.6f}")
            print(f"    (original max: {np.expm1(firstTime_stats.get('log_max', 0)):.6f})")
        else:
            print(f"    min: {firstTime_stats.get('min', 'N/A'):.6f}")
            print(f"    max: {firstTime_stats.get('max', 'N/A'):.6f}")
    else:
        npe_stats = None
        firstTime_stats = None
        print("\nWarning: Normalization statistics not available")
    
    # Extract single sample if batch
    if B > 1:
        print(f"Warning: Using first sample from batch of size {B}")
        x0_sig = x0_sig[0:1]
        geom = geom[0:1]
        label = label[0:1]
    
    # Convert to numpy for visualization
    geom_np = geom[0].cpu().numpy()  # (3, L)
    label_np = label[0].cpu().numpy()  # (6,)
    
    print(f"\n{'='*80}")
    print(f"Forward Diffusion Visualization")
    print(f"{'='*80}")
    print(f"Signal shape: {x0_sig.shape}")
    print(f"Geometry shape: {geom.shape}")
    print(f"Timesteps to visualize: {timesteps}")
    print(f"Schedules: {[s[0] for s in schedules]}")
    print(f"Output directory: {output_path}")
    
    # Process each schedule
    for schedule_name, schedule_kwargs in schedules:
        print(f"\n{'='*80}")
        print(f"Processing schedule: {schedule_name}")
        print(f"{'='*80}")
        
        # Get noise schedule
        timesteps_total = max(timesteps) + 1
        betas = get_noise_schedule(
            schedule_name=schedule_name,
            timesteps=timesteps_total,
            **schedule_kwargs
        )
        alpha_schedule = compute_alpha_schedule(betas)
        
        # Create schedule-specific output directory
        schedule_output = output_path / schedule_name
        schedule_output.mkdir(exist_ok=True)
        
        # Process each timestep
        for t_val in timesteps:
            print(f"\n  Timestep t={t_val}...")
            
            # Apply forward diffusion
            t_tensor = torch.full((1,), t_val, device=device, dtype=torch.long)
            x_t = apply_forward_diffusion(
                x0=x0_sig,
                betas=betas,
                timesteps=t_tensor,
            )
            
            # Convert to numpy
            x_t_np = x_t[0].cpu().numpy()  # (2, L) - normalized [0, 1]
            
            # Denormalize for visualization (back to original scale)
            # Forward diffusion uses normalized data, but visualization shows original scale
            if npe_stats is not None and firstTime_stats is not None:
                from utils.normalize import denormalize_log_minmax
                x_t_vis = x_t_np.copy()
                # Denormalize npe channel: [0, 1] -> log scale -> original scale
                x_t_vis[0, :] = denormalize_log_minmax(
                    torch.from_numpy(x_t_np[0, :]),
                    npe_stats.get('log_min', 0.0),
                    npe_stats.get('log_max', np.log1p(225.0)),
                    feature_range=(0, 1)
                ).numpy()
                # Denormalize firstTime channel: [0, 1] -> log scale -> original scale
                x_t_vis[1, :] = denormalize_log_minmax(
                    torch.from_numpy(x_t_np[1, :]),
                    firstTime_stats.get('log_min', 0.0),
                    firstTime_stats.get('log_max', np.log1p(20676.0)),
                    feature_range=(0, 1)
                ).numpy()
            else:
                # Fallback: use normalized values if stats not available
                x_t_vis = x_t_np.copy()
            
            # Extract nPE and firstTime (denormalized to original scale)
            npe = x_t_vis[0]  # (L,) - denormalized to original scale
            ftime = x_t_vis[1]  # (L,) - denormalized to original scale
            
            # Sanitize firstTime
            ftime = np.nan_to_num(ftime, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Create signal array for visualization (2, L)
            sig_vis = np.stack([npe, ftime], axis=0)
            
            # Save 3D plot
            if save_3d:
                try:
                    png_path = schedule_output / f"forward_t{t_val}_3d.png"
                    show_event_dual_plot(
                        sig=sig_vis,
                        geo=geom_np,
                        label=label_np,
                        output_path=str(png_path),
                        figure_size=(20, 10),
                        marker_size=20.0,
                        show_detector_hull=True,
                        show=False,
                        title_prefix=f"t={t_val} ({schedule_name}) - "
                    )
                    print(f"     3D plot saved: {png_path}")
                except Exception as e:
                    print(f"      3D plot failed: {e}")
            
            # Save histograms
            if save_histograms:
                try:
                    hist_path = plot_histograms(
                        npe_data=npe,
                        ftime_data=ftime,
                        output_path=schedule_output,
                        t_val=t_val,
                        schedule_name=schedule_name,
                    )
                    print(f"     Histogram saved: {hist_path}")
                except Exception as e:
                    print(f"      Histogram failed: {e}")
    
    print(f"\n{'='*80}")
    print(f" Visualization complete!")
    print(f" Files saved to: {output_path}")
    print(f"{'='*80}")
