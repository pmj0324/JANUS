#!/usr/bin/env python3
"""
event_dataloader_viz.py

Event visualization directly from dataloader with automatic denormalization.
Perfect for debugging training data, analyzing specific events, and data quality inspection.

Key Features:
- Direct integration with PMTSignalsH5 dataloader
- Automatic denormalization from config
- NPZ file generation for compatibility
- Fast 3D visualization
- Comprehensive event statistics
- CLI interface for easy usage

Usage:
    from utils.event_visualization.event_dataloader_viz import visualize_event_from_dataloader
    
    # Visualize specific event
    visualize_event_from_dataloader(
        config_path="configs/default.yaml",
        event_index=42,
        output_dir="outputs/event_viz"
    )
"""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from config import load_config_from_file
from dataloader.pmt_dataloader import PMTSignalsH5
from .event_fast import plot_event_fast


def denormalize_signal(
    x_sig_norm: np.ndarray,
    charge_offset: float,
    charge_scale: float,
    time_offset: float,
    time_scale: float,
    time_transform: str = "ln",
) -> np.ndarray:
    """
    Denormalize signal data back to original scale.
    
    Args:
        x_sig_norm: Normalized signal (2, L)
        charge_offset: Charge offset parameter
        charge_scale: Charge scale parameter
        time_offset: Time offset parameter
        time_scale: Time scale parameter
        time_transform: Time transformation type ("ln", "log10", None)
        
    Returns:
        Denormalized signal (2, L)
    """
    x_sig_denorm = x_sig_norm.copy()
    
    # Denormalize charge
    x_sig_denorm[0] = (x_sig_norm[0] * charge_scale) + charge_offset
    
    # Denormalize time
    time_denorm = (x_sig_norm[1] * time_scale) + time_offset
    
    # Apply inverse time transformation
    if time_transform == "ln":
        time_denorm = np.exp(time_denorm) - 1e-10
    elif time_transform == "log10":
        time_denorm = np.power(10.0, time_denorm) - 1e-10
    
    # Handle zeros (very negative values in log space)
    zero_mask = time_denorm < 0.1
    time_denorm[zero_mask] = 0.0
    
    x_sig_denorm[1] = time_denorm
    
    return x_sig_denorm


def denormalize_labels(
    labels_norm: np.ndarray,
    label_offsets: np.ndarray,
    label_scales: np.ndarray,
) -> np.ndarray:
    """
    Denormalize label data back to original scale.
    
    Args:
        labels_norm: Normalized labels (6,)
        label_offsets: Label offset parameters (6,)
        label_scales: Label scale parameters (6,)
        
    Returns:
        Denormalized labels (6,)
    """
    return (labels_norm * label_scales) + label_offsets


def visualize_event_from_dataloader(
    config_path: str,
    event_index: int,
    output_dir: str = "./outputs/event_viz",
    detector_csv: str = "./configs/detector_geometry.csv",
    save_npz: bool = True,
    save_png: bool = True,
    plot_type: str = "both",
    figure_size: tuple = (16, 8),
    sphere_size: float = 2.0,
    alpha: float = 0.8,
    show_stats: bool = True,
) -> dict:
    """
    Visualize event directly from dataloader with automatic denormalization.
    
    Args:
        config_path: YAML configuration file path
        event_index: Index of event to visualize
        output_dir: Output directory for files
        detector_csv: Detector geometry CSV file path
        save_npz: Whether to save NPZ file
        save_png: Whether to save PNG visualization
        plot_type: Type of plot ("npe", "time", "both")
        figure_size: Figure size tuple
        sphere_size: Sphere size for visualization
        alpha: Transparency for visualization
        show_stats: Whether to print event statistics
        
    Returns:
        dict: Event data and file paths
        
    Example:
        >>> result = visualize_event_from_dataloader(
        ...     config_path="configs/default.yaml",
        ...     event_index=42,
        ...     output_dir="outputs/event_viz"
        ... )
        >>> print(f"Event info: {result['labels']}")
    """
    print("\n" + "="*80)
    print("📊 Event Visualization from DataLoader")
    print("="*80)
    
    # Load configuration
    print(f"\n📂 Loading configuration from: {config_path}")
    config = load_config_from_file(config_path)
    
    # Extract normalization parameters from config
    # affine_offsets/scales: [charge, time, x, y, z]
    # label_offsets/scales: [Energy, Zenith, Azimuth, X, Y, Z]
    affine_offsets = list(config.data.affine_offsets)
    affine_scales = list(config.data.affine_scales)
    label_offsets = list(config.data.label_offsets)
    label_scales = list(config.data.label_scales)
    
    # Create dataset
    print(f"📂 Loading dataset from: {config.data.h5_path}")
    print(f"   Time transform: {config.data.time_transform}")
    
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
        raise ValueError(f"Invalid event index: {event_index} (dataset size: {len(dataset)})")
    
    # Load event from dataset
    print(f"\n🔍 Loading event at index: {event_index}")
    x_sig, geom, labels, real_idx = dataset[event_index]
    
    # Convert to numpy
    x_sig_norm = x_sig.numpy()    # (2, 5160) - normalized
    geom_norm = geom.numpy()      # (3, 5160) - normalized geometry
    labels_norm = labels.numpy()  # (6,) - normalized labels
    
    print(f"   Real HDF5 index: {real_idx}")
    print(f"   Signal shape: {x_sig_norm.shape}")
    print(f"   Geometry shape: {geom_norm.shape}")
    print(f"   Labels shape: {labels_norm.shape}")
    
    # Denormalize signal data
    print(f"\n🔄 Denormalizing data...")
    print(f"   Charge: offset={affine_offsets[0]}, scale={affine_scales[0]}")
    print(f"   Time: offset={affine_offsets[1]}, scale={affine_scales[1]}, transform={config.data.time_transform}")
    
    x_sig_denorm = denormalize_signal(
        x_sig_norm,
        charge_offset=affine_offsets[0],
        charge_scale=affine_scales[0],
        time_offset=affine_offsets[1],
        time_scale=affine_scales[1],
        time_transform=config.data.time_transform,
    )
    
    # Denormalize labels
    labels_denorm = denormalize_labels(
        labels_norm,
        label_offsets=np.array(label_offsets, dtype=np.float32),
        label_scales=np.array(label_scales, dtype=np.float32),
    )
    
    # Print event information
    label_names = ['energy', 'zenith', 'azimuth', 'x', 'y', 'z']
    print(f"\n📊 Event Information:")
    for i, name in enumerate(label_names):
        print(f"   {name}: {labels_denorm[i]:.3f}")
    
    # Statistics
    if show_stats:
        print(f"\n📈 Signal Statistics (denormalized):")
        charge = x_sig_denorm[0]
        time = x_sig_denorm[1]
        
        nonzero_charge = charge[charge > 0]
        nonzero_time = time[charge > 0]  # Time where charge exists
        
        print(f"   Charge (NPE):")
        print(f"      Non-zero count: {len(nonzero_charge)} / {len(charge)} ({100*len(nonzero_charge)/len(charge):.1f}%)")
        if len(nonzero_charge) > 0:
            print(f"      Mean: {nonzero_charge.mean():.2f}")
            print(f"      Std: {nonzero_charge.std():.2f}")
            print(f"      Min/Max: {nonzero_charge.min():.2f} / {nonzero_charge.max():.2f}")
        
        print(f"   Time (ns) where charge > 0:")
        if len(nonzero_time) > 0:
            finite_time = nonzero_time[np.isfinite(nonzero_time)]
            if len(finite_time) > 0:
                print(f"      Count: {len(finite_time)}")
                print(f"      Mean: {finite_time.mean():.2f}")
                print(f"      Std: {finite_time.std():.2f}")
                print(f"      Min/Max: {finite_time.min():.2f} / {finite_time.max():.2f}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result = {
        'event_index': event_index,
        'real_h5_index': real_idx,
        'signal_normalized': x_sig_norm,
        'signal_denormalized': x_sig_denorm,
        'geometry_normalized': geom_norm,
        'labels_normalized': labels_norm,
        'labels_denormalized': labels_denorm,
        'config_path': config_path,
        'output_dir': str(output_path)
    }
    
    # Save as NPZ
    if save_npz:
        npz_path = output_path / f"event_{event_index}.npz"
        print(f"\n💾 Saving NPZ file to: {npz_path}")
        
        np.savez(
            npz_path,
            input=x_sig_denorm,  # (2, 5160)
            label=labels_denorm,  # (6,)
            info=labels_denorm,   # Same as label for compatibility
        )
        result['npz_path'] = str(npz_path)
    
    # Create fast 3D visualization
    if save_png:
        png_path = output_path / f"event_{event_index}.png"
        print(f"🎨 Creating fast 3D visualization...")
        
        try:
            # Load geometry from CSV
            geometry = np.loadtxt(detector_csv, delimiter=',', skiprows=1, usecols=(1,2,3))
            
            # Direct fast plotting
            fig, axes = plot_event_fast(
                charge_data=x_sig_denorm[0],
                time_data=x_sig_denorm[1],
                geometry=geometry,
                labels=labels_denorm,
                output_path=str(png_path),
                plot_type=plot_type,
                figure_size=figure_size,
                show_detector_hull=True,
                show_background=True,
                sphere_size=sphere_size,
                alpha=alpha
            )
            print(f"✅ Fast 3D visualization saved to: {png_path}")
            result['png_path'] = str(png_path)
            
        except Exception as e:
            print(f"⚠️  Fast 3D visualization failed: {e}")
            result['png_error'] = str(e)
    
    print(f"\n✅ Event visualization completed!")
    print(f"   Event index: {event_index}")
    print(f"   Output directory: {output_path}")
    
    return result


def main():
    """CLI for event visualization from dataloader."""
    parser = argparse.ArgumentParser(
        description="Visualize events directly from dataloader with automatic denormalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize event 42
  python event_dataloader_viz.py --config configs/default.yaml --index 42

  # Visualize with custom output directory
  python event_dataloader_viz.py --config configs/default.yaml --index 100 --output outputs/my_events

  # NPE only visualization
  python event_dataloader_viz.py --config configs/default.yaml --index 42 --plot-type npe

  # Time only visualization
  python event_dataloader_viz.py --config configs/default.yaml --index 42 --plot-type time
        """
    )
    
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--index", type=int, required=True, help="Event index to visualize")
    parser.add_argument("--output", default="./outputs/event_viz", help="Output directory")
    parser.add_argument("--detector-csv", default="./configs/detector_geometry.csv", help="Detector geometry CSV file")
    parser.add_argument("--plot-type", choices=["npe", "time", "both"], default="both", help="Type of plot to generate")
    parser.add_argument("--figure-size", nargs=2, type=int, default=[16, 8], help="Figure size (width height)")
    parser.add_argument("--sphere-size", type=float, default=2.0, help="Sphere size")
    parser.add_argument("--alpha", type=float, default=0.8, help="Transparency (0-1)")
    parser.add_argument("--no-npz", action="store_true", help="Don't save NPZ file")
    parser.add_argument("--no-png", action="store_true", help="Don't save PNG visualization")
    parser.add_argument("--no-stats", action="store_true", help="Don't print event statistics")
    
    args = parser.parse_args()
    
    try:
        result = visualize_event_from_dataloader(
            config_path=args.config,
            event_index=args.index,
            output_dir=args.output,
            detector_csv=args.detector_csv,
            save_npz=not args.no_npz,
            save_png=not args.no_png,
            plot_type=args.plot_type,
            figure_size=tuple(args.figure_size),
            sphere_size=args.sphere_size,
            alpha=args.alpha,
            show_stats=not args.no_stats
        )
        
        print("\n🎉 Visualization completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
