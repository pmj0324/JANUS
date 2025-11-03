#!/usr/bin/env python3
"""
event_fast.py

Ultra-fast 3D event visualization optimized for speed.
Direct plotting without NPZ files, designed for training loops and real-time visualization.

Key Features:
- Ultra-fast rendering optimized for speed
- Direct array input (no NPZ files needed)
- Side-by-side NPE and time plots
- Background PMT visualization
- Detector hull outline
- Configurable sphere sizes and transparency

Usage:
    from utils.event_visualization.event_fast import plot_event_fast, plot_event_comparison_fast
    
    # Single event visualization
    plot_event_fast(
        charge_data,      # (5160,) charge values
        time_data,        # (5160,) time values  
        geometry,         # (5160, 3) x,y,z coordinates
        labels,           # (6,) event labels
        output_path,      # output PNG path
        plot_type="both"  # "npe", "time", or "both"
    )
    
    # Comparison visualization
    plot_event_comparison_fast(
        real_data, generated_data, output_path
    )
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
import time
from typing import Optional, Tuple, Union
from pathlib import Path


def plot_event_fast(
    charge_data: np.ndarray,
    time_data: np.ndarray,
    geometry: np.ndarray,
    labels: np.ndarray,
    output_path: Optional[str] = None,
    plot_type: str = "both",
    figure_size: Tuple[int, int] = (20, 8),
    show_detector_hull: bool = True,
    show_background: bool = True,
    sphere_size: float = 10.0,
    alpha: float = 0.8,
) -> Tuple[plt.Figure, Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]]:
    """
    Ultra-fast 3D event visualization.
    
    Args:
        charge_data: (5160,) charge values
        time_data: (5160,) time values
        geometry: (5160, 3) x,y,z coordinates
        labels: (6,) event labels [energy, zenith, azimuth, x, y, z]
        output_path: output PNG path (None to not save)
        plot_type: "npe", "time", or "both"
        figure_size: figure size tuple
        show_detector_hull: show detector outline
        show_background: show background PMTs
        sphere_size: size of spheres
        alpha: transparency
        
    Returns:
        (fig, axes): matplotlib Figure and Axes objects
        
    Example:
        >>> plot_event_fast(
        ...     charge_data=event_charges,
        ...     time_data=event_times,
        ...     geometry=detector_geometry,
        ...     labels=event_labels,
        ...     output_path="fast_event.png",
        ...     plot_type="both"
        ... )
    """
    start_time = time.perf_counter()
    
    # Extract labels
    energy, zenith, azimuth, x_pos, y_pos, z_pos = labels
    
    # Sanitize time data
    time_data = np.where(np.isinf(time_data), 0.0, time_data)
    
    print(f"🎨 Creating fast 3D plot...")
    print(f"   Plot type: {plot_type}")
    print(f"   PMTs: {len(charge_data)}")
    print(f"   Energy: {energy:.3f}, Zenith: {zenith:.3f}, Azimuth: {azimuth:.3f}")
    
    if plot_type == "both":
        # Create side-by-side plots
        fig = plt.figure(figsize=figure_size)
        
        # Common title
        title = f"Energy={energy:.3f}, Zenith={zenith:.3f}, Azimuth={azimuth:.3f}\nX={x_pos:.2f}, Y={y_pos:.2f}, Z={z_pos:.2f}"
        fig.suptitle(title, fontsize=12, y=0.95)
        
        # NPE plot (left)
        ax1 = fig.add_subplot(121, projection="3d")
        _plot_single_channel_fast(
            ax1, charge_data, time_data, geometry, "npe",
            show_detector_hull, show_background, sphere_size, alpha
        )
        
        # Time plot (right)
        ax2 = fig.add_subplot(122, projection="3d")
        _plot_single_channel_fast(
            ax2, charge_data, time_data, geometry, "time",
            show_detector_hull, show_background, sphere_size, alpha
        )
        
        axes = (ax1, ax2)
        
    else:
        # Single plot
        fig = plt.figure(figsize=(figure_size[0]//2, figure_size[1]))
        
        title = f"{plot_type.upper()} Distribution\nEnergy={energy:.3f}, Zenith={zenith:.3f}, Azimuth={azimuth:.3f}"
        fig.suptitle(title, fontsize=12, y=0.95)
        
        ax = fig.add_subplot(111, projection="3d")
        _plot_single_channel_fast(
            ax, charge_data, time_data, geometry, plot_type,
            show_detector_hull, show_background, sphere_size, alpha
        )
        
        axes = ax
    
    # Save if requested
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"📁 Saved: {output_path}")
    
    end_time = time.perf_counter()
    print(f"✅ Fast 3D plot completed in {end_time - start_time:.3f} seconds")
    
    return fig, axes


def plot_event_comparison_fast(
    real_data: dict,
    generated_data: dict,
    output_path: Optional[str] = None,
    max_events: int = 4,
    figure_size: Tuple[int, int] = (20, 16),
    show_detector_hull: bool = True,
    show_background: bool = True,
    sphere_size: float = 8.0,
    alpha: float = 0.7,
) -> plt.Figure:
    """
    Fast comparison visualization between real and generated events.
    
    Args:
        real_data: dict with keys 'charge', 'time', 'geometry', 'labels'
        generated_data: dict with keys 'charge', 'time', 'geometry', 'labels'
        output_path: output PNG path (None to not save)
        max_events: maximum number of events to compare
        figure_size: figure size tuple
        show_detector_hull: show detector outline
        show_background: show background PMTs
        sphere_size: size of spheres
        alpha: transparency
        
    Returns:
        fig: matplotlib Figure object
        
    Example:
        >>> plot_event_comparison_fast(
        ...     real_data={'charge': real_charges, 'time': real_times, ...},
        ...     generated_data={'charge': gen_charges, 'time': gen_times, ...},
        ...     output_path="comparison.png"
        ... )
    """
    start_time = time.perf_counter()
    
    batch_size = min(len(real_data['charge']), len(generated_data['charge']), max_events)
    
    print(f"🎨 Creating fast comparison plot...")
    print(f"   Events: {batch_size}")
    
    fig = plt.figure(figsize=figure_size)
    fig.suptitle("Real vs Generated Events Comparison", fontsize=16, y=0.98)
    
    for i in range(batch_size):
        # Real event
        ax_real = fig.add_subplot(batch_size, 2, 2*i + 1, projection="3d")
        _plot_single_channel_fast(
            ax_real, 
            real_data['charge'][i], 
            real_data['time'][i], 
            real_data['geometry'][i],
            "both",
            show_detector_hull, show_background, sphere_size, alpha
        )
        ax_real.set_title(f"Real Event {i+1}", fontsize=10)
        
        # Generated event
        ax_gen = fig.add_subplot(batch_size, 2, 2*i + 2, projection="3d")
        _plot_single_channel_fast(
            ax_gen, 
            generated_data['charge'][i], 
            generated_data['time'][i], 
            generated_data['geometry'][i],
            "both",
            show_detector_hull, show_background, sphere_size, alpha
        )
        ax_gen.set_title(f"Generated Event {i+1}", fontsize=10)
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"📁 Saved comparison: {output_path}")
    
    end_time = time.perf_counter()
    print(f"✅ Fast comparison plot completed in {end_time - start_time:.3f} seconds")
    
    return fig


def _plot_single_channel_fast(
    ax, charge_data, time_data, geometry, channel_type,
    show_detector_hull, show_background, sphere_size, alpha
):
    """Plot single channel (NPE or time) on given axis - optimized version."""
    x, y, z = geometry[:, 0], geometry[:, 1], geometry[:, 2]
    
    # Detector hull
    if show_detector_hull:
        _draw_detector_hull_fast(ax, x, y, z)
    
    # Background PMTs
    if show_background:
        ax.scatter(x, y, z, s=0.5, c="lightgray", alpha=0.3)
    
    if channel_type == "npe":
        # NPE plot: charge as color
        valid_mask = charge_data > 0
        if not valid_mask.any():
            print("⚠️  No valid NPE data")
            return
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        charge_valid = charge_data[valid_mask]
        
        # Color mapping
        norm = Normalize(vmin=0, vmax=np.max(charge_valid))
        cmap = colormaps["viridis"]
        colors = cmap(norm(charge_valid))
        
        # Scatter plot with varying sizes
        sizes = sphere_size * (1 + charge_valid / np.max(charge_valid))
        ax.scatter(x_valid, y_valid, z_valid, c=colors, s=sizes, alpha=alpha)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("NPE", fontsize=8)
        
        ax.set_title("NPE Distribution", fontsize=10)
        
    elif channel_type == "time":
        # Time plot: time as color
        valid_mask = (time_data > 0) & np.isfinite(time_data)
        if not valid_mask.any():
            print("⚠️  No valid time data")
            return
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        time_valid = time_data[valid_mask]
        
        # Color mapping
        norm = Normalize(vmin=np.min(time_valid), vmax=np.max(time_valid))
        cmap = colormaps["plasma"]
        colors = cmap(norm(time_valid))
        
        # Scatter plot
        ax.scatter(x_valid, y_valid, z_valid, c=colors, s=sphere_size, alpha=alpha)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Time (ns)", fontsize=8)
        
        ax.set_title("Time Distribution", fontsize=10)
        
    elif channel_type == "both":
        # Combined plot: NPE as size, time as color
        valid_mask = (charge_data > 0) & (time_data > 0) & np.isfinite(time_data)
        if not valid_mask.any():
            print("⚠️  No valid combined data")
            return
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        charge_valid = charge_data[valid_mask]
        time_valid = time_data[valid_mask]
        
        # Size mapping (NPE)
        size_norm = Normalize(vmin=0, vmax=np.max(charge_valid))
        sizes = sphere_size * (1 + size_norm(charge_valid))
        
        # Color mapping (time)
        color_norm = Normalize(vmin=np.min(time_valid), vmax=np.max(time_valid))
        cmap = colormaps["plasma"]
        colors = cmap(color_norm(time_valid))
        
        # Scatter plot
        ax.scatter(x_valid, y_valid, z_valid, c=colors, s=sizes, alpha=alpha)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Time (ns)", fontsize=8)
        
        ax.set_title("NPE (size) + Time (color)", fontsize=10)
    
    # Style axes
    _style_axes_fast(ax)


def _draw_detector_hull_fast(ax, x, y, z):
    """Draw detector hull outline - optimized version."""
    # Simplified hull for speed
    try:
        # Find boundary points
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)
        
        # Draw bounding box
        corners = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ]
        
        # Draw edges
        edges = [
            [0,1], [1,2], [2,3], [3,0],  # bottom
            [4,5], [5,6], [6,7], [7,4],  # top
            [0,4], [1,5], [2,6], [3,7]   # vertical
        ]
        
        for edge in edges:
            p1, p2 = corners[edge[0]], corners[edge[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                   'k-', alpha=0.5, linewidth=1)
    except Exception as e:
        print(f"⚠️  Hull drawing failed: {e}")


def _style_axes_fast(ax):
    """Style 3D axes - optimized version."""
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_zlabel("Z (m)", fontsize=8)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def main():
    """CLI for fast event visualization."""
    parser = argparse.ArgumentParser(
        description="Ultra-fast 3D event visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize single event
  python event_fast.py --charge data/charges.npy --time data/times.npy --geometry data/geometry.npy --labels data/labels.npy --output event.png

  # NPE only visualization
  python event_fast.py --charge data/charges.npy --time data/times.npy --geometry data/geometry.npy --labels data/labels.npy --plot-type npe --output npe.png

  # Time only visualization
  python event_fast.py --charge data/charges.npy --time data/times.npy --geometry data/geometry.npy --labels data/labels.npy --plot-type time --output time.png
        """
    )
    
    parser.add_argument("--charge", required=True, help="Path to charge data (5160,) numpy array")
    parser.add_argument("--time", required=True, help="Path to time data (5160,) numpy array")
    parser.add_argument("--geometry", required=True, help="Path to geometry data (5160, 3) numpy array")
    parser.add_argument("--labels", required=True, help="Path to labels data (6,) numpy array")
    parser.add_argument("--output", help="Output PNG path")
    parser.add_argument("--plot-type", choices=["npe", "time", "both"], default="both", help="Type of plot to generate")
    parser.add_argument("--figure-size", nargs=2, type=int, default=[20, 8], help="Figure size (width height)")
    parser.add_argument("--sphere-size", type=float, default=10.0, help="Sphere size")
    parser.add_argument("--alpha", type=float, default=0.8, help="Transparency (0-1)")
    parser.add_argument("--no-hull", action="store_true", help="Don't show detector hull")
    parser.add_argument("--no-background", action="store_true", help="Don't show background PMTs")
    
    args = parser.parse_args()
    
    try:
        # Load data
        print("📁 Loading data...")
        charge_data = np.load(args.charge)
        time_data = np.load(args.time)
        geometry = np.load(args.geometry)
        labels = np.load(args.labels)
        
        print(f"✅ Loaded data shapes:")
        print(f"   Charge: {charge_data.shape}")
        print(f"   Time: {time_data.shape}")
        print(f"   Geometry: {geometry.shape}")
        print(f"   Labels: {labels.shape}")
        
        # Create visualization
        fig, axes = plot_event_fast(
            charge_data=charge_data,
            time_data=time_data,
            geometry=geometry,
            labels=labels,
            output_path=args.output,
            plot_type=args.plot_type,
            figure_size=tuple(args.figure_size),
            show_detector_hull=not args.no_hull,
            show_background=not args.no_background,
            sphere_size=args.sphere_size,
            alpha=args.alpha
        )
        
        print("✅ Visualization completed!")
        
        # Show if no output specified
        if not args.output:
            plt.show()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
