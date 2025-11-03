#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Grid - Grid Visualization for NPE and Time

This module provides grid visualization showing NPE and Time responses separately
in a side-by-side or grid layout for better comparison and analysis.

Usage:
    # As a library
    from utils.event_visualization.event_grid import show_event_grid
    fig, axes = show_event_grid("event.npz")
    
    # As a script
    python event_grid.py --npz-path event.npz --output event_grid.png
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def _find_project_root():
    """Find the project root directory by looking for .git or configs folder."""
    from pathlib import Path
    current = Path.cwd()
    
    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "configs").exists():
            return parent
    
    # Fallback to current directory
    return current


def show_event_grid(
    npz_path: Union[str, Path],
    detector_csv: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    grid_layout: str = "side_by_side",  # "side_by_side", "stacked", "separate"
    sphere_resolution: Tuple[int, int] = (40, 20),
    base_radius: float = 5.0,
    radius_scale: float = 0.2,
    figure_size: Tuple[int, int] = (24, 10),
    skip_nonfinite: bool = True,
    scatter_background: bool = True,
    show_detector_hull: bool = True,
    show: bool = False,
    title_prefix: str = "",
    **kwargs
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Show grid visualization of an event with separate NPE and Time plots.
    
    Creates side-by-side or stacked visualization showing PMT responses
    for NPE (size) and Time (color) separately for better analysis.
    
    Args:
        npz_path: Path to NPZ file with 'input' and 'label' keys
        detector_csv: Path to detector geometry CSV. If None, uses default.
        output_path: Path to save the plot (PNG/PDF/SVG). If None, doesn't save.
        grid_layout: Layout type - "side_by_side", "stacked", or "separate"
        sphere_resolution: Resolution for sphere rendering (u_steps, v_steps)
        base_radius: Base radius for PMT spheres
        radius_scale: Scaling factor for sphere radius based on NPE
        figure_size: Figure size for plots
        skip_nonfinite: Skip non-finite values in visualization
        scatter_background: Show background dots for all PMTs
        show_detector_hull: Show detector hull outline
        show: If True, display the plot with plt.show()
        title_prefix: Prefix for the plot title
        **kwargs: Additional arguments (for compatibility)
    
    Returns:
        (fig, axes): matplotlib Figure and list of Axes3D objects
        
    Example:
        >>> fig, axes = show_event_grid(
        ...     "outputs/samples/sample_0000.npz",
        ...     output_path="sample_grid.png",
        ...     grid_layout="side_by_side"
        ... )
    """
    # Set default detector CSV path
    if detector_csv is None:
        project_root = _find_project_root()
        detector_csv = project_root / "configs" / "detector_geometry.csv"
    
    # Load detector geometry
    try:
        df_geo = pd.read_csv(detector_csv)
        x = np.asarray(df_geo["x"], dtype=np.float32)
        y = np.asarray(df_geo["y"], dtype=np.float32)
        z = np.asarray(df_geo["z"], dtype=np.float32)
        L = len(x)
    except Exception as e:
        raise FileNotFoundError(f"Could not load detector geometry from {detector_csv}: {e}")

    # Load event NPZ
    try:
        with np.load(npz_path) as data:
            arr = data["input"]  # shape (2, L)
            label = data["label"]  # shape (6,)
    except Exception as e:
        raise ValueError(f"Could not load NPZ file {npz_path}: {e}")
    
    # Validate input shape
    if arr.shape != (2, L):
        raise ValueError(f"Input shape must be (2, {L}), got {arr.shape}")
    
    # Extract event data
    energy, zenith, azimuth, x_pos, y_pos, z_pos = label
    npe = arr[0, :].astype(np.float32)
    ftime = arr[1, :].astype(np.float32)

    # Sanitize firstTime: ±inf → 0
    ftime[np.isinf(ftime)] = 0.0

    # Create figure based on layout
    if grid_layout == "side_by_side":
        fig = plt.figure(figsize=figure_size)
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        axes = [ax1, ax2]
    elif grid_layout == "stacked":
        fig = plt.figure(figsize=(figure_size[0], figure_size[1] * 2))
        ax1 = fig.add_subplot(211, projection="3d")
        ax2 = fig.add_subplot(212, projection="3d")
        axes = [ax1, ax2]
    elif grid_layout == "separate":
        fig = plt.figure(figsize=figure_size)
        ax1 = fig.add_subplot(111, projection="3d")
        axes = [ax1]
    else:
        raise ValueError(f"Unknown grid_layout: {grid_layout}")

    # Create visualizations
    if grid_layout == "separate":
        # Combined view (default behavior)
        _create_combined_plot(axes[0], x, y, z, npe, ftime, sphere_resolution,
                            base_radius, radius_scale, skip_nonfinite,
                            scatter_background, show_detector_hull,
                            energy, zenith, azimuth, x_pos, y_pos, z_pos,
                            title_prefix + "Combined View")
    else:
        # NPE-only plot (left/top)
        _create_npe_plot(axes[0], x, y, z, npe, sphere_resolution,
                        base_radius, radius_scale, skip_nonfinite,
                        scatter_background, show_detector_hull,
                        energy, zenith, azimuth, x_pos, y_pos, z_pos,
                        title_prefix + "NPE Response")
        
        # Time-only plot (right/bottom)
        _create_time_plot(axes[1], x, y, z, ftime, sphere_resolution,
                         base_radius, radius_scale, skip_nonfinite,
                         scatter_background, show_detector_hull,
                         energy, zenith, azimuth, x_pos, y_pos, z_pos,
                         title_prefix + "Time Response")

    # Overall title
    fig.suptitle(f"Event Grid Visualization\n"
                f"Energy = {energy:.3f}, Zenith = {zenith:.3f}, Azimuth = {azimuth:.3f}\n"
                f"Position = ({x_pos:.2f}, {y_pos:.2f}, {z_pos:.2f})", 
                fontsize=16, y=0.95)
    
    # Save if requested
    if output_path:
        fig.savefig(output_path, transparent=True, bbox_inches="tight")
        print(f"Event grid visualization saved to {output_path}")
    
    if show:
        plt.show()
    
    return fig, axes


def _create_combined_plot(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray,
    npe: np.ndarray, ftime: np.ndarray,
    sphere_resolution: Tuple[int, int], base_radius: float, radius_scale: float,
    skip_nonfinite: bool, scatter_background: bool, show_detector_hull: bool,
    energy: float, zenith: float, azimuth: float, x_pos: float, y_pos: float, z_pos: float,
    title: str
):
    """Create combined NPE (size) and Time (color) visualization."""
    # Color scale for time
    nonzero_mask = (ftime != 0) & np.isfinite(ftime)
    if not nonzero_mask.any():
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(ftime[nonzero_mask]))
        vmax = float(np.max(ftime[nonzero_mask]))
        if vmin == vmax:
            vmax = vmin + 1.0
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps["jet"]
    
    # Detector hull
    if show_detector_hull:
        _draw_detector_hull(ax, x, y, z)
    
    # Background dots
    if scatter_background:
        ax.scatter(x, y, z, s=1, c="gray", alpha=0.5)
    
    # PMT spheres
    _draw_pmt_spheres(ax, x, y, z, npe, ftime, norm, cmap,
                     sphere_resolution, base_radius, radius_scale, skip_nonfinite)
    
    # Styling
    ax.set_title(title, fontsize=14)
    _style_axes(ax)
    
    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Time (ns)', rotation=270, labelpad=20)


def _create_npe_plot(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray,
    npe: np.ndarray, sphere_resolution: Tuple[int, int],
    base_radius: float, radius_scale: float, skip_nonfinite: bool,
    scatter_background: bool, show_detector_hull: bool,
    energy: float, zenith: float, azimuth: float, x_pos: float, y_pos: float, z_pos: float,
    title: str
):
    """Create NPE-only visualization (size-based)."""
    # Use uniform color for NPE visualization
    cmap = colormaps["viridis"]
    norm = Normalize(vmin=0, vmax=np.max(npe) if np.max(npe) > 0 else 1.0)
    
    # Detector hull
    if show_detector_hull:
        _draw_detector_hull(ax, x, y, z)
    
    # Background dots
    if scatter_background:
        ax.scatter(x, y, z, s=1, c="gray", alpha=0.5)
    
    # PMT spheres with NPE-based size and color
    _draw_pmt_spheres(ax, x, y, z, npe, npe, norm, cmap,
                     sphere_resolution, base_radius, radius_scale, skip_nonfinite)
    
    # Styling
    ax.set_title(title, fontsize=14)
    _style_axes(ax)
    
    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('NPE', rotation=270, labelpad=20)


def _create_time_plot(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray,
    ftime: np.ndarray, sphere_resolution: Tuple[int, int],
    base_radius: float, radius_scale: float, skip_nonfinite: bool,
    scatter_background: bool, show_detector_hull: bool,
    energy: float, zenith: float, azimuth: float, x_pos: float, y_pos: float, z_pos: float,
    title: str
):
    """Create Time-only visualization (color-based)."""
    # Color scale for time
    nonzero_mask = (ftime != 0) & np.isfinite(ftime)
    if not nonzero_mask.any():
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(ftime[nonzero_mask]))
        vmax = float(np.max(ftime[nonzero_mask]))
        if vmin == vmax:
            vmax = vmin + 1.0
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps["jet"]
    
    # Detector hull
    if show_detector_hull:
        _draw_detector_hull(ax, x, y, z)
    
    # Background dots
    if scatter_background:
        ax.scatter(x, y, z, s=1, c="gray", alpha=0.5)
    
    # PMT spheres with uniform size but time-based color
    uniform_npe = np.ones_like(ftime) * 10  # Uniform size
    _draw_pmt_spheres(ax, x, y, z, uniform_npe, ftime, norm, cmap,
                     sphere_resolution, base_radius, radius_scale, skip_nonfinite)
    
    # Styling
    ax.set_title(title, fontsize=14)
    _style_axes(ax)
    
    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Time (ns)', rotation=270, labelpad=20)


def _draw_detector_hull(ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Draw detector hull outline."""
    edge_string_idx = [1, 6, 50, 74, 73, 78, 75, 31]
    top_xy, bottom_xy = [], []
    
    for i in edge_string_idx:
        if (i - 1) * 60 < len(x):
            top_xy.append([x[(i - 1) * 60], y[(i - 1) * 60]])
        if (i - 1) * 60 + 59 < len(x):
            bottom_xy.append([x[(i - 1) * 60 + 59], y[(i - 1) * 60 + 59]])
    
    if top_xy and bottom_xy:
        top_xy = np.array(top_xy)
        bottom_xy = np.array(bottom_xy)
        
        ax.plot(top_xy[:, 0], top_xy[:, 1], z.max(), 'k-', linewidth=2, alpha=0.7)
        ax.plot(bottom_xy[:, 0], bottom_xy[:, 1], z.min(), 'k-', linewidth=2, alpha=0.7)


def _draw_pmt_spheres(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray,
    npe: np.ndarray, ftime: np.ndarray, norm: Normalize, cmap,
    sphere_resolution: Tuple[int, int], base_radius: float,
    radius_scale: float, skip_nonfinite: bool
):
    """Draw PMT spheres with size based on NPE and color based on time."""
    u_steps, v_steps = sphere_resolution
    
    # Create sphere coordinates
    u = np.linspace(0, 2 * np.pi, u_steps)
    v = np.linspace(0, np.pi, v_steps)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Base sphere
    x_sphere = np.cos(u_grid) * np.sin(v_grid)
    y_sphere = np.sin(u_grid) * np.sin(v_grid)
    z_sphere = np.cos(v_grid)
    
    # Draw spheres for each PMT
    for i in range(len(x)):
        if skip_nonfinite and not np.isfinite(ftime[i]):
            continue
            
        if npe[i] <= 0:
            continue
            
        # Calculate radius based on NPE
        radius = base_radius + radius_scale * npe[i]
        
        # Get color based on time
        color = cmap(norm(ftime[i]))
        
        # Scale and translate sphere
        sphere_x = x[i] + radius * x_sphere
        sphere_y = y[i] + radius * y_sphere
        sphere_z = z[i] + radius * z_sphere
        
        # Draw sphere
        ax.plot_surface(sphere_x, sphere_y, sphere_z, 
                       color=color, alpha=0.8, linewidth=0)


def _style_axes(ax: plt.Axes):
    """Style the 3D axes."""
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.grid(True, alpha=0.3)


def main():
    """Command line interface for event grid visualization."""
    parser = argparse.ArgumentParser(
        description="Create grid visualization of IceCube neutrino events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--npz-path", "-n",
        type=str,
        required=True,
        help="Path to NPZ file containing event data"
    )
    
    parser.add_argument(
        "--detector-csv", "-d",
        type=str,
        default=None,
        help="Path to detector geometry CSV file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for the visualization (PNG/PDF/SVG)"
    )
    
    parser.add_argument(
        "--grid-layout",
        type=str,
        choices=["side_by_side", "stacked", "separate"],
        default="side_by_side",
        help="Grid layout type"
    )
    
    parser.add_argument(
        "--sphere-resolution",
        type=int,
        nargs=2,
        default=[40, 20],
        metavar=("U_STEPS", "V_STEPS"),
        help="Sphere rendering resolution"
    )
    
    parser.add_argument(
        "--base-radius",
        type=float,
        default=5.0,
        help="Base radius for PMT spheres"
    )
    
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=0.2,
        help="Scaling factor for sphere radius based on NPE"
    )
    
    parser.add_argument(
        "--figure-size",
        type=int,
        nargs=2,
        default=[24, 10],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches"
    )
    
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Don't show background PMT dots"
    )
    
    parser.add_argument(
        "--no-hull",
        action="store_true",
        help="Don't show detector hull outline"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively"
    )
    
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="",
        help="Prefix for the plot title"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.npz_path).exists():
        print(f"Error: NPZ file not found: {args.npz_path}")
        sys.exit(1)
    
    try:
        # Create visualization
        fig, axes = show_event_grid(
            npz_path=args.npz_path,
            detector_csv=args.detector_csv,
            output_path=args.output,
            grid_layout=args.grid_layout,
            sphere_resolution=tuple(args.sphere_resolution),
            base_radius=args.base_radius,
            radius_scale=args.radius_scale,
            figure_size=tuple(args.figure_size),
            scatter_background=not args.no_background,
            show_detector_hull=not args.no_hull,
            show=args.show,
            title_prefix=args.title_prefix
        )
        
        print("✅ Event grid visualization completed successfully!")
        
    except Exception as e:
        print(f"❌ Error creating grid visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
