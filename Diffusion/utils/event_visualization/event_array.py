#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Array - Direct Visualization from NumPy Arrays

This module provides direct visualization functionality for NumPy arrays
without requiring NPZ files. Perfect for training loops and real-time visualization.

Usage:
    # As a library
    from utils.event_visualization.event_array import show_event_from_array
    fig, ax = show_event_from_array(npe_array, time_array, geometry, labels)
    
    # As a script
    python event_array.py --npe-array npe.npy --time-array time.npy --geometry geometry.npy
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, List
import numpy as np
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


def show_event_from_array(
    npe_array: np.ndarray,
    time_array: np.ndarray,
    geometry: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    sphere_resolution: Tuple[int, int] = (40, 20),
    base_radius: float = 5.0,
    radius_scale: float = 0.2,
    figure_size: Tuple[int, int] = (15, 10),
    skip_nonfinite: bool = True,
    scatter_background: bool = True,
    show_detector_hull: bool = True,
    show: bool = False,
    title_prefix: str = "",
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Show 3D visualization directly from NumPy arrays.
    
    Perfect for training loops, real-time visualization, or when you have
    data in memory as arrays rather than saved files.
    
    Args:
        npe_array: NPE values array (L,) or (1, L)
        time_array: Time values array (L,) or (1, L)
        geometry: PMT geometry array (3, L) or (L, 3)
        labels: Event labels array (6,) or (1, 6) - [energy, zenith, azimuth, x, y, z]
        output_path: Path to save the plot (PNG/PDF/SVG). If None, doesn't save.
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
        (fig, ax): matplotlib Figure and Axes3D objects
        
    Example:
        >>> # From training data
        >>> fig, ax = show_event_from_array(
        ...     npe_data[0],  # (5160,)
        ...     time_data[0], # (5160,)
        ...     geometry,     # (3, 5160)
        ...     labels[0]     # (6,)
        ... )
    """
    # Normalize array shapes
    npe_array, time_array, geometry, labels = _normalize_array_shapes(
        npe_array, time_array, geometry, labels
    )
    
    # Extract coordinates
    x, y, z = geometry[0], geometry[1], geometry[2]
    L = len(x)
    
    # Validate array lengths
    if len(npe_array) != L or len(time_array) != L:
        raise ValueError(f"Array length mismatch: NPE={len(npe_array)}, "
                        f"Time={len(time_array)}, Geometry={L}")

    # Sanitize time array: ±inf → 0
    time_array = time_array.copy()
    time_array[np.isinf(time_array)] = 0.0

    # Color scale: exclude zeros for better visualization
    nonzero_mask = (time_array != 0) & np.isfinite(time_array)
    if not nonzero_mask.any():
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(time_array[nonzero_mask]))
        vmax = float(np.max(time_array[nonzero_mask]))
        if vmin == vmax:
            vmax = vmin + 1.0
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps["jet"]
    
    # Create figure
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection="3d")
    
    # Title
    if labels is not None:
        energy, zenith, azimuth, x_pos, y_pos, z_pos = labels
        title_line1 = f"{title_prefix}Energy = {energy:.3f}"
        title_line2 = f"Zenith = {zenith:.3f}, Azimuth = {azimuth:.3f}"
        title_line3 = f"Position = ({x_pos:.2f}, {y_pos:.2f}, {z_pos:.2f})"
        fig.suptitle(f"{title_line1}\n{title_line2}\n{title_line3}", fontsize=14, y=0.98)
    else:
        fig.suptitle(f"{title_prefix}Event Visualization from Arrays", fontsize=14, y=0.98)
    
    # Detector hull (optional)
    if show_detector_hull:
        _draw_detector_hull(ax, x, y, z)
    
    # Background dots
    if scatter_background:
        ax.scatter(x, y, z, s=1, c="gray", alpha=0.5)
    
    # PMT spheres
    _draw_pmt_spheres(ax, x, y, z, npe_array, time_array, norm, cmap, 
                     sphere_resolution, base_radius, radius_scale, skip_nonfinite)
    
    # Colorbar
    _add_colorbar(fig, norm, cmap)
    
    # Style axes
    _style_axes(ax)
    
    # Save if requested
    if output_path:
        fig.savefig(output_path, transparent=True, bbox_inches="tight")
        print(f"Event visualization saved to {output_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def show_event_grid_from_array(
    npe_array: np.ndarray,
    time_array: np.ndarray,
    geometry: np.ndarray,
    labels: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, Path]] = None,
    grid_layout: str = "side_by_side",
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
    Show grid visualization directly from NumPy arrays.
    
    Args:
        npe_array: NPE values array (L,) or (1, L)
        time_array: Time values array (L,) or (1, L)
        geometry: PMT geometry array (3, L) or (L, 3)
        labels: Event labels array (6,) or (1, 6)
        output_path: Path to save the plot
        grid_layout: Layout type - "side_by_side", "stacked", or "separate"
        sphere_resolution: Resolution for sphere rendering
        base_radius: Base radius for PMT spheres
        radius_scale: Scaling factor for sphere radius based on NPE
        figure_size: Figure size for plots
        skip_nonfinite: Skip non-finite values in visualization
        scatter_background: Show background dots for all PMTs
        show_detector_hull: Show detector hull outline
        show: If True, display the plot with plt.show()
        title_prefix: Prefix for the plot title
        **kwargs: Additional arguments
    
    Returns:
        (fig, axes): matplotlib Figure and list of Axes3D objects
    """
    # Normalize array shapes
    npe_array, time_array, geometry, labels = _normalize_array_shapes(
        npe_array, time_array, geometry, labels
    )
    
    # Extract coordinates
    x, y, z = geometry[0], geometry[1], geometry[2]
    L = len(x)
    
    # Validate array lengths
    if len(npe_array) != L or len(time_array) != L:
        raise ValueError(f"Array length mismatch: NPE={len(npe_array)}, "
                        f"Time={len(time_array)}, Geometry={L}")

    # Sanitize time array
    time_array = time_array.copy()
    time_array[np.isinf(time_array)] = 0.0

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
        # Combined view
        _create_combined_plot_from_arrays(axes[0], x, y, z, npe_array, time_array, 
                                        sphere_resolution, base_radius, radius_scale,
                                        skip_nonfinite, scatter_background, 
                                        show_detector_hull, labels, title_prefix + "Combined View")
    else:
        # NPE-only plot
        _create_npe_plot_from_arrays(axes[0], x, y, z, npe_array, sphere_resolution,
                                   base_radius, radius_scale, skip_nonfinite,
                                   scatter_background, show_detector_hull, labels,
                                   title_prefix + "NPE Response")
        
        # Time-only plot
        _create_time_plot_from_arrays(axes[1], x, y, z, time_array, sphere_resolution,
                                    base_radius, radius_scale, skip_nonfinite,
                                    scatter_background, show_detector_hull, labels,
                                    title_prefix + "Time Response")

    # Overall title
    if labels is not None:
        energy, zenith, azimuth, x_pos, y_pos, z_pos = labels
        fig.suptitle(f"Event Grid Visualization from Arrays\n"
                    f"Energy = {energy:.3f}, Zenith = {zenith:.3f}, Azimuth = {azimuth:.3f}\n"
                    f"Position = ({x_pos:.2f}, {y_pos:.2f}, {z_pos:.2f})", 
                    fontsize=16, y=0.95)
    else:
        fig.suptitle("Event Grid Visualization from Arrays", fontsize=16, y=0.95)
    
    # Save if requested
    if output_path:
        fig.savefig(output_path, transparent=True, bbox_inches="tight")
        print(f"Event grid visualization saved to {output_path}")
    
    if show:
        plt.show()
    
    return fig, axes


def _normalize_array_shapes(npe_array, time_array, geometry, labels):
    """Normalize array shapes to standard format."""
    # Handle batch dimensions
    if npe_array.ndim == 2 and npe_array.shape[0] == 1:
        npe_array = npe_array[0]
    if time_array.ndim == 2 and time_array.shape[0] == 1:
        time_array = time_array[0]
    
    # Handle geometry format
    if geometry.ndim == 3 and geometry.shape[0] == 3:
        geometry = geometry[:, 0]  # Take first batch
    elif geometry.ndim == 2 and geometry.shape[1] == 3:
        geometry = geometry.T  # Transpose to (3, L)
    
    # Handle labels
    if labels is not None:
        if labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels[0]
    
    return npe_array, time_array, geometry, labels


def _create_combined_plot_from_arrays(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray,
    npe_array: np.ndarray, time_array: np.ndarray,
    sphere_resolution: Tuple[int, int], base_radius: float, radius_scale: float,
    skip_nonfinite: bool, scatter_background: bool, show_detector_hull: bool,
    labels: Optional[np.ndarray], title: str
):
    """Create combined visualization from arrays."""
    # Color scale for time
    nonzero_mask = (time_array != 0) & np.isfinite(time_array)
    if not nonzero_mask.any():
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(time_array[nonzero_mask]))
        vmax = float(np.max(time_array[nonzero_mask]))
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
    _draw_pmt_spheres(ax, x, y, z, npe_array, time_array, norm, cmap,
                     sphere_resolution, base_radius, radius_scale, skip_nonfinite)
    
    # Styling
    ax.set_title(title, fontsize=14)
    _style_axes(ax)
    
    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Time (ns)', rotation=270, labelpad=20)


def _create_npe_plot_from_arrays(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray,
    npe_array: np.ndarray, sphere_resolution: Tuple[int, int],
    base_radius: float, radius_scale: float, skip_nonfinite: bool,
    scatter_background: bool, show_detector_hull: bool, labels: Optional[np.ndarray],
    title: str
):
    """Create NPE-only visualization from arrays."""
    cmap = colormaps["viridis"]
    norm = Normalize(vmin=0, vmax=np.max(npe_array) if np.max(npe_array) > 0 else 1.0)
    
    if show_detector_hull:
        _draw_detector_hull(ax, x, y, z)
    
    if scatter_background:
        ax.scatter(x, y, z, s=1, c="gray", alpha=0.5)
    
    _draw_pmt_spheres(ax, x, y, z, npe_array, npe_array, norm, cmap,
                     sphere_resolution, base_radius, radius_scale, skip_nonfinite)
    
    ax.set_title(title, fontsize=14)
    _style_axes(ax)
    
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('NPE', rotation=270, labelpad=20)


def _create_time_plot_from_arrays(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray,
    time_array: np.ndarray, sphere_resolution: Tuple[int, int],
    base_radius: float, radius_scale: float, skip_nonfinite: bool,
    scatter_background: bool, show_detector_hull: bool, labels: Optional[np.ndarray],
    title: str
):
    """Create Time-only visualization from arrays."""
    nonzero_mask = (time_array != 0) & np.isfinite(time_array)
    if not nonzero_mask.any():
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(time_array[nonzero_mask]))
        vmax = float(np.max(time_array[nonzero_mask]))
        if vmin == vmax:
            vmax = vmin + 1.0
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps["jet"]
    
    if show_detector_hull:
        _draw_detector_hull(ax, x, y, z)
    
    if scatter_background:
        ax.scatter(x, y, z, s=1, c="gray", alpha=0.5)
    
    uniform_npe = np.ones_like(time_array) * 10
    _draw_pmt_spheres(ax, x, y, z, uniform_npe, time_array, norm, cmap,
                     sphere_resolution, base_radius, radius_scale, skip_nonfinite)
    
    ax.set_title(title, fontsize=14)
    _style_axes(ax)
    
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Time (ns)', rotation=270, labelpad=20)


# Import helper functions locally
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
    ax: plt.Axes, 
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    npe: np.ndarray, ftime: np.ndarray,
    norm: Normalize, cmap,
    sphere_resolution: Tuple[int, int],
    base_radius: float,
    radius_scale: float,
    skip_nonfinite: bool
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


def _add_colorbar(fig: plt.Figure, norm: Normalize, cmap):
    """Add colorbar to the figure."""
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    # Add colorbar
    cbar = fig.colorbar(sm, ax=fig.axes[0], shrink=0.5, aspect=20)
    cbar.set_label('Time (ns)', rotation=270, labelpad=20)


def _style_axes(ax: plt.Axes):
    """Style the 3D axes."""
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.grid(True, alpha=0.3)


def main():
    """Command line interface for array-based event visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize IceCube neutrino events directly from NumPy arrays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--npe-array", "-n",
        type=str,
        required=True,
        help="Path to NPE array file (.npy)"
    )
    
    parser.add_argument(
        "--time-array", "-t",
        type=str,
        required=True,
        help="Path to Time array file (.npy)"
    )
    
    parser.add_argument(
        "--geometry", "-g",
        type=str,
        required=True,
        help="Path to geometry array file (.npy)"
    )
    
    parser.add_argument(
        "--labels", "-l",
        type=str,
        default=None,
        help="Path to labels array file (.npy)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for the visualization (PNG/PDF/SVG)"
    )
    
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Create grid visualization instead of single plot"
    )
    
    parser.add_argument(
        "--grid-layout",
        type=str,
        choices=["side_by_side", "stacked", "separate"],
        default="side_by_side",
        help="Grid layout type (only used with --grid)"
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
        default=[15, 10],
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
    
    # Validate input files
    for file_path, name in [(args.npe_array, "NPE"), (args.time_array, "Time"), 
                           (args.geometry, "Geometry")]:
        if not Path(file_path).exists():
            print(f"Error: {name} array file not found: {file_path}")
            sys.exit(1)
    
    if args.labels and not Path(args.labels).exists():
        print(f"Error: Labels array file not found: {args.labels}")
        sys.exit(1)
    
    try:
        # Load arrays
        npe_array = np.load(args.npe_array)
        time_array = np.load(args.time_array)
        geometry = np.load(args.geometry)
        labels = np.load(args.labels) if args.labels else None
        
        # Create visualization
        if args.grid:
            fig, axes = show_event_grid_from_array(
                npe_array=npe_array,
                time_array=time_array,
                geometry=geometry,
                labels=labels,
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
        else:
            fig, ax = show_event_from_array(
                npe_array=npe_array,
                time_array=time_array,
                geometry=geometry,
                labels=labels,
                output_path=args.output,
                sphere_resolution=tuple(args.sphere_resolution),
                base_radius=args.base_radius,
                radius_scale=args.radius_scale,
                figure_size=tuple(args.figure_size),
                scatter_background=not args.no_background,
                show_detector_hull=not args.no_hull,
                show=args.show,
                title_prefix=args.title_prefix
            )
        
        print("✅ Event visualization from arrays completed successfully!")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
