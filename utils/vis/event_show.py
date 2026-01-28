#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Show - Basic Event Visualization from NPZ Files

This module provides the fundamental event visualization functionality for NPZ files.
It's the core visualization tool that other modules build upon.

Usage:
    # As a library
    from utils.event_visualization.event_show import show_event_from_npz
    fig, ax = show_event_from_npz("event.npz")
    
    # As a script
    python event_show.py --npz-path event.npz --output event.png
"""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def show_event_from_npz(
    npz_path: Union[str, Path],
    detector_csv: Optional[Union[str, Path]] = None,
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
    Show a 3D visualization of an event from NPZ file.
    
    This is the fundamental event visualization function that loads NPZ files
    and creates beautiful 3D sphere plots showing PMT responses.
    
    Args:
        npz_path: Path to NPZ file with 'input' and 'label' keys
        detector_csv: Path to detector geometry CSV. If None, uses default.
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
        >>> fig, ax = show_event_from_npz(
        ...     "outputs/samples/sample_0000.npz",
        ...     output_path="sample_3d.png",
        ...     show=True
        ... )
    """
    # Set default detector CSV path
    if detector_csv is None:
        # Try to find detector geometry relative to project root
        project_root = _find_project_root()
        detector_csv = project_root /  "detector_geometry.csv"
    
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

    # Color scale: exclude zeros for better visualization
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
    
    # Create figure
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection="3d")
    
    # Title
    title_line1 = f"{title_prefix}"
    title_line2 = f"Energy = {energy:.3f}, Zenith = {zenith:.3f}, Azimuth = {azimuth:.3f}"
    title_line3 = f"Position = ({x_pos:.2f}, {y_pos:.2f}, {z_pos:.2f})"
    fig.suptitle(f"{title_line1}\n{title_line2}\n{title_line3}", fontsize=14, y=0.98)
    
    # Detector hull (optional)
    if show_detector_hull:
        _draw_detector_hull(ax, x, y, z)
    
    # Background dots
    if scatter_background:
        ax.scatter(x, y, z, s=1, c="gray", alpha=0.5)
    
    # PMT spheres
    _draw_pmt_spheres(ax, x, y, z, npe, ftime, norm, cmap, 
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


def _find_project_root() -> Path:
    """Find the project root directory by looking for .git or configs folder."""
    current = Path.cwd()
    
    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "configs").exists():
            return parent
    
    # Fallback to current directory
    return current


def _draw_detector_hull(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    alpha: float = 0.25,
    linewidth: float = 1.5,
):
    """Draw detector hull outline.

    Notes:
    - We approximate the IceCube-like outer boundary using a small set of "edge strings".
    - We explicitly *close* the top/bottom polygons (last -> first) to avoid missing edges.
    - We also draw vertical edges between top and bottom.
    """
    # Simplified detector hull (IceCube-like)
    edge_string_idx = [1, 6, 50, 74, 73, 78, 75, 31]
    top_xy, bottom_xy = [], []
    
    for i in edge_string_idx:
        if (i - 1) * 60 < len(x):
            top_xy.append([x[(i - 1) * 60], y[(i - 1) * 60]])
        if (i - 1) * 60 + 59 < len(x):
            bottom_xy.append([x[(i - 1) * 60 + 59], y[(i - 1) * 60 + 59]])
    
    if top_xy and bottom_xy:
        top_xy = np.asarray(top_xy, dtype=np.float32)
        bottom_xy = np.asarray(bottom_xy, dtype=np.float32)

        z_top = float(np.nanmax(z))
        z_bottom = float(np.nanmin(z))

        # Close the polygons (avoid missing last->first edge)
        if top_xy.shape[0] >= 2:
            top_closed = np.vstack([top_xy, top_xy[0]])
            bottom_closed = np.vstack([bottom_xy, bottom_xy[0]])

            ax.plot(
                top_closed[:, 0],
                top_closed[:, 1],
                np.full(top_closed.shape[0], z_top, dtype=np.float32),
                color="black",
                linewidth=linewidth,
                alpha=alpha,
            )
            ax.plot(
                bottom_closed[:, 0],
                bottom_closed[:, 1],
                np.full(bottom_closed.shape[0], z_bottom, dtype=np.float32),
                color="black",
                linewidth=linewidth,
                alpha=alpha,
            )

        # Vertical edges
        if top_xy.shape[0] == bottom_xy.shape[0]:
            for (tx, ty), (bx, by) in zip(top_xy, bottom_xy):
                ax.plot(
                    [tx, bx],
                    [ty, by],
                    [z_top, z_bottom],
                    color="black",
                    linewidth=max(1.0, linewidth * 0.8),
                    alpha=alpha,
                )


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


def show_event_dual_plot(
    sig: np.ndarray,
    geo: np.ndarray,
    label: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    figure_size: Tuple[int, int] = (20, 10),
    marker_size: float = 20.0,
    show_detector_hull: bool = True,
    show: bool = False,
    title_prefix: str = "",
    **kwargs
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Show a dual 3D visualization of an event with two side-by-side plots.
    
    Left plot: firstTime colored by time value
    Right plot: npe colored by npe value
    
    Uses circular markers (not spheres) for non-zero values, colored by the respective values.
    
    Args:
        sig: Signal array with shape (2, L) - [npe, firstTime]
        geo: Geometry array with shape (3, L) - [x, y, z]
        label: Label array with shape (6,) - [Energy, Zenith, Azimuth, X, Y, Z]
        output_path: Path to save the plot (PNG/PDF/SVG). If None, doesn't save.
        figure_size: Figure size for plots
        marker_size: Size of markers for non-zero values
        show_detector_hull: Show detector hull outline
        show: If True, display the plot with plt.show()
        title_prefix: Prefix for the plot title
        **kwargs: Additional arguments (for compatibility)
    
    Returns:
        (fig, (ax1, ax2)): matplotlib Figure and tuple of two Axes3D objects
    """
    # Optional labels/titles (kept in kwargs for backward compatibility)
    firsttime_title = kwargs.pop("firsttime_title", "FirstTime")
    npe_title = kwargs.pop("npe_title", "NPE")
    firsttime_cbar_label = kwargs.pop("firsttime_cbar_label", "FirstTime (ns)")
    npe_cbar_label = kwargs.pop("npe_cbar_label", "NPE")
    # Optional fixed color range (None = use data min/max)
    firsttime_vmin = kwargs.pop("firsttime_vmin", None)
    firsttime_vmax = kwargs.pop("firsttime_vmax", None)
    npe_vmin = kwargs.pop("npe_vmin", None)
    npe_vmax = kwargs.pop("npe_vmax", None)

    # Validate input shapes
    if sig.shape[0] != 2:
        raise ValueError(f"sig must have shape (2, L), got {sig.shape}")
    if geo.shape[0] != 3:
        raise ValueError(f"geo must have shape (3, L), got {geo.shape}")
    if label.shape[0] != 6:
        raise ValueError(f"label must have shape (6,), got {label.shape}")
    
    L = sig.shape[1]
    if geo.shape[1] != L:
        raise ValueError(f"sig and geo must have same L dimension: sig.shape={sig.shape}, geo.shape={geo.shape}")
    
    # Extract geometry
    x = np.asarray(geo[0, :], dtype=np.float32)
    y = np.asarray(geo[1, :], dtype=np.float32)
    z = np.asarray(geo[2, :], dtype=np.float32)
    
    # Extract event data
    energy, zenith, azimuth, x_pos, y_pos, z_pos = label
    npe = np.asarray(sig[0, :], dtype=np.float32)
    ftime = np.asarray(sig[1, :], dtype=np.float32)
    
    # Sanitize firstTime: ±inf → 0
    ftime[np.isinf(ftime)] = 0.0
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figure_size)
    ax1 = fig.add_subplot(121, projection="3d")  # Left: firstTime
    ax2 = fig.add_subplot(122, projection="3d")  # Right: npe
    
    # Title - Convert units
    energy_pev = energy / 1e6  # MeV to PeV
    zenith_deg = np.degrees(zenith)  # rad to degrees
    azimuth_deg = np.degrees(azimuth)  # rad to degrees
    
    title_line1 = f"{title_prefix}"
    title_line2 = f"Energy = {energy_pev:.3f} PeV, Zenith = {zenith_deg:.3f}°, Azimuth = {azimuth_deg:.3f}°"
    title_line3 = f"Vertex Position (m) = ({x_pos:.2f}, {y_pos:.2f}, {z_pos:.2f})"
    fig.suptitle(f"{title_line1}\n{title_line2}\n{title_line3}", fontsize=14, y=0.98)
    
    # Detector hull (optional)
    if show_detector_hull:
        _draw_detector_hull(ax1, x, y, z)
        _draw_detector_hull(ax2, x, y, z)
    
    # Background dots (all PMTs)
    ax1.scatter(x, y, z, s=1, c="gray", alpha=0.3)
    ax2.scatter(x, y, z, s=1, c="gray", alpha=0.3)
    
    # Left plot: firstTime
    nonzero_ftime_mask = (ftime != 0) & np.isfinite(ftime)
    if nonzero_ftime_mask.any():
        x_ftime = x[nonzero_ftime_mask]
        y_ftime = y[nonzero_ftime_mask]
        z_ftime = z[nonzero_ftime_mask]
        ftime_vals = ftime[nonzero_ftime_mask]
        
        # Color scale for firstTime (fixed range if provided)
        if firsttime_vmin is not None and firsttime_vmax is not None:
            vmin_ftime, vmax_ftime = firsttime_vmin, firsttime_vmax
        else:
            vmin_ftime = float(np.min(ftime_vals))
            vmax_ftime = float(np.max(ftime_vals))
            if vmin_ftime == vmax_ftime:
                vmax_ftime = vmin_ftime + 1.0
        
        norm_ftime = Normalize(vmin=vmin_ftime, vmax=vmax_ftime)
        # Use the same colormap for both channels for consistency
        cmap_ftime = colormaps["jet"]
        
        # Scatter plot with colored markers
        scatter1 = ax1.scatter(x_ftime, y_ftime, z_ftime, 
                              c=ftime_vals, s=marker_size, 
                              cmap=cmap_ftime, norm=norm_ftime,
                              alpha=0.8, edgecolors='none')
        
        # Colorbar for firstTime
        cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=20, pad=0.15)
        cbar1.set_label(firsttime_cbar_label, rotation=270, labelpad=20)
    
    ax1.set_title(firsttime_title, fontsize=12)
    _style_axes(ax1)
    
    # Right plot: npe
    nonzero_npe_mask = (npe != 0) & np.isfinite(npe)
    if nonzero_npe_mask.any():
        x_npe = x[nonzero_npe_mask]
        y_npe = y[nonzero_npe_mask]
        z_npe = z[nonzero_npe_mask]
        npe_vals = npe[nonzero_npe_mask]
        
        # Color scale for npe (fixed range if provided)
        if npe_vmin is not None and npe_vmax is not None:
            vmin_npe, vmax_npe = npe_vmin, npe_vmax
        else:
            vmin_npe = float(np.min(npe_vals))
            vmax_npe = float(np.max(npe_vals))
            if vmin_npe == vmax_npe:
                vmax_npe = vmin_npe + 1.0
        
        norm_npe = Normalize(vmin=vmin_npe, vmax=vmax_npe)
        # Use the same colormap for both channels for consistency
        cmap_npe = colormaps["jet"]
        
        # Scatter plot with colored markers
        scatter2 = ax2.scatter(x_npe, y_npe, z_npe,
                              c=npe_vals, s=marker_size,
                              cmap=cmap_npe, norm=norm_npe,
                              alpha=0.8, edgecolors='none')
        
        # Colorbar for npe
        cbar2 = fig.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=20, pad=0.15)
        cbar2.set_label(npe_cbar_label, rotation=270, labelpad=20)
    
    ax2.set_title(npe_title, fontsize=12)
    _style_axes(ax2)
    
    # Save if requested
    if output_path:
        fig.savefig(output_path, transparent=True, bbox_inches="tight")
        print(f"Event dual visualization saved to {output_path}")
    
    if show:
        plt.show()
    
    return fig, (ax1, ax2)


def main():
    """Command line interface for event visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize IceCube neutrino events from NPZ files",
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
        default="event_visualization.png",
        help="Output path for the visualization (PNG/PDF/SVG). Default: event_visualization.png"
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
    
    # Validate input file
    if not Path(args.npz_path).exists():
        print(f"Error: NPZ file not found: {args.npz_path}")
        sys.exit(1)
    
    try:
        # Create visualization
        fig, ax = show_event_from_npz(
            npz_path=args.npz_path,
            detector_csv=args.detector_csv,
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
        
        print(" Event visualization completed successfully!")
        
    except Exception as e:
        print(f" Error creating visualization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    event = load_event_from_h5()
    show_event_from_npy(event)
