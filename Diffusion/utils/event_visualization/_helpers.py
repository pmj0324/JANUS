#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Visualization Helper Functions

This module contains shared helper functions used across all event visualization modules.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Tuple


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


def _draw_detector_hull(ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Draw detector hull outline."""
    # Simplified detector hull (IceCube-like)
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


def _compute_time_color_scale(ftime: np.ndarray) -> Tuple[float, float]:
    """Compute color scale for time data."""
    nonzero_mask = (ftime != 0) & np.isfinite(ftime)
    if not nonzero_mask.any():
        return 0.0, 1.0
    
    vmin = float(np.min(ftime[nonzero_mask]))
    vmax = float(np.max(ftime[nonzero_mask]))
    if vmin == vmax:
        vmax = vmin + 1.0
    
    return vmin, vmax


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
