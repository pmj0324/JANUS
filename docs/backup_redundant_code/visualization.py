#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities for IceCube neutrino events.

This module provides visualization functions that integrate with the npz-show-event.py
format for displaying generated and real neutrino events.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Optional, Tuple, Union, List
from pathlib import Path
import torch


def create_3d_event_plot(
    npz_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    detector_csv: Optional[str] = None,
    show: bool = False,
    **kwargs
):
    """
    Create a 3D visualization of an event from NPZ file.
    
    This is a wrapper around npz_show_event.show_event() for easy integration.
    
    Args:
        npz_path: Path to NPZ file with 'input' and 'label' keys
        output_path: Path to save the plot (PNG/PDF/SVG). If None, doesn't save.
        detector_csv: Path to detector geometry CSV. If None, uses default.
        show: If True, display the plot with plt.show()
        **kwargs: Additional arguments passed to show_event()
    
    Returns:
        (fig, ax): matplotlib Figure and Axes3D objects
        
    Example:
        >>> create_3d_event_plot(
        ...     "outputs/samples/sample_0000.npz",
        ...     "outputs/samples/sample_0000_3d.png"
        ... )
    """
    # Import new event visualization module
    from .event_visualization.event_show import show_event_from_npz
    
    # Call show_event_from_npz
    fig, ax = show_event_from_npz(
        npz_path=npz_path,
        detector_csv=detector_csv,
        output_path=output_path,
        show=show,
        **kwargs
    )
    
    return fig, ax


class EventVisualizer:
    """
    Legacy EventVisualizer class.
    
    Note: This class is maintained for backward compatibility.
    For new code, consider using the modules in utils.event_visualization:
    - event_show.py for NPZ-based visualization
    - event_array.py for direct array visualization  
    - event_grid.py for grid layouts
    - event_dataloader.py for dataloader integration
    """
    
    def __init__(
        self,
        detector_geometry_path: str = "./utils/csv/detector_geometry.csv",
        sphere_resolution: Tuple[int, int] = (40, 20),
        base_radius: float = 5.0,
        radius_scale: float = 0.2,
        figure_size: Tuple[int, int] = (15, 10),
        time_transform: Optional[str] = "ln",  # "log10", "ln", None
        exclude_zero_time: bool = True,
    ):
        """
        Initialize the event visualizer.
        
        Args:
            detector_geometry_path: Path to detector geometry CSV file
            sphere_resolution: Resolution for sphere rendering (u_steps, v_steps)
            base_radius: Base radius for PMT spheres
            radius_scale: Scaling factor for sphere radius based on NPE
            figure_size: Figure size for plots
        """
        self.detector_geometry_path = detector_geometry_path
        self.sphere_resolution = sphere_resolution
        self.base_radius = base_radius
        self.radius_scale = radius_scale
        self.figure_size = figure_size
        self.time_transform = time_transform
        self.exclude_zero_time = exclude_zero_time
        
        # Load detector geometry
        self._load_detector_geometry()
    
    def denormalize_signal(self, x_sig: torch.Tensor, affine_offsets: Tuple[float, ...], 
                          affine_scales: Tuple[float, ...]) -> torch.Tensor:
        """
        Denormalize signal from normalized range back to original scale.
        Formula: x_original = (x_normalized * scale) + offset
        
        Args:
            x_sig: Normalized signal tensor (B, 2, L)
            affine_offsets: Offset values for [npe, time, x, y, z]
            affine_scales: Scale values for [npe, time, x, y, z]
            
        Returns:
            Denormalized signal tensor
        """
        # Only use charge and time offsets/scales (first 2 elements)
        off = torch.tensor(affine_offsets[:2], dtype=torch.float32).view(1, 2, 1)
        scl = torch.tensor(affine_scales[:2], dtype=torch.float32).view(1, 2, 1)
        
        # Apply inverse normalization: x_original = (x_normalized * scale) + offset
        x_original = (x_sig * scl) + off
        
        # If time was log-transformed, apply inverse transformation
        if self.time_transform is not None and x_sig.shape[1] >= 2:
            time_original = x_original[:, 1, :]  # Time channel
            
            if self.time_transform == "log10":
                time_original = torch.pow(10.0, time_original) - 1e-10
            elif self.time_transform == "ln":
                time_original = torch.exp(time_original) - 1e-10
            
            # Handle the case where 0 values were excluded
            if self.exclude_zero_time:
                # Very negative values in log space correspond to original 0 values
                zero_mask = time_original < 0.1  # Very small values
                time_original[zero_mask] = 0.0
            
            x_original[:, 1, :] = time_original
        
        return x_original
    
    def _load_detector_geometry(self):
        """Load detector geometry from CSV file."""
        if os.path.exists(self.detector_geometry_path):
            df_geo = pd.read_csv(self.detector_geometry_path)
            self.x = np.asarray(df_geo["x"], dtype=np.float32)
            self.y = np.asarray(df_geo["y"], dtype=np.float32)
            self.z = np.asarray(df_geo["z"], dtype=np.float32)
            self.num_pmts = len(self.x)
        else:
            # Create synthetic geometry if file doesn't exist
            print(f"Warning: Detector geometry file not found at {self.detector_geometry_path}")
            print("Creating synthetic detector geometry...")
            self._create_synthetic_geometry()
    
    def _create_synthetic_geometry(self):
        """Create synthetic detector geometry."""
        # Simple cubic grid geometry
        grid_size = int(np.ceil(5160**(1/3)))
        x = np.linspace(-500, 500, grid_size)
        y = np.linspace(-500, 500, grid_size)
        z = np.linspace(-500, 500, grid_size)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        self.x = xx.flatten()[:5160]
        self.y = yy.flatten()[:5160]
        self.z = zz.flatten()[:5160]
        self.num_pmts = 5160
    
    def visualize_event(
        self,
        pmt_signals: Union[torch.Tensor, np.ndarray],
        event_conditions: Union[torch.Tensor, np.ndarray],
        output_path: Optional[str] = None,
        title_prefix: str = "",
        skip_nonfinite: bool = True,
        scatter_background: bool = True,
        show_detector_hull: bool = True,
        denormalize: bool = True,
        affine_offsets: Optional[Tuple[float, ...]] = None,
        affine_scales: Optional[Tuple[float, ...]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize a single neutrino event in 3D.
        
        Args:
            pmt_signals: PMT signals (2, L) [npe, time]
            event_conditions: Event conditions (6,) [Energy, Zenith, Azimuth, X, Y, Z]
            output_path: Path to save the visualization
            title_prefix: Prefix for the plot title
            skip_nonfinite: Skip non-finite values
            scatter_background: Show background PMT positions
            show_detector_hull: Show detector hull outline
            
        Returns:
            Tuple of (figure, axes)
        """
        # Convert to torch tensor if needed for denormalization
        if isinstance(pmt_signals, np.ndarray):
            pmt_signals_tensor = torch.from_numpy(pmt_signals)
        else:
            pmt_signals_tensor = pmt_signals
        
        # Denormalize if requested
        if denormalize and affine_offsets is not None and affine_scales is not None:
            if pmt_signals_tensor.ndim == 2:
                pmt_signals_tensor = pmt_signals_tensor.unsqueeze(0)  # Add batch dimension
            
            pmt_signals_tensor = self.denormalize_signal(pmt_signals_tensor, affine_offsets, affine_scales)
            
            if pmt_signals_tensor.ndim == 3:
                pmt_signals_tensor = pmt_signals_tensor.squeeze(0)  # Remove batch dimension
        
        # Convert to numpy for visualization
        pmt_signals = pmt_signals_tensor.cpu().numpy()
        
        if isinstance(event_conditions, torch.Tensor):
            event_conditions = event_conditions.cpu().numpy()
        
        # Ensure correct shapes
        if pmt_signals.ndim == 3:
            pmt_signals = pmt_signals[0]  # Take first event
        if event_conditions.ndim == 2:
            event_conditions = event_conditions[0]  # Take first event
        
        assert pmt_signals.shape == (2, self.num_pmts), f"Expected signals shape (2, {self.num_pmts}), got {pmt_signals.shape}"
        assert event_conditions.shape == (6,), f"Expected conditions shape (6,), got {event_conditions.shape}"
        
        # Extract data
        npe = pmt_signals[0, :].astype(np.float32)
        ftime = pmt_signals[1, :].astype(np.float32)
        
        # Extract event conditions
        energy, zenith, azimuth, x_pos, y_pos, z_pos = event_conditions
        
        # Sanitize firstTime: ±inf → 0
        ftime[np.isinf(ftime)] = 0.0
        
        # Color scale: 0 제외하고 min/max 계산
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
        fig = plt.figure(figsize=self.figure_size)
        ax = fig.add_subplot(111, projection="3d")
        
        # Title
        title_line1 = f"{title_prefix}Energy = {energy:.3f}"
        title_line2 = f"Zenith = {zenith:.3f}, Azimuth = {azimuth:.3f}"
        title_line3 = f"X = {x_pos:.2f}, Y = {y_pos:.2f}, Z = {z_pos:.2f}"
        fig.suptitle(f"{title_line1}\n{title_line2}\n{title_line3}", fontsize=14, y=0.98)
        
        # Detector hull (optional)
        if show_detector_hull:
            self._draw_detector_hull(ax)
        
        # Background dots
        if scatter_background:
            ax.scatter(self.x, self.y, self.z, s=1, c="gray", alpha=0.5)
        
        # PMT spheres
        self._draw_pmt_spheres(ax, npe, ftime, norm, cmap, skip_nonfinite)
        
        # Colorbar
        self._add_colorbar(fig, norm, cmap)
        
        # Style axes
        self._style_axes(ax)
        
        # Save if requested
        if output_path:
            fig.savefig(output_path, transparent=True, bbox_inches="tight")
            print(f"Visualization saved to {output_path}")
        
        return fig, ax
    
    def _draw_detector_hull(self, ax: plt.Axes):
        """Draw detector hull outline."""
        # Simplified detector hull (IceCube-like)
        edge_string_idx = [1, 6, 50, 74, 73, 78, 75, 31]
        top_xy, bottom_xy = [], []
        
        for i in edge_string_idx:
            if (i - 1) * 60 < len(self.x):
                top_xy.append([self.x[(i - 1) * 60], self.y[(i - 1) * 60]])
            if (i - 1) * 60 + 59 < len(self.x):
                bottom_xy.append([self.x[(i - 1) * 60 + 59], self.y[(i - 1) * 60 + 59]])
        
        if top_xy and bottom_xy:
            top_xy.append(top_xy[0])
            bottom_xy.append(bottom_xy[0])
            
            z_bottom, z_top = -500, 500
            for poly in (top_xy, bottom_xy):
                for i in range(len(poly) - 1):
                    x0, y0 = poly[i]
                    x1, y1 = poly[i + 1]
                    zc = z_top if poly is top_xy else z_bottom
                    ax.plot([x0, x1], [y0, y1], [zc, zc], color="gray", alpha=0.7)
            
            for _x, _y in top_xy[:-1]:
                ax.plot([_x, _x], [_y, _y], [z_bottom, z_top], color="gray", alpha=0.7)
    
    def _draw_pmt_spheres(
        self,
        ax: plt.Axes,
        npe: np.ndarray,
        ftime: np.ndarray,
        norm: Normalize,
        cmap,
        skip_nonfinite: bool
    ):
        """Draw PMT spheres with NPE-based size and time-based color."""
        u_steps, v_steps = self.sphere_resolution
        u, v = np.mgrid[0:2 * np.pi:complex(u_steps), 0:np.pi:complex(v_steps)]
        
        for xi, yi, zi, ri, ci in zip(self.x, self.y, self.z, npe, ftime):
            # Skip non-finite values if requested
            if skip_nonfinite and ((ri <= 0) or (not np.isfinite(ci)) or (ci == 0)):
                continue
            
            radius = self.base_radius + self.radius_scale * (1.0 + ri)
            color = cmap(norm(ci))
            
            xs = radius * np.cos(u) * np.sin(v) + xi
            ys = radius * np.sin(u) * np.sin(v) + yi
            zs = radius * np.cos(v) + zi
            
            facecolor = np.ones(xs.shape + (4,), dtype=np.float32)
            facecolor[...] = color
            
            ax.plot_surface(
                xs, ys, zs,
                facecolors=facecolor,
                rstride=1, cstride=1,
                linewidth=0, antialiased=True,
                shade=True, alpha=0.9
            )
    
    def _add_colorbar(self, fig: plt.Figure, norm: Normalize, cmap):
        """Add colorbar to the figure."""
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.373, 0.15, 0.3, 0.02])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label("FirstTime (ns)", fontsize=16)
    
    def _style_axes(self, ax: plt.Axes):
        """Style the 3D axes."""
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_visible(False)
            axis.line.set_color((0.0, 0.0, 0.0, 0.0))
        
        ax.dist = 3
    
    def visualize_batch(
        self,
        pmt_signals: Union[torch.Tensor, np.ndarray],
        event_conditions: Union[torch.Tensor, np.ndarray],
        output_dir: str,
        max_events: int = 4,
        prefix: str = "event"
    ) -> List[str]:
        """
        Visualize a batch of events.
        
        Args:
            pmt_signals: PMT signals (B, 2, L)
            event_conditions: Event conditions (B, 6)
            output_dir: Directory to save visualizations
            max_events: Maximum number of events to visualize
            prefix: Prefix for output filenames
            
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to numpy if needed
        if isinstance(pmt_signals, torch.Tensor):
            pmt_signals = pmt_signals.cpu().numpy()
        if isinstance(event_conditions, torch.Tensor):
            event_conditions = event_conditions.cpu().numpy()
        
        batch_size = min(pmt_signals.shape[0], max_events)
        output_paths = []
        
        for i in range(batch_size):
            output_path = os.path.join(output_dir, f"{prefix}_{i:03d}.png")
            
            fig, ax = self.visualize_event(
                pmt_signals[i],
                event_conditions[i],
                output_path=output_path,
                title_prefix=f"Event {i+1} - "
            )
            
            plt.close(fig)  # Close to free memory
            output_paths.append(output_path)
        
        print(f"Visualized {batch_size} events in {output_dir}")
        return output_paths
    
    def compare_events(
        self,
        real_signals: Union[torch.Tensor, np.ndarray],
        real_conditions: Union[torch.Tensor, np.ndarray],
        generated_signals: Union[torch.Tensor, np.ndarray],
        generated_conditions: Union[torch.Tensor, np.ndarray],
        output_path: Optional[str] = None,
        max_events: int = 4
    ) -> plt.Figure:
        """
        Create a comparison visualization between real and generated events.
        
        Args:
            real_signals: Real PMT signals (B, 2, L)
            real_conditions: Real event conditions (B, 6)
            generated_signals: Generated PMT signals (B, 2, L)
            generated_conditions: Generated event conditions (B, 6)
            output_path: Path to save the comparison
            max_events: Maximum number of events to compare
            
        Returns:
            Figure with comparison plots
        """
        # Convert to numpy if needed
        if isinstance(real_signals, torch.Tensor):
            real_signals = real_signals.cpu().numpy()
        if isinstance(real_conditions, torch.Tensor):
            real_conditions = real_conditions.cpu().numpy()
        if isinstance(generated_signals, torch.Tensor):
            generated_signals = generated_signals.cpu().numpy()
        if isinstance(generated_conditions, torch.Tensor):
            generated_conditions = generated_conditions.cpu().numpy()
        
        batch_size = min(real_signals.shape[0], generated_signals.shape[0], max_events)
        
        # Create subplot grid
        fig = plt.figure(figsize=(20, 5 * batch_size))
        
        for i in range(batch_size):
            # Real event
            ax1 = fig.add_subplot(batch_size, 2, 2*i + 1, projection="3d")
            self._plot_single_event(
                ax1, real_signals[i], real_conditions[i],
                title=f"Real Event {i+1}"
            )
            
            # Generated event
            ax2 = fig.add_subplot(batch_size, 2, 2*i + 2, projection="3d")
            self._plot_single_event(
                ax2, generated_signals[i], generated_conditions[i],
                title=f"Generated Event {i+1}"
            )
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Comparison saved to {output_path}")
        
        return fig
    
    def _plot_single_event(
        self,
        ax: plt.Axes,
        pmt_signals: np.ndarray,
        event_conditions: np.ndarray,
        title: str
    ):
        """Plot a single event on given axes."""
        npe = pmt_signals[0, :]
        ftime = pmt_signals[1, :]
        energy, zenith, azimuth, x_pos, y_pos, z_pos = event_conditions
        
        # Color scale
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
        
        # Background
        ax.scatter(self.x, self.y, self.z, s=1, c="gray", alpha=0.3)
        
        # PMT spheres
        u_steps, v_steps = 20, 10  # Lower resolution for subplots
        u, v = np.mgrid[0:2 * np.pi:complex(u_steps), 0:np.pi:complex(v_steps)]
        
        for xi, yi, zi, ri, ci in zip(self.x, self.y, self.z, npe, ftime):
            if (ri <= 0) or (not np.isfinite(ci)) or (ci == 0):
                continue
            
            radius = 3.0 + 0.1 * (1.0 + ri)  # Smaller spheres for subplots
            color = cmap(norm(ci))
            
            xs = radius * np.cos(u) * np.sin(v) + xi
            ys = radius * np.sin(u) * np.sin(v) + yi
            zs = radius * np.cos(v) + zi
            
            facecolor = np.ones(xs.shape + (4,), dtype=np.float32)
            facecolor[...] = color
            
            ax.plot_surface(
                xs, ys, zs,
                facecolors=facecolor,
                rstride=1, cstride=1,
                linewidth=0, antialiased=True,
                shade=True, alpha=0.8
            )
        
        # Style
        ax.set_title(f"{title}\nE={energy:.2f}, θ={zenith:.2f}, φ={azimuth:.2f}", fontsize=10)
        self._style_axes(ax)


def create_event_visualizer(
    detector_geometry_path: Optional[str] = None,
    **kwargs
) -> EventVisualizer:
    """
    Convenience function to create an EventVisualizer.
    
    Args:
        detector_geometry_path: Path to detector geometry CSV
        **kwargs: Additional arguments for EventVisualizer
        
    Returns:
        EventVisualizer instance
    """
    if detector_geometry_path is None:
        # Try to find the geometry file
        possible_paths = [
            "./utils/csv/detector_geometry.csv",
            "./csv/detector_geometry.csv",
            "../utils/csv/detector_geometry.csv",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                detector_geometry_path = path
                break
        
        if detector_geometry_path is None:
            detector_geometry_path = "./utils/csv/detector_geometry.csv"
    
    return EventVisualizer(detector_geometry_path=detector_geometry_path, **kwargs)


if __name__ == "__main__":
    # Test the visualizer
    visualizer = create_event_visualizer()
    
    # Create synthetic data
    num_pmts = 5160
    pmt_signals = np.random.rand(2, num_pmts)
    pmt_signals[0, :] = np.abs(pmt_signals[0, :]) * 10  # NPE
    pmt_signals[1, :] = np.abs(pmt_signals[1, :]) * 1000  # Time
    
    event_conditions = np.array([100.0, 1.5, 2.0, 0.0, 0.0, 0.0])  # Energy, zenith, azimuth, x, y, z
    
    # Visualize
    fig, ax = visualizer.visualize_event(
        pmt_signals,
        event_conditions,
        output_path="test_event.png",
        title_prefix="Test - "
    )
    
    plt.show()
    print("Test visualization completed!")
