#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Dataloader - Visualization Integrated with Dataloader Classes

This module provides visualization functionality integrated with dataloader classes,
making it easy to visualize events directly from training/validation data.

Usage:
    # As a library
    from utils.event_visualization.event_dataloader import show_event_from_dataloader
    fig, ax = show_event_from_dataloader(dataloader, event_index=0)
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, List, Any, Dict
import numpy as np
import matplotlib.pyplot as plt
import torch


def _find_project_root():
    """Find the project root directory by looking for .git or configs folder."""
    current = Path.cwd()
    
    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "configs").exists():
            return parent
    
    # Fallback to current directory
    return current


def show_event_from_dataloader(
    dataloader: Any,
    event_index: int = 0,
    batch_index: int = 0,
    denormalize: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    grid_layout: str = "side_by_side",
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
) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
    """
    Show event visualization directly from dataloader.
    
    Extracts data from dataloader, optionally denormalizes it, and creates
    visualization. Perfect for training monitoring and debugging.
    
    Args:
        dataloader: PyTorch DataLoader or similar object
        event_index: Index of event within the batch
        batch_index: Index of batch within the dataloader (default: 0)
        denormalize: Whether to denormalize data before visualization
        output_path: Path to save the plot (PNG/PDF/SVG). If None, doesn't save.
        grid_layout: Layout for grid visualization ("side_by_side", "stacked", "separate")
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
        (fig, ax_or_axes): matplotlib Figure and Axes3D object(s)
        
    Example:
        >>> from dataloader.pmt_dataloader import make_dataloader
        >>> dataloader = make_dataloader(config)
        >>> fig, ax = show_event_from_dataloader(dataloader, event_index=5)
    """
    # Import the array visualization functions
    from .event_array import show_event_from_array, show_event_grid_from_array
    
    # Extract data from dataloader
    data_batch = _extract_data_from_dataloader(dataloader, batch_index)
    
    # Get specific event
    x_sig, geom, label = _get_event_from_batch(data_batch, event_index)
    
    # Denormalize if requested
    if denormalize:
        x_sig, label = _denormalize_event_data(x_sig, label, data_batch.get('config'))
    
    # Convert to numpy arrays
    npe_array = x_sig[0].cpu().numpy() if torch.is_tensor(x_sig[0]) else x_sig[0]
    time_array = x_sig[1].cpu().numpy() if torch.is_tensor(x_sig[1]) else x_sig[1]
    geometry = geom.cpu().numpy() if torch.is_tensor(geom) else geom
    labels = label.cpu().numpy() if torch.is_tensor(label) else label
    
    # Create title prefix with dataloader info
    full_title_prefix = f"{title_prefix}Dataloader Event {event_index} (Batch {batch_index}) - "
    
    # Choose visualization type
    if grid_layout == "separate":
        # Single combined plot
        fig, ax = show_event_from_array(
            npe_array=npe_array,
            time_array=time_array,
            geometry=geometry,
            labels=labels,
            output_path=output_path,
            sphere_resolution=sphere_resolution,
            base_radius=base_radius,
            radius_scale=radius_scale,
            figure_size=figure_size,
            skip_nonfinite=skip_nonfinite,
            scatter_background=scatter_background,
            show_detector_hull=show_detector_hull,
            show=show,
            title_prefix=full_title_prefix,
            **kwargs
        )
        return fig, ax
    else:
        # Grid plot
        fig, axes = show_event_grid_from_array(
            npe_array=npe_array,
            time_array=time_array,
            geometry=geometry,
            labels=labels,
            output_path=output_path,
            grid_layout=grid_layout,
            sphere_resolution=sphere_resolution,
            base_radius=base_radius,
            radius_scale=radius_scale,
            figure_size=figure_size,
            skip_nonfinite=skip_nonfinite,
            scatter_background=scatter_background,
            show_detector_hull=show_detector_hull,
            show=show,
            title_prefix=full_title_prefix,
            **kwargs
        )
        return fig, axes


def _extract_data_from_dataloader(dataloader: Any, batch_index: int = 0) -> Dict[str, Any]:
    """Extract data batch from dataloader."""
    try:
        # Handle different dataloader types
        if hasattr(dataloader, '__iter__'):
            # Standard PyTorch DataLoader
            data_iter = iter(dataloader)
            for i, batch in enumerate(data_iter):
                if i == batch_index:
                    return batch
            raise IndexError(f"Batch index {batch_index} out of range")
        elif hasattr(dataloader, 'get_batch'):
            # Custom dataloader with get_batch method
            return dataloader.get_batch(batch_index)
        else:
            raise ValueError(f"Unsupported dataloader type: {type(dataloader)}")
    except Exception as e:
        raise ValueError(f"Could not extract data from dataloader: {e}")


def _get_event_from_batch(batch: Dict[str, Any], event_index: int) -> Tuple[Any, Any, Any]:
    """Extract specific event from batch."""
    # Handle different batch formats
    if 'input' in batch and 'label' in batch:
        # Standard format: input, label
        x_sig = batch['input'][event_index]  # (2, L)
        geom = batch.get('geometry', batch.get('geom'))[event_index]  # (3, L)
        label = batch['label'][event_index]  # (6,)
    elif 'x_sig' in batch and 'label' in batch:
        # Alternative format: x_sig, label
        x_sig = batch['x_sig'][event_index]  # (2, L)
        geom = batch.get('geom', batch.get('geometry'))[event_index]  # (3, L)
        label = batch['label'][event_index]  # (6,)
    else:
        raise ValueError(f"Unknown batch format. Available keys: {list(batch.keys())}")
    
    # Add config if available
    if 'config' not in batch:
        batch['config'] = None
    
    return x_sig, geom, label


def _denormalize_event_data(x_sig: Any, label: Any, config: Any = None) -> Tuple[Any, Any]:
    """Denormalize event data if config is available."""
    if config is None:
        return x_sig, label
    
    try:
        # Try to import denormalization functions
        from utils.denormalization import denormalize_signal, denormalize_label
        
        # Denormalize signal
        x_sig_denorm = denormalize_signal(
            x_sig,
            affine_offsets=tuple(config.data.affine_offsets),
            affine_scales=tuple(config.data.affine_scales),
            time_transform=config.data.time_transform,
            channels="signal"
        )
        
        # Denormalize label
        label_denorm = denormalize_label(
            label,
            label_offsets=tuple(config.data.label_offsets),
            label_scales=tuple(config.data.label_scales)
        )
        
        return x_sig_denorm, label_denorm
        
    except ImportError:
        print("Warning: Could not import denormalization functions. Using normalized data.")
        return x_sig, label
    except Exception as e:
        print(f"Warning: Could not denormalize data: {e}. Using normalized data.")
        return x_sig, label


def main():
    """Command line interface for dataloader-based event visualization."""
    print("Note: This module is primarily designed for library use.")
    print("For CLI usage, use the other event visualization modules.")
    print("Example: python -c \"from utils.event_visualization.event_dataloader import show_event_from_dataloader\"")
    return 0


if __name__ == "__main__":
    exit(main())