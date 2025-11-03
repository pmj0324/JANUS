#!/usr/bin/env python3
"""
Denormalization utilities for GENESIS

Provides functions to convert normalized signals back to their original physical scale.
Handles both affine normalization and log transformations.
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional


def denormalize_signal(
    x_normalized: Union[torch.Tensor, np.ndarray],
    affine_offsets: Tuple[float, ...],
    affine_scales: Tuple[float, ...],
    time_transform: Optional[str] = "ln",
    channels: str = "signal"  # "signal" for (charge, time) or "geometry" for (x, y, z)
) -> Union[torch.Tensor, np.ndarray]:
    """
    Denormalize signal from normalized range back to original scale.
    
    The forward normalization process is:
    1. Raw data → log transform (if time_transform is set)
    2. Log data → affine normalization: (x - offset) / scale
    
    The inverse process (this function) is:
    1. Normalized data → inverse affine: x_original = (x_normalized * scale) + offset
    2. If time_transform: exp(x_original) to get back to raw scale
    
    Args:
        x_normalized: Normalized signal tensor
            - For signal: (B, 2, L) or (2, L) for [charge, time]
            - For geometry: (B, 3, L) or (3, L) for [x, y, z]
        affine_offsets: Offset values for denormalization
            - For signal: [charge_offset, time_offset] (first 2)
            - For geometry: [x_offset, y_offset, z_offset] (last 3)
        affine_scales: Scale values for denormalization
            - For signal: [charge_scale, time_scale] (first 2)
            - For geometry: [x_scale, y_scale, z_scale] (last 3)
        time_transform: Type of log transform applied ("ln", "log10", or None)
            Only applies to time channel (index 1) in signal mode
        channels: "signal" or "geometry" to determine which channels to process
    
    Returns:
        Denormalized signal tensor in original scale
        
    Examples:
        >>> # Denormalize signal (charge, time)
        >>> x_sig_normalized = torch.randn(4, 2, 5160)  # (B, 2, L)
        >>> offsets = (0.0, 0.0, 0.0, 0.0, 0.0)
        >>> scales = (200.0, 10.0, 500.0, 500.0, 500.0)
        >>> x_sig_original = denormalize_signal(
        ...     x_sig_normalized, offsets, scales, time_transform="ln", channels="signal"
        ... )
        
        >>> # Denormalize geometry
        >>> geom_normalized = torch.randn(4, 3, 5160)  # (B, 3, L)
        >>> geom_original = denormalize_signal(
        ...     geom_normalized, offsets, scales, channels="geometry"
        ... )
    """
    is_torch = isinstance(x_normalized, torch.Tensor)
    
    if channels == "signal":
        # Use first 2 (charge, time)
        idx_start, idx_end = 0, 2
    elif channels == "geometry":
        # Use last 3 (x, y, z)
        idx_start, idx_end = 2, 5
    else:
        raise ValueError(f"channels must be 'signal' or 'geometry', got {channels}")
    
    # Extract relevant offsets and scales
    offsets = affine_offsets[idx_start:idx_end]
    scales = affine_scales[idx_start:idx_end]
    
    # Convert to tensors/arrays
    if is_torch:
        device = x_normalized.device
        # Reshape for broadcasting: (1, C, 1) or (C, 1)
        if x_normalized.ndim == 3:  # (B, C, L)
            off = torch.tensor(offsets, dtype=x_normalized.dtype, device=device).view(1, -1, 1)
            scl = torch.tensor(scales, dtype=x_normalized.dtype, device=device).view(1, -1, 1)
        else:  # (C, L)
            off = torch.tensor(offsets, dtype=x_normalized.dtype, device=device).view(-1, 1)
            scl = torch.tensor(scales, dtype=x_normalized.dtype, device=device).view(-1, 1)
    else:
        if x_normalized.ndim == 3:  # (B, C, L)
            off = np.array(offsets, dtype=x_normalized.dtype).reshape(1, -1, 1)
            scl = np.array(scales, dtype=x_normalized.dtype).reshape(1, -1, 1)
        else:  # (C, L)
            off = np.array(offsets, dtype=x_normalized.dtype).reshape(-1, 1)
            scl = np.array(scales, dtype=x_normalized.dtype).reshape(-1, 1)
    
    # Step 1: Inverse affine normalization
    # Forward: normalized = (original - offset) / scale
    # Inverse: original = (normalized * scale) + offset
    x_denorm = (x_normalized * scl) + off
    
    # Step 2: Inverse log transform (only for time channel in signal mode)
    if channels == "signal" and time_transform is not None:
        # Time is channel 1
        if x_normalized.ndim == 3:  # (B, C, L)
            time_channel = x_denorm[:, 1:2, :]  # Keep dimension
        else:  # (C, L)
            time_channel = x_denorm[1:2, :]  # Keep dimension
        
        if time_transform == "ln":
            # Inverse of ln(1 + x) is exp(y) - 1
            # Forward:  y = ln(1 + x)
            # Inverse:  x = exp(y) - 1
            # Example:  y=0 → x = exp(0) - 1 = 0 (zero time preserved)
            # NOTE: Clamp to prevent overflow (exp(50) = 5e21, exp(100) = inf)
            if is_torch:
                time_channel_clamped = torch.clamp(time_channel, min=-20, max=20)
                time_original = torch.exp(time_channel_clamped) - 1.0
            else:
                time_channel_clamped = np.clip(time_channel, -20, 20)
                time_original = np.exp(time_channel_clamped) - 1.0
        elif time_transform == "log10":
            # Inverse of log10(1 + x) is 10^y - 1
            # Forward:  y = log10(1 + x)
            # Inverse:  x = 10^y - 1
            # Example:  y=0 → x = 10^0 - 1 = 0 (zero time preserved)
            # NOTE: Clamp to prevent overflow (10^20 = 1e20, 10^50 = inf)
            if is_torch:
                time_channel_clamped = torch.clamp(time_channel, min=-20, max=20)
                time_original = torch.pow(10.0, time_channel_clamped) - 1.0
            else:
                time_channel_clamped = np.clip(time_channel, -20, 20)
                time_original = np.power(10.0, time_channel_clamped) - 1.0
        else:
            raise ValueError(f"Unknown time_transform: {time_transform}")
        
        # Replace time channel with denormalized version
        if x_normalized.ndim == 3:
            x_denorm[:, 1:2, :] = time_original
        else:
            x_denorm[1:2, :] = time_original
    
    return x_denorm


def denormalize_label(
    label_normalized: Union[torch.Tensor, np.ndarray],
    label_offsets: Tuple[float, ...],
    label_scales: Tuple[float, ...]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Denormalize labels from normalized range back to original scale.
    
    Args:
        label_normalized: Normalized label tensor (B, 6) or (6,)
            [Energy, Zenith, Azimuth, X, Y, Z]
        label_offsets: Offset values for denormalization
        label_scales: Scale values for denormalization
    
    Returns:
        Denormalized label tensor in original scale
        
    Example:
        >>> labels_normalized = torch.randn(4, 6)  # (B, 6)
        >>> label_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> label_scales = (1e-7, 1.0, 1.0, 0.01, 0.01, 0.01)
        >>> labels_original = denormalize_label(
        ...     labels_normalized, label_offsets, label_scales
        ... )
    """
    is_torch = isinstance(label_normalized, torch.Tensor)
    
    # Convert to tensors/arrays
    if is_torch:
        device = label_normalized.device
        if label_normalized.ndim == 2:  # (B, 6)
            off = torch.tensor(label_offsets, dtype=label_normalized.dtype, device=device).view(1, -1)
            scl = torch.tensor(label_scales, dtype=label_normalized.dtype, device=device).view(1, -1)
        else:  # (6,)
            off = torch.tensor(label_offsets, dtype=label_normalized.dtype, device=device)
            scl = torch.tensor(label_scales, dtype=label_normalized.dtype, device=device)
    else:
        if label_normalized.ndim == 2:  # (B, 6)
            off = np.array(label_offsets, dtype=label_normalized.dtype).reshape(1, -1)
            scl = np.array(label_scales, dtype=label_normalized.dtype).reshape(1, -1)
        else:  # (6,)
            off = np.array(label_offsets, dtype=label_normalized.dtype)
            scl = np.array(label_scales, dtype=label_normalized.dtype)
    
    # Inverse affine normalization
    # Forward: normalized = (original - offset) / scale
    # Inverse: original = (normalized * scale) + offset
    label_denorm = (label_normalized * scl) + off
    
    return label_denorm


def denormalize_full_event(
    x_sig_normalized: Union[torch.Tensor, np.ndarray],
    geom_normalized: Union[torch.Tensor, np.ndarray],
    label_normalized: Union[torch.Tensor, np.ndarray],
    affine_offsets: Tuple[float, ...],
    affine_scales: Tuple[float, ...],
    label_offsets: Tuple[float, ...],
    label_scales: Tuple[float, ...],
    time_transform: Optional[str] = "ln"
) -> Tuple[Union[torch.Tensor, np.ndarray], ...]:
    """
    Denormalize a complete event (signal, geometry, label) back to original scale.
    
    Args:
        x_sig_normalized: Normalized signal (B, 2, L) or (2, L)
        geom_normalized: Normalized geometry (B, 3, L) or (3, L)
        label_normalized: Normalized label (B, 6) or (6,)
        affine_offsets: Signal+geometry offsets [charge, time, x, y, z]
        affine_scales: Signal+geometry scales [charge, time, x, y, z]
        label_offsets: Label offsets [Energy, Zenith, Azimuth, X, Y, Z]
        label_scales: Label scales [Energy, Zenith, Azimuth, X, Y, Z]
        time_transform: Log transform type ("ln", "log10", or None)
    
    Returns:
        Tuple of (signal_original, geometry_original, label_original)
        
    Example:
        >>> x_sig_norm = torch.randn(4, 2, 5160)
        >>> geom_norm = torch.randn(4, 3, 5160)
        >>> label_norm = torch.randn(4, 6)
        >>> 
        >>> # From config
        >>> affine_offsets = (0.0, 0.0, 0.0, 0.0, 0.0)
        >>> affine_scales = (200.0, 10.0, 500.0, 500.0, 500.0)
        >>> label_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> label_scales = (1e-7, 1.0, 1.0, 0.01, 0.01, 0.01)
        >>> 
        >>> x_sig_orig, geom_orig, label_orig = denormalize_full_event(
        ...     x_sig_norm, geom_norm, label_norm,
        ...     affine_offsets, affine_scales,
        ...     label_offsets, label_scales,
        ...     time_transform="ln"
        ... )
    """
    # Denormalize signal (charge, time)
    x_sig_original = denormalize_signal(
        x_sig_normalized, affine_offsets, affine_scales,
        time_transform=time_transform, channels="signal"
    )
    
    # Denormalize geometry (x, y, z)
    geom_original = denormalize_signal(
        geom_normalized, affine_offsets, affine_scales,
        time_transform=None, channels="geometry"
    )
    
    # Denormalize label
    label_original = denormalize_label(
        label_normalized, label_offsets, label_scales
    )
    
    return x_sig_original, geom_original, label_original


# Convenience function for config-based denormalization
def denormalize_from_config(
    x_sig_normalized: Union[torch.Tensor, np.ndarray],
    geom_normalized: Optional[Union[torch.Tensor, np.ndarray]] = None,
    label_normalized: Optional[Union[torch.Tensor, np.ndarray]] = None,
    config = None
):
    """
    Denormalize using config object.
    
    Args:
        x_sig_normalized: Normalized signal
        geom_normalized: Normalized geometry (optional)
        label_normalized: Normalized label (optional)
        config: Config object with model parameters
    
    Returns:
        Denormalized signal (and optionally geometry, label)
    """
    if config is None:
        raise ValueError("config is required")
    
    # Extract parameters from config
    affine_offsets = config.model.affine_offsets
    affine_scales = config.model.affine_scales
    label_offsets = config.model.label_offsets
    label_scales = config.model.label_scales
    time_transform = config.model.time_transform
    
    # Denormalize signal
    x_sig_original = denormalize_signal(
        x_sig_normalized, affine_offsets, affine_scales,
        time_transform=time_transform, channels="signal"
    )
    
    results = [x_sig_original]
    
    # Optionally denormalize geometry
    if geom_normalized is not None:
        geom_original = denormalize_signal(
            geom_normalized, affine_offsets, affine_scales,
            time_transform=None, channels="geometry"
        )
        results.append(geom_original)
    
    # Optionally denormalize label
    if label_normalized is not None:
        label_original = denormalize_label(
            label_normalized, label_offsets, label_scales
        )
        results.append(label_original)
    
    if len(results) == 1:
        return results[0]
    return tuple(results)

