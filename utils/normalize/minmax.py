#!/usr/bin/env python3
"""
Min-Max Normalization
=====================

Min-Max normalization scales data to a specified range [min, max].
Default range is [0, 1].

Formula: (x - min) / (max - min) * (max_range - min_range) + min_range
"""

from __future__ import annotations
from typing import Union, Tuple, Optional, Sequence
import torch
import numpy as np
from .common import is_tensor, is_array


def _as_broadcastable_minmax(data, data_min, data_max, is_torch):
    """Reshape data_min, data_max for broadcasting with data (C, L) or (B, C, L)."""
    if is_torch:
        if data.dim() == 2:  # (C, L)
            return data_min.view(-1, 1), data_max.view(-1, 1)
        else:  # (B, C, L)
            return data_min.view(1, -1, 1), data_max.view(1, -1, 1)
    else:
        if data.ndim == 2:
            return np.reshape(data_min, (-1, 1)), np.reshape(data_max, (-1, 1))
        else:
            return np.reshape(data_min, (1, -1, 1)), np.reshape(data_max, (1, -1, 1))


def apply_minmax(
    data: Union[torch.Tensor, np.ndarray], 
    feature_range: Tuple[float, float] = (0, 1),
    data_min: Optional[Union[float, Sequence[float], np.ndarray, torch.Tensor]] = None,
    data_max: Optional[Union[float, Sequence[float], np.ndarray, torch.Tensor]] = None
) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply Min-Max normalization to data.
    
    Args:
        data: Input tensor or array (any shape, or (C, L) / (B, C, L) for per-channel min/max)
        feature_range: Target range (min, max), default (0, 1)
        data_min: Fixed minimum value(s). Scalar for global, or array-like (C,) for per-channel (geo: x,y,z).
        data_max: Fixed maximum value(s). Same shape as data_min.
    
    Returns:
        Normalized data in the specified range
    
    Examples:
        >>> import torch
        >>> data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> normalized = apply_minmax(data, feature_range=(0, 1))
        
        >>> # Per-channel (e.g. geo (3, L)) with min/max from H5
        >>> geo_min = np.array([x_min, y_min, z_min])
        >>> geo_max = np.array([x_max, y_max, z_max])
        >>> geo_norm = apply_minmax(geo, data_min=geo_min, data_max=geo_max)
    """
    # Per-channel min/max: data_min, data_max are (C,) and data is (C, L) or (B, C, L)
    if data_min is not None and data_max is not None:
        try:
            n = len(data_min)
        except TypeError:
            n = None
        if n is not None and n > 0:
            if is_tensor(data):
                if (data.dim() == 2 and data.shape[0] == n) or (data.dim() == 3 and data.shape[1] == n):
                    dm = data_min if is_tensor(data_min) else torch.as_tensor(data_min, dtype=data.dtype, device=data.device)
                    dx = data_max if is_tensor(data_max) else torch.as_tensor(data_max, dtype=data.dtype, device=data.device)
                    dm, dx = _as_broadcastable_minmax(data, dm, dx, True)
                    span = dx - dm
                    span = torch.where(span == 0, torch.ones_like(span), span)
                    normalized = (data - dm) / span
                    normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
                    return normalized
            else:
                if (data.ndim == 2 and data.shape[0] == n) or (data.ndim == 3 and data.shape[1] == n):
                    dm = np.asarray(data_min, dtype=data.dtype).reshape(-1, 1) if data.ndim == 2 else np.asarray(data_min, dtype=data.dtype).reshape(1, -1, 1)
                    dx = np.asarray(data_max, dtype=data.dtype).reshape(-1, 1) if data.ndim == 2 else np.asarray(data_max, dtype=data.dtype).reshape(1, -1, 1)
                    span = dx - dm
                    span = np.where(span == 0, 1.0, span)
                    normalized = (data - dm) / span
                    normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
                    return normalized.astype(data.dtype)
            # fall through to scalar path if shape didn't match
            if n == 1:
                data_min, data_max = (data_min[0] if hasattr(data_min, '__getitem__') else data_min), (data_max[0] if hasattr(data_max, '__getitem__') else data_max)
            else:
                data_min, data_max = float(np.min(data_min)), float(np.max(data_max))  # fallback: global min/max
        else:
            data_min = float(data_min)
            data_max = float(data_max)

    # Scalar min/max path
    if data_min is None or data_max is None:
        if is_tensor(data):
            computed_min = data.min().item() if data_min is None else (float(data_min) if not hasattr(data_min, '__len__') else float(np.min(data_min)))
            computed_max = data.max().item() if data_max is None else (float(data_max) if not hasattr(data_max, '__len__') else float(np.max(data_max)))
        else:
            computed_min = data.min() if data_min is None else (float(data_min) if not hasattr(data_min, '__len__') else float(np.min(data_min)))
            computed_max = data.max() if data_max is None else (float(data_max) if not hasattr(data_max, '__len__') else float(np.max(data_max)))
    else:
        computed_min = float(data_min)
        computed_max = float(data_max)
    
    if is_tensor(data):
        if computed_max - computed_min == 0:
            return torch.zeros_like(data) + feature_range[0]
        normalized = (data - computed_min) / (computed_max - computed_min)
        normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        return normalized
    elif is_array(data):
        if computed_max - computed_min == 0:
            return np.zeros_like(data) + feature_range[0]
        normalized = (data - computed_min) / (computed_max - computed_min)
        normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        return normalized
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor or numpy.ndarray")


def denormalize_minmax(
    data: Union[torch.Tensor, np.ndarray],
    data_min: Union[float, Sequence[float], np.ndarray, torch.Tensor],
    data_max: Union[float, Sequence[float], np.ndarray, torch.Tensor],
    feature_range: Tuple[float, float] = (0, 1)
) -> Union[torch.Tensor, np.ndarray]:
    """
    Denormalize Min-Max normalized data back to original scale.
    Supports per-channel data_min/data_max (e.g. shape (3,) for geo [x,y,z]).
    """
    try:
        n = len(data_min)
    except TypeError:
        n = None
    use_per_channel = (
        n is not None and n > 0 and
        ((is_tensor(data) and ((data.dim() == 2 and data.shape[0] == n) or (data.dim() == 3 and data.shape[1] == n))) or
         (is_array(data) and ((data.ndim == 2 and data.shape[0] == n) or (data.ndim == 3 and data.shape[1] == n))))
    )
    if use_per_channel:
        if is_tensor(data):
            dm = data_min if is_tensor(data_min) else torch.as_tensor(data_min, dtype=data.dtype, device=data.device)
            dx = data_max if is_tensor(data_max) else torch.as_tensor(data_max, dtype=data.dtype, device=data.device)
            dm, dx = _as_broadcastable_minmax(data, dm, dx, True)
            scaled = (data - feature_range[0]) / (feature_range[1] - feature_range[0])
            span = dx - dm
            span = torch.where(span == 0, torch.ones_like(span), span)
            return scaled * span + dm
        else:
            dm = np.asarray(data_min, dtype=data.dtype).reshape(-1, 1) if data.ndim == 2 else np.asarray(data_min, dtype=data.dtype).reshape(1, -1, 1)
            dx = np.asarray(data_max, dtype=data.dtype).reshape(-1, 1) if data.ndim == 2 else np.asarray(data_max, dtype=data.dtype).reshape(1, -1, 1)
            scaled = (data - feature_range[0]) / (feature_range[1] - feature_range[0])
            span = dx - dm
            span = np.where(span == 0, 1.0, span)
            return (scaled * span + dm).astype(data.dtype)
    data_min = float(np.min(data_min) if hasattr(data_min, '__len__') else data_min)
    data_max = float(np.max(data_max) if hasattr(data_max, '__len__') else data_max)
    if is_tensor(data):
        scaled = (data - feature_range[0]) / (feature_range[1] - feature_range[0])
        if data_max - data_min == 0:
            return torch.zeros_like(data) + data_min
        return scaled * (data_max - data_min) + data_min
    elif is_array(data):
        scaled = (data - feature_range[0]) / (feature_range[1] - feature_range[0])
        if data_max - data_min == 0:
            return np.zeros_like(data) + data_min
        return scaled * (data_max - data_min) + data_min
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor or numpy.ndarray")


def apply_minmax_geo(
    geo: Union[torch.Tensor, np.ndarray],
    geo_min: Union[np.ndarray, torch.Tensor, Sequence[float]],
    geo_max: Union[np.ndarray, torch.Tensor, Sequence[float]],
    feature_range: Tuple[float, float] = (0, 1),
) -> Union[torch.Tensor, np.ndarray]:
    """
    Min-Max normalize geometry (x, y, z) per channel using dataset min/max.
    geo: (3, L) or (B, 3, L); geo_min, geo_max: (3,) from H5Dataset.get_geo_minmax().
    """
    return apply_minmax(geo, feature_range=feature_range, data_min=geo_min, data_max=geo_max)


def denormalize_minmax_geo(
    geo_norm: Union[torch.Tensor, np.ndarray],
    geo_min: Union[np.ndarray, torch.Tensor, Sequence[float]],
    geo_max: Union[np.ndarray, torch.Tensor, Sequence[float]],
    feature_range: Tuple[float, float] = (0, 1),
) -> Union[torch.Tensor, np.ndarray]:
    """Denormalize geometry normalized with apply_minmax_geo."""
    return denormalize_minmax(geo_norm, geo_min, geo_max, feature_range)
