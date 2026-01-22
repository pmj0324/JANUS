#!/usr/bin/env python3
"""
Min-Max Normalization
=====================

Min-Max normalization scales data to a specified range [min, max].
Default range is [0, 1].

Formula: (x - min) / (max - min) * (max_range - min_range) + min_range
"""

from __future__ import annotations
from typing import Union, Tuple
import torch
import numpy as np
from .common import is_tensor, is_array


def apply_minmax(
    data: Union[torch.Tensor, np.ndarray], 
    feature_range: Tuple[float, float] = (0, 1)
) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply Min-Max normalization to data.
    
    Args:
        data: Input tensor or array
        feature_range: Target range (min, max), default (0, 1)
    
    Returns:
        Normalized data in the specified range
    
    Examples:
        >>> import torch
        >>> data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> normalized = apply_minmax(data, feature_range=(0, 1))
        >>> print(normalized)
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
    """
    if is_tensor(data):
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min == 0:
            # Avoid division by zero
            return torch.zeros_like(data) + feature_range[0]
        
        normalized = (data - data_min) / (data_max - data_min)
        # Scale to feature_range
        normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        return normalized
    elif is_array(data):
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min == 0:
            return np.zeros_like(data) + feature_range[0]
        
        normalized = (data - data_min) / (data_max - data_min)
        normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        return normalized
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor or numpy.ndarray")


def denormalize_minmax(
    data: Union[torch.Tensor, np.ndarray],
    data_min: float,
    data_max: float,
    feature_range: Tuple[float, float] = (0, 1)
) -> Union[torch.Tensor, np.ndarray]:
    """
    Denormalize Min-Max normalized data back to original scale.
    
    Args:
        data: Normalized data
        data_min: Original minimum value
        data_max: Original maximum value
        feature_range: Target range used for normalization, default (0, 1)
    
    Returns:
        Denormalized data in original scale
    
    Examples:
        >>> import torch
        >>> normalized = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        >>> original = denormalize_minmax(normalized, data_min=1.0, data_max=5.0)
        >>> print(original)
        tensor([1.0000, 2.0000, 3.0000, 4.0000, 5.0000])
    """
    if is_tensor(data):
        # Reverse the scaling
        scaled = (data - feature_range[0]) / (feature_range[1] - feature_range[0])
        # Reverse the normalization
        if data_max - data_min == 0:
            return torch.zeros_like(data) + data_min
        denormalized = scaled * (data_max - data_min) + data_min
        return denormalized
    elif is_array(data):
        scaled = (data - feature_range[0]) / (feature_range[1] - feature_range[0])
        if data_max - data_min == 0:
            return np.zeros_like(data) + data_min
        denormalized = scaled * (data_max - data_min) + data_min
        return denormalized
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor or numpy.ndarray")
