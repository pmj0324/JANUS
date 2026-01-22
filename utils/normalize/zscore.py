#!/usr/bin/env python3
"""
Z-Score Normalization (Standardization)
========================================

Z-score normalization standardizes data to have mean=0 and std=1.

Formula: (x - mean) / std
"""

from __future__ import annotations
from typing import Union
import torch
import numpy as np
from .common import is_tensor, is_array


def apply_zscore(data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply Z-score normalization (standardization) to data.
    
    Args:
        data: Input tensor or array
    
    Returns:
        Normalized data with mean=0, std=1
    
    Examples:
        >>> import torch
        >>> data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> normalized = apply_zscore(data)
        >>> print(f"Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
        Mean: 0.0000, Std: 1.0000
    """
    if is_tensor(data):
        mean = data.mean()
        std = data.std()
        if std == 0:
            return torch.zeros_like(data)
        return (data - mean) / std
    elif is_array(data):
        mean = data.mean()
        std = data.std()
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor or numpy.ndarray")


def denormalize_zscore(
    data: Union[torch.Tensor, np.ndarray],
    mean: float,
    std: float
) -> Union[torch.Tensor, np.ndarray]:
    """
    Denormalize Z-score normalized data back to original scale.
    
    Args:
        data: Normalized data (mean=0, std=1)
        mean: Original mean value
        std: Original standard deviation
    
    Returns:
        Denormalized data in original scale
    
    Examples:
        >>> import torch
        >>> normalized = torch.tensor([-1.41, -0.71, 0.0, 0.71, 1.41])
        >>> original = denormalize_zscore(normalized, mean=3.0, std=1.41)
        >>> print(original)
        tensor([1.0000, 2.0000, 3.0000, 4.0000, 5.0000])
    """
    if is_tensor(data):
        if std == 0:
            return torch.zeros_like(data) + mean
        return data * std + mean
    elif is_array(data):
        if std == 0:
            return np.zeros_like(data) + mean
        return data * std + mean
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor or numpy.ndarray")
