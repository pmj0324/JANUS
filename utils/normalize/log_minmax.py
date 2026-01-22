#!/usr/bin/env python3
"""
Log + Min-Max Normalization
============================

Combines log(1+x) transformation followed by Min-Max normalization.
This is useful for data with skewed distributions that need to be normalized
to a specific range.

Process:
1. Apply log(1+x) transformation
2. Apply Min-Max normalization to specified range
"""

from __future__ import annotations
from typing import Union, Tuple, Optional
import torch
import numpy as np
from .common import is_tensor, is_array
from .log import apply_log_transform
from .minmax import apply_minmax


def apply_log_minmax(
    data: Union[torch.Tensor, np.ndarray],
    feature_range: Tuple[float, float] = (0, 1),
    data_min: Optional[float] = None,
    data_max: Optional[float] = None
) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply log(1+x) transformation followed by minmax normalization.
    
    Args:
        data: Input tensor or array
        feature_range: Target range (min, max), default (0, 1)
        data_min: Fixed minimum value for log-transformed data (if None, uses log_data.min())
        data_max: Fixed maximum value for log-transformed data (if None, uses log_data.max())
    
    Returns:
        Log-transformed and normalized data
    
    Examples:
        >>> import torch
        >>> data = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        >>> normalized = apply_log_minmax(data, feature_range=(0, 1))
        >>> print(normalized)
        # Data is first log-transformed, then minmax normalized
        
        >>> # Use fixed min/max for log-transformed data
        >>> normalized = apply_log_minmax(data, feature_range=(0, 1), data_min=0.0, data_max=np.log1p(225.0))
    """
    # First apply log transform
    log_data = apply_log_transform(data)
    # Then apply minmax with fixed min/max if provided
    return apply_minmax(log_data, feature_range, data_min=data_min, data_max=data_max)


def denormalize_log_minmax(
    data: Union[torch.Tensor, np.ndarray],
    log_data_min: float,
    log_data_max: float,
    feature_range: Tuple[float, float] = (0, 1)
) -> Union[torch.Tensor, np.ndarray]:
    """
    Denormalize log_minmax normalized data back to original scale.
    
    Process:
    1. Denormalize minmax (reverse)
    2. Denormalize log (exp(x) - 1)
    
    Args:
        data: Normalized data
        log_data_min: Minimum of log-transformed original data
        log_data_max: Maximum of log-transformed original data
        feature_range: Target range used for normalization, default (0, 1)
    
    Returns:
        Denormalized data in original scale
    
    Examples:
        >>> import torch
        >>> # First denormalize minmax, then denormalize log
        >>> normalized = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        >>> original = denormalize_log_minmax(normalized, log_data_min=0.0, log_data_max=1.609)
    """
    from .minmax import denormalize_minmax
    from .log import denormalize_log
    
    # First denormalize minmax
    log_data = denormalize_minmax(data, log_data_min, log_data_max, feature_range)
    # Then denormalize log
    return denormalize_log(log_data)
