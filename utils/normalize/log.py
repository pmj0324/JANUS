#!/usr/bin/env python3
"""
Log Transformation
==================

Log transformation using log(1+x) to handle zero values gracefully.
This is equivalent to numpy.log1p() and torch.log1p().

Formula: log(1 + x)
"""

from __future__ import annotations
from typing import Union
import torch
import numpy as np
from .common import is_tensor, is_array


def apply_log_transform(data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply log(1+x) transformation to data.
    
    This transformation is useful for data with skewed distributions.
    The log1p function is used to handle zero values gracefully.
    
    Args:
        data: Input tensor or array
    
    Returns:
        Log-transformed data
    
    Examples:
        >>> import torch
        >>> data = torch.tensor([0.0, 1.0, 2.0, 3.0])
        >>> transformed = apply_log_transform(data)
        >>> print(transformed)
        tensor([0.0000, 0.6931, 1.0986, 1.3863])
    """
    if is_tensor(data):
        return torch.log1p(data)
    elif is_array(data):
        return np.log1p(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor or numpy.ndarray")


def denormalize_log(data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Denormalize log(1+x) transformed data back to original scale.
    
    Inverse of log(1+x) is exp(x) - 1.
    
    Args:
        data: Log-transformed data
    
    Returns:
        Denormalized data in original scale
    
    Examples:
        >>> import torch
        >>> transformed = torch.tensor([0.0, 0.6931, 1.0986, 1.3863])
        >>> original = denormalize_log(transformed)
        >>> print(original)
        tensor([0.0000, 1.0000, 2.0000, 3.0000])
    """
    if is_tensor(data):
        return torch.expm1(data)  # exp(x) - 1
    elif is_array(data):
        return np.expm1(data)  # exp(x) - 1
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected torch.Tensor or numpy.ndarray")
