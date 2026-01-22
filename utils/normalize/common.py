#!/usr/bin/env python3
"""
Common Utilities for Data Normalization
========================================

Common functions used across different normalization methods.
Includes decorator logic and type checking utilities.
"""

from __future__ import annotations
from functools import wraps
from typing import Callable, Union, Optional, Tuple, Dict, List
import torch
import numpy as np


def is_tensor(data):
    """Check if data is a torch tensor."""
    return isinstance(data, torch.Tensor)


def is_array(data):
    """Check if data is a numpy array."""
    return isinstance(data, np.ndarray)


def is_numeric_data(data):
    """Check if data is a tensor or array."""
    return is_tensor(data) or is_array(data)


def get_normalization_function(
    method: str, 
    feature_range: Tuple[float, float] = (0, 1),
    data_min: Optional[float] = None,
    data_max: Optional[float] = None
):
    """
    Get normalization function by method name.
    
    Args:
        method: Normalization method name ('minmax', 'zscore', 'log', 'log_minmax')
        feature_range: Target range for minmax normalization, default (0, 1)
        data_min: Fixed minimum value for minmax normalization (if None, uses data.min())
        data_max: Fixed maximum value for minmax normalization (if None, uses data.max())
    
    Returns:
        Normalization function
    """
    if method == 'minmax':
        from .minmax import apply_minmax
        return lambda x: apply_minmax(x, feature_range, data_min=data_min, data_max=data_max)
    elif method == 'zscore':
        from .zscore import apply_zscore
        return apply_zscore
    elif method == 'log':
        from .log import apply_log_transform
        return apply_log_transform
    elif method == 'log_minmax':
        from .log_minmax import apply_log_minmax
        return lambda x: apply_log_minmax(x, feature_range, data_min=data_min, data_max=data_max)
    else:
        raise ValueError(f"Unknown normalization method: {method}. Choose from: minmax, zscore, log, log_minmax")


def get_denormalization_function(method: str, stats: dict, feature_range: Tuple[float, float] = (0, 1)):
    """
    Get denormalization function by method name.
    
    Args:
        method: Normalization method name ('minmax', 'zscore', 'log', 'log_minmax')
        stats: Statistics dictionary containing normalization parameters
        feature_range: Target range used for normalization, default (0, 1)
    
    Returns:
        Denormalization function
    """
    if method == 'minmax':
        from .minmax import denormalize_minmax
        return lambda x: denormalize_minmax(x, stats['min'], stats['max'], feature_range)
    elif method == 'zscore':
        from .zscore import denormalize_zscore
        return lambda x: denormalize_zscore(x, stats['mean'], stats['std'])
    elif method == 'log':
        from .log import denormalize_log
        return denormalize_log
    elif method == 'log_minmax':
        from .log_minmax import denormalize_log_minmax
        return lambda x: denormalize_log_minmax(x, stats['log_min'], stats['log_max'], feature_range)
    else:
        raise ValueError(f"Unknown normalization method: {method}. Choose from: minmax, zscore, log, log_minmax")


def _compute_normalization_stats(data: Union[torch.Tensor, np.ndarray], method: str) -> dict:
    """
    Compute statistics needed for denormalization.
    
    Args:
        data: Input data
        method: Normalization method
    
    Returns:
        Dictionary with statistics
    """
    stats = {}
    
    if method == 'minmax':
        if is_tensor(data):
            stats['min'] = float(data.min().item())
            stats['max'] = float(data.max().item())
        else:
            stats['min'] = float(data.min())
            stats['max'] = float(data.max())
    
    elif method == 'zscore':
        if is_tensor(data):
            stats['mean'] = float(data.mean().item())
            stats['std'] = float(data.std().item())
        else:
            stats['mean'] = float(data.mean())
            stats['std'] = float(data.std())
    
    elif method == 'log':
        # Log transform doesn't need stats for denormalization
        pass
    
    elif method == 'log_minmax':
        from .log import apply_log_transform
        log_data = apply_log_transform(data)
        if is_tensor(log_data):
            stats['log_min'] = float(log_data.min().item())
            stats['log_max'] = float(log_data.max().item())
        else:
            stats['log_min'] = float(log_data.min())
            stats['log_max'] = float(log_data.max())
    
    return stats


def _denormalize_signal_channels(
    sig: Union[torch.Tensor, np.ndarray],
    channel_methods: Union[List[str], Dict[str, str]],
    channel_stats: List[dict],
    feature_ranges: Optional[Union[List[Tuple[float, float]], Dict[str, Tuple[float, float]]]] = None
) -> Union[torch.Tensor, np.ndarray]:
    """
    Denormalize signal channels separately.
    
    Args:
        sig: Normalized signal tensor/array with shape (B, 2, L) or (2, L)
        channel_methods: List of methods for each channel [npe_method, firstTime_method]
                       or dict {'npe': method, 'firstTime': method}
        channel_stats: List of statistics dictionaries for each channel
        feature_ranges: Optional list of feature ranges for each channel or dict
    
    Returns:
        Denormalized signal with same shape
    """
    if feature_ranges is None:
        feature_ranges = [(0, 1), (0, 1)]
    
    # Convert dict to list format
    if isinstance(channel_methods, dict):
        npe_method = channel_methods.get('npe', 'minmax')
        firstTime_method = channel_methods.get('firstTime', 'minmax')
        channel_methods = [npe_method, firstTime_method]
        
        if isinstance(feature_ranges, dict):
            npe_range = feature_ranges.get('npe', (0, 1))
            firstTime_range = feature_ranges.get('firstTime', (0, 1))
            feature_ranges = [npe_range, firstTime_range]
    
    # Get denormalization functions for each channel
    npe_func = get_denormalization_function(channel_methods[0], channel_stats[0], feature_ranges[0])
    firstTime_func = get_denormalization_function(channel_methods[1], channel_stats[1], feature_ranges[1])
    
    # Handle different input shapes
    if is_tensor(sig):
        if sig.dim() == 3:  # (B, 2, L)
            denormalized_sig = sig.clone()
            denormalized_sig[:, 0, :] = npe_func(sig[:, 0, :])  # npe channel
            denormalized_sig[:, 1, :] = firstTime_func(sig[:, 1, :])  # firstTime channel
            return denormalized_sig
        elif sig.dim() == 2:  # (2, L)
            denormalized_sig = sig.clone()
            denormalized_sig[0, :] = npe_func(sig[0, :])  # npe channel
            denormalized_sig[1, :] = firstTime_func(sig[1, :])  # firstTime channel
            return denormalized_sig
        else:
            raise ValueError(f"Expected signal with shape (B, 2, L) or (2, L), got {sig.shape}")
    elif is_array(sig):
        if sig.ndim == 3:  # (B, 2, L)
            denormalized_sig = sig.copy()
            denormalized_sig[:, 0, :] = npe_func(sig[:, 0, :])  # npe channel
            denormalized_sig[:, 1, :] = firstTime_func(sig[:, 1, :])  # firstTime channel
            return denormalized_sig
        elif sig.ndim == 2:  # (2, L)
            denormalized_sig = sig.copy()
            denormalized_sig[0, :] = npe_func(sig[0, :])  # npe channel
            denormalized_sig[1, :] = firstTime_func(sig[1, :])  # firstTime channel
            return denormalized_sig
        else:
            raise ValueError(f"Expected signal with shape (B, 2, L) or (2, L), got {sig.shape}")
    else:
        raise TypeError(f"Unsupported data type: {type(sig)}. Expected torch.Tensor or numpy.ndarray")


def _normalize_signal_channels(
    sig: Union[torch.Tensor, np.ndarray],
    channel_methods: Union[List[str], Dict[str, str]],
    feature_ranges: Optional[Union[List[Tuple[float, float]], Dict[str, Tuple[float, float]]]] = None,
    channel_mins: Optional[Union[List[float], Dict[str, float]]] = None,
    channel_maxs: Optional[Union[List[float], Dict[str, float]]] = None
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize signal channels separately.
    
    Args:
        sig: Signal tensor/array with shape (B, 2, L) or (2, L)
        channel_methods: List of methods for each channel [npe_method, firstTime_method]
                       or dict {'npe': method, 'firstTime': method}
        feature_ranges: Optional list of feature ranges for each channel or dict
        channel_mins: Optional list of fixed min values [npe_min, firstTime_min] or dict
        channel_maxs: Optional list of fixed max values [npe_max, firstTime_max] or dict
    
    Returns:
        Normalized signal with same shape
    """
    if feature_ranges is None:
        feature_ranges = [(0, 1), (0, 1)]
    
    # Convert dict to list format
    if isinstance(channel_methods, dict):
        npe_method = channel_methods.get('npe', 'minmax')
        firstTime_method = channel_methods.get('firstTime', 'minmax')
        channel_methods = [npe_method, firstTime_method]
        
        if isinstance(feature_ranges, dict):
            npe_range = feature_ranges.get('npe', (0, 1))
            firstTime_range = feature_ranges.get('firstTime', (0, 1))
            feature_ranges = [npe_range, firstTime_range]
        
        if isinstance(channel_mins, dict):
            npe_min = channel_mins.get('npe')
            firstTime_min = channel_mins.get('firstTime')
            channel_mins = [npe_min, firstTime_min]
        
        if isinstance(channel_maxs, dict):
            npe_max = channel_maxs.get('npe')
            firstTime_max = channel_maxs.get('firstTime')
            channel_maxs = [npe_max, firstTime_max]
    
    # Ensure we have methods for both channels
    if len(channel_methods) < 2:
        channel_methods = channel_methods + [channel_methods[0]] * (2 - len(channel_methods))
    if len(feature_ranges) < 2:
        feature_ranges = feature_ranges + [(0, 1)] * (2 - len(feature_ranges))
    if channel_mins is not None and len(channel_mins) < 2:
        channel_mins = channel_mins + [channel_mins[0]] * (2 - len(channel_mins))
    if channel_maxs is not None and len(channel_maxs) < 2:
        channel_maxs = channel_maxs + [channel_maxs[0]] * (2 - len(channel_maxs))
    
    # Get normalization functions for each channel
    npe_min = channel_mins[0] if channel_mins is not None else None
    npe_max = channel_maxs[0] if channel_maxs is not None else None
    firstTime_min = channel_mins[1] if channel_mins is not None and len(channel_mins) > 1 else None
    firstTime_max = channel_maxs[1] if channel_maxs is not None and len(channel_maxs) > 1 else None
    
    npe_func = get_normalization_function(channel_methods[0], feature_ranges[0], data_min=npe_min, data_max=npe_max)
    firstTime_func = get_normalization_function(channel_methods[1], feature_ranges[1], data_min=firstTime_min, data_max=firstTime_max)
    
    # Handle different input shapes
    if is_tensor(sig):
        if sig.dim() == 3:  # (B, 2, L)
            normalized_sig = sig.clone()
            normalized_sig[:, 0, :] = npe_func(sig[:, 0, :])  # npe channel
            normalized_sig[:, 1, :] = firstTime_func(sig[:, 1, :])  # firstTime channel
            return normalized_sig
        elif sig.dim() == 2:  # (2, L)
            normalized_sig = sig.clone()
            normalized_sig[0, :] = npe_func(sig[0, :])  # npe channel
            normalized_sig[1, :] = firstTime_func(sig[1, :])  # firstTime channel
            return normalized_sig
        else:
            raise ValueError(f"Expected signal with shape (B, 2, L) or (2, L), got {sig.shape}")
    elif is_array(sig):
        if sig.ndim == 3:  # (B, 2, L)
            normalized_sig = sig.copy()
            normalized_sig[:, 0, :] = npe_func(sig[:, 0, :])  # npe channel
            normalized_sig[:, 1, :] = firstTime_func(sig[:, 1, :])  # firstTime channel
            return normalized_sig
        elif sig.ndim == 2:  # (2, L)
            normalized_sig = sig.copy()
            normalized_sig[0, :] = npe_func(sig[0, :])  # npe channel
            normalized_sig[1, :] = firstTime_func(sig[1, :])  # firstTime channel
            return normalized_sig
        else:
            raise ValueError(f"Expected signal with shape (B, 2, L) or (2, L), got {sig.shape}")
    else:
        raise TypeError(f"Unsupported data type: {type(sig)}. Expected torch.Tensor or numpy.ndarray")


def normalize(
    method: Optional[str] = None,
    channel_methods: Optional[Union[List[str], Dict[str, str]]] = None,
    arg_index: Optional[int] = None,
    feature_range: Tuple[float, float] = (0, 1),
    feature_ranges: Optional[Union[List[Tuple[float, float]], Dict[str, Tuple[float, float]]]] = None,
    normalize_all: bool = False,
    normalize_signal_channels: bool = False,
    denormalize: bool = False,
    channel_stats: Optional[Union[List[dict], Dict[str, dict]]] = None
):
    """
    Decorator for automatic data normalization.
    
    Args:
        method: Normalization method (used when channel_methods is None). Options:
            - 'minmax': Min-Max normalization to [feature_range[0], feature_range[1]]
            - 'zscore': Z-score normalization (mean=0, std=1)
            - 'log': Log transformation log(1+x)
            - 'log_minmax': Log transformation followed by minmax normalization
        channel_methods: For signal normalization, specify methods for each channel.
                        List format: [npe_method, firstTime_method]
                        Dict format: {'npe': method, 'firstTime': method}
                        If provided, normalize_signal_channels is automatically True.
        arg_index: Index of argument to normalize (0-based). If None, normalizes first argument.
                  If normalize_all=True, this is ignored.
        feature_range: Target range for minmax normalization, default (0, 1)
                      (used when method is specified and feature_ranges is None)
        feature_ranges: For signal normalization, specify ranges for each channel.
                       List format: [npe_range, firstTime_range]
                       Dict format: {'npe': range, 'firstTime': range}
        normalize_all: If True, normalize all tensor/array arguments. If False, only normalize
                      the argument at arg_index (or first argument if arg_index is None).
        normalize_signal_channels: If True, treat the target argument as signal (B, 2, L) and
                                  normalize each channel separately. If channel_methods is provided,
                                  this is automatically True.
        denormalize: If True, automatically denormalize the function output back to original scale.
                    The output must have the same shape as the normalized input.
        channel_stats: Pre-computed statistics for each channel (for fixed min/max values).
                      List format: [{'min': npe_min, 'max': npe_max}, {'min': firstTime_min, 'max': firstTime_max}]
                      Dict format: {'npe': {'min': min, 'max': max}, 'firstTime': {'min': min, 'max': max}}
                      If None, statistics are computed from data automatically.
    
    Returns:
        Decorator function
    
    Examples:
        # Normalize first argument with minmax
        @normalize(method='minmax')
        def process(data):
            return data
        
        # Normalize sig channels separately: npe with minmax, firstTime with log_minmax
        @normalize(channel_methods=['minmax', 'log_minmax'], arg_index=0)
        def process(sig, geo, label):
            return sig, geo, label
        
        # Using dict format
        @normalize(channel_methods={'npe': 'minmax', 'firstTime': 'log_minmax'})
        def process(sig, geo, label):
            return sig, geo, label
        
        # With custom feature ranges
        @normalize(
            channel_methods=['minmax', 'log_minmax'],
            feature_ranges=[(0, 1), (-1, 1)]
        )
        def process(sig):
            return sig
        
        # With denormalize option (auto denormalize output)
        @normalize(
            channel_methods=['minmax', 'log_minmax'],
            denormalize=True
        )
        def process_with_denorm(sig):
            # sig is normalized, output is automatically denormalized
            return sig
    """
    # Determine if we're doing channel-wise normalization
    if channel_methods is not None:
        normalize_signal_channels = True
        if method is not None:
            import warnings
            warnings.warn("Both 'method' and 'channel_methods' specified. 'channel_methods' will be used.")
    elif method is None:
        method = 'minmax'  # default
    
    # Validate methods
    valid_methods = ['minmax', 'zscore', 'log', 'log_minmax']
    
    if channel_methods is not None:
        if isinstance(channel_methods, list):
            for m in channel_methods:
                if m not in valid_methods:
                    raise ValueError(f"Unknown normalization method: {m}. Choose from {valid_methods}")
        elif isinstance(channel_methods, dict):
            for m in channel_methods.values():
                if m not in valid_methods:
                    raise ValueError(f"Unknown normalization method: {m}. Choose from {valid_methods}")
    elif method not in valid_methods:
        raise ValueError(f"Unknown normalization method: {method}. Choose from {valid_methods}")
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Storage for normalization statistics (for denormalization)
            norm_stats = None
            computed_channel_stats = None  # Computed or provided channel stats
            original_target_arg = None  # Store original data before normalization
            
            # Normalize arguments
            if normalize_all:
                # Normalize all tensor/array arguments
                normalized_args = []
                for arg in args:
                    if is_numeric_data(arg):
                        if normalize_signal_channels and channel_methods is not None:
                            # Check if this looks like a signal (B, 2, L) or (2, L)
                            if (is_tensor(arg) and arg.dim() >= 2) or (is_array(arg) and arg.ndim >= 2):
                                if (is_tensor(arg) and arg.shape[-2] == 2) or (is_array(arg) and arg.shape[-2] == 2):
                                    normalized_args.append(_normalize_signal_channels(arg, channel_methods, feature_ranges))
                                else:
                                    # Not a signal, use single method
                                    norm_func = get_normalization_function(method, feature_range)
                                    normalized_args.append(norm_func(arg))
                            else:
                                norm_func = get_normalization_function(method, feature_range)
                                normalized_args.append(norm_func(arg))
                        else:
                            norm_func = get_normalization_function(method, feature_range)
                            normalized_args.append(norm_func(arg))
                    else:
                        normalized_args.append(arg)
                args = tuple(normalized_args)
            else:
                # Normalize specific argument
                if arg_index is None:
                    target_index = 0
                else:
                    target_index = arg_index
                
                if target_index < len(args):
                    target_arg = args[target_index]
                    if is_numeric_data(target_arg):
                        if normalize_signal_channels and channel_methods is not None:
                            # Check if this looks like a signal (B, 2, L) or (2, L)
                            if (is_tensor(target_arg) and target_arg.dim() >= 2) or (is_array(target_arg) and target_arg.ndim >= 2):
                                if (is_tensor(target_arg) and target_arg.shape[-2] == 2) or (is_array(target_arg) and target_arg.shape[-2] == 2):
                                    # Store original data and stats for denormalization
                                    original_target_arg = target_arg.clone() if is_tensor(target_arg) else target_arg.copy()
                                    
                                    # Use provided channel_stats or compute from data
                                    if channel_stats is not None:
                                        # Convert dict to list format if needed
                                        if isinstance(channel_stats, dict):
                                            npe_stats = channel_stats.get('npe', {})
                                            firstTime_stats = channel_stats.get('firstTime', {})
                                            computed_channel_stats = [npe_stats, firstTime_stats]
                                        else:
                                            computed_channel_stats = list(channel_stats)
                                        # Ensure we have stats for both channels
                                        if len(computed_channel_stats) < 2:
                                            computed_channel_stats = computed_channel_stats + [computed_channel_stats[0]] * (2 - len(computed_channel_stats))
                                    else:
                                        # Compute stats from data
                                        if isinstance(channel_methods, dict):
                                            npe_method = channel_methods.get('npe', 'minmax')
                                            firstTime_method = channel_methods.get('firstTime', 'minmax')
                                        else:
                                            npe_method = channel_methods[0]
                                            firstTime_method = channel_methods[1]
                                        
                                        if target_arg.dim() == 3:  # (B, 2, L)
                                            computed_channel_stats = [
                                                _compute_normalization_stats(target_arg[:, 0, :], npe_method),
                                                _compute_normalization_stats(target_arg[:, 1, :], firstTime_method)
                                            ]
                                        else:  # (2, L)
                                            computed_channel_stats = [
                                                _compute_normalization_stats(target_arg[0, :], npe_method),
                                                _compute_normalization_stats(target_arg[1, :], firstTime_method)
                                            ]
                                    
                                    # Extract min/max from stats for normalization
                                    # For log_minmax, use log_min/log_max; for others, use min/max
                                    if isinstance(channel_methods, dict):
                                        npe_method = channel_methods.get('npe', 'minmax')
                                        firstTime_method = channel_methods.get('firstTime', 'minmax')
                                    else:
                                        npe_method = channel_methods[0]
                                        firstTime_method = channel_methods[1]
                                    
                                    if npe_method == 'log_minmax':
                                        npe_min = computed_channel_stats[0].get('log_min') if 'log_min' in computed_channel_stats[0] else None
                                        npe_max = computed_channel_stats[0].get('log_max') if 'log_max' in computed_channel_stats[0] else None
                                    else:
                                        npe_min = computed_channel_stats[0].get('min') if 'min' in computed_channel_stats[0] else None
                                        npe_max = computed_channel_stats[0].get('max') if 'max' in computed_channel_stats[0] else None
                                    
                                    if firstTime_method == 'log_minmax':
                                        firstTime_min = computed_channel_stats[1].get('log_min') if 'log_min' in computed_channel_stats[1] else None
                                        firstTime_max = computed_channel_stats[1].get('log_max') if 'log_max' in computed_channel_stats[1] else None
                                    else:
                                        firstTime_min = computed_channel_stats[1].get('min') if 'min' in computed_channel_stats[1] else None
                                        firstTime_max = computed_channel_stats[1].get('max') if 'max' in computed_channel_stats[1] else None
                                    
                                    normalized_arg = _normalize_signal_channels(
                                        target_arg, 
                                        channel_methods, 
                                        feature_ranges,
                                        channel_mins=[npe_min, firstTime_min] if npe_min is not None else None,
                                        channel_maxs=[npe_max, firstTime_max] if npe_max is not None else None
                                    )
                                    
                                    # Store stats as function attribute for access inside function
                                    # Store on the original function object, not wrapper
                                    func._normalization_stats = computed_channel_stats
                                    func._original_data = original_target_arg
                                    
                                    args = args[:target_index] + (normalized_arg,) + args[target_index + 1:]
                                else:
                                    # Not a signal, use single method
                                    if denormalize:
                                        norm_stats = _compute_normalization_stats(target_arg, method)
                                    norm_func = get_normalization_function(method, feature_range)
                                    normalized_arg = norm_func(target_arg)
                                    args = args[:target_index] + (normalized_arg,) + args[target_index + 1:]
                            else:
                                if denormalize:
                                    norm_stats = _compute_normalization_stats(target_arg, method)
                                norm_func = get_normalization_function(method, feature_range)
                                normalized_arg = norm_func(target_arg)
                                args = args[:target_index] + (normalized_arg,) + args[target_index + 1:]
                        else:
                            if denormalize:
                                norm_stats = _compute_normalization_stats(target_arg, method)
                            norm_func = get_normalization_function(method, feature_range)
                            normalized_arg = norm_func(target_arg)
                            args = args[:target_index] + (normalized_arg,) + args[target_index + 1:]
            
            # Call original function with normalized arguments
            result = func(*args, **kwargs)
            
            # Denormalize output if requested
            if denormalize and is_numeric_data(result):
                if normalize_signal_channels and channel_methods is not None and computed_channel_stats is not None:
                    # Check if result looks like a signal (B, 2, L) or (2, L)
                    if (is_tensor(result) and result.dim() >= 2) or (is_array(result) and result.ndim >= 2):
                        if (is_tensor(result) and result.shape[-2] == 2) or (is_array(result) and result.shape[-2] == 2):
                            result = _denormalize_signal_channels(result, channel_methods, computed_channel_stats, feature_ranges)
                        else:
                            # Not a signal, use single method
                            if norm_stats is not None:
                                denorm_func = get_denormalization_function(method, norm_stats, feature_range)
                                result = denorm_func(result)
                else:
                    # Single method denormalization
                    if norm_stats is not None:
                        denorm_func = get_denormalization_function(method, norm_stats, feature_range)
                        result = denorm_func(result)
            
            return result
        
        return wrapper
    return decorator
