"""
Data Normalization Decorator
============================

This module provides decorators for automatic data normalization.
Supports multiple normalization methods: minmax, zscore, log(1+x), and log(1+x) + minmax.

Different normalization methods:
- MinMax: Min-Max normalization to specified range (default: [0, 1])
- ZScore: Z-score normalization (mean=0, std=1)
- Log: Log transformation log(1+x)
- LogMinMax: Log transformation followed by minmax normalization

Each normalization method is implemented in its own module, similar to diffusion/schedules.

Special feature: Channel-wise normalization for signal data (B, 2, L) where:
- Channel 0: npe (number of photoelectrons)
- Channel 1: firstTime (first hit time)

Usage:
    # Basic usage - normalize first argument with minmax
    from utils.normalize import normalize
    
    @normalize(method='minmax')
    def my_function(data):
        # data is automatically normalized
        return data
    
    # Channel-wise normalization for sig (B, 2, L)
    # npe with minmax, firstTime with log_minmax
    @normalize(channel_methods=['minmax', 'log_minmax'], arg_index=0)
    def process(sig, geo, label):
        # sig[:, 0, :] (npe) normalized with minmax
        # sig[:, 1, :] (firstTime) normalized with log_minmax
        return sig, geo, label
    
    # Using dict format for clarity
    @normalize(
        channel_methods={'npe': 'minmax', 'firstTime': 'log_minmax'},
        feature_ranges={'npe': (0, 1), 'firstTime': (-1, 1)}
    )
    def process(sig):
        return sig
    
    # Import individual normalization functions
    from utils.normalize import apply_minmax, apply_zscore
    normalized_data = apply_minmax(data, feature_range=(0, 1))
    
    # Use convenience decorators
    from utils.normalize import minmax_normalize, zscore_normalize
    @minmax_normalize(feature_range=(-1, 1))
    def process(data):
        return data
"""

# Import common utilities and decorator
from .common import (
    normalize,
    is_tensor,
    is_array,
    is_numeric_data,
)

# Import individual normalization functions
from .minmax import apply_minmax, denormalize_minmax
from .zscore import apply_zscore, denormalize_zscore
from .log import apply_log_transform, denormalize_log
from .log_minmax import apply_log_minmax, denormalize_log_minmax

# Convenience decorators
def minmax_normalize(
    arg_index=None, 
    feature_range=(0, 1), 
    normalize_all=False,
    channel_methods=None,
    feature_ranges=None
):
    """Convenience decorator for minmax normalization."""
    return normalize(
        method='minmax', 
        arg_index=arg_index, 
        feature_range=feature_range, 
        normalize_all=normalize_all,
        channel_methods=channel_methods,
        feature_ranges=feature_ranges
    )


def zscore_normalize(
    arg_index=None, 
    normalize_all=False,
    channel_methods=None
):
    """Convenience decorator for zscore normalization."""
    return normalize(
        method='zscore', 
        arg_index=arg_index, 
        normalize_all=normalize_all,
        channel_methods=channel_methods
    )


def log_normalize(
    arg_index=None, 
    normalize_all=False,
    channel_methods=None
):
    """Convenience decorator for log(1+x) transformation."""
    return normalize(
        method='log', 
        arg_index=arg_index, 
        normalize_all=normalize_all,
        channel_methods=channel_methods
    )


def log_minmax_normalize(
    arg_index=None, 
    feature_range=(0, 1), 
    normalize_all=False,
    channel_methods=None,
    feature_ranges=None
):
    """Convenience decorator for log(1+x) + minmax normalization."""
    return normalize(
        method='log_minmax', 
        arg_index=arg_index,
        feature_range=feature_range, 
        normalize_all=normalize_all,
        channel_methods=channel_methods,
        feature_ranges=feature_ranges
    )


__all__ = [
    # Main decorator
    'normalize',
    
    # Individual normalization functions
    'apply_minmax',
    'apply_zscore',
    'apply_log_transform',
    'apply_log_minmax',
    
    # Individual denormalization functions
    'denormalize_minmax',
    'denormalize_zscore',
    'denormalize_log',
    'denormalize_log_minmax',
    
    # Convenience decorators
    'minmax_normalize',
    'zscore_normalize',
    'log_normalize',
    'log_minmax_normalize',
    
    # Utility functions
    'is_tensor',
    'is_array',
    'is_numeric_data',
]
