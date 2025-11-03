#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5 Time Align - Time Alignment Utilities for HDF5 Data

This module provides utilities for aligning and processing time data in HDF5 files
containing IceCube neutrino event data.

Usage:
    # As a library
    from utils.h5.h5_time_align import align_time_data, process_time_series
    
    # As a script
    python h5_time_align.py --path data.h5 --dataset input --time-channel 1
"""

import h5py
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import matplotlib.pyplot as plt


def align_time_data(
    time_data: np.ndarray,
    method: str = "first_hit",
    threshold: float = 0.0,
    reference_time: Optional[float] = None
) -> np.ndarray:
    """
    Align time data using various methods.
    
    Args:
        time_data: Time data array (can be 1D or 2D)
        method: Alignment method ("first_hit", "median", "mean", "reference")
        threshold: Minimum time threshold for valid hits
        reference_time: Reference time for "reference" method
    
    Returns:
        Aligned time data
    """
    if time_data.ndim == 1:
        # Single event
        return _align_single_event(time_data, method, threshold, reference_time)
    elif time_data.ndim == 2:
        # Multiple events
        aligned_data = np.zeros_like(time_data)
        for i in range(time_data.shape[0]):
            aligned_data[i] = _align_single_event(
                time_data[i], method, threshold, reference_time
            )
        return aligned_data
    else:
        raise ValueError(f"Unsupported time data dimensions: {time_data.ndim}")


def _align_single_event(
    time_data: np.ndarray,
    method: str,
    threshold: float,
    reference_time: Optional[float]
) -> np.ndarray:
    """Align time data for a single event."""
    # Filter valid times
    valid_mask = (time_data >= threshold) & np.isfinite(time_data)
    valid_times = time_data[valid_mask]
    
    if len(valid_times) == 0:
        # No valid times, return original data
        return time_data
    
    # Calculate alignment offset
    if method == "first_hit":
        offset = np.min(valid_times)
    elif method == "median":
        offset = np.median(valid_times)
    elif method == "mean":
        offset = np.mean(valid_times)
    elif method == "reference":
        if reference_time is None:
            raise ValueError("Reference time must be provided for 'reference' method")
        offset = reference_time
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    # Apply alignment
    aligned_data = time_data.copy()
    aligned_data[valid_mask] = time_data[valid_mask] - offset
    
    return aligned_data


def process_time_series(
    time_data: np.ndarray,
    operations: List[str],
    **kwargs
) -> np.ndarray:
    """
    Apply various processing operations to time series data.
    
    Args:
        time_data: Time data array
        operations: List of operations to apply
        **kwargs: Additional parameters for operations
    
    Returns:
        Processed time data
    """
    processed_data = time_data.copy()
    
    for operation in operations:
        if operation == "log_transform":
            processed_data = _apply_log_transform(processed_data, **kwargs)
        elif operation == "normalize":
            processed_data = _apply_normalization(processed_data, **kwargs)
        elif operation == "filter_outliers":
            processed_data = _filter_outliers(processed_data, **kwargs)
        elif operation == "interpolate_missing":
            processed_data = _interpolate_missing(processed_data, **kwargs)
        else:
            print(f"Warning: Unknown operation '{operation}'")
    
    return processed_data


def _apply_log_transform(
    time_data: np.ndarray,
    base: str = "ln",
    offset: float = 1.0,
    threshold: float = 0.0
) -> np.ndarray:
    """Apply logarithmic transformation to time data."""
    processed_data = time_data.copy()
    
    # Apply offset and threshold
    valid_mask = (time_data >= threshold) & np.isfinite(time_data)
    processed_data[valid_mask] = time_data[valid_mask] + offset
    
    # Apply log transform
    if base == "ln":
        processed_data[valid_mask] = np.log(processed_data[valid_mask])
    elif base == "log10":
        processed_data[valid_mask] = np.log10(processed_data[valid_mask])
    else:
        raise ValueError(f"Unknown log base: {base}")
    
    return processed_data


def _apply_normalization(
    time_data: np.ndarray,
    method: str = "z_score",
    **kwargs
) -> np.ndarray:
    """Apply normalization to time data."""
    processed_data = time_data.copy()
    valid_mask = np.isfinite(time_data)
    
    if not np.any(valid_mask):
        return processed_data
    
    valid_data = time_data[valid_mask]
    
    if method == "z_score":
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        if std > 0:
            processed_data[valid_mask] = (valid_data - mean) / std
    elif method == "min_max":
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        if max_val > min_val:
            processed_data[valid_mask] = (valid_data - min_val) / (max_val - min_val)
    elif method == "robust":
        median = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median))
        if mad > 0:
            processed_data[valid_mask] = (valid_data - median) / mad
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return processed_data


def _filter_outliers(
    time_data: np.ndarray,
    method: str = "iqr",
    factor: float = 1.5,
    **kwargs
) -> np.ndarray:
    """Filter outliers from time data."""
    processed_data = time_data.copy()
    valid_mask = np.isfinite(time_data)
    
    if not np.any(valid_mask):
        return processed_data
    
    valid_data = time_data[valid_mask]
    
    if method == "iqr":
        q25 = np.percentile(valid_data, 25)
        q75 = np.percentile(valid_data, 75)
        iqr = q75 - q25
        lower_bound = q25 - factor * iqr
        upper_bound = q75 + factor * iqr
        
        outlier_mask = (valid_data < lower_bound) | (valid_data > upper_bound)
        processed_data[valid_mask][outlier_mask] = np.nan
    
    elif method == "z_score":
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        if std > 0:
            z_scores = np.abs((valid_data - mean) / std)
            outlier_mask = z_scores > factor
            processed_data[valid_mask][outlier_mask] = np.nan
    
    else:
        raise ValueError(f"Unknown outlier filtering method: {method}")
    
    return processed_data


def _interpolate_missing(
    time_data: np.ndarray,
    method: str = "linear",
    **kwargs
) -> np.ndarray:
    """Interpolate missing values in time data."""
    processed_data = time_data.copy()
    
    # Only interpolate if there are finite values
    finite_mask = np.isfinite(time_data)
    if np.sum(finite_mask) < 2:
        return processed_data
    
    # Create interpolation indices
    finite_indices = np.where(finite_mask)[0]
    all_indices = np.arange(len(time_data))
    
    # Interpolate
    if method == "linear":
        processed_data = np.interp(all_indices, finite_indices, time_data[finite_indices])
    elif method == "nearest":
        from scipy.interpolate import interp1d
        f = interp1d(finite_indices, time_data[finite_indices], 
                    kind='nearest', bounds_error=False, fill_value='extrapolate')
        processed_data = f(all_indices)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return processed_data


def analyze_time_alignment(
    h5_file_path: str,
    dataset_name: str,
    time_channel: int = 1,
    sample_size: Optional[int] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze time alignment characteristics of a dataset.
    
    Args:
        h5_file_path: Path to the HDF5 file
        dataset_name: Name of the dataset to analyze
        time_channel: Index of the time channel (0-based)
        sample_size: Number of events to analyze
        output_path: Path to save analysis results
    
    Returns:
        Dictionary containing alignment analysis results
    """
    analysis = {
        'file_path': h5_file_path,
        'dataset_name': dataset_name,
        'time_channel': time_channel,
        'sample_size': sample_size,
        'alignment_stats': {},
        'recommendations': []
    }
    
    with h5py.File(h5_file_path, 'r') as f:
        dataset = f[dataset_name]
        
        # Determine sample size
        if sample_size is None:
            sample_size = min(dataset.shape[0], 1000)  # Default to 1000 events
        
        # Sample events
        if dataset.shape[0] > sample_size:
            indices = np.random.choice(dataset.shape[0], sample_size, replace=False)
            sample_data = dataset[indices]
        else:
            sample_data = dataset[:]
        
        # Extract time channel
        if dataset.ndim == 3:  # (batch, channels, length)
            time_data = sample_data[:, time_channel, :]
        elif dataset.ndim == 2:  # (batch, length) - assume single channel
            time_data = sample_data
        else:
            raise ValueError(f"Unsupported dataset shape: {dataset.shape}")
        
        # Analyze alignment
        analysis['alignment_stats'] = _compute_alignment_stats(time_data)
        analysis['recommendations'] = _generate_alignment_recommendations(
            analysis['alignment_stats']
        )
    
    # Print analysis results
    _print_alignment_analysis(analysis)
    
    # Save results if requested
    if output_path:
        _save_alignment_analysis(analysis, output_path)
    
    return analysis


def _compute_alignment_stats(time_data: np.ndarray) -> Dict[str, Any]:
    """Compute alignment statistics for time data."""
    stats = {
        'total_events': time_data.shape[0],
        'total_pmts': time_data.shape[1],
        'first_hit_times': [],
        'hit_counts': [],
        'time_ranges': [],
        'alignment_quality': {}
    }
    
    for i in range(time_data.shape[0]):
        event_times = time_data[i]
        
        # Filter valid times
        valid_mask = (event_times > 0) & np.isfinite(event_times)
        valid_times = event_times[valid_mask]
        
        if len(valid_times) > 0:
            stats['first_hit_times'].append(np.min(valid_times))
            stats['hit_counts'].append(len(valid_times))
            stats['time_ranges'].append(np.max(valid_times) - np.min(valid_times))
        else:
            stats['first_hit_times'].append(np.nan)
            stats['hit_counts'].append(0)
            stats['time_ranges'].append(0)
    
    # Convert to arrays for analysis
    stats['first_hit_times'] = np.array(stats['first_hit_times'])
    stats['hit_counts'] = np.array(stats['hit_counts'])
    stats['time_ranges'] = np.array(stats['time_ranges'])
    
    # Compute quality metrics
    valid_first_hits = stats['first_hit_times'][np.isfinite(stats['first_hit_times'])]
    if len(valid_first_hits) > 0:
        stats['alignment_quality'] = {
            'first_hit_mean': float(np.mean(valid_first_hits)),
            'first_hit_std': float(np.std(valid_first_hits)),
            'first_hit_range': float(np.max(valid_first_hits) - np.min(valid_first_hits)),
            'hit_count_mean': float(np.mean(stats['hit_counts'])),
            'hit_count_std': float(np.std(stats['hit_counts'])),
            'time_range_mean': float(np.mean(stats['time_ranges'])),
            'time_range_std': float(np.std(stats['time_ranges']))
        }
    
    return stats


def _generate_alignment_recommendations(stats: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on alignment statistics."""
    recommendations = []
    
    quality = stats['alignment_quality']
    if not quality:
        recommendations.append("No valid time data found - check data quality")
        return recommendations
    
    # First hit time recommendations
    if quality['first_hit_std'] > quality['first_hit_mean'] * 0.5:
        recommendations.append("High variance in first hit times - consider 'first_hit' alignment")
    elif quality['first_hit_range'] > 1000:  # 1000 ns
        recommendations.append("Large range in first hit times - 'first_hit' alignment recommended")
    
    # Hit count recommendations
    if quality['hit_count_std'] > quality['hit_count_mean'] * 0.3:
        recommendations.append("High variance in hit counts - check event quality")
    
    # Time range recommendations
    if quality['time_range_mean'] < 100:  # 100 ns
        recommendations.append("Short time ranges - events may be poorly reconstructed")
    elif quality['time_range_mean'] > 10000:  # 10000 ns
        recommendations.append("Long time ranges - consider time filtering")
    
    return recommendations


def _print_alignment_analysis(analysis: Dict[str, Any]) -> None:
    """Print alignment analysis results."""
    print(f"\n{'='*60}")
    print(f"Time Alignment Analysis")
    print(f"{'='*60}")
    print(f"File: {analysis['file_path']}")
    print(f"Dataset: {analysis['dataset_name']}")
    print(f"Time Channel: {analysis['time_channel']}")
    print(f"Sample Size: {analysis['sample_size']}")
    
    stats = analysis['alignment_stats']
    quality = stats['alignment_quality']
    
    if quality:
        print(f"\nAlignment Quality:")
        print(f"  First Hit Time: {quality['first_hit_mean']:.1f} ± {quality['first_hit_std']:.1f} ns")
        print(f"  First Hit Range: {quality['first_hit_range']:.1f} ns")
        print(f"  Hit Count: {quality['hit_count_mean']:.1f} ± {quality['hit_count_std']:.1f}")
        print(f"  Time Range: {quality['time_range_mean']:.1f} ± {quality['time_range_std']:.1f} ns")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"  {i}. {rec}")


def _save_alignment_analysis(analysis: Dict[str, Any], output_path: str) -> None:
    """Save alignment analysis to file."""
    with open(output_path, 'w') as f:
        f.write("Time Alignment Analysis Report\n")
        f.write("============================\n\n")
        f.write(f"File: {analysis['file_path']}\n")
        f.write(f"Dataset: {analysis['dataset_name']}\n")
        f.write(f"Time Channel: {analysis['time_channel']}\n")
        f.write(f"Sample Size: {analysis['sample_size']}\n\n")
        
        # Write detailed statistics
        stats = analysis['alignment_stats']
        f.write("Alignment Statistics:\n")
        for key, value in stats.items():
            if key != 'alignment_quality':
                f.write(f"  {key}: {value}\n")
        
        quality = stats['alignment_quality']
        if quality:
            f.write("\nQuality Metrics:\n")
            for key, value in quality.items():
                f.write(f"  {key}: {value}\n")
        
        # Write recommendations
        f.write("\nRecommendations:\n")
        for i, rec in enumerate(analysis['recommendations'], 1):
            f.write(f"  {i}. {rec}\n")
    
    print(f"Alignment analysis saved to: {output_path}")


def main():
    """Command line interface for H5 time alignment utilities."""
    parser = argparse.ArgumentParser(
        description="Time alignment utilities for HDF5 datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--path", "-p",
        type=str,
        required=True,
        help="Path to the HDF5 file"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Name of the dataset to analyze"
    )
    
    parser.add_argument(
        "--time-channel", "-t",
        type=int,
        default=1,
        help="Index of the time channel (0-based)"
    )
    
    parser.add_argument(
        "--sample-size", "-s",
        type=int,
        default=None,
        help="Number of events to analyze"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save analysis results"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.path).exists():
        print(f"Error: HDF5 file not found: {args.path}")
        return 1
    
    try:
        # Perform alignment analysis
        analysis = analyze_time_alignment(
            h5_file_path=args.path,
            dataset_name=args.dataset,
            time_channel=args.time_channel,
            sample_size=args.sample_size,
            output_path=args.output
        )
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())