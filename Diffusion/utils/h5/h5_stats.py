#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5 Stats - Statistical Analysis for HDF5 Datasets

This module provides comprehensive statistical analysis tools for HDF5 datasets
containing IceCube neutrino event data.

Usage:
    # As a library
    from utils.h5.h5_stats import analyze_h5_dataset, get_dataset_stats
    
    # As a script
    python h5_stats.py --path data.h5 --dataset input
"""

import h5py
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import matplotlib.pyplot as plt


def get_dataset_stats(
    h5_file_path: str,
    dataset_name: str,
    sample_size: Optional[int] = None,
    chunk_size: int = 10000
) -> Dict[str, Any]:
    """
    Get comprehensive statistics for a specific dataset in HDF5 file.
    
    Args:
        h5_file_path: Path to the HDF5 file
        dataset_name: Name of the dataset to analyze
        sample_size: Number of samples to analyze (None for all)
        chunk_size: Size of chunks for processing large datasets
    
    Returns:
        Dictionary containing statistical information
    """
    stats = {
        'dataset_name': dataset_name,
        'file_path': h5_file_path,
        'shape': None,
        'dtype': None,
        'total_size': 0,
        'sample_size': sample_size,
        'statistics': {},
        'distributions': {},
        'quality_metrics': {}
    }
    
    with h5py.File(h5_file_path, 'r') as f:
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in HDF5 file")
        
        dataset = f[dataset_name]
        stats['shape'] = dataset.shape
        stats['dtype'] = str(dataset.dtype)
        stats['total_size'] = dataset.size
        
        # Determine sample size
        if sample_size is None or sample_size > dataset.size:
            sample_size = min(dataset.size, 1000000)  # Max 1M samples
        
        stats['sample_size'] = sample_size
        
        # Sample data for analysis
        if dataset.size <= sample_size:
            # Load all data
            data = dataset[:]
        else:
            # Sample data
            if dataset.ndim == 1:
                indices = np.random.choice(dataset.size, sample_size, replace=False)
                data = dataset[indices]
            else:
                # For multi-dimensional arrays, sample along first dimension
                n_samples = min(sample_size, dataset.shape[0])
                indices = np.random.choice(dataset.shape[0], n_samples, replace=False)
                data = dataset[indices]
        
        # Basic statistics
        if np.issubdtype(dataset.dtype, np.number):
            stats['statistics'] = _compute_basic_stats(data)
            stats['distributions'] = _compute_distributions(data)
            stats['quality_metrics'] = _compute_quality_metrics(data)
        
        return stats


def _compute_basic_stats(data: np.ndarray) -> Dict[str, Any]:
    """Compute basic statistical measures."""
    stats = {}
    
    # Handle different array shapes
    if data.ndim == 1:
        arrays_to_analyze = [data]
        names = ['data']
    elif data.ndim == 2:
        arrays_to_analyze = [data[:, i] for i in range(data.shape[1])]
        names = [f'channel_{i}' for i in range(data.shape[1])]
    else:
        # Flatten for analysis
        arrays_to_analyze = [data.flatten()]
        names = ['flattened']
    
    for array, name in zip(arrays_to_analyze, names):
        finite_mask = np.isfinite(array)
        finite_data = array[finite_mask]
        
        if len(finite_data) == 0:
            stats[name] = {'error': 'No finite values found'}
            continue
        
        stats[name] = {
            'count': int(len(finite_data)),
            'count_total': int(len(array)),
            'finite_ratio': float(len(finite_data) / len(array)),
            'min': float(np.min(finite_data)),
            'max': float(np.max(finite_data)),
            'mean': float(np.mean(finite_data)),
            'median': float(np.median(finite_data)),
            'std': float(np.std(finite_data)),
            'var': float(np.var(finite_data)),
            'q25': float(np.percentile(finite_data, 25)),
            'q75': float(np.percentile(finite_data, 75)),
            'iqr': float(np.percentile(finite_data, 75) - np.percentile(finite_data, 25)),
            'skewness': float(_compute_skewness(finite_data)),
            'kurtosis': float(_compute_kurtosis(finite_data))
        }
        
        # Additional metrics
        stats[name].update({
            'zero_count': int(np.sum(array == 0)),
            'positive_count': int(np.sum(array > 0)),
            'negative_count': int(np.sum(array < 0)),
            'inf_count': int(np.sum(np.isinf(array))),
            'nan_count': int(np.sum(np.isnan(array)))
        })
    
    return stats


def _compute_distributions(data: np.ndarray) -> Dict[str, Any]:
    """Compute distribution characteristics."""
    distributions = {}
    
    if data.ndim == 1:
        arrays_to_analyze = [data]
        names = ['data']
    elif data.ndim == 2:
        arrays_to_analyze = [data[:, i] for i in range(data.shape[1])]
        names = [f'channel_{i}' for i in range(data.shape[1])]
    else:
        arrays_to_analyze = [data.flatten()]
        names = ['flattened']
    
    for array, name in zip(arrays_to_analyze, names):
        finite_mask = np.isfinite(array)
        finite_data = array[finite_mask]
        
        if len(finite_data) == 0:
            distributions[name] = {'error': 'No finite values found'}
            continue
        
        # Log transform analysis
        positive_mask = finite_data > 0
        if np.sum(positive_mask) > 0:
            log_data = np.log(finite_data[positive_mask])
            log10_data = np.log10(finite_data[positive_mask])
            
            distributions[name] = {
                'log_mean': float(np.mean(log_data)),
                'log_std': float(np.std(log_data)),
                'log10_mean': float(np.mean(log10_data)),
                'log10_std': float(np.std(log10_data))
            }
        else:
            distributions[name] = {'error': 'No positive values for log analysis'}
    
    return distributions


def _compute_quality_metrics(data: np.ndarray) -> Dict[str, Any]:
    """Compute data quality metrics."""
    quality = {}
    
    total_elements = data.size
    
    quality = {
        'total_elements': int(total_elements),
        'finite_elements': int(np.sum(np.isfinite(data))),
        'infinite_elements': int(np.sum(np.isinf(data))),
        'nan_elements': int(np.sum(np.isnan(data))),
        'zero_elements': int(np.sum(data == 0)),
        'positive_elements': int(np.sum(data > 0)),
        'negative_elements': int(np.sum(data < 0)),
        'finite_ratio': float(np.sum(np.isfinite(data)) / total_elements),
        'zero_ratio': float(np.sum(data == 0) / total_elements),
        'positive_ratio': float(np.sum(data > 0) / total_elements),
        'negative_ratio': float(np.sum(data < 0) / total_elements)
    }
    
    return quality


def _compute_skewness(data: np.ndarray) -> float:
    """Compute skewness of the data."""
    if len(data) < 3:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    
    return np.mean(((data - mean) / std) ** 3)


def _compute_kurtosis(data: np.ndarray) -> float:
    """Compute kurtosis of the data."""
    if len(data) < 4:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    
    return np.mean(((data - mean) / std) ** 4) - 3


def analyze_h5_dataset(
    h5_file_path: str,
    dataset_name: str,
    sample_size: Optional[int] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of an HDF5 dataset.
    
    Args:
        h5_file_path: Path to the HDF5 file
        dataset_name: Name of the dataset to analyze
        sample_size: Number of samples to analyze (None for all)
        output_path: Path to save analysis report
    
    Returns:
        Dictionary containing comprehensive analysis results
    """
    print(f"Analyzing dataset '{dataset_name}' in {h5_file_path}")
    
    # Get statistics
    stats = get_dataset_stats(h5_file_path, dataset_name, sample_size)
    
    # Print summary
    _print_stats_summary(stats)
    
    # Save report if requested
    if output_path:
        _save_stats_report(stats, output_path)
    
    return stats


def _print_stats_summary(stats: Dict[str, Any]) -> None:
    """Print a formatted summary of statistics."""
    print(f"\n{'='*60}")
    print(f"Dataset Analysis Summary")
    print(f"{'='*60}")
    print(f"Dataset: {stats['dataset_name']}")
    print(f"File: {stats['file_path']}")
    print(f"Shape: {stats['shape']}")
    print(f"Data Type: {stats['dtype']}")
    print(f"Total Size: {stats['total_size']:,}")
    print(f"Sample Size: {stats['sample_size']:,}")
    
    # Quality metrics
    quality = stats['quality_metrics']
    print(f"\nData Quality:")
    print(f"  Finite Ratio: {quality['finite_ratio']:.4f}")
    print(f"  Zero Ratio: {quality['zero_ratio']:.4f}")
    print(f"  Positive Ratio: {quality['positive_ratio']:.4f}")
    print(f"  Negative Ratio: {quality['negative_ratio']:.4f}")
    
    # Statistics for each channel
    for name, channel_stats in stats['statistics'].items():
        if 'error' in channel_stats:
            continue
            
        print(f"\n{name} Statistics:")
        print(f"  Range: [{channel_stats['min']:.3f}, {channel_stats['max']:.3f}]")
        print(f"  Mean: {channel_stats['mean']:.3f} ± {channel_stats['std']:.3f}")
        print(f"  Median: {channel_stats['median']:.3f}")
        print(f"  IQR: [{channel_stats['q25']:.3f}, {channel_stats['q75']:.3f}]")
        print(f"  Skewness: {channel_stats['skewness']:.3f}")
        print(f"  Kurtosis: {channel_stats['kurtosis']:.3f}")


def _save_stats_report(stats: Dict[str, Any], output_path: str) -> None:
    """Save statistics report to file."""
    with open(output_path, 'w') as f:
        f.write(f"Dataset Analysis Report\n")
        f.write(f"======================\n\n")
        f.write(f"Dataset: {stats['dataset_name']}\n")
        f.write(f"File: {stats['file_path']}\n")
        f.write(f"Shape: {stats['shape']}\n")
        f.write(f"Data Type: {stats['dtype']}\n")
        f.write(f"Total Size: {stats['total_size']:,}\n")
        f.write(f"Sample Size: {stats['sample_size']:,}\n\n")
        
        # Quality metrics
        quality = stats['quality_metrics']
        f.write(f"Data Quality:\n")
        for key, value in quality.items():
            f.write(f"  {key}: {value}\n")
        
        # Statistics
        f.write(f"\nStatistics:\n")
        for name, channel_stats in stats['statistics'].items():
            f.write(f"\n{name}:\n")
            for key, value in channel_stats.items():
                f.write(f"  {key}: {value}\n")
    
    print(f"Analysis report saved to: {output_path}")


def main():
    """Command line interface for H5 statistics analysis."""
    parser = argparse.ArgumentParser(
        description="Statistical analysis of HDF5 datasets",
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
        "--sample-size", "-s",
        type=int,
        default=None,
        help="Number of samples to analyze (None for all)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save analysis report"
    )
    
    args = parser.parse_args()

    # Validate input file
    if not Path(args.path).exists():
        print(f"Error: HDF5 file not found: {args.path}")
        return 1
    
    try:
        # Perform analysis
        stats = analyze_h5_dataset(
            h5_file_path=args.path,
            dataset_name=args.dataset,
            sample_size=args.sample_size,
            output_path=args.output
        )
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())