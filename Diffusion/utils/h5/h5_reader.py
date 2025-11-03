#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5 Reader - HDF5 File Reading Utilities

This module provides comprehensive utilities for reading and analyzing HDF5 files
containing IceCube neutrino event data.

Usage:
    # As a library
    from utils.h5.h5_reader import read_h5_event, read_h5_batch, print_h5_structure
    
    # As a script
    python h5_reader.py --path data.h5 --event-index 0
"""

import h5py
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List


def print_h5_structure(h5_file_path: str) -> None:
    """
    Print the internal structure of an HDF5 file.
    
    Args:
        h5_file_path: Path to the HDF5 file
    """
    def recursively_print(name, obj):
        indent = '  ' * (name.count('/') - 1)
        if isinstance(obj, h5py.Group):
            print(f"{indent}[Group] {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}[Dataset] {name} - shape: {obj.shape}, dtype: {obj.dtype}")

    with h5py.File(h5_file_path, 'r') as f:
        print(f"HDF5 file: {h5_file_path}")
        f.visititems(recursively_print)


def read_h5_event(
    h5_file_path: str, 
    event_index: int = 0,
    keys: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Read a specific event from HDF5 file.
    
    Args:
        h5_file_path: Path to the HDF5 file
        event_index: Index of the event to read
        keys: List of keys to read. If None, reads all available keys.
    
    Returns:
        Dictionary containing event data
    """
    with h5py.File(h5_file_path, 'r') as f:
        if keys is None:
            keys = list(f.keys())
        
        event_data = {}
        for key in keys:
            if key in f:
                dataset = f[key]
                if dataset.ndim >= 1:
                    event_data[key] = dataset[event_index]
                else:
                    event_data[key] = dataset[()]
            else:
                print(f"Warning: Key '{key}' not found in HDF5 file")
        
        return event_data


def read_h5_batch(
    h5_file_path: str,
    start_index: int = 0,
    batch_size: int = 1,
    keys: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Read a batch of events from HDF5 file.
    
    Args:
        h5_file_path: Path to the HDF5 file
        start_index: Starting index of the batch
        batch_size: Number of events to read
        keys: List of keys to read. If None, reads all available keys.
    
    Returns:
        Dictionary containing batch data
    """
    with h5py.File(h5_file_path, 'r') as f:
        if keys is None:
            keys = list(f.keys())
        
        batch_data = {}
        for key in keys:
            if key in f:
                dataset = f[key]
                if dataset.ndim >= 1:
                    end_index = min(start_index + batch_size, dataset.shape[0])
                    batch_data[key] = dataset[start_index:end_index]
                else:
                    batch_data[key] = np.array([dataset[()]] * batch_size)
            else:
                print(f"Warning: Key '{key}' not found in HDF5 file")
        
        return batch_data


def get_dataset_info(h5_file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive information about all datasets in HDF5 file.
    
    Args:
        h5_file_path: Path to the HDF5 file
    
    Returns:
        Dictionary containing dataset information
    """
    info = {}
    
    with h5py.File(h5_file_path, 'r') as f:
        for key in f.keys():
            dataset = f[key]
            info[key] = {
                'shape': dataset.shape,
                'dtype': str(dataset.dtype),
                'size': dataset.size,
                'ndim': dataset.ndim,
                'chunks': dataset.chunks,
                'compression': dataset.compression,
                'compression_opts': dataset.compression_opts
            }
    
    return info


def analyze_h5_file(h5_file_path: str) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of HDF5 file.
    
    Args:
        h5_file_path: Path to the HDF5 file
    
    Returns:
        Dictionary containing analysis results
    """
    analysis = {
        'file_path': h5_file_path,
        'file_size': Path(h5_file_path).stat().st_size,
        'datasets': {},
        'summary': {}
    }
    
    with h5py.File(h5_file_path, 'r') as f:
        # Basic file info
        analysis['file_size_mb'] = analysis['file_size'] / (1024 * 1024)
        
        # Dataset analysis
        for key in f.keys():
            dataset = f[key]
            
            # Basic info
            dataset_info = {
                'shape': dataset.shape,
                'dtype': str(dataset.dtype),
                'size': dataset.size,
                'memory_size_mb': dataset.size * dataset.dtype.itemsize / (1024 * 1024)
            }
            
            # Statistical analysis for numerical data
            if np.issubdtype(dataset.dtype, np.number):
                try:
                    # Sample for large datasets
                    if dataset.size > 1000000:
                        sample_indices = np.random.choice(dataset.size, 100000, replace=False)
                        sample_data = dataset.flat[sample_indices]
                    else:
                        sample_data = dataset[:]
                    
                    dataset_info['stats'] = {
                        'min': float(np.min(sample_data)),
                        'max': float(np.max(sample_data)),
                        'mean': float(np.mean(sample_data)),
                        'std': float(np.std(sample_data)),
                        'finite_count': int(np.sum(np.isfinite(sample_data))),
                        'inf_count': int(np.sum(np.isinf(sample_data))),
                        'nan_count': int(np.sum(np.isnan(sample_data)))
                    }
                except Exception as e:
                    dataset_info['stats'] = {'error': str(e)}
            
            analysis['datasets'][key] = dataset_info
        
        # Summary
        analysis['summary'] = {
            'total_datasets': len(f.keys()),
            'total_memory_mb': sum(info['memory_size_mb'] for info in analysis['datasets'].values()),
            'largest_dataset': max(analysis['datasets'].keys(), 
                                 key=lambda k: analysis['datasets'][k]['size'])
        }
    
    return analysis


def print_analysis_report(analysis: Dict[str, Any]) -> None:
    """
    Print a formatted analysis report.
    
    Args:
        analysis: Analysis results from analyze_h5_file()
    """
    print(f"\n{'='*60}")
    print(f"HDF5 File Analysis Report")
    print(f"{'='*60}")
    print(f"File: {analysis['file_path']}")
    print(f"File Size: {analysis['file_size_mb']:.2f} MB")
    print(f"Total Datasets: {analysis['summary']['total_datasets']}")
    print(f"Total Memory: {analysis['summary']['total_memory_mb']:.2f} MB")
    print(f"Largest Dataset: {analysis['summary']['largest_dataset']}")
    
    print(f"\n{'Dataset Details':-^60}")
    for key, info in analysis['datasets'].items():
        print(f"\n{key}:")
        print(f"  Shape: {info['shape']}")
        print(f"  Dtype: {info['dtype']}")
        print(f"  Size: {info['size']:,}")
        print(f"  Memory: {info['memory_size_mb']:.2f} MB")
        
        if 'stats' in info and 'error' not in info['stats']:
            stats = info['stats']
            print(f"  Statistics:")
            print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"    Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"    Finite: {stats['finite_count']:,}")
            if stats['inf_count'] > 0:
                print(f"    Infinite: {stats['inf_count']:,}")
            if stats['nan_count'] > 0:
                print(f"    NaN: {stats['nan_count']:,}")


def main():
    """Command line interface for H5 reader utilities."""
    parser = argparse.ArgumentParser(
        description="HDF5 file reading and analysis utilities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--path", "-p",
        type=str,
        required=True,
        help="Path to the HDF5 file"
    )
    
    parser.add_argument(
        "--event-index", "-e",
        type=int,
        default=None,
        help="Index of specific event to read"
    )
    
    parser.add_argument(
        "--batch-start", "-s",
        type=int,
        default=0,
        help="Starting index for batch reading"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Number of events to read in batch"
    )
    
    parser.add_argument(
        "--keys", "-k",
        type=str,
        nargs="+",
        default=None,
        help="Specific keys to read"
    )
    
    parser.add_argument(
        "--structure",
        action="store_true",
        help="Print HDF5 file structure"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print dataset information"
    )
    
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Perform comprehensive analysis"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.path).exists():
        print(f"Error: HDF5 file not found: {args.path}")
        return 1
    
    try:
        # Print structure
        if args.structure:
            print_h5_structure(args.path)
        
        # Print dataset info
        if args.info:
            info = get_dataset_info(args.path)
            print("\nDataset Information:")
            for key, data in info.items():
                print(f"\n{key}:")
                for attr, value in data.items():
                    print(f"  {attr}: {value}")
        
        # Comprehensive analysis
        if args.analyze:
            analysis = analyze_h5_file(args.path)
            print_analysis_report(analysis)
        
        # Read specific event
        if args.event_index is not None:
            event_data = read_h5_event(args.path, args.event_index, args.keys)
            print(f"\nEvent {args.event_index} data:")
            for key, data in event_data.items():
                print(f"{key}: shape={data.shape}, dtype={data.dtype}")
                if data.size <= 20:  # Show small arrays
                    print(f"  data: {data}")
                else:
                    print(f"  sample: {data.flat[:10]}...")
        
        # Read batch
        if args.batch_size > 1:
            batch_data = read_h5_batch(args.path, args.batch_start, args.batch_size, args.keys)
            print(f"\nBatch [{args.batch_start}:{args.batch_start + args.batch_size}] data:")
            for key, data in batch_data.items():
                print(f"{key}: shape={data.shape}, dtype={data.dtype}")
        
        # If no specific action requested, show structure by default
        if not any([args.structure, args.info, args.analyze, args.event_index is not None, args.batch_size > 1]):
            print_h5_structure(args.path)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())