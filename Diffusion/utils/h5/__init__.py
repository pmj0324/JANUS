#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5 Utilities Package for GENESIS

This package provides comprehensive HDF5 file utilities for IceCube neutrino event data.

Modules:
- h5_hist: Beautiful histogram generation from HDF5 files
- h5_stats: Statistical analysis of HDF5 datasets
- h5_time_align: Time alignment utilities for HDF5 data
- h5_reader: HDF5 file reading utilities
- h5_add_xyz: Add XYZ coordinates to HDF5 files
- h5_replace_inf: Replace infinite values in HDF5 files
- h5_separate: Separate HDF5 datasets
"""

from .h5_hist import plot_hist_pair
from .h5_stats import analyze_h5_dataset, get_dataset_stats
from .h5_time_align import align_time_data
from .h5_reader import read_h5_event, read_h5_batch

__all__ = [
    "plot_hist_pair",
    "create_histogram_plot", 
    "analyze_h5_dataset",
    "get_dataset_stats",
    "align_time_data",
    "read_h5_event",
    "read_h5_batch"
]

__version__ = "1.0.0"
__author__ = "Minje Park"
