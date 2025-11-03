#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Visualization Package for GENESIS

This package provides comprehensive event visualization tools for IceCube neutrino events.
All modules are designed to be used both as libraries and standalone scripts.

Modules:
- event_show: Basic event visualization from NPZ files
- event_grid: Grid visualization showing NPE and Time separately  
- event_array: Direct visualization from NumPy arrays
- event_dataloader: Visualization integrated with dataloader classes
"""

from .event_show import show_event_from_npz
from .event_grid import show_event_grid
from .event_array import show_event_from_array
from .event_dataloader import show_event_from_dataloader
from .event_fast import plot_event_fast, plot_event_comparison_fast
from .event_dataloader_viz import visualize_event_from_dataloader

__all__ = [
    "show_event_from_npz",
    "show_event_grid", 
    "show_event_from_array",
    "show_event_from_dataloader",
    "plot_event_fast",
    "plot_event_comparison_fast",
    "visualize_event_from_dataloader"
]

__version__ = "1.0.0"
__author__ = "Minje Park"
