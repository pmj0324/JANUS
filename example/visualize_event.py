#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Event Dual - Command-line script for dual plot event visualization

This script uses utils.io.h5_2_npy to load events from H5 files and
visualizes them with two side-by-side plots (firstTime and npe) using utils.vis.event_show module.

Usage:
    python visualize_event.py  # Uses default H5 file from GENESIS-data/
    python visualize_event.py --h5-path data.h5 --event-index 0 --output event_dual.png
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.io.h5_2_npy import load_event_from_h5
from utils.vis.event_show import show_event_dual_plot


def _find_default_h5_file():
    """Find the first H5 file in GENESIS-data directory."""
    genesis_data_dir = Path(__file__).parent.parent / "GENESIS-data"
    if genesis_data_dir.exists():
        h5_files = list(genesis_data_dir.glob("*.h5"))
        if h5_files:
            return str(h5_files[0])
    return None


def main():
    """Command line interface for dual plot event visualization."""
    default_h5 = _find_default_h5_file()
    
    parser = argparse.ArgumentParser(
        description="Visualize IceCube neutrino events from H5 files with dual plots (firstTime and npe)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--h5-path", "-f",
        type=str,
        default=default_h5,
        required=default_h5 is None,
        help=f"Path to H5 file (default: first .h5 file in GENESIS-data/)"
    )
    
    parser.add_argument(
        "--event-index", "-i",
        type=int,
        default=0,
        help="Event index to visualize"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="event_dual_visualization.png",
        help="Output path for the visualization (PNG/PDF/SVG). Default: event_dual_visualization.png"
    )
    
    parser.add_argument(
        "--figure-size",
        type=int,
        nargs=2,
        default=[20, 10],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches"
    )
    
    parser.add_argument(
        "--marker-size",
        type=float,
        default=20.0,
        help="Size of markers for non-zero values"
    )
    
    parser.add_argument(
        "--no-hull",
        action="store_true",
        help="Don't show detector hull outline"
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively"
    )
    
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="",
        help="Prefix for the plot title"
    )
    
    args = parser.parse_args()
    
    # Set default if not provided
    if args.h5_path is None:
        args.h5_path = _find_default_h5_file()
        if args.h5_path is None:
            print("Error: No H5 file found in GENESIS-data/ and --h5-path not provided")
            sys.exit(1)
    
    # Validate input file
    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        print(f"Error: H5 file not found: {args.h5_path}")
        sys.exit(1)
    
    try:
        # Load event data from H5 using h5_2_npy
        print(f"Loading event {args.event_index} from {h5_path}...")
        sig, geo, label = load_event_from_h5(str(h5_path), args.event_index)
        
        print(f"Loaded event data:")
        print(f"  sig shape: {sig.shape}")
        print(f"  geo shape: {geo.shape}")
        print(f"  label shape: {label.shape}")
        
        # Create dual visualization using event_show module
        print(f"Creating dual visualization...")
        fig, (ax1, ax2) = show_event_dual_plot(
            sig=sig,
            geo=geo,
            label=label,
            output_path=args.output,
            figure_size=tuple(args.figure_size),
            marker_size=args.marker_size,
            show_detector_hull=not args.no_hull,
            show=args.show,
            title_prefix=args.title_prefix
        )
        
        print("Event dual visualization completed successfully!")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
