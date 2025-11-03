#!/usr/bin/env python3
"""
h5_distribution.py

H5 Data Distribution Visualization Script
========================================

Script for visualizing H5 data distributions and generating histograms.
Creates beautiful histograms showing charge (NPE) and time distributions
from HDF5 files with various styling options.

Usage:
    python h5_distribution.py -p data.h5 --bins 200 --log-time ln --style modern
"""

import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.h5.h5_hist import plot_hist_pair
import argparse


def main():
    parser = argparse.ArgumentParser(description="Visualize IceCube data distributions")
    parser.add_argument("-p", "--path", required=True, help="Path to HDF5 file")
    parser.add_argument("--bins", type=int, default=200, help="Number of histogram bins")
    parser.add_argument("--log-time", choices=["log10", "ln"], default="log10",
                       help="Log transformation for time values")
    parser.add_argument("--exclude-zero", action="store_true", help="Exclude zero values")
    parser.add_argument("--min-time", type=float, help="Minimum time threshold (ns)")
    parser.add_argument("--style", choices=["modern", "elegant", "classic"], default="modern",
                       help="Visual style")
    parser.add_argument("--logy", action="store_true", help="Use log-scale on y-axis")
    parser.add_argument("--figsize", type=int, nargs=2, default=(12, 8),
                       metavar=("WIDTH", "HEIGHT"), help="Figure size")
    parser.add_argument("--out", default="hist_input", help="Output file prefix")
    
    args = parser.parse_args()
    
    print(f"📊 Analyzing data from {args.path}")
    print(f"🎨 Style: {args.style}")
    print(f"🔄 Time transform: {args.log_time}")
    
    # Create histograms
    plot_hist_pair(
        h5_path=args.path,
        bins=args.bins,
        out_prefix=args.out,
        log_time_transform=args.log_time,
        exclude_zero=args.exclude_zero,
        min_time_threshold=args.min_time,
        style=args.style,
        logy=args.logy,
        figsize=tuple(args.figsize),
    )
    
    print("✅ Visualization complete!")


if __name__ == "__main__":
    main()
