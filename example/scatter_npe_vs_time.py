#!/usr/bin/env python3
"""
Scatter Plot: nPE vs FirstTime
===============================

Create a 2D scatter plot of nPE (x-axis) vs firstTime (y-axis) for all hits across all events.

Usage:
    python scatter_npe_vs_time.py \
        --config dit_linear_cosine.yaml \
        --output scatter_npe_vs_time.png

    # With custom markers and limits
    python scatter_npe_vs_time.py \
        --config dit_linear_cosine.yaml \
        --output scatter_npe_vs_time.png \
        --marker-size 10 \
        --alpha 0.5

    # Run from GENESIS root directory:
    cd /path/to/GENESIS
    python example/scatter_npe_vs_time.py --config example/dit_linear_cosine.yaml
    
    # Or from example folder:
    cd example
    python scatter_npe_vs_time.py --config dit_linear_cosine.yaml
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory to path for imports
script_path = Path(__file__).resolve()
if script_path.parent.name == "example":
    project_root = script_path.parent.parent
else:
    project_root = script_path.parent

sys.path.insert(0, str(project_root))

import yaml
from types import SimpleNamespace

try:
    from dataloader.h5 import H5Dataset
except ImportError as e:
    print(f"Error: Could not import 'dataloader.h5': {e}")
    sys.exit(1)


def load_config_from_file(config_path: str):
    """
    Load YAML config file and return as SimpleNamespace object.
    
    Args:
        config_path: Path to YAML config file (can be relative or absolute)
    
    Returns:
        SimpleNamespace object with config attributes
    """
    config_path = Path(config_path)
    
    # If relative path, try relative to current directory first, then script directory
    if not config_path.is_absolute():
        # Try current directory
        if not config_path.exists():
            # Try relative to script directory
            script_dir = Path(__file__).parent
            alt_path = script_dir / config_path
            if alt_path.exists():
                config_path = alt_path
            # Try relative to project root
            else:
                alt_path = project_root / config_path
                if alt_path.exists():
                    config_path = alt_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert nested dicts to SimpleNamespace recursively
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d
    
    return dict_to_namespace(config_dict)


def process_event_batch(args):
    """
    Process a batch of events and return nPE and firstTime values.
    
    Args:
        args: Tuple of (h5_path, start_idx, end_idx)
    
    Returns:
        Tuple of (npe_values, ftime_values)
    """
    h5_path, start_idx, end_idx = args
    
    # Import here to avoid issues with multiprocessing
    from dataloader.h5 import H5Dataset
    
    dataset = H5Dataset(h5_path=h5_path)
    
    npe_list = []
    ftime_list = []
    
    for event_idx in range(start_idx, end_idx):
        try:
            sig, _geo, _label = dataset[event_idx]  # (sig, geo, label)
            
            # Extract nPE and firstTime
            npe = sig[0, :].numpy()  # (L,)
            ftime = sig[1, :].numpy()  # (L,)
            
            # Collect all values
            npe_list.extend(npe.tolist())
            ftime_list.extend(ftime.tolist())
            
        except Exception as e:
            continue
    
    # Convert to numpy arrays
    npe_array = np.array(npe_list) if npe_list else np.array([])
    ftime_array = np.array(ftime_list) if ftime_list else np.array([])
    
    return npe_array, ftime_array


def create_scatter_plot(
    npe_values: np.ndarray,
    ftime_values: np.ndarray,
    output_path: Path,
    figsize: tuple = (12, 8),
    marker_size: float = 20.0,
    alpha: float = 0.6,
    cmap: str = 'viridis',
    log_scale: bool = False
):
    """
    Create a 2D scatter plot of nPE vs firstTime.
    
    Args:
        npe_values: Array of nPE values (x-axis)
        ftime_values: Array of firstTime values (y-axis)
        output_path: Output file path for the plot
        figsize: Figure size in inches (width, height)
        marker_size: Size of scatter plot markers
        alpha: Transparency of markers (0-1)
        cmap: Colormap name
        log_scale: Whether to use log scale for x-axis
    """
    
    # Filter valid values
    valid_npe = npe_values[npe_values > 0]
    valid_ftime = ftime_values[(ftime_values > 0) & np.isfinite(ftime_values)]
    
    # Create pairs of valid nPE and firstTime
    # Both must be valid to be included
    mask = (npe_values > 0) & (ftime_values > 0) & np.isfinite(ftime_values)
    npe_plot = npe_values[mask]
    ftime_plot = ftime_values[mask]
    
    print(f"\n Creating scatter plot...")
    print(f"   Total points: {len(npe_plot):,}")
    print(f"   nPE range: [{np.min(npe_plot):.2f}, {np.max(npe_plot):.2f}]")
    print(f"   firstTime range: [{np.min(ftime_plot):.1f}, {np.max(ftime_plot):.1f}] ns")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot with density coloring
    scatter = ax.scatter(
        npe_plot,
        ftime_plot,
        s=marker_size,
        alpha=alpha,
        c=np.log10(npe_plot + 1),  # Color by log(nPE+1) for better visualization
        cmap=cmap,
        edgecolors='none'
    )
    
    ax.set_xlabel('nPE', fontsize=12, fontweight='bold')
    ax.set_ylabel('FirstTime (ns)', fontsize=12, fontweight='bold')
    ax.set_title(f'nPE vs FirstTime ({len(npe_plot):,} hits)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set log scale for x-axis if requested
    if log_scale:
        ax.set_xscale('log')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('log₁₀(nPE + 1)', fontsize=11, fontweight='bold')
    
    # Add statistics box
    stats_text = f'N features: {len(npe_plot):,}\n'
    stats_text += f'nPE - Mean: {np.mean(npe_plot):.2f}, Median: {np.median(npe_plot):.2f}\n'
    stats_text += f'Time - Mean: {np.mean(ftime_plot):.1f} ns, Median: {np.median(ftime_plot):.1f} ns\n'
    stats_text += f'Correlation: {np.corrcoef(npe_plot, ftime_plot)[0, 1]:.3f}'
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, family='monospace')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f" Scatter plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create 2D scatter plot of nPE vs firstTime for all hits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Default config file in example folder
    default_config = Path(__file__).parent / "dit_linear_cosine.yaml"
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=str(default_config),
        help=f"Path to YAML config file (default: {default_config})"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="scatter_npe_vs_time.png",
        help="Output file path for scatter plot"
    )
    
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum number of events to process (None = all events)"
    )
    
    parser.add_argument(
        "--marker-size",
        type=float,
        default=20.0,
        help="Size of scatter plot markers"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Transparency of markers (0-1)"
    )
    
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[12, 8],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches"
    )
    
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap name"
    )
    
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale for nPE x-axis"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of events to process in each batch"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" Scatter Plot: nPE vs FirstTime (Parallel Processing)")
    print("="*80)
    
    # Load config
    print(f"\n Loading configuration from: {args.config}")
    config = load_config_from_file(args.config)
    
    # Handle h5_path: if relative, make it relative to project root
    h5_path = config.data.h5_path
    if not os.path.isabs(h5_path):
        script_path = Path(__file__).resolve()
        if script_path.parent.name == "example":
            project_root = script_path.parent.parent
        else:
            project_root = script_path.parent
        h5_path = project_root / h5_path
        h5_path = str(h5_path.resolve())
    
    print(f" Loading dataset from: {h5_path}")
    dataset = H5Dataset(h5_path=h5_path)
    
    total_events = len(dataset)
    max_events = args.max_events if args.max_events is not None else total_events
    max_events = min(max_events, total_events)
    
    print(f" Dataset loaded: {total_events} events total")
    print(f" Processing {max_events} events...")
    
    # Determine number of workers
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    print(f" Using {num_workers} parallel workers")
    
    # Prepare batch arguments for parallel processing
    batch_size = args.batch_size
    num_batches = (max_events + batch_size - 1) // batch_size
    
    batch_args = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, max_events)
        batch_args.append((h5_path, start_idx, end_idx))
    
    # Process batches in parallel
    print(f"\n Processing {num_batches} batches in parallel...")
    npe_all_list = []
    ftime_all_list = []
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_event_batch, batch_args),
            total=num_batches,
            desc="Processing batches"
        ))
    
    # Combine all batches
    for npe_vals, ftime_vals in results:
        if len(npe_vals) > 0:
            npe_all_list.append(npe_vals)
        if len(ftime_vals) > 0:
            ftime_all_list.append(ftime_vals)
    
    # Convert to numpy arrays
    npe_all = np.concatenate(npe_all_list) if npe_all_list else np.array([])
    ftime_all = np.concatenate(ftime_all_list) if ftime_all_list else np.array([])
    
    if len(npe_all) == 0 or len(ftime_all) == 0:
        print(" Error: No valid data collected!")
        sys.exit(1)
    
    print(f"\n Collected data:")
    print(f"   Total points: {len(npe_all):,}")
    print(f"   Valid nPE values: {np.sum(npe_all > 0):,}")
    print(f"   Valid firstTime values: {np.sum((ftime_all > 0) & np.isfinite(ftime_all)):,}")
    
    # Determine output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        # If relative, save to current directory or script directory
        if not output_path.parent.exists():
            output_path = Path(__file__).parent / output_path.name
    
    # Create scatter plot
    create_scatter_plot(
        npe_values=npe_all,
        ftime_values=ftime_all,
        output_path=output_path,
        figsize=tuple(args.figsize),
        marker_size=args.marker_size,
        alpha=args.alpha,
        cmap=args.cmap,
        log_scale=args.log_scale
    )
    
    print("\n" + "="*80)
    print(" Scatter plot creation complete!")
    print(f" Output saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
