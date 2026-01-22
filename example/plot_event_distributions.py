#!/usr/bin/env python3
"""
Plot Event Distributions
========================

Plot histograms of nPE and firstTime distributions across all events in the dataset.

Usage:
    #     (: )
    python plot_event_distributions.py \
        --config dit_linear_cosine.yaml \
        --output event_distributions.png

    #     
    python plot_event_distributions.py \
        --load-data event_distributions.npz \
        --output event_distributions_quick.png
    
    #     
    python plot_event_distributions.py \
        --config dit_linear_cosine.yaml \
        --no-save-data \
        --output event_distributions.png
    
    # Run from GENESIS root directory:
    cd /path/to/GENESIS
    python example/plot_event_distributions.py --config example/dit_linear_cosine.yaml
    
    # Or from example folder:
    cd example
    python plot_event_distributions.py --config dit_linear_cosine.yaml
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
    Process a batch of events and return histogram counts.
    
    Args:
        args: Tuple of (h5_path, start_idx, end_idx, npe_bins, ftime_bins, npe_range, ftime_range)
    
    Returns:
        Tuple of (npe_hist, ftime_hist, npe_values, ftime_values) for statistics
    """
    h5_path, start_idx, end_idx, npe_bins, ftime_bins, npe_range, ftime_range = args
    
    # Import here to avoid issues with multiprocessing
    from dataloader.h5 import H5Dataset
    
    dataset = H5Dataset(h5_path=h5_path)
    
    npe_list = []
    ftime_list = []
    
    for event_idx in range(start_idx, end_idx):
        try:
            sample_dict = dataset[event_idx]
            sig = sample_dict["sig"]  # (2, L)
            
            # Extract nPE and firstTime
            npe = sig[0, :].numpy()  # (L,)
            ftime = sig[1, :].numpy()  # (L,)
            
            # Filter valid values
            npe_valid = npe[npe > 0]
            ftime_valid = ftime[(ftime > 0) & np.isfinite(ftime)]
            
            npe_list.extend(npe_valid.tolist())
            ftime_list.extend(ftime_valid.tolist())
            
        except Exception as e:
            continue
    
    # Convert to numpy arrays
    npe_array = np.array(npe_list) if npe_list else np.array([])
    ftime_array = np.array(ftime_list) if ftime_list else np.array([])
    
    # Compute histograms with fixed bins
    npe_hist = np.histogram(npe_array, bins=npe_bins, range=npe_range)[0] if len(npe_array) > 0 else np.zeros(npe_bins)
    ftime_hist = np.histogram(ftime_array, bins=ftime_bins, range=ftime_range)[0] if len(ftime_array) > 0 else np.zeros(ftime_bins)
    
    return npe_hist, ftime_hist, npe_array, ftime_array


def plot_distributions(
    npe_hist: np.ndarray,
    ftime_hist: np.ndarray,
    npe_edges: np.ndarray,
    ftime_edges: np.ndarray,
    npe_all: np.ndarray,
    ftime_all: np.ndarray,
    output_path: Path,
    histogram_data_path: Path = None,
    title_suffix: str = ""
):
    """
    Create histogram plots for nPE and firstTime distributions.
    
    Args:
        npe_hist: Histogram counts for nPE
        ftime_hist: Histogram counts for firstTime
        npe_edges: Bin edges for nPE
        ftime_edges: Bin edges for firstTime
        npe_all: All nPE values for statistics
        ftime_all: All firstTime values for statistics
        output_path: Output file path
        histogram_data_path: Path to save histogram data (npz format)
        title_suffix: Additional title suffix
    """
    # Save histogram data for later reuse
    if histogram_data_path is not None:
        np.savez(
            str(histogram_data_path),
            npe_hist=npe_hist,
            ftime_hist=ftime_hist,
            npe_edges=npe_edges,
            ftime_edges=ftime_edges,
            npe_all=npe_all,
            ftime_all=ftime_all
        )
        print(f" Histogram data saved: {histogram_data_path}")
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top row: linear x-axis, bottom row: log x-axis
    # Left column: nPE, right column: firstTime
    
    npe_valid = npe_all[npe_all > 0]
    ftime_valid = ftime_all[(ftime_all > 0) & np.isfinite(ftime_all)]
    
    # Compute statistics
    if len(npe_valid) > 0:
        mean_npe = np.mean(npe_valid)
        median_npe = np.median(npe_valid)
        std_npe = np.std(npe_valid)
    else:
        mean_npe = median_npe = std_npe = 0.0
    
    if len(ftime_valid) > 0:
        mean_ftime = np.mean(ftime_valid)
        median_ftime = np.median(ftime_valid)
        std_ftime = np.std(ftime_valid)
    else:
        mean_ftime = median_ftime = std_ftime = 0.0
    
    # Plot 1: nPE with linear x-axis (top-left)
    ax1 = axes[0, 0]
    if len(npe_valid) > 0:
        npe_centers = (npe_edges[:-1] + npe_edges[1:]) / 2
        ax1.bar(npe_centers, npe_hist, width=np.diff(npe_edges), alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('NPE')
        ax1.set_ylabel('Count')
        ax1.set_title(f'NPE Distribution (Linear Scale) {title_suffix}')
        ax1.set_yscale('log')
        ax1.set_xscale('linear')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics lines
        ax1.axvline(mean_npe, color='red', linestyle='--', linewidth=2)
        ax1.axvline(median_npe, color='green', linestyle='--', linewidth=2)
        ax1.axvline(mean_npe + std_npe, color='orange', linestyle='--', alpha=0.7)
        ax1.axvline(mean_npe - std_npe, color='orange', linestyle='--', alpha=0.7)
        
        # Add nPE statistics text box
        npe_stats_text = f'Total: {len(npe_valid):,}\n'
        npe_stats_text += f'Mean: {mean_npe:.2f}\n'
        npe_stats_text += f'Median: {median_npe:.2f}\n'
        npe_stats_text += f'Std: {std_npe:.2f}\n'
        npe_stats_text += f'Min: {np.min(npe_valid):.2f}\n'
        npe_stats_text += f'Max: {np.max(npe_valid):.2f}'
        ax1.text(0.98, 0.98, npe_stats_text, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)
    
    # Plot 2: firstTime with linear x-axis (top-right)
    ax2 = axes[0, 1]
    if len(ftime_valid) > 0:
        ftime_centers = (ftime_edges[:-1] + ftime_edges[1:]) / 2
        ax2.bar(ftime_centers, ftime_hist, width=np.diff(ftime_edges), alpha=0.7, color='green', edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('FirstTime (ns)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'FirstTime Distribution (Linear Scale) {title_suffix}')
        ax2.set_yscale('log')
        ax2.set_xscale('linear')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics lines
        ax2.axvline(mean_ftime, color='red', linestyle='--', linewidth=2)
        ax2.axvline(median_ftime, color='green', linestyle='--', linewidth=2)
        ax2.axvline(mean_ftime + std_ftime, color='orange', linestyle='--', alpha=0.7)
        ax2.axvline(mean_ftime - std_ftime, color='orange', linestyle='--', alpha=0.7)
        
        # Add firstTime statistics text box
        ftime_stats_text = f'Total: {len(ftime_valid):,}\n'
        ftime_stats_text += f'Mean: {mean_ftime:.1f} ns\n'
        ftime_stats_text += f'Median: {median_ftime:.1f} ns\n'
        ftime_stats_text += f'Std: {std_ftime:.1f} ns\n'
        ftime_stats_text += f'Min: {np.min(ftime_valid):.1f} ns\n'
        ftime_stats_text += f'Max: {np.max(ftime_valid):.1f} ns'
        ax2.text(0.98, 0.98, ftime_stats_text, transform=ax2.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)
    
    # Plot 3: nPE with log x-axis (bottom-left)
    ax3 = axes[1, 0]
    if len(npe_valid) > 0:
        npe_centers = (npe_edges[:-1] + npe_edges[1:]) / 2
        ax3.bar(npe_centers, npe_hist, width=np.diff(npe_edges), alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('NPE')
        ax3.set_ylabel('Count')
        ax3.set_title(f'NPE Distribution (Log Scale) {title_suffix}')
        ax3.set_yscale('log')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics lines
        ax3.axvline(mean_npe, color='red', linestyle='--', linewidth=2)
        ax3.axvline(median_npe, color='green', linestyle='--', linewidth=2)
        ax3.axvline(mean_npe + std_npe, color='orange', linestyle='--', alpha=0.7)
        ax3.axvline(mean_npe - std_npe, color='orange', linestyle='--', alpha=0.7)
        
        # Add nPE statistics text box
        npe_stats_text = f'Total: {len(npe_valid):,}\n'
        npe_stats_text += f'Mean: {mean_npe:.2f}\n'
        npe_stats_text += f'Median: {median_npe:.2f}\n'
        npe_stats_text += f'Std: {std_npe:.2f}\n'
        npe_stats_text += f'Min: {np.min(npe_valid):.2f}\n'
        npe_stats_text += f'Max: {np.max(npe_valid):.2f}'
        ax3.text(0.98, 0.98, npe_stats_text, transform=ax3.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)
    
    # Plot 4: firstTime with log x-axis (bottom-right)
    ax4 = axes[1, 1]
    if len(ftime_valid) > 0:
        ftime_centers = (ftime_edges[:-1] + ftime_edges[1:]) / 2
        ax4.bar(ftime_centers, ftime_hist, width=np.diff(ftime_edges), alpha=0.7, color='green', edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('FirstTime (ns)')
        ax4.set_ylabel('Count')
        ax4.set_title(f'FirstTime Distribution (Log Scale) {title_suffix}')
        ax4.set_yscale('log')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics lines
        ax4.axvline(mean_ftime, color='red', linestyle='--', linewidth=2)
        ax4.axvline(median_ftime, color='green', linestyle='--', linewidth=2)
        ax4.axvline(mean_ftime + std_ftime, color='orange', linestyle='--', alpha=0.7)
        ax4.axvline(mean_ftime - std_ftime, color='orange', linestyle='--', alpha=0.7)
        
        # Add firstTime statistics text box
        ftime_stats_text = f'Total: {len(ftime_valid):,}\n'
        ftime_stats_text += f'Mean: {mean_ftime:.1f} ns\n'
        ftime_stats_text += f'Median: {median_ftime:.1f} ns\n'
        ftime_stats_text += f'Std: {std_ftime:.1f} ns\n'
        ftime_stats_text += f'Min: {np.min(ftime_valid):.1f} ns\n'
        ftime_stats_text += f'Max: {np.max(ftime_valid):.1f} ns'
        ax4.text(0.98, 0.98, ftime_stats_text, transform=ax4.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)
    
    # Add unified legend at the bottom
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=f'Mean (nPE: {mean_npe:.2f}, Time: {mean_ftime:.1f})'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=2, label=f'Median (nPE: {median_npe:.2f}, Time: {median_ftime:.1f})'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'±1σ (nPE: {std_npe:.2f}, Time: {std_ftime:.1f})'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02), fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for legend
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f" Histogram saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot nPE and firstTime distributions for all events",
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
        default="event_distributions.png",
        help="Output file path for histogram"
    )
    
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum number of events to process (None = all events)"
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
    
    parser.add_argument(
        "--npe-bins",
        type=int,
        default=200,
        help="Number of bins for nPE histogram"
    )
    
    parser.add_argument(
        "--ftime-bins",
        type=int,
        default=200,
        help="Number of bins for firstTime histogram"
    )
    
    parser.add_argument(
        "--no-save-data",
        dest="save_data",
        action="store_false",
        default=True,
        help="Do not save histogram data to NPZ file (default: save data)"
    )
    
    parser.add_argument(
        "--load-data",
        type=str,
        default=None,
        help="Load histogram data from NPZ file instead of computing"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" Event Distribution Analysis (Parallel Processing)")
    print("="*80)
    
    # Check if loading from saved data
    if args.load_data:
        print(f"\n Loading histogram data from: {args.load_data}")
        data = np.load(args.load_data)
        npe_hist_combined = data['npe_hist']
        ftime_hist_combined = data['ftime_hist']
        npe_edges = data['npe_edges']
        ftime_edges = data['ftime_edges']
        npe_all = data['npe_all']
        ftime_all = data['ftime_all']
        
        print(f" Loaded histogram data:")
        print(f"   nPE values: {len(npe_all):,}")
        print(f"   firstTime values: {len(ftime_all):,}")
    else:
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
        
        # First pass: determine bin ranges by sampling
        print(f"\n Sampling data to determine bin ranges...")
        sample_size = min(10000, max_events)
        sample_indices = np.linspace(0, max_events - 1, sample_size, dtype=int)
        
        npe_sample = []
        ftime_sample = []
        
        for idx in tqdm(sample_indices, desc="Sampling"):
            try:
                sample_dict = dataset[int(idx)]
                sig = sample_dict["sig"]
                npe = sig[0, :].numpy()
                ftime = sig[1, :].numpy()
                
                npe_valid = npe[npe > 0]
                ftime_valid = ftime[(ftime > 0) & np.isfinite(ftime)]
                
                npe_sample.extend(npe_valid.tolist())
                ftime_sample.extend(ftime_valid.tolist())
            except:
                continue
        
        if len(npe_sample) == 0 or len(ftime_sample) == 0:
            print(" Error: No valid data in sample!")
            sys.exit(1)
        
        # Determine bin ranges
        npe_min, npe_max = np.min(npe_sample), np.max(npe_sample)
        ftime_min, ftime_max = np.min(ftime_sample), np.max(ftime_sample)
        
        # Add small margin
        npe_range = (npe_min * 0.9, npe_max * 1.1)
        ftime_range = (ftime_min * 0.9, ftime_max * 1.1)
        
        print(f"   nPE range: [{npe_min:.2f}, {npe_max:.2f}]")
        print(f"   firstTime range: [{ftime_min:.1f}, {ftime_max:.1f}] ns")
        
        # Prepare batch arguments for parallel processing
        batch_size = args.batch_size
        num_batches = (max_events + batch_size - 1) // batch_size
        
        batch_args = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, max_events)
            batch_args.append((
                h5_path,
                start_idx,
                end_idx,
                args.npe_bins,
                args.ftime_bins,
                npe_range,
                ftime_range
            ))
        
        # Process batches in parallel
        print(f"\n Processing {num_batches} batches in parallel...")
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_event_batch, batch_args),
                total=num_batches,
                desc="Processing batches"
            ))
        
        # Combine histograms and collect all values for statistics
        npe_hist_combined = np.zeros(args.npe_bins)
        ftime_hist_combined = np.zeros(args.ftime_bins)
        npe_all_list = []
        ftime_all_list = []
        
        for npe_hist, ftime_hist, npe_vals, ftime_vals in results:
            npe_hist_combined += npe_hist
            ftime_hist_combined += ftime_hist
            if len(npe_vals) > 0:
                npe_all_list.append(npe_vals)
            if len(ftime_vals) > 0:
                ftime_all_list.append(ftime_vals)
        
        # Combine all values for statistics
        npe_all = np.concatenate(npe_all_list) if npe_all_list else np.array([])
        ftime_all = np.concatenate(ftime_all_list) if ftime_all_list else np.array([])
        
        print(f"\n Collected data:")
        print(f"   nPE values: {len(npe_all):,}")
        print(f"   firstTime values: {len(ftime_all):,}")
        
        if len(npe_all) == 0 or len(ftime_all) == 0:
            print(" Error: No valid data collected!")
            sys.exit(1)
        
        # Create bin edges
        npe_edges = np.linspace(npe_range[0], npe_range[1], args.npe_bins + 1)
        ftime_edges = np.linspace(ftime_range[0], ftime_range[1], args.ftime_bins + 1)
        
        max_events = len(npe_all)  # For title
    
    # Determine output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        # If relative, save to current directory or script directory
        if not output_path.parent.exists():
            output_path = Path(__file__).parent / output_path.name
    
    # Determine histogram data path
    histogram_data_path = None
    if args.save_data:
        histogram_data_path = output_path.with_suffix('.npz')
    
    # Create histogram
    print(f"\n Creating histograms...")
    plot_distributions(
        npe_hist=npe_hist_combined,
        ftime_hist=ftime_hist_combined,
        npe_edges=npe_edges,
        ftime_edges=ftime_edges,
        npe_all=npe_all,
        ftime_all=ftime_all,
        output_path=output_path,
        histogram_data_path=histogram_data_path,
        title_suffix=f"({max_events:,} events)"
    )
    
    print("\n" + "="*80)
    print(" Distribution analysis complete!")
    print(f" Histogram saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
