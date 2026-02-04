#!/usr/bin/env python3
"""
Sample events by zenith bins and run sample_cfg.py for each selected event.

Flow:
  1. Load labels from HDF5 (label[1] = zenith)
  2. Split zenith range into n bins
  3. Sample m events per bin
  4. Run sample_cfg.py for each selected event

Usage (required: -c, -n, -m):
  python sample_by_zenith.py -c <checkpoint.pt> -n <n_bins> -m <m_per_bin> [options]

Options:
  -c, --checkpoint     (required) Checkpoint .pt path
  -n, --n_bins         (required) Number of zenith bins
  -m, --m_per_bin      (required) Events to sample per bin
  -d, --data_dir       Data dir with H5 files          [default: ./GENESIS-data]
  -H, --h5_file        Specific H5 file                [default: first .h5 in data_dir]
  -k, --label_key      HDF5 label key                  [default: label]
  -o, --output_dir     Output root dir                 [default: tasks/output_zenith_sampling]
  -N, --num_samples    Samples per event               [default: 1]
  -g, --gpu            GPU ID                          [default: auto]
  -W, --histogram      Save histograms                 [default: True]; use --no-histogram to disable
  -p, --cut_npe        nPE cut (vis)                   [default: 0.0]
  -t, --cut_firsttime  FirstTime cut (vis)             [default: 0.0]
  -C, --cfg_scale      CFG scale override              [default: from checkpoint]
  -s, --seed           Random seed                     [default: 42]
  -x, --skip_existing  Skip if output dir exists       [default: False]
"""

import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import h5py
from typing import List, Tuple

try:
    import argcomplete
except ImportError:
    argcomplete = None

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))


def find_h5_files(data_dir: Path) -> List[Path]:
    """Find all H5 files in the data directory."""
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {data_dir}")
    return h5_files


def load_labels(h5_path: Path, label_key: str = "label") -> Tuple[np.ndarray, int]:
    """
    Load labels from HDF5 file.
    
    Returns:
        labels: (N, 6) array with [Energy, Zenith, Azimuth, X, Y, Z]
        total_events: Total number of events
    """
    print(f"Loading labels from: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        if label_key not in f:
            raise KeyError(f"Key '{label_key}' not found in HDF5 file. Available keys: {list(f.keys())}")
        labels = np.asarray(f[label_key], dtype=np.float32)
        total_events = labels.shape[0]
    
    print(f"Loaded {total_events} events")
    print(f"Zenith range: [{labels[:, 1].min():.4f}, {labels[:, 1].max():.4f}]")
    return labels, total_events


def divide_zenith_bins(labels: np.ndarray, n_bins: int) -> List[np.ndarray]:
    """
    Divide events into n bins based on zenith (2nd column, index 1).
    
    Returns:
        List of arrays, each containing event indices for that bin
    """
    zenith = labels[:, 1]
    bin_edges = np.linspace(zenith.min(), zenith.max(), n_bins + 1)
    bin_indices = np.digitize(zenith, bin_edges) - 1
    # Handle edge case: last bin includes right edge
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bins = []
    for i in range(n_bins):
        mask = bin_indices == i
        event_indices = np.where(mask)[0]
        bins.append(event_indices)
        print(f"Bin {i+1}/{n_bins} [{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f}]: {len(event_indices)} events")
    
    return bins


def sample_from_bins(bins: List[np.ndarray], m_per_bin: int, seed: int = 42) -> List[int]:
    """
    Sample m events from each bin.
    
    Returns:
        List of selected event indices
    """
    rng = np.random.default_rng(seed)
    selected_indices = []
    
    for i, bin_indices in enumerate(bins):
        if len(bin_indices) == 0:
            print(f"Warning: Bin {i+1} is empty, skipping...")
            continue
        
        n_sample = min(m_per_bin, len(bin_indices))
        sampled = rng.choice(bin_indices, size=n_sample, replace=False)
        selected_indices.extend(sampled.tolist())
        print(f"Bin {i+1}: sampled {n_sample} events (indices: {sampled.tolist()})")
    
    return sorted(selected_indices)


def run_sample_cfg(
    checkpoint: str,
    ref_idx: int,
    output_dir: Path,
    num_samples: int = 1,
    gpu: int = None,
    histogram: bool = False,
    cut_npe: float = 0.0,
    cut_firsttime: float = 0.0,
    cfg_scale: float = None,
    additional_args: List[str] = None,
) -> bool:
    """
    Run sample_cfg.py for a specific event index.
    
    Returns:
        True if successful, False otherwise
    """
    sample_cfg_path = _ROOT / "sample_cfg.py"
    if not sample_cfg_path.exists():
        raise FileNotFoundError(f"sample_cfg.py not found at {sample_cfg_path}")
    
    cmd = [
        sys.executable,
        str(sample_cfg_path),
        "--checkpoint", checkpoint,
        "--ref_idx", str(ref_idx),
        "--output_dir", str(output_dir),
        "--num_samples", str(num_samples),
    ]
    
    if gpu is not None:
        cmd.extend(["--gpu", str(gpu)])
    
    if histogram:
        cmd.append("--histogram")
    
    if cut_npe > 0:
        cmd.extend(["--cut_npe", str(cut_npe)])
    
    if cut_firsttime > 0:
        cmd.extend(["--cut_firsttime", str(cut_firsttime)])
    
    if cfg_scale is not None:
        cmd.extend(["--cfg_scale", str(cfg_scale)])
    
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"\n{'='*80}")
    print(f"Running sample_cfg.py for event index {ref_idx}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Successfully completed event {ref_idx}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running sample_cfg.py for event {ref_idx}: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Sample events by zenith bins and run sample_cfg.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("-d", "--data_dir", type=str, default="./GENESIS-data", help="Directory containing H5 files (default: %(default)s)")
    parser.add_argument("-H", "--h5_file", type=str, default=None, help="Specific H5 file path; if not set, use first .h5 in data_dir (default: %(default)s)")
    parser.add_argument("-k", "--label_key", type=str, default="label", help="Key for labels in HDF5 file (default: %(default)s)")
    parser.add_argument("-n", "--n_bins", type=int, required=True, help="Number of zenith bins")
    parser.add_argument("-m", "--m_per_bin", type=int, required=True, help="Number of events to sample per bin")
    parser.add_argument("-o", "--output_dir", type=str, default="tasks/output_zenith_sampling", help="Base output directory (default: %(default)s)")
    parser.add_argument("-N", "--num_samples", type=int, default=1, help="Number of samples per event for sample_cfg.py (default: %(default)s)")
    parser.add_argument("-g", "--gpu", type=int, default=None, help="GPU ID (default: auto)")
    parser.add_argument("-W", "--histogram", action="store_true", default=True, help="Save histograms (default: %(default)s)")
    parser.add_argument("--no-histogram", dest="histogram", action="store_false", help="Do not save histograms")
    parser.add_argument("-p", "--cut_npe", type=float, default=0.0, help="nPE cut for visualization (default: %(default)s)")
    parser.add_argument("-t", "--cut_firsttime", type=float, default=0.0, help="FirstTime cut for visualization (default: %(default)s)")
    parser.add_argument("-C", "--cfg_scale", type=float, default=None, help="CFG scale override (default: use checkpoint)")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for sampling (default: %(default)s)")
    parser.add_argument("-x", "--skip_existing", action="store_true", default=False, help="Skip events if output directory already exists (default: %(default)s)")
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    # Find H5 file
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if args.h5_file:
        h5_path = Path(args.h5_file)
        if not h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
    else:
        h5_files = find_h5_files(data_dir)
        h5_path = h5_files[0]
        if len(h5_files) > 1:
            print(f"Warning: Multiple H5 files found, using: {h5_path}")
    
    # Load labels
    labels, total_events = load_labels(h5_path, args.label_key)
    
    # Divide into bins
    print(f"\nDividing {total_events} events into {args.n_bins} zenith bins...")
    bins = divide_zenith_bins(labels, args.n_bins)
    
    # Sample from bins
    print(f"\nSampling {args.m_per_bin} events per bin (seed={args.seed})...")
    selected_indices = sample_from_bins(bins, args.m_per_bin, seed=args.seed)
    
    print(f"\nTotal selected events: {len(selected_indices)}")
    print(f"Selected indices: {selected_indices}")
    
    # Create base output directory
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run sample_cfg.py for each selected event
    print(f"\n{'='*80}")
    print(f"Running sample_cfg.py for {len(selected_indices)} events...")
    print(f"{'='*80}\n")
    
    successful = 0
    failed = 0
    
    for idx, event_idx in enumerate(selected_indices, 1):
        # Create subdirectory for this event
        event_output_dir = base_output_dir / f"event_{event_idx:05d}"
        
        if args.skip_existing and event_output_dir.exists():
            print(f"Skipping event {event_idx} (output directory exists)")
            continue
        
        print(f"\n[{idx}/{len(selected_indices)}] Processing event {event_idx}...")
        print(f"  Zenith: {labels[event_idx, 1]:.4f}")
        print(f"  Label: {labels[event_idx]}")
        
        success = run_sample_cfg(
            checkpoint=args.checkpoint,
            ref_idx=event_idx,
            output_dir=event_output_dir,
            num_samples=args.num_samples,
            gpu=args.gpu,
            histogram=args.histogram,
            cut_npe=args.cut_npe,
            cut_firsttime=args.cut_firsttime,
            cfg_scale=args.cfg_scale,
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total events processed: {len(selected_indices)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {base_output_dir.absolute()}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
