import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


LABEL_NAMES: Sequence[str] = ("Energy", "Zenith", "Azimuth", "X", "Y", "Z")
LABEL_UNITS: Sequence[str] = ("MeV", "rad", "rad", "m", "m", "m")


def _find_default_h5_file():
    """Find the first H5 file in GENESIS-data directory."""
    genesis_data_dir = Path(__file__).parent.parent / "GENESIS-data"
    if genesis_data_dir.exists():
        h5_files = list(genesis_data_dir.glob("*.h5"))
        if h5_files:
            return str(h5_files[0])
    return None


def load_labels(
    h5_path: str,
    label_key: str = "label",
    max_samples: Optional[int] = None,
    seed: int = 1234,
) -> np.ndarray:
    """
    Load (optionally sampled) label array of shape (N, 6) from an HDF5 file.
    Uses deterministic random sampling for speed on very large files.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as f:
        ds = f[label_key]
        n = ds.shape[0]
        if max_samples is None or max_samples >= n:
            arr = ds[...]
        else:
            idx = rng.choice(n, size=max_samples, replace=False)
            idx.sort()
            arr = ds[idx]
    return np.asarray(arr, dtype=np.float32)


def plot_histograms(
    labels: np.ndarray,
    out_path: str,
    bins: int = 120,
    title: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    if labels.ndim != 2 or labels.shape[1] != 6:
        raise ValueError(f"Expected labels shape (N, 6), got {labels.shape}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        x = labels[:, i]
        ax.hist(x, bins=bins, color="#4C78A8", alpha=0.85)
        ax.set_title(LABEL_NAMES[i])
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("count")
        ax.set_xlabel(f"{LABEL_UNITS[i]}")

        # Quick numeric summary in the corner
        q05, q50, q95 = np.quantile(x, [0.05, 0.5, 0.95])
        ax.text(
            1.02,
            0.98,
            f"n={len(x)}\nmin={x.min():.3g}\nq05={q05:.3g}\nmed={q50:.3g}\nq95={q95:.3g}\nmax={x.max():.3g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
            clip_on=False,
        )

    if title:
        fig.suptitle(title, fontsize=14)

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    default_h5 = _find_default_h5_file()
    
    p = argparse.ArgumentParser(
        description="Plot 6 label histograms from an HDF5 file (single figure).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "-p", "--path",
        type=str,
        default=default_h5,
        required=default_h5 is None,
        help="Path to the HDF5 file (default: first .h5 file in GENESIS-data/)"
    )
    p.add_argument("--label-key", default="label", help="Dataset key for labels (default: label)")
    p.add_argument("-o", "--out", default="label_histograms.png", help="Output image path (png/pdf/etc.)")
    p.add_argument("--bins", type=int, default=120, help="Histogram bins per subplot")
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max number of events to sample for plotting. Use 0 for ALL events (default).",
    )
    p.add_argument("--seed", type=int, default=1234, help="RNG seed for sampling")
    args = p.parse_args()

    # Set default if not provided
    if args.path is None:
        args.path = _find_default_h5_file()
        if args.path is None:
            print("Error: No H5 file found in GENESIS-data/ and --path not provided")
            sys.exit(1)

    # Validate input file
    h5_path = Path(args.path)
    if not h5_path.exists():
        print(f"Error: H5 file not found: {args.path}")
        sys.exit(1)

    max_samples = None if args.max_samples == 0 else args.max_samples
    labels = load_labels(
        str(h5_path),
        label_key=args.label_key,
        max_samples=max_samples,
        seed=args.seed,
    )
    plot_histograms(
        labels,
        out_path=args.out,
        bins=args.bins,
        title=h5_path.name,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
