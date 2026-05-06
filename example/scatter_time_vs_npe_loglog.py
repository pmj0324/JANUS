#!/usr/bin/env python3
"""Scatter/Density Plot: FirstTime vs nPE (log-log)
==================================================

Reads the HDF5 dataset (default key: "input") and plots hit-level pairs:
  - x-axis: FirstTime (ns)
  - y-axis: nPE

Plot modes:
  - scatter: downsampled scatter plot (good for quick sanity checks)
  - hexbin: hexbin density in log-log (approximate if downsampled)
  - contour: 2D histogram in log-log with filled contours (probability-like)

Usage:
  python example/scatter_time_vs_npe_loglog.py \
    --h5-path /path/to/data.h5 \
    --output time_vs_npe_contour.png \
    --mode contour
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def _shift_npe(values: np.ndarray, npe_shift: float) -> np.ndarray:
    """Apply shift to positive, finite nPE values only."""
    if npe_shift == 0.0:
        return values
    out = np.array(values, copy=True)
    mask = np.isfinite(out) & (out > 0.0)
    out[mask] += npe_shift
    return out


def _reservoir_add(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    filled: int,
    seen_total: int,
    x_new: np.ndarray,
    y_new: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Reservoir-sample new points into fixed-size (sample_x, sample_y)."""
    k = int(sample_x.shape[0])
    if x_new.size == 0:
        return filled, seen_total

    if filled < k:
        take = min(k - filled, int(x_new.size))
        if take > 0:
            sample_x[filled : filled + take] = x_new[:take]
            sample_y[filled : filled + take] = y_new[:take]
            filled += take
            seen_total += take
            x_new = x_new[take:]
            y_new = y_new[take:]

    if x_new.size > 0 and filled == k:
        m = int(x_new.size)
        i = np.arange(seen_total, seen_total + m, dtype=np.int64)
        prob = k / (i + 1.0)
        keep = rng.random(m) < prob
        keep_count = int(np.count_nonzero(keep))
        if keep_count > 0:
            replace_idx = rng.integers(0, k, size=keep_count, endpoint=False)
            sample_x[replace_idx] = x_new[keep]
            sample_y[replace_idx] = y_new[keep]
        seen_total += m

    return filled, seen_total


def _iter_valid_pairs(
    sig_ds,
    stop: int,
    chunk_size: int,
    npe_shift: float,
    desc: str,
):
    for start in tqdm(range(0, stop, chunk_size), desc=desc):
        end = min(start + chunk_size, stop)
        sig = np.asarray(sig_ds[start:end], dtype=np.float32)

        npe = _shift_npe(sig[:, 0, :].ravel(), npe_shift)
        ftime = sig[:, 1, :].ravel()

        mask = np.isfinite(npe) & np.isfinite(ftime) & (npe > 0.0) & (ftime > 0.0)
        x = ftime[mask]
        y = npe[mask]
        if x.size:
            yield x, y


def _scan_log_ranges(
    sig_ds,
    stop: int,
    chunk_size: int,
    npe_shift: float,
) -> tuple[float, float, float, float, int]:
    lx_min = np.inf
    lx_max = -np.inf
    ly_min = np.inf
    ly_max = -np.inf
    total_valid = 0

    for x, y in _iter_valid_pairs(sig_ds, stop, chunk_size, npe_shift, desc="Scanning ranges"):
        lx = np.log10(x)
        ly = np.log10(y)
        lx_min = min(lx_min, float(np.min(lx)))
        lx_max = max(lx_max, float(np.max(lx)))
        ly_min = min(ly_min, float(np.min(ly)))
        ly_max = max(ly_max, float(np.max(ly)))
        total_valid += int(x.size)

    return lx_min, lx_max, ly_min, ly_max, total_valid


def _build_hist2d_logspace(
    sig_ds,
    stop: int,
    chunk_size: int,
    npe_shift: float,
    xedges_log10: np.ndarray,
    yedges_log10: np.ndarray,
) -> np.ndarray:
    hist = np.zeros((len(xedges_log10) - 1, len(yedges_log10) - 1), dtype=np.int64)
    for x, y in _iter_valid_pairs(sig_ds, stop, chunk_size, npe_shift, desc="Building hist2d"):
        lx = np.log10(x)
        ly = np.log10(y)
        h, _, _ = np.histogram2d(lx, ly, bins=[xedges_log10, yedges_log10])
        hist += h.astype(np.int64, copy=False)
    return hist


def main() -> int:
    p = argparse.ArgumentParser(description="FirstTime vs nPE visualization on log-log axes.")
    p.add_argument("--h5-path", type=Path, required=True, help="Path to input HDF5 file")
    p.add_argument("--output", type=Path, default=Path("time_vs_npe.png"), help="Output image path")
    p.add_argument("--dataset-key", type=str, default="input", help='HDF5 dataset key (default: "input")')
    p.add_argument("--max-events", type=int, default=0, help="Max events to read (0 = all)")
    p.add_argument("--chunk-size", type=int, default=512, help="Events per read chunk")
    p.add_argument("--npe-shift", type=float, default=0.0, help="Shift added to nPE (positive values only)")

    p.add_argument(
        "--mode",
        type=str,
        default="scatter",
        choices=["scatter", "hexbin", "contour"],
        help="Plot mode",
    )
    p.add_argument("--bins", type=int, default=200, help="Bins (contour) or gridsize (hexbin)")
    p.add_argument("--levels", type=int, default=12, help="Contour levels (contour mode)")
    p.add_argument("--cmap", type=str, default="magma", help="Colormap")

    # For scatter/hexbin we often want a downsample.
    p.add_argument(
        "--max-points",
        type=int,
        default=300_000,
        help="Max points to plot in scatter/hexbin (0 = plot all; may use lots of RAM)",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for downsampling")
    p.add_argument("--marker-size", type=float, default=2.0, help="Scatter marker size")
    p.add_argument("--alpha", type=float, default=0.25, help="Scatter alpha")
    p.add_argument("--normalize", action="store_true", help="Normalize hist to probability per bin (contour)")

    p.add_argument("--dpi", type=int, default=200, help="Output DPI")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    with h5py.File(args.h5_path, "r") as f:
        if args.dataset_key not in f:
            raise KeyError(f"Dataset key not found in H5 file: {args.dataset_key!r}")
        sig_ds = f[args.dataset_key]

        total_events = int(sig_ds.shape[0])
        stop = total_events if args.max_events <= 0 else min(total_events, args.max_events)

        if stop <= 0:
            raise RuntimeError("No events to read (stop <= 0).")

        if args.mode == "contour":
            lx_min, lx_max, ly_min, ly_max, total_valid = _scan_log_ranges(
                sig_ds,
                stop=stop,
                chunk_size=args.chunk_size,
                npe_shift=args.npe_shift,
            )
            if not np.isfinite(lx_min) or not np.isfinite(ly_min) or total_valid <= 0:
                raise RuntimeError("No valid (positive, finite) FirstTime/nPE pairs found.")

            # Small padding helps avoid edge effects when almost-constant.
            if lx_min == lx_max:
                lx_max = lx_min + 1e-3
            if ly_min == ly_max:
                ly_max = ly_min + 1e-3

            xedges = np.linspace(lx_min, lx_max, args.bins + 1)
            yedges = np.linspace(ly_min, ly_max, args.bins + 1)
            hist = _build_hist2d_logspace(
                sig_ds,
                stop=stop,
                chunk_size=args.chunk_size,
                npe_shift=args.npe_shift,
                xedges_log10=xedges,
                yedges_log10=yedges,
            )

            z = hist.T.astype(np.float64, copy=False)
            if args.normalize:
                s = float(np.sum(z))
                if s > 0:
                    z /= s

            z_pos = z[z > 0]
            if z_pos.size == 0:
                raise RuntimeError("Histogram is empty after filtering; nothing to contour.")

            levels = np.geomspace(float(np.min(z_pos)), float(np.max(z_pos)), max(2, int(args.levels)))

            xc_log = 0.5 * (xedges[:-1] + xedges[1:])
            yc_log = 0.5 * (yedges[:-1] + yedges[1:])
            xc = np.power(10.0, xc_log)
            yc = np.power(10.0, yc_log)
            xx, yy = np.meshgrid(xc, yc)

            fig, ax = plt.subplots(figsize=(10, 8))
            cf = ax.contourf(xx, yy, z, levels=levels, cmap=args.cmap)
            ax.contour(xx, yy, z, levels=levels, colors="k", linewidths=0.4, alpha=0.4)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("FirstTime (ns)")
            ax.set_ylabel("nPE")
            title_stat = "prob/bin" if args.normalize else "count/bin"
            ax.set_title(f"FirstTime vs nPE (log-log) contour  {title_stat}  events={stop:,}")
            ax.grid(True, which="both", alpha=0.2, linestyle="--")
            cbar = fig.colorbar(cf, ax=ax)
            cbar.set_label(title_stat)

        else:
            all_x = []
            all_y = []
            use_reservoir = args.max_points is not None and args.max_points > 0
            if use_reservoir:
                sample_x = np.empty(args.max_points, dtype=np.float32)
                sample_y = np.empty(args.max_points, dtype=np.float32)
                filled = 0
                seen_total = 0

            total_valid = 0
            for x, y in _iter_valid_pairs(sig_ds, stop, args.chunk_size, args.npe_shift, desc="Reading hits"):
                total_valid += int(x.size)
                if use_reservoir:
                    filled, seen_total = _reservoir_add(sample_x, sample_y, filled, seen_total, x, y, rng)
                else:
                    all_x.append(x)
                    all_y.append(y)

            if use_reservoir:
                x_plot = sample_x[:filled]
                y_plot = sample_y[:filled]
            else:
                x_plot = np.concatenate(all_x) if all_x else np.array([], dtype=np.float32)
                y_plot = np.concatenate(all_y) if all_y else np.array([], dtype=np.float32)

            if x_plot.size == 0:
                raise RuntimeError("No valid (positive, finite) FirstTime/nPE pairs found to plot.")

            fig, ax = plt.subplots(figsize=(10, 8))
            if args.mode == "hexbin":
                hb = ax.hexbin(
                    x_plot,
                    y_plot,
                    gridsize=int(args.bins),
                    xscale="log",
                    yscale="log",
                    bins="log",
                    mincnt=1,
                    cmap=args.cmap,
                )
                cbar = fig.colorbar(hb, ax=ax)
                cbar.set_label("log10(count)")
            else:
                ax.scatter(
                    x_plot,
                    y_plot,
                    s=float(args.marker_size),
                    alpha=float(args.alpha),
                    color="#4C78A8",
                    edgecolors="none",
                    rasterized=True,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("FirstTime (ns)")
            ax.set_ylabel("nPE")
            ax.set_title(
                f"FirstTime vs nPE (log-log) {args.mode}  points={x_plot.size:,}  valid={total_valid:,}"
            )
            ax.grid(True, which="both", alpha=0.25, linestyle="--")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=int(args.dpi))
    print(f"Saved figure to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
