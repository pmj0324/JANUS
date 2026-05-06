#!/usr/bin/env python3
"""
Plot signal histograms before and after the active-only normalization.

This script mirrors the updated preprocessing:
  - nPE: add an optional shift, then log1p, then divide by the active-sample percentile
  - FirstTime: log1p, then divide by the active-sample percentile
  - inactive detectors are excluded from normalization entirely

The figure shows:
  - row 1: raw all values
  - row 2: log1p(all values)
  - row 3: raw active values
  - row 4: raw active values on log x-axis
  - row 5: log1p(active values)
  - row 6: normalized active-sample distributions

Usage:
    python example/plot_signal_normalization_histograms.py \
        --h5-path ./GENESIS-data/22644_0921_time_shift.h5 \
        --output signal_norm_hist.png \
        --npe-shift 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.insert(0, str(project_root))


def _safe_edges(x_min: float, x_max: float, num_bins: int) -> np.ndarray:
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("Cannot build histogram edges from non-finite bounds")
    if x_min == x_max:
        pad = 0.5 if x_min == 0.0 else max(abs(x_min) * 0.05, 0.5)
        x_min -= pad
        x_max += pad
    return np.linspace(x_min, x_max, num_bins + 1)


def _safe_log_edges(x_min: float, x_max: float, num_bins: int) -> np.ndarray:
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("Cannot build log histogram edges from non-finite bounds")
    if x_max <= 0.0:
        x_min, x_max = 1.0, 10.0
    else:
        x_min = max(x_min, np.finfo(np.float32).tiny)
        x_max = max(x_max, x_min * 1.0001)
    if x_min == x_max:
        x_max = x_min * 1.1
    return np.logspace(np.log10(x_min), np.log10(x_max), num_bins + 1)


def _active_log_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return active values and their log1p transform."""
    active = np.isfinite(values) & (values > 0.0)
    active_values = values[active]
    if active_values.size == 0:
        return active_values, active_values
    return active_values, np.log1p(active_values)


def _shift_npe(values: np.ndarray, npe_shift: float) -> np.ndarray:
    """Apply the configured nPE shift to positive nPE values only."""
    if npe_shift == 0.0:
        return values
    out = np.array(values, copy=True)
    mask = np.isfinite(out) & (out > 0.0)
    out[mask] += npe_shift
    return out


def _compute_active_stats(
    h5_path: Path,
    max_events: int | None,
    chunk_size: int,
    npe_shift: float,
):
    stats = {
        "npe_shift": float(npe_shift),
        "all_npe_min": np.inf,
        "all_npe_max": -np.inf,
        "all_ftime_min": np.inf,
        "all_ftime_max": -np.inf,
        "raw_npe_min": np.inf,
        "raw_npe_max": -np.inf,
        "raw_ftime_min": np.inf,
        "raw_ftime_max": -np.inf,
        "log_all_npe_min": np.inf,
        "log_all_npe_max": -np.inf,
        "log_all_ftime_min": np.inf,
        "log_all_ftime_max": -np.inf,
        "log_npe_min": np.inf,
        "log_npe_max": -np.inf,
        "log_ftime_min": np.inf,
        "log_ftime_max": -np.inf,
        "log_npe_scale": np.nan,
        "log_ftime_scale": np.nan,
        "active_npe_count": 0,
        "active_ftime_count": 0,
    }
    active_log_npe_values = []
    active_log_ftime_values = []

    with h5py.File(h5_path, "r") as f:
        sig_ds = f["input"]
        total = sig_ds.shape[0]
        stop = total if max_events is None or max_events <= 0 else min(total, max_events)

        for start in tqdm(range(0, stop, chunk_size), desc="Scanning stats"):
            end = min(start + chunk_size, stop)
            sig = np.asarray(sig_ds[start:end], dtype=np.float32)

            npe_raw = _shift_npe(sig[:, 0, :].ravel(), npe_shift)
            ftime_raw = sig[:, 1, :].ravel()

            npe_all = npe_raw[np.isfinite(npe_raw)]
            ftime_all = ftime_raw[np.isfinite(ftime_raw)]
            npe_active, npe_log = _active_log_stats(npe_raw)
            ftime_active, ftime_log = _active_log_stats(ftime_raw)
            npe_log_all = np.log1p(npe_all) if npe_all.size > 0 else npe_all
            ftime_log_all = np.log1p(ftime_all) if ftime_all.size > 0 else ftime_all

            if npe_all.size > 0:
                stats["all_npe_min"] = min(stats["all_npe_min"], float(np.min(npe_all)))
                stats["all_npe_max"] = max(stats["all_npe_max"], float(np.max(npe_all)))
                stats["log_all_npe_min"] = min(stats["log_all_npe_min"], float(np.min(npe_log_all)))
                stats["log_all_npe_max"] = max(stats["log_all_npe_max"], float(np.max(npe_log_all)))
            if ftime_all.size > 0:
                stats["all_ftime_min"] = min(stats["all_ftime_min"], float(np.min(ftime_all)))
                stats["all_ftime_max"] = max(stats["all_ftime_max"], float(np.max(ftime_all)))
                stats["log_all_ftime_min"] = min(stats["log_all_ftime_min"], float(np.min(ftime_log_all)))
                stats["log_all_ftime_max"] = max(stats["log_all_ftime_max"], float(np.max(ftime_log_all)))

            if npe_active.size > 0:
                stats["raw_npe_min"] = min(stats["raw_npe_min"], float(np.min(npe_active)))
                stats["raw_npe_max"] = max(stats["raw_npe_max"], float(np.max(npe_active)))
                stats["log_npe_min"] = min(stats["log_npe_min"], float(np.min(npe_log)))
                stats["log_npe_max"] = max(stats["log_npe_max"], float(np.max(npe_log)))
                stats["active_npe_count"] += int(npe_log.size)
                active_log_npe_values.append(npe_log)

            if ftime_active.size > 0:
                stats["raw_ftime_min"] = min(stats["raw_ftime_min"], float(np.min(ftime_active)))
                stats["raw_ftime_max"] = max(stats["raw_ftime_max"], float(np.max(ftime_active)))
                stats["log_ftime_min"] = min(stats["log_ftime_min"], float(np.min(ftime_log)))
                stats["log_ftime_max"] = max(stats["log_ftime_max"], float(np.max(ftime_log)))
                stats["active_ftime_count"] += int(ftime_log.size)
                active_log_ftime_values.append(ftime_log)

    if active_log_npe_values:
        active_log_npe = np.concatenate(active_log_npe_values)
        stats["log_npe_scale"] = float(np.percentile(active_log_npe, 99))
    if active_log_ftime_values:
        active_log_ftime = np.concatenate(active_log_ftime_values)
        stats["log_ftime_scale"] = float(np.percentile(active_log_ftime, 99))

    if stats["all_npe_min"] == np.inf:
        stats["all_npe_min"] = 0.0
        stats["all_npe_max"] = 1.0
        stats["log_all_npe_min"] = 0.0
        stats["log_all_npe_max"] = 1.0
    if stats["all_ftime_min"] == np.inf:
        stats["all_ftime_min"] = 0.0
        stats["all_ftime_max"] = 1.0
        stats["log_all_ftime_min"] = 0.0
        stats["log_all_ftime_max"] = 1.0
    if stats["raw_npe_min"] == np.inf:
        stats["raw_npe_min"] = 0.0
        stats["raw_npe_max"] = 1.0
        stats["log_npe_min"] = 0.0
        stats["log_npe_max"] = 1.0
    if stats["raw_ftime_min"] == np.inf:
        stats["raw_ftime_min"] = 0.0
        stats["raw_ftime_max"] = 1.0
        stats["log_ftime_min"] = 0.0
        stats["log_ftime_max"] = 1.0
    if not np.isfinite(stats["log_npe_scale"]) or stats["log_npe_scale"] <= 0:
        stats["log_npe_scale"] = 1.0
    if not np.isfinite(stats["log_ftime_scale"]) or stats["log_ftime_scale"] <= 0:
        stats["log_ftime_scale"] = 1.0

    return stats


def add_hist(ax, counts: np.ndarray, edges: np.ndarray, title: str, xlabel: str, color: str, density: bool = True):
    total = float(np.sum(counts))
    if total <= 0:
        ax.text(0.5, 0.5, "No finite values", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density" if density else "Count")
        return

    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    if density:
        heights = counts / (total * widths)
        ax.set_ylabel("Density")
    else:
        heights = counts
        ax.set_ylabel("Count")
    ax.bar(centers, heights, width=widths, color=color, alpha=0.8, edgecolor="black", linewidth=0.4, align="center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.25)


def collect_histograms(
    h5_path: Path,
    max_events: int | None,
    chunk_size: int,
    num_bins: int,
    npe_shift: float,
):
    stats = _compute_active_stats(
        h5_path,
        max_events=max_events,
        chunk_size=chunk_size,
        npe_shift=npe_shift,
    )

    all_npe_counts = None
    all_ftime_counts = None
    log_all_npe_counts = None
    log_all_ftime_counts = None
    raw_npe_counts = None
    raw_ftime_counts = None
    raw_log_npe_counts = None
    raw_log_ftime_counts = None
    log_npe_counts = None
    log_ftime_counts = None
    norm_npe_counts = None
    norm_ftime_counts = None
    stats["norm_npe_min"] = np.inf
    stats["norm_npe_max"] = -np.inf
    stats["norm_ftime_min"] = np.inf
    stats["norm_ftime_max"] = -np.inf

    all_npe_edges = _safe_edges(stats["all_npe_min"], stats["all_npe_max"], num_bins)
    all_ftime_edges = _safe_edges(stats["all_ftime_min"], stats["all_ftime_max"], num_bins)
    log_all_npe_edges = _safe_edges(stats["log_all_npe_min"], stats["log_all_npe_max"], num_bins)
    log_all_ftime_edges = _safe_edges(stats["log_all_ftime_min"], stats["log_all_ftime_max"], num_bins)
    raw_npe_edges = _safe_edges(stats["raw_npe_min"], stats["raw_npe_max"], num_bins)
    raw_ftime_edges = _safe_edges(stats["raw_ftime_min"], stats["raw_ftime_max"], num_bins)
    raw_log_npe_edges = _safe_log_edges(stats["raw_npe_min"], stats["raw_npe_max"], num_bins)
    raw_log_ftime_edges = _safe_log_edges(stats["raw_ftime_min"], stats["raw_ftime_max"], num_bins)
    log_npe_edges = _safe_edges(stats["log_npe_min"], stats["log_npe_max"], num_bins)
    log_ftime_edges = _safe_edges(stats["log_ftime_min"], stats["log_ftime_max"], num_bins)
    norm_npe_edges = _safe_edges(stats["log_npe_min"] / stats["log_npe_scale"], stats["log_npe_max"] / stats["log_npe_scale"], num_bins)
    norm_ftime_edges = _safe_edges(stats["log_ftime_min"] / stats["log_ftime_scale"], stats["log_ftime_max"] / stats["log_ftime_scale"], num_bins)

    with h5py.File(h5_path, "r") as f:
        sig_ds = f["input"]
        total = sig_ds.shape[0]
        stop = total if max_events is None or max_events <= 0 else min(total, max_events)

        for start in tqdm(range(0, stop, chunk_size), desc="Building histograms"):
            end = min(start + chunk_size, stop)
            sig = np.asarray(sig_ds[start:end], dtype=np.float32)
            npe_raw = _shift_npe(sig[:, 0, :].ravel(), npe_shift)
            ftime_raw = sig[:, 1, :].ravel()

            npe_all = npe_raw[np.isfinite(npe_raw)]
            ftime_all = ftime_raw[np.isfinite(ftime_raw)]
            npe_mask = np.isfinite(npe_raw) & (npe_raw > 0.0)
            ftime_mask = np.isfinite(ftime_raw) & (ftime_raw > 0.0)

            npe_active = npe_raw[npe_mask]
            ftime_active = ftime_raw[ftime_mask]
            npe_log_all = np.log1p(npe_all) if npe_all.size > 0 else npe_all
            ftime_log_all = np.log1p(ftime_all) if ftime_all.size > 0 else ftime_all
            npe_log = np.log1p(npe_active)
            ftime_log = np.log1p(ftime_active)
            npe_norm = npe_log / stats["log_npe_scale"]
            ftime_norm = ftime_log / stats["log_ftime_scale"]

            if npe_all.size > 0:
                stats["all_npe_min"] = min(stats["all_npe_min"], float(np.min(npe_all)))
                stats["all_npe_max"] = max(stats["all_npe_max"], float(np.max(npe_all)))
                stats["log_all_npe_min"] = min(stats["log_all_npe_min"], float(np.min(npe_log_all)))
                stats["log_all_npe_max"] = max(stats["log_all_npe_max"], float(np.max(npe_log_all)))
            if ftime_all.size > 0:
                stats["all_ftime_min"] = min(stats["all_ftime_min"], float(np.min(ftime_all)))
                stats["all_ftime_max"] = max(stats["all_ftime_max"], float(np.max(ftime_all)))
                stats["log_all_ftime_min"] = min(stats["log_all_ftime_min"], float(np.min(ftime_log_all)))
                stats["log_all_ftime_max"] = max(stats["log_all_ftime_max"], float(np.max(ftime_log_all)))
            if npe_all.size > 0:
                all_npe_chunk, _ = np.histogram(npe_all, bins=all_npe_edges)
                log_all_npe_chunk, _ = np.histogram(npe_log_all, bins=log_all_npe_edges)
            else:
                all_npe_chunk = np.zeros(num_bins, dtype=np.int64)
                log_all_npe_chunk = np.zeros(num_bins, dtype=np.int64)
            if ftime_all.size > 0:
                all_ftime_chunk, _ = np.histogram(ftime_all, bins=all_ftime_edges)
                log_all_ftime_chunk, _ = np.histogram(ftime_log_all, bins=log_all_ftime_edges)
            else:
                all_ftime_chunk = np.zeros(num_bins, dtype=np.int64)
                log_all_ftime_chunk = np.zeros(num_bins, dtype=np.int64)

            if npe_log.size > 0:
                stats["log_npe_min"] = min(stats["log_npe_min"], float(np.min(npe_log)))
                stats["log_npe_max"] = max(stats["log_npe_max"], float(np.max(npe_log)))
            if ftime_log.size > 0:
                stats["log_ftime_min"] = min(stats["log_ftime_min"], float(np.min(ftime_log)))
                stats["log_ftime_max"] = max(stats["log_ftime_max"], float(np.max(ftime_log)))
            if npe_norm.size > 0:
                stats["norm_npe_min"] = min(stats["norm_npe_min"], float(np.min(npe_norm)))
                stats["norm_npe_max"] = max(stats["norm_npe_max"], float(np.max(npe_norm)))
            if ftime_norm.size > 0:
                stats["norm_ftime_min"] = min(stats["norm_ftime_min"], float(np.min(ftime_norm)))
                stats["norm_ftime_max"] = max(stats["norm_ftime_max"], float(np.max(ftime_norm)))

            raw_npe_chunk, _ = np.histogram(npe_active, bins=raw_npe_edges)
            raw_ftime_chunk, _ = np.histogram(ftime_active, bins=raw_ftime_edges)
            raw_log_npe_chunk, _ = np.histogram(npe_active, bins=raw_log_npe_edges)
            raw_log_ftime_chunk, _ = np.histogram(ftime_active, bins=raw_log_ftime_edges)
            log_npe_chunk, _ = np.histogram(npe_log, bins=log_npe_edges)
            log_ftime_chunk, _ = np.histogram(ftime_log, bins=log_ftime_edges)
            norm_npe_chunk, _ = np.histogram(npe_norm, bins=norm_npe_edges)
            norm_ftime_chunk, _ = np.histogram(ftime_norm, bins=norm_ftime_edges)

            if all_npe_counts is None:
                all_npe_counts = all_npe_chunk.astype(np.int64)
                all_ftime_counts = all_ftime_chunk.astype(np.int64)
                log_all_npe_counts = log_all_npe_chunk.astype(np.int64)
                log_all_ftime_counts = log_all_ftime_chunk.astype(np.int64)
                raw_npe_counts = raw_npe_chunk.astype(np.int64)
                raw_ftime_counts = raw_ftime_chunk.astype(np.int64)
                raw_log_npe_counts = raw_log_npe_chunk.astype(np.int64)
                raw_log_ftime_counts = raw_log_ftime_chunk.astype(np.int64)
                log_npe_counts = log_npe_chunk.astype(np.int64)
                log_ftime_counts = log_ftime_chunk.astype(np.int64)
                norm_npe_counts = norm_npe_chunk.astype(np.int64)
                norm_ftime_counts = norm_ftime_chunk.astype(np.int64)
            else:
                all_npe_counts += all_npe_chunk
                all_ftime_counts += all_ftime_chunk
                log_all_npe_counts += log_all_npe_chunk
                log_all_ftime_counts += log_all_ftime_chunk
                raw_npe_counts += raw_npe_chunk
                raw_ftime_counts += raw_ftime_chunk
                raw_log_npe_counts += raw_log_npe_chunk
                raw_log_ftime_counts += raw_log_ftime_chunk
                log_npe_counts += log_npe_chunk
                log_ftime_counts += log_ftime_chunk
                norm_npe_counts += norm_npe_chunk
                norm_ftime_counts += norm_ftime_chunk

    return (
        all_npe_counts,
        all_ftime_counts,
        log_all_npe_counts,
        log_all_ftime_counts,
        raw_npe_counts,
        raw_ftime_counts,
        raw_log_npe_counts,
        raw_log_ftime_counts,
        log_npe_counts,
        log_ftime_counts,
        norm_npe_counts,
        norm_ftime_counts,
        all_npe_edges,
        all_ftime_edges,
        log_all_npe_edges,
        log_all_ftime_edges,
        raw_npe_edges,
        raw_ftime_edges,
        raw_log_npe_edges,
        raw_log_ftime_edges,
        log_npe_edges,
        log_ftime_edges,
        norm_npe_edges,
        norm_ftime_edges,
        stats,
    )


def plot_histograms(
    all_npe_counts: np.ndarray,
    all_ftime_counts: np.ndarray,
    log_all_npe_counts: np.ndarray,
    log_all_ftime_counts: np.ndarray,
    raw_npe_counts: np.ndarray,
    raw_ftime_counts: np.ndarray,
    raw_log_npe_counts: np.ndarray,
    raw_log_ftime_counts: np.ndarray,
    log_npe_counts: np.ndarray,
    log_ftime_counts: np.ndarray,
    norm_npe_counts: np.ndarray,
    norm_ftime_counts: np.ndarray,
    all_npe_edges: np.ndarray,
    all_ftime_edges: np.ndarray,
    log_all_npe_edges: np.ndarray,
    log_all_ftime_edges: np.ndarray,
    raw_npe_edges: np.ndarray,
    raw_ftime_edges: np.ndarray,
    raw_log_npe_edges: np.ndarray,
    raw_log_ftime_edges: np.ndarray,
    log_npe_edges: np.ndarray,
    log_ftime_edges: np.ndarray,
    norm_npe_edges: np.ndarray,
    norm_ftime_edges: np.ndarray,
    output_path: Path,
    *,
    num_bins: int = 100,
    log_y: bool = True,
    npe_shift: float = 0.0,
):
    fig, axes = plt.subplots(6, 2, figsize=(16, 26))

    add_hist(
        axes[0, 0],
        all_npe_counts,
        all_npe_edges,
        "Raw all nPE (nPE+shift)",
        "nPE+shift",
        "slateblue",
        density=True,
    )
    add_hist(
        axes[0, 1],
        all_ftime_counts,
        all_ftime_edges,
        "Raw all FirstTime",
        "FirstTime",
        "saddlebrown",
        density=True,
    )
    add_hist(
        axes[1, 0],
        log_all_npe_counts,
        log_all_npe_edges,
        "log1p(all) nPE",
        "log1p(nPE)",
        "mediumpurple",
        density=True,
    )
    add_hist(
        axes[1, 1],
        log_all_ftime_counts,
        log_all_ftime_edges,
        "log1p(all) FirstTime",
        "log1p(FirstTime)",
        "peru",
        density=True,
    )
    add_hist(
        axes[2, 0],
        raw_npe_counts,
        raw_npe_edges,
        "Raw active nPE (nPE+shift)",
        "nPE+shift",
        "steelblue",
        density=True,
    )
    add_hist(
        axes[2, 1],
        raw_ftime_counts,
        raw_ftime_edges,
        "Raw active FirstTime",
        "FirstTime",
        "darkorange",
        density=True,
    )
    add_hist(
        axes[3, 0],
        log_npe_counts,
        log_npe_edges,
        "log1p(active) nPE",
        "log1p(nPE+shift)",
        "royalblue",
        density=True,
    )
    add_hist(
        axes[3, 1],
        log_ftime_counts,
        log_ftime_edges,
        "log1p(active) FirstTime",
        "log1p(FirstTime)",
        "orangered",
        density=True,
    )
    add_hist(
        axes[4, 0],
        raw_log_npe_counts,
        raw_log_npe_edges,
        "Raw active nPE (log x)",
        "nPE+shift",
        "steelblue",
        density=True,
    )
    add_hist(
        axes[4, 1],
        raw_log_ftime_counts,
        raw_log_ftime_edges,
        "Raw active FirstTime (log x)",
        "FirstTime",
        "darkorange",
        density=True,
    )
    axes[4, 0].set_xscale("log")
    axes[4, 1].set_xscale("log")
    add_hist(
        axes[5, 0],
        norm_npe_counts,
        norm_npe_edges,
        "Normalized active nPE",
        "log1p(nPE+shift) / p99",
        "royalblue",
        density=True,
    )
    add_hist(
        axes[5, 1],
        norm_ftime_counts,
        norm_ftime_edges,
        "Normalized active FirstTime",
        "log1p(FirstTime) / p99",
        "orangered",
        density=True,
    )

    if log_y:
        for ax in axes.flat:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-5)

    fig.suptitle(
        "Signal histograms before/after active-only normalization\n"
        "top: raw all / log1p(all) | middle: raw active / log1p(active) | "
        f"new row: raw active log scale | bottom: active log1p normalized by 99th percentile | nPE shift = {npe_shift:g}",
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram figure to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot raw and normalized signal histograms")
    parser.add_argument(
        "--h5-path",
        type=str,
        default="./GENESIS-data/22644_0921_time_shift.h5",
        help="Path to the HDF5 dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="signal_normalization_histograms.png",
        help="Output image path",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=100,
        help="Number of histogram bins",
    )
    parser.add_argument(
        "--npe-shift",
        type=float,
        default=0.0,
        help="Shift added to nPE before all histogram/normalization steps",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap on number of events to scan",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of events read per HDF5 chunk",
    )
    parser.add_argument(
        "--no-log-y",
        action="store_true",
        help="Disable log scale on the y-axis",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    print(f"Loading dataset: {h5_path}")
    print(f"Using active-only log1p + p99 normalization with nPE shift = {args.npe_shift:g}")

    (
        all_npe_counts,
        all_ftime_counts,
        log_all_npe_counts,
        log_all_ftime_counts,
        raw_npe_counts,
        raw_ftime_counts,
        raw_log_npe_counts,
        raw_log_ftime_counts,
        log_npe_counts,
        log_ftime_counts,
        norm_npe_counts,
        norm_ftime_counts,
        all_npe_edges,
        all_ftime_edges,
        log_all_npe_edges,
        log_all_ftime_edges,
        raw_npe_edges,
        raw_ftime_edges,
        raw_log_npe_edges,
        raw_log_ftime_edges,
        log_npe_edges,
        log_ftime_edges,
        norm_npe_edges,
        norm_ftime_edges,
        stats,
    ) = collect_histograms(
        h5_path=h5_path,
        max_events=args.max_events,
        chunk_size=max(1, args.chunk_size),
        num_bins=max(1, args.num_bins),
        npe_shift=float(args.npe_shift),
    )

    print("Collected histogram counts:")
    print(f"  all nPE total: {int(all_npe_counts.sum()):,}")
    print(f"  all FirstTime total: {int(all_ftime_counts.sum()):,}")
    print(f"  log1p(all) nPE total: {int(log_all_npe_counts.sum()):,}")
    print(f"  log1p(all) FirstTime total: {int(log_all_ftime_counts.sum()):,}")
    print(f"  raw nPE total: {int(raw_npe_counts.sum()):,}")
    print(f"  raw FirstTime total: {int(raw_ftime_counts.sum()):,}")
    print(f"  raw log-x nPE total: {int(raw_log_npe_counts.sum()):,}")
    print(f"  raw log-x FirstTime total: {int(raw_log_ftime_counts.sum()):,}")
    print(f"  log1p(active) nPE total: {int(log_npe_counts.sum()):,}")
    print(f"  log1p(active) FirstTime total: {int(log_ftime_counts.sum()):,}")
    print(f"  norm nPE total: {int(norm_npe_counts.sum()):,}")
    print(f"  norm FirstTime total: {int(norm_ftime_counts.sum()):,}")
    print("Active-only p99 stats:")
    print(
        f"  nPE      p99={stats['log_npe_scale']:.6g}, "
        f"active={stats['active_npe_count']:,}"
    )
    print(
        f"  FirstTime p99={stats['log_ftime_scale']:.6g}, "
        f"active={stats['active_ftime_count']:,}"
    )
    print("Observed value ranges:")
    print(f"  raw nPE (shifted by {stats['npe_shift']:g}): [{stats['raw_npe_min']:.6g}, {stats['raw_npe_max']:.6g}]")
    print(f"  raw FirstTime: [{stats['raw_ftime_min']:.6g}, {stats['raw_ftime_max']:.6g}]")
    print(f"  norm nPE     : [{stats['norm_npe_min']:.6g}, {stats['norm_npe_max']:.6g}]")
    print(f"  norm FirstTime: [{stats['norm_ftime_min']:.6g}, {stats['norm_ftime_max']:.6g}]")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_histograms(
        all_npe_counts=all_npe_counts,
        all_ftime_counts=all_ftime_counts,
        log_all_npe_counts=log_all_npe_counts,
        log_all_ftime_counts=log_all_ftime_counts,
        raw_npe_counts=raw_npe_counts,
        raw_ftime_counts=raw_ftime_counts,
        raw_log_npe_counts=raw_log_npe_counts,
        raw_log_ftime_counts=raw_log_ftime_counts,
        log_npe_counts=log_npe_counts,
        log_ftime_counts=log_ftime_counts,
        norm_npe_counts=norm_npe_counts,
        norm_ftime_counts=norm_ftime_counts,
        all_npe_edges=all_npe_edges,
        all_ftime_edges=all_ftime_edges,
        log_all_npe_edges=log_all_npe_edges,
        log_all_ftime_edges=log_all_ftime_edges,
        raw_npe_edges=raw_npe_edges,
        raw_ftime_edges=raw_ftime_edges,
        raw_log_npe_edges=raw_log_npe_edges,
        raw_log_ftime_edges=raw_log_ftime_edges,
        log_npe_edges=log_npe_edges,
        log_ftime_edges=log_ftime_edges,
        norm_npe_edges=norm_npe_edges,
        norm_ftime_edges=norm_ftime_edges,
        output_path=output_path,
        num_bins=args.num_bins,
        log_y=not args.no_log_y,
        npe_shift=float(args.npe_shift),
    )


if __name__ == "__main__":
    main()
