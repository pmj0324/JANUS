#!/usr/bin/env python3
"""make_sample_new.py

Extended experiment runner for GENESIS sample sweeps.

This version preserves the original DOM-event evaluation mode and adds a
multifield-map evaluation mode tailored to ensemble-level cosmology-style
validation:
- log-space one-point PDF
- auto-power spectrum
- cross-power spectrum
- coherence
- optional bispectrum diagnostics
- diversity / cosmic-variance checks
- covariance-weighted summary statistics
- optional 1-parameter response plots when labels are available
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

_CSV_LIMIT = sys.maxsize
while True:
    try:
        csv.field_size_limit(_CSV_LIMIT)
        break
    except OverflowError:
        _CSV_LIMIT //= 10


# -----------------------------
# Range helpers
# -----------------------------
def frange(start: float, stop: float, step: float) -> List[float]:
    if step == 0:
        raise ValueError("step must be non-zero")
    if (stop - start) * step < 0:
        raise ValueError("step sign must move start toward stop")

    n = int(math.floor((stop - start) / step + 1e-12)) + 1
    values = [start + i * step for i in range(n)]

    def _round(x: float) -> float:
        return float(f"{x:.10g}")

    values = [_round(v) for v in values]
    if abs(values[-1] - stop) > 1e-9 and ((stop - values[-1]) * step) > 0:
        values.append(_round(stop))
    else:
        values[-1] = _round(stop)
    return values


def irange(start: int, stop: int, step: int) -> List[int]:
    if step == 0:
        raise ValueError("step must be non-zero")
    if (stop - start) * step < 0:
        raise ValueError("step sign must move start toward stop")
    return list(range(start, stop + (1 if step > 0 else -1), step))


# -----------------------------
# Project import setup
# -----------------------------
def setup_project_imports(tasks_root: Path):
    if str(tasks_root) not in sys.path:
        sys.path.insert(0, str(tasks_root))
    from dataloader.h5 import H5Dataset
    return H5Dataset


# -----------------------------
# Generic helpers
# -----------------------------
def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def _to_float(x: Any, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _to_int(x: Any, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def _safe_nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def parse_str_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def as_jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [as_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): as_jsonable(v) for k, v in x.items()}
    try:
        return float(x)
    except Exception:
        return str(x)


def stable_label_key(label_vector: Sequence[float], precision: int = 8) -> str:
    rounded = [round(float(v), precision) for v in label_vector]
    return "|".join(f"{v:.{precision}g}" for v in rounded)


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


# -----------------------------
# Existing event-level helpers
# -----------------------------
NPE_CLIP = 1000.0
FTIME_CLIP = 21000.0


def clamp_actual_signal(sig: np.ndarray) -> np.ndarray:
    out = sig.copy()
    out[0] = np.clip(out[0], 0.0, NPE_CLIP)
    out[1] = np.clip(out[1], 0.0, FTIME_CLIP)
    return out


def apply_eval_cuts(sig: np.ndarray, cut_npe: float, cut_firsttime: float) -> np.ndarray:
    out = sig.copy()
    if cut_npe > 0:
        out[0] = np.where(out[0] <= cut_npe, 0.0, out[0])
    if cut_firsttime > 0:
        out[1] = np.where(out[1] <= cut_firsttime, 0.0, out[1])
    return out


def apply_event_style_eval_cuts(sig: np.ndarray, *, cut_npe: float, cut_firsttime: float) -> np.ndarray:
    """event 평가와 동일: clip 후 eval cut 적용. multifield 입력 텐서의 채널 0·1이 npe/firsttime과 동일 의미일 때 사용."""
    return apply_eval_cuts(clamp_actual_signal(np.asarray(sig, dtype=float)), cut_npe=cut_npe, cut_firsttime=cut_firsttime)


def safe_stats(x: np.ndarray) -> Dict[str, float]:
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "std": np.nan,
            "q90": np.nan,
            "q99": np.nan,
            "sum": 0.0,
        }
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "std": float(np.std(x)),
        "q90": float(np.quantile(x, 0.90)),
        "q99": float(np.quantile(x, 0.99)),
        "sum": float(np.sum(x)),
    }


def wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    u = u[np.isfinite(u)]
    v = v[np.isfinite(v)]
    if u.size == 0 and v.size == 0:
        return 0.0
    if u.size == 0 or v.size == 0:
        return np.nan

    all_vals = np.sort(np.unique(np.concatenate([u, v])))
    if all_vals.size == 1:
        return 0.0

    u_sorted = np.sort(u)
    v_sorted = np.sort(v)

    u_cdf = np.searchsorted(u_sorted, all_vals, side="right") / u_sorted.size
    v_cdf = np.searchsorted(v_sorted, all_vals, side="right") / v_sorted.size

    dx = np.diff(all_vals)
    cdf_diff = np.abs(u_cdf[:-1] - v_cdf[:-1])
    return float(np.sum(cdf_diff * dx))


def ks_statistic_1d(u: np.ndarray, v: np.ndarray) -> float:
    u = u[np.isfinite(u)]
    v = v[np.isfinite(v)]
    if u.size == 0 and v.size == 0:
        return 0.0
    if u.size == 0 or v.size == 0:
        return np.nan

    all_vals = np.sort(np.unique(np.concatenate([u, v])))
    u_sorted = np.sort(u)
    v_sorted = np.sort(v)

    u_cdf = np.searchsorted(u_sorted, all_vals, side="right") / u_sorted.size
    v_cdf = np.searchsorted(v_sorted, all_vals, side="right") / v_sorted.size
    return float(np.max(np.abs(u_cdf - v_cdf)))


def iou_from_masks(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def log_mae(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(np.log1p(a[valid]) - np.log1p(b[valid]))))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(a[valid] - b[valid])))


def build_ers_score(metrics: Dict[str, float]) -> float:
    actual_npe_median = metrics.get("actual_npe_median", np.nan)
    actual_time_median = metrics.get("actual_time_median", np.nan)

    actual_npe_scale = max(float(actual_npe_median) if np.isfinite(actual_npe_median) else 1.0, 1.0)
    actual_time_scale = max(float(actual_time_median) if np.isfinite(actual_time_median) else 1.0, 1.0)

    npe_log = 0.0 if np.isnan(metrics["npe_log_mae"]) else metrics["npe_log_mae"]
    time_mae_norm = 0.0 if np.isnan(metrics["time_mae"]) else metrics["time_mae"] / actual_time_scale
    iou_penalty = 0.0 if np.isnan(metrics["active_dom_iou"]) else (1.0 - metrics["active_dom_iou"])
    npe_w1_norm = 0.0 if np.isnan(metrics["npe_wasserstein"]) else metrics["npe_wasserstein"] / actual_npe_scale
    time_w1_norm = 0.0 if np.isnan(metrics["time_wasserstein"]) else metrics["time_wasserstein"] / actual_time_scale

    ers = (
        0.30 * npe_log
        + 0.25 * time_mae_norm
        + 0.20 * iou_penalty
        + 0.15 * npe_w1_norm
        + 0.10 * time_w1_norm
    )
    return float(ers)


def evaluate_event_sample_vs_actual(
    actual_sig_raw: np.ndarray,
    sample_sig_raw: np.ndarray,
    *,
    cut_npe: float,
    cut_firsttime: float,
) -> Dict[str, float]:
    actual = clamp_actual_signal(actual_sig_raw)
    sample = clamp_actual_signal(sample_sig_raw)

    actual_eval = apply_eval_cuts(actual, cut_npe=cut_npe, cut_firsttime=cut_firsttime)
    sample_eval = apply_eval_cuts(sample, cut_npe=cut_npe, cut_firsttime=cut_firsttime)

    actual_npe = actual_eval[0]
    actual_time = actual_eval[1]
    sample_npe = sample_eval[0]
    sample_time = sample_eval[1]

    actual_active = actual_npe > 0
    sample_active = sample_npe > 0
    both_active = actual_active & sample_active

    actual_npe_stats = safe_stats(actual_npe)
    sample_npe_stats = safe_stats(sample_npe)
    actual_time_stats = safe_stats(actual_time)
    sample_time_stats = safe_stats(sample_time)

    metrics = {
        "actual_active_count": int(actual_active.sum()),
        "sample_active_count": int(sample_active.sum()),
        "active_dom_iou": iou_from_masks(actual_active, sample_active),
        "actual_npe_mean": actual_npe_stats["mean"],
        "sample_npe_mean": sample_npe_stats["mean"],
        "actual_npe_median": actual_npe_stats["median"],
        "sample_npe_median": sample_npe_stats["median"],
        "actual_npe_sum": actual_npe_stats["sum"],
        "sample_npe_sum": sample_npe_stats["sum"],
        "actual_time_mean": actual_time_stats["mean"],
        "sample_time_mean": sample_time_stats["mean"],
        "actual_time_median": actual_time_stats["median"],
        "sample_time_median": sample_time_stats["median"],
        "actual_time_min": actual_time_stats["min"],
        "sample_time_min": sample_time_stats["min"],
        "npe_mean_abs_err": abs(sample_npe_stats["mean"] - actual_npe_stats["mean"])
        if np.isfinite(sample_npe_stats["mean"]) and np.isfinite(actual_npe_stats["mean"]) else np.nan,
        "npe_median_abs_err": abs(sample_npe_stats["median"] - actual_npe_stats["median"])
        if np.isfinite(sample_npe_stats["median"]) and np.isfinite(actual_npe_stats["median"]) else np.nan,
        "npe_sum_abs_err": abs(sample_npe_stats["sum"] - actual_npe_stats["sum"]),
        "time_mean_abs_err": abs(sample_time_stats["mean"] - actual_time_stats["mean"])
        if np.isfinite(sample_time_stats["mean"]) and np.isfinite(actual_time_stats["mean"]) else np.nan,
        "time_median_abs_err": abs(sample_time_stats["median"] - actual_time_stats["median"])
        if np.isfinite(sample_time_stats["median"]) and np.isfinite(actual_time_stats["median"]) else np.nan,
        "time_min_abs_err": abs(sample_time_stats["min"] - actual_time_stats["min"])
        if np.isfinite(sample_time_stats["min"]) and np.isfinite(actual_time_stats["min"]) else np.nan,
    }

    metrics["npe_log_mae"] = log_mae(actual_npe, sample_npe)

    if both_active.sum() > 0:
        metrics["time_mae"] = mae(actual_time[both_active], sample_time[both_active])
    else:
        metrics["time_mae"] = np.nan

    actual_npe_pos = actual_npe[(actual_npe > 0) & np.isfinite(actual_npe)]
    sample_npe_pos = sample_npe[(sample_npe > 0) & np.isfinite(sample_npe)]
    actual_time_pos = actual_time[(actual_time > 0) & np.isfinite(actual_time)]
    sample_time_pos = sample_time[(sample_time > 0) & np.isfinite(sample_time)]

    metrics["npe_wasserstein"] = wasserstein_1d(actual_npe_pos, sample_npe_pos)
    metrics["time_wasserstein"] = wasserstein_1d(actual_time_pos, sample_time_pos)
    metrics["npe_ks"] = ks_statistic_1d(actual_npe_pos, sample_npe_pos)
    metrics["time_ks"] = ks_statistic_1d(actual_time_pos, sample_time_pos)

    metrics["ers_score"] = build_ers_score(metrics)
    metrics["total_score"] = metrics["ers_score"]
    return metrics


# -----------------------------
# Multifield map evaluation helpers
# -----------------------------
@dataclass
class MultifieldEvalConfig:
    channel_names: List[str]
    channel_types: List[str]
    pdf_eps: List[float]
    pdf_bins: int
    map_box_size: float
    k_bins: int
    log_k_bins: bool
    enable_bispectrum: bool
    bispectrum_bin_triplets: List[Tuple[int, int, int]]


def parse_multifield_config(args: argparse.Namespace) -> MultifieldEvalConfig:
    channel_names = parse_str_list(args.channel_names)
    channel_types = parse_str_list(args.channel_types)
    pdf_eps = parse_float_list(args.pdf_eps)

    if not channel_names:
        raise ValueError("--channel_names must not be empty in multifield_map mode")
    if len(channel_types) != len(channel_names):
        raise ValueError("--channel_types length must match --channel_names")
    if len(pdf_eps) == 1 and len(channel_names) > 1:
        pdf_eps = pdf_eps * len(channel_names)
    if len(pdf_eps) != len(channel_names):
        raise ValueError("--pdf_eps length must be 1 or match number of channels")

    triplets: List[Tuple[int, int, int]] = []
    if args.enable_bispectrum and args.bispectrum_triplets.strip():
        for block in args.bispectrum_triplets.split(";"):
            block = block.strip()
            if not block:
                continue
            vals = [int(v.strip()) for v in block.split(",") if v.strip()]
            if len(vals) != 3:
                raise ValueError("Each bispectrum triplet must have exactly three bin indices")
            triplets.append((vals[0], vals[1], vals[2]))
    if args.enable_bispectrum and not triplets:
        triplets = [(2, 2, 2), (2, 6, 6), (3, 5, 7), (4, 8, 8)]

    return MultifieldEvalConfig(
        channel_names=channel_names,
        channel_types=channel_types,
        pdf_eps=pdf_eps,
        pdf_bins=int(args.pdf_bins),
        map_box_size=float(args.map_box_size),
        k_bins=int(args.k_bins),
        log_k_bins=bool(args.log_k_bins),
        enable_bispectrum=bool(args.enable_bispectrum),
        bispectrum_bin_triplets=triplets,
    )


def infer_channel_count_from_signal(sig: np.ndarray) -> Optional[int]:
    arr = np.asarray(sig)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        candidates = [int(arr.shape[0]), int(arr.shape[1])]
    elif arr.ndim == 3:
        candidates = [int(s) for s in arr.shape]
    else:
        return None

    plausible = [c for c in candidates if 1 < c <= 16]
    if not plausible:
        return None
    return int(plausible[0])


def _resize_with_fallback(values: Sequence[Any], target_len: int, fallback_prefix: str, fallback_value: Any) -> List[Any]:
    out = list(values[:target_len])
    while len(out) < target_len:
        if fallback_prefix == "channel":
            out.append(f"channel_{len(out)}")
        else:
            out.append(fallback_value)
    return out


def prepare_multifield_map(arr: np.ndarray, expected_channels: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and 1 in arr.shape:
        squeezed = np.squeeze(arr)
        if squeezed.ndim in (2, 3):
            arr = squeezed

    if arr.ndim == 2:
        if arr.shape[0] == expected_channels:
            out = arr[:, None, :]
        elif arr.shape[1] == expected_channels:
            out = np.moveaxis(arr, -1, 0)[:, None, :]
        else:
            raise ValueError(
                f"Could not infer channel axis for 2D shape={arr.shape}; expected {expected_channels} channels"
            )
    elif arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D array for multifield_map, got shape={arr.shape}")
    else:
        if arr.shape[0] == expected_channels:
            out = arr
        elif arr.shape[-1] == expected_channels:
            out = np.moveaxis(arr, -1, 0)
        else:
            raise ValueError(
                f"Could not infer channel axis for shape={arr.shape}; expected {expected_channels} channels"
            )

    if out.ndim != 3:
        raise ValueError(f"Prepared multifield map must be (C,H,W), got {out.shape}")
    return np.asarray(out, dtype=float)


def compute_log_transform(x: np.ndarray, eps: float) -> np.ndarray:
    return np.log10(np.maximum(x + eps, eps))


def compute_channel_representation(x: np.ndarray, channel_type: str, eps: float) -> np.ndarray:
    ctype = channel_type.lower().strip()
    if ctype == "density":
        mean = float(np.mean(x))
        if not np.isfinite(mean) or abs(mean) < 1e-12:
            return x - mean
        return (x - mean) / mean
    if ctype in {"temperature", "log_temperature", "log-temp", "temp"}:
        u = compute_log_transform(x, eps)
        mu = float(np.mean(u))
        sigma = float(np.std(u))
        if not np.isfinite(sigma) or sigma < 1e-12:
            return u - mu
        return (u - mu) / sigma
    if ctype in {"log", "generic_log"}:
        u = compute_log_transform(x, eps)
        return u - float(np.mean(u))
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if not np.isfinite(sigma) or sigma < 1e-12:
        return x - mu
    return (x - mu) / sigma


def compute_pdf_hist_pair(actual: np.ndarray, sample: np.ndarray, eps: float, bins: int) -> Dict[str, Any]:
    u_actual = compute_log_transform(actual, eps).ravel()
    u_sample = compute_log_transform(sample, eps).ravel()
    valid = np.concatenate([u_actual[np.isfinite(u_actual)], u_sample[np.isfinite(u_sample)]])
    if valid.size == 0:
        edges = np.linspace(-1.0, 1.0, bins + 1)
        actual_hist = np.zeros(bins, dtype=float)
        sample_hist = np.zeros(bins, dtype=float)
    else:
        lo = float(np.min(valid))
        hi = float(np.max(valid))
        if hi <= lo:
            hi = lo + 1e-6
        pad = 0.02 * (hi - lo)
        edges = np.linspace(lo - pad, hi + pad, bins + 1)
        actual_hist, _ = np.histogram(u_actual[np.isfinite(u_actual)], bins=edges, density=True)
        sample_hist, _ = np.histogram(u_sample[np.isfinite(u_sample)], bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "centers": centers.tolist(),
        "edges": edges.tolist(),
        "actual": actual_hist.tolist(),
        "sample": sample_hist.tolist(),
    }


def build_kbin_structure(h: int, w: int, box_size: float, k_bins: int, log_k_bins: bool) -> Dict[str, Any]:
    dx = box_size / float(w)
    dy = box_size / float(h)
    kx = 2.0 * np.pi * np.fft.fftfreq(w, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(h, d=dy)
    kxg, kyg = np.meshgrid(kx, ky)
    kmag = np.sqrt(kxg**2 + kyg**2)

    positive = kmag[np.isfinite(kmag) & (kmag > 0)]
    if positive.size == 0:
        raise ValueError("No positive Fourier modes found")
    kmin = float(np.min(positive))
    kmax = float(np.max(positive))

    if log_k_bins:
        edges = np.geomspace(kmin, kmax, k_bins + 1)
    else:
        edges = np.linspace(kmin, kmax, k_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:]) if log_k_bins else 0.5 * (edges[:-1] + edges[1:])
    bin_index = np.digitize(kmag.ravel(), edges) - 1
    mode_counts = np.array([(bin_index == i).sum() for i in range(k_bins)], dtype=int)
    return {
        "kx": kxg,
        "ky": kyg,
        "kmag": kmag,
        "edges": edges,
        "centers": centers,
        "bin_index": bin_index,
        "mode_counts": mode_counts,
        "shape": (h, w),
    }


def radial_bin_average(array2d: np.ndarray, kstruct: Dict[str, Any]) -> np.ndarray:
    flat = np.asarray(array2d, dtype=float).ravel()
    idx = kstruct["bin_index"]
    nbins = len(kstruct["centers"])
    out = np.full(nbins, np.nan, dtype=float)
    for i in range(nbins):
        mask = idx == i
        if np.any(mask):
            vals = flat[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                out[i] = float(np.mean(vals))
    return out


def compute_fft(field: np.ndarray) -> np.ndarray:
    return np.fft.fft2(np.asarray(field, dtype=float))


def compute_auto_power(field: np.ndarray, kstruct: Dict[str, Any]) -> np.ndarray:
    f = compute_fft(field)
    power2d = (np.abs(f) ** 2) / float(field.size ** 2)
    return radial_bin_average(power2d, kstruct)


def compute_cross_power(field_a: np.ndarray, field_b: np.ndarray, kstruct: Dict[str, Any]) -> np.ndarray:
    fa = compute_fft(field_a)
    fb = compute_fft(field_b)
    cross2d = np.real(fa * np.conjugate(fb)) / float(field_a.size ** 2)
    return radial_bin_average(cross2d, kstruct)


def compute_coherence(cross: np.ndarray, auto_a: np.ndarray, auto_b: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    denom = np.sqrt(np.maximum(auto_a, 0.0) * np.maximum(auto_b, 0.0))
    out = np.full_like(cross, np.nan, dtype=float)
    valid = np.isfinite(cross) & np.isfinite(denom) & (denom > eps)
    out[valid] = cross[valid] / denom[valid]
    out = np.clip(out, -1.0, 1.0, where=np.isfinite(out), out=out)
    return out


def relative_curve_error(actual: np.ndarray, sample: np.ndarray, eps: float = 1e-12) -> float:
    actual = np.asarray(actual, dtype=float)
    sample = np.asarray(sample, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(sample)
    if valid.sum() == 0:
        return np.nan
    a = np.abs(actual[valid])
    robust = max(float(np.nanmedian(a)), eps)
    denom = np.maximum(a, 0.1 * robust + eps)
    return float(np.mean(np.abs(sample[valid] - actual[valid]) / denom))


def l1_hist_error(actual_hist: Sequence[float], sample_hist: Sequence[float]) -> float:
    a = np.asarray(actual_hist, dtype=float)
    b = np.asarray(sample_hist, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(a[valid] - b[valid])))


def bandpass_mask(kstruct: Dict[str, Any], bin_idx: int) -> np.ndarray:
    edges = kstruct["edges"]
    kmag = kstruct["kmag"]
    lo = edges[bin_idx]
    hi = edges[bin_idx + 1]
    return (kmag >= lo) & (kmag < hi)


def filtered_field_from_bin(field: np.ndarray, kstruct: Dict[str, Any], bin_idx: int) -> np.ndarray:
    f = compute_fft(field)
    mask = bandpass_mask(kstruct, bin_idx)
    return np.real(np.fft.ifft2(f * mask))


def representative_bispectrum(field: np.ndarray, kstruct: Dict[str, Any], triplets: Sequence[Tuple[int, int, int]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    cache: Dict[int, np.ndarray] = {}
    n_bins = len(kstruct["centers"])
    for i, j, k in triplets:
        if min(i, j, k) < 0 or max(i, j, k) >= n_bins:
            continue
        if i not in cache:
            cache[i] = filtered_field_from_bin(field, kstruct, i)
        if j not in cache:
            cache[j] = filtered_field_from_bin(field, kstruct, j)
        if k not in cache:
            cache[k] = filtered_field_from_bin(field, kstruct, k)
        bis = float(np.mean(cache[i] * cache[j] * cache[k]))
        auto_i = float(np.mean(cache[i] ** 2))
        auto_j = float(np.mean(cache[j] ** 2))
        auto_k = float(np.mean(cache[k] ** 2))
        denom = math.sqrt(max(auto_i * auto_j * auto_k, 1e-30))
        red = bis / denom if denom > 0 else np.nan
        tag = f"{i}-{j}-{k}"
        out[f"B_{tag}"] = bis
        out[f"Q_{tag}"] = red
    return out


def effective_bispectrum_triplets(
    requested: Sequence[Tuple[int, int, int]],
    n_bins: int,
) -> List[Tuple[int, int, int]]:
    """Keep only triplets inside [0, n_bins); if none remain, fall back to in-range defaults."""
    if n_bins <= 0:
        return []
    valid = [t for t in requested if len(t) == 3 and all(0 <= int(x) < n_bins for x in t)]
    if valid:
        return valid
    i0 = 0
    i1 = min(max(n_bins // 3, 0), n_bins - 1)
    i2 = min(max(2 * n_bins // 3, 0), n_bins - 1)
    return [(i0, i0, min(i2, n_bins - 1)), (i1, i1, i2)]


def aggregate_weighted_score(score_map: Dict[str, float], weights: Dict[str, float]) -> float:
    vals = []
    ws = []
    for key, weight in weights.items():
        value = score_map.get(key, np.nan)
        if np.isfinite(value):
            vals.append(float(value))
            ws.append(float(weight))
    if not vals:
        return np.nan
    wsum = np.sum(ws)
    if wsum <= 0:
        return np.nan
    return float(np.dot(vals, ws) / wsum)


def extract_label_vector(dataset_item_1: Any, dataset_item_2: Any) -> Optional[List[float]]:
    candidate = dataset_item_1
    if candidate is None or (isinstance(candidate, (list, tuple, dict)) and len(candidate) == 0):
        candidate = dataset_item_2

    if isinstance(candidate, dict):
        vals = []
        for k in sorted(candidate.keys()):
            try:
                vals.append(float(candidate[k]))
            except Exception:
                return None
        return vals if vals else None

    arr = to_numpy(candidate)
    if arr.size == 0:
        return None
    arr = np.asarray(arr).reshape(-1)
    try:
        vals = [float(v) for v in arr]
        return vals
    except Exception:
        return None


def evaluate_multifield_sample_vs_actual(
    actual_map: np.ndarray,
    sample_map: np.ndarray,
    config: MultifieldEvalConfig,
    label_vector: Optional[List[float]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    n_channels = len(config.channel_names)
    if actual_map.shape[0] != n_channels or sample_map.shape[0] != n_channels:
        raise ValueError(
            f"Channel mismatch: expected {n_channels}, actual={actual_map.shape}, sample={sample_map.shape}"
        )
    if actual_map.shape[1:] != sample_map.shape[1:]:
        raise ValueError(f"Spatial shape mismatch: actual={actual_map.shape}, sample={sample_map.shape}")

    h, w = actual_map.shape[1:]
    kstruct = build_kbin_structure(h, w, config.map_box_size, config.k_bins, config.log_k_bins)

    bundle: Dict[str, Any] = {
        "eval_mode": "multifield_map",
        "label_vector": label_vector,
        "channel_names": config.channel_names,
        "channel_types": config.channel_types,
        "pdf_eps": config.pdf_eps,
        "k_centers": kstruct["centers"].tolist(),
        "k_edges": kstruct["edges"].tolist(),
        "mode_counts": kstruct["mode_counts"].tolist(),
        "pdf": {},
        "auto": {"actual": {}, "sample": {}},
        "cross": {"actual": {}, "sample": {}},
        "coherence": {"actual": {}, "sample": {}},
        "representations": {"actual": {}, "sample": {}},
        "scalar_scores": {},
    }

    actual_repr: Dict[str, np.ndarray] = {}
    sample_repr: Dict[str, np.ndarray] = {}
    pdf_channel_scores: List[float] = []
    auto_channel_scores: List[float] = []

    for idx, name in enumerate(config.channel_names):
        eps = config.pdf_eps[idx]
        ctype = config.channel_types[idx]
        pdf_info = compute_pdf_hist_pair(actual_map[idx], sample_map[idx], eps, config.pdf_bins)
        pdf_score = l1_hist_error(pdf_info["actual"], pdf_info["sample"])
        pdf_channel_scores.append(pdf_score)
        bundle["pdf"][name] = pdf_info

        a_repr = compute_channel_representation(actual_map[idx], ctype, eps)
        s_repr = compute_channel_representation(sample_map[idx], ctype, eps)
        actual_repr[name] = a_repr
        sample_repr[name] = s_repr
        bundle["representations"]["actual"][name] = {
            "mean": float(np.mean(a_repr)),
            "std": float(np.std(a_repr)),
        }
        bundle["representations"]["sample"][name] = {
            "mean": float(np.mean(s_repr)),
            "std": float(np.std(s_repr)),
        }

        a_auto = compute_auto_power(a_repr, kstruct)
        s_auto = compute_auto_power(s_repr, kstruct)
        auto_score = relative_curve_error(a_auto, s_auto)
        auto_channel_scores.append(auto_score)
        bundle["auto"]["actual"][name] = a_auto.tolist()
        bundle["auto"]["sample"][name] = s_auto.tolist()

    pair_names: List[str] = []
    cross_scores: List[float] = []
    coh_scores: List[float] = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            name_i = config.channel_names[i]
            name_j = config.channel_names[j]
            pair = f"{name_i}__{name_j}"
            pair_names.append(pair)
            a_cross = compute_cross_power(actual_repr[name_i], actual_repr[name_j], kstruct)
            s_cross = compute_cross_power(sample_repr[name_i], sample_repr[name_j], kstruct)
            a_auto_i = np.asarray(bundle["auto"]["actual"][name_i], dtype=float)
            a_auto_j = np.asarray(bundle["auto"]["actual"][name_j], dtype=float)
            s_auto_i = np.asarray(bundle["auto"]["sample"][name_i], dtype=float)
            s_auto_j = np.asarray(bundle["auto"]["sample"][name_j], dtype=float)
            a_coh = compute_coherence(a_cross, a_auto_i, a_auto_j)
            s_coh = compute_coherence(s_cross, s_auto_i, s_auto_j)
            cross_scores.append(relative_curve_error(a_cross, s_cross))
            coh_scores.append(relative_curve_error(a_coh, s_coh))
            bundle["cross"]["actual"][pair] = a_cross.tolist()
            bundle["cross"]["sample"][pair] = s_cross.tolist()
            bundle["coherence"]["actual"][pair] = a_coh.tolist()
            bundle["coherence"]["sample"][pair] = s_coh.tolist()

    bispectrum_scores: List[float] = []
    if config.enable_bispectrum:
        bundle["bispectrum"] = {"actual": {}, "sample": {}}
        n_k_bins = len(kstruct["centers"])
        triplets_use = effective_bispectrum_triplets(config.bispectrum_bin_triplets, n_k_bins)
        for idx, name in enumerate(config.channel_names):
            a_bis = representative_bispectrum(actual_repr[name], kstruct, triplets_use)
            s_bis = representative_bispectrum(sample_repr[name], kstruct, triplets_use)
            bundle["bispectrum"]["actual"][name] = a_bis
            bundle["bispectrum"]["sample"][name] = s_bis
            keys = sorted(set(a_bis.keys()) & set(s_bis.keys()))
            if keys:
                a = np.array([a_bis[k] for k in keys], dtype=float)
                s = np.array([s_bis[k] for k in keys], dtype=float)
                bispectrum_scores.append(relative_curve_error(a, s))

    score_map = {
        "pdf_score": nanmean(pdf_channel_scores),
        "auto_score": nanmean(auto_channel_scores),
        "cross_score": nanmean(cross_scores),
        "coherence_score": nanmean(coh_scores),
        "bispectrum_score": nanmean(bispectrum_scores) if bispectrum_scores else np.nan,
    }
    total = aggregate_weighted_score(
        score_map,
        {
            "pdf_score": 0.25,
            "auto_score": 0.25,
            "cross_score": 0.20,
            "coherence_score": 0.20,
            "bispectrum_score": 0.10,
        },
    )
    score_map["total_score"] = total
    bundle["scalar_scores"] = score_map.copy()

    scalar_metrics = {
        "pdf_score": score_map["pdf_score"],
        "auto_score": score_map["auto_score"],
        "cross_score": score_map["cross_score"],
        "coherence_score": score_map["coherence_score"],
        "bispectrum_score": score_map["bispectrum_score"],
        # multifield에는 event용 DOM ERS가 없으므로, 요약/플롯용으로 가중 total과 동일한 스칼라를 기록한다.
        "ers_score": score_map["total_score"],
        "total_score": score_map["total_score"],
        "n_channels": n_channels,
        "n_pairs": len(pair_names),
        "map_height": h,
        "map_width": w,
    }
    return scalar_metrics, bundle


# -----------------------------
# CSV logging
# -----------------------------
CSV_FIELDS = [
    "timestamp",
    "run_id",
    "mode",
    "eval_mode",
    "checkpoint_tag",
    "checkpoint_path",
    "ref_idx",
    "cfg_scale",
    "cfg_tag",
    "sample_idx",
    "num_samples_requested",
    "gen_cut_npe",
    "gen_cut_firsttime",
    "cut_npe",
    "cut_firsttime",
    "output_dir",
    "sample_path",
    "bundle_path",
    "label_json",
    "label_key",
    # event metrics
    "actual_active_count",
    "sample_active_count",
    "active_dom_iou",
    "actual_npe_mean",
    "sample_npe_mean",
    "actual_npe_median",
    "sample_npe_median",
    "actual_npe_sum",
    "sample_npe_sum",
    "actual_time_mean",
    "sample_time_mean",
    "actual_time_median",
    "sample_time_median",
    "actual_time_min",
    "sample_time_min",
    "npe_mean_abs_err",
    "npe_median_abs_err",
    "npe_sum_abs_err",
    "time_mean_abs_err",
    "time_median_abs_err",
    "time_min_abs_err",
    "npe_log_mae",
    "time_mae",
    "npe_wasserstein",
    "time_wasserstein",
    "npe_ks",
    "time_ks",
    "ers_score",
    # multifield metrics
    "pdf_score",
    "auto_score",
    "cross_score",
    "coherence_score",
    "bispectrum_score",
    "parameter_response_consistency",
    "cv_score",
    "ees_score",
    "overall_score",
    "n_channels",
    "n_pairs",
    "map_height",
    "map_width",
    # common
    "total_score",
]


def append_row_to_csv(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_logged_eval_keys(csv_path: Path) -> Set[Tuple[str, float, float, str]]:
    rows = read_csv_rows(csv_path)
    logged: Set[Tuple[str, float, float, str]] = set()
    for row in rows:
        sample_path = row.get("sample_path", "").strip()
        if not sample_path:
            continue
        cut_npe = _to_float(row.get("cut_npe", np.nan))
        cut_firsttime = _to_float(row.get("cut_firsttime", np.nan))
        eval_mode = str(row.get("eval_mode", "")).strip()
        logged.add((str(Path(sample_path).resolve()), cut_npe, cut_firsttime, eval_mode))
    return logged


def make_eval_cut_lists(args: argparse.Namespace) -> Tuple[List[float], List[float]]:
    eval_cut_npe_list = frange(
        args.eval_cut_npe_start,
        args.eval_cut_npe_end,
        args.eval_cut_npe_step,
    )
    eval_cut_firsttime_list = frange(
        args.eval_cut_firsttime_start,
        args.eval_cut_firsttime_end,
        args.eval_cut_firsttime_step,
    )
    return eval_cut_npe_list, eval_cut_firsttime_list


def mean_of(rows: List[Dict[str, str]], key: str) -> float:
    vals = np.array([_to_float(r.get(key, np.nan)) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))


# -----------------------------
# Legacy event summary helpers
# -----------------------------
def rows_to_feature_matrices(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    actual_feats = []
    sample_feats = []
    for r in rows:
        a = [
            _to_float(r["actual_active_count"]),
            _to_float(r["actual_npe_mean"]),
            _to_float(r["actual_npe_median"]),
            _to_float(r["actual_npe_sum"]),
            _to_float(r["actual_time_mean"]),
            _to_float(r["actual_time_median"]),
            _to_float(r["actual_time_min"]),
        ]
        s = [
            _to_float(r["sample_active_count"]),
            _to_float(r["sample_npe_mean"]),
            _to_float(r["sample_npe_median"]),
            _to_float(r["sample_npe_sum"]),
            _to_float(r["sample_time_mean"]),
            _to_float(r["sample_time_median"]),
            _to_float(r["sample_time_min"]),
        ]
        if np.all(np.isfinite(a)) and np.all(np.isfinite(s)):
            actual_feats.append(a)
            sample_feats.append(s)
    if not actual_feats:
        return np.empty((0, 7)), np.empty((0, 7))
    return np.asarray(actual_feats, dtype=float), np.asarray(sample_feats, dtype=float)


def _sqrtm_psd(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mat = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance_gaussians(x_real: np.ndarray, x_fake: np.ndarray) -> float:
    if len(x_real) < 2 or len(x_fake) < 2:
        return np.nan
    mu_r = np.mean(x_real, axis=0)
    mu_f = np.mean(x_fake, axis=0)
    cov_r = np.cov(x_real, rowvar=False)
    cov_f = np.cov(x_fake, rowvar=False)
    diff = mu_r - mu_f
    cov_r = 0.5 * (cov_r + cov_r.T)
    cov_f = 0.5 * (cov_f + cov_f.T)
    sqrt_cov_r = _sqrtm_psd(cov_r)
    middle = sqrt_cov_r @ cov_f @ sqrt_cov_r
    sqrt_middle = _sqrtm_psd(middle)
    fd = diff @ diff + np.trace(cov_r + cov_f - 2.0 * sqrt_middle)
    return float(np.real(fd))


def covariance_error(x_real: np.ndarray, x_fake: np.ndarray) -> float:
    if len(x_real) < 2 or len(x_fake) < 2:
        return np.nan
    cov_r = np.cov(x_real, rowvar=False)
    cov_f = np.cov(x_fake, rowvar=False)
    return float(np.linalg.norm(cov_r - cov_f, ord="fro"))


def separation_power_from_hist(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    lo = min(np.min(a), np.min(b))
    hi = max(np.max(a), np.max(b))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0
    hist_a, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hist_b, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    denom = hist_a + hist_b + 1e-12
    sp = 0.5 * np.sum(((hist_a - hist_b) ** 2) / denom) * (edges[1] - edges[0])
    return float(sp)


def compute_cfg_global_metrics(rows: List[Dict[str, str]]) -> Dict[str, float]:
    x_real, x_fake = rows_to_feature_matrices(rows)
    fpd_like = frechet_distance_gaussians(x_real, x_fake)
    cov_err = covariance_error(x_real, x_fake)
    real_npe_sum = np.array([_to_float(r["actual_npe_sum"]) for r in rows], dtype=float)
    fake_npe_sum = np.array([_to_float(r["sample_npe_sum"]) for r in rows], dtype=float)
    real_time_med = np.array([_to_float(r["actual_time_median"]) for r in rows], dtype=float)
    fake_time_med = np.array([_to_float(r["sample_time_median"]) for r in rows], dtype=float)
    sep_npe = separation_power_from_hist(real_npe_sum, fake_npe_sum, bins=40)
    sep_time = separation_power_from_hist(real_time_med, fake_time_med, bins=40)
    mean_ers = mean_of(rows, "ers_score")
    return {
        "mean_ers_score": mean_ers,
        "fpd_like": fpd_like,
        "cov_err": cov_err,
        "sep_power_npe": sep_npe,
        "sep_power_time": sep_time,
    }


def group_rows_by_cfg(rows: List[Dict[str, str]]) -> Dict[float, List[Dict[str, str]]]:
    groups = defaultdict(list)
    for row in rows:
        groups[_to_float(row["cfg_scale"])].append(row)
    return dict(groups)


def group_rows_by_cfg_cut(rows: List[Dict[str, str]]) -> Dict[Tuple[float, float, float], List[Dict[str, str]]]:
    groups = defaultdict(list)
    for row in rows:
        key = (
            _to_float(row.get("cfg_scale", np.nan)),
            _to_float(row.get("cut_npe", np.nan)),
            _to_float(row.get("cut_firsttime", np.nan)),
        )
        groups[key].append(row)
    return dict(groups)


def group_rows_by_ref_cfg(rows: List[Dict[str, str]]) -> Dict[Tuple[int, float], List[Dict[str, str]]]:
    groups = defaultdict(list)
    for row in rows:
        key = (_to_int(row["ref_idx"]), _to_float(row["cfg_scale"]))
        groups[key].append(row)
    return dict(groups)


def group_rows_by_ref_cfg_cut(rows: List[Dict[str, str]]) -> Dict[Tuple[int, float, float, float], List[Dict[str, str]]]:
    groups = defaultdict(list)
    for row in rows:
        key = (
            _to_int(row.get("ref_idx", 0)),
            _to_float(row.get("cfg_scale", np.nan)),
            _to_float(row.get("cut_npe", np.nan)),
            _to_float(row.get("cut_firsttime", np.nan)),
        )
        groups[key].append(row)
    return dict(groups)


def compute_event_ers_term_means(group: List[Dict[str, str]]) -> Dict[str, float]:
    npe_log_vals = []
    time_mae_norm_vals = []
    iou_penalty_vals = []
    npe_w1_norm_vals = []
    time_w1_norm_vals = []

    for row in group:
        npe_log = _to_float(row.get("npe_log_mae", np.nan), np.nan)
        time_mae = _to_float(row.get("time_mae", np.nan), np.nan)
        active_dom_iou = _to_float(row.get("active_dom_iou", np.nan), np.nan)
        npe_w1 = _to_float(row.get("npe_wasserstein", np.nan), np.nan)
        time_w1 = _to_float(row.get("time_wasserstein", np.nan), np.nan)
        actual_npe_median = _to_float(row.get("actual_npe_median", np.nan), np.nan)
        actual_time_median = _to_float(row.get("actual_time_median", np.nan), np.nan)

        actual_npe_scale = max(actual_npe_median if np.isfinite(actual_npe_median) else 1.0, 1.0)
        actual_time_scale = max(actual_time_median if np.isfinite(actual_time_median) else 1.0, 1.0)

        npe_log_vals.append(0.0 if not np.isfinite(npe_log) else npe_log)
        time_mae_norm_vals.append(0.0 if not np.isfinite(time_mae) else time_mae / actual_time_scale)
        iou_penalty_vals.append(0.0 if not np.isfinite(active_dom_iou) else (1.0 - active_dom_iou))
        npe_w1_norm_vals.append(0.0 if not np.isfinite(npe_w1) else npe_w1 / actual_npe_scale)
        time_w1_norm_vals.append(0.0 if not np.isfinite(time_w1) else time_w1 / actual_time_scale)

    mean_npe_log = _safe_nanmean(npe_log_vals)
    mean_time_mae_norm = _safe_nanmean(time_mae_norm_vals)
    mean_iou_penalty = _safe_nanmean(iou_penalty_vals)
    mean_npe_w1_norm = _safe_nanmean(npe_w1_norm_vals)
    mean_time_w1_norm = _safe_nanmean(time_w1_norm_vals)

    return {
        "mean_ers_term_npe_log": mean_npe_log,
        "mean_ers_term_time_mae_norm": mean_time_mae_norm,
        "mean_ers_term_iou_penalty": mean_iou_penalty,
        "mean_ers_term_npe_w1_norm": mean_npe_w1_norm,
        "mean_ers_term_time_w1_norm": mean_time_w1_norm,
        "mean_ers_contrib_npe_log": 0.30 * mean_npe_log if np.isfinite(mean_npe_log) else np.nan,
        "mean_ers_contrib_time_mae_norm": 0.25 * mean_time_mae_norm if np.isfinite(mean_time_mae_norm) else np.nan,
        "mean_ers_contrib_iou_penalty": 0.20 * mean_iou_penalty if np.isfinite(mean_iou_penalty) else np.nan,
        "mean_ers_contrib_npe_w1_norm": 0.15 * mean_npe_w1_norm if np.isfinite(mean_npe_w1_norm) else np.nan,
        "mean_ers_contrib_time_w1_norm": 0.10 * mean_time_w1_norm if np.isfinite(mean_time_w1_norm) else np.nan,
    }


def aggregate_metric_by_single_axis(
    rows_in: List[Dict[str, Any]],
    axis_key: str,
    metric_key: str,
) -> Dict[float, float]:
    """
    axis_key별로 metric_key 값들을 모아 nan-safe 평균.
    다른 축(cfg_scale / cut_npe / cut_firsttime)은 모두 같은 버킷 안에서 평균으로 접힌다.
    """
    buckets: Dict[float, List[float]] = defaultdict(list)
    for row in rows_in:
        ax = _to_float(row.get(axis_key, np.nan), np.nan)
        if not np.isfinite(ax):
            continue
        mv = _to_float(row.get(metric_key, np.nan), np.nan)
        buckets[ax].append(mv)
    return {k: _safe_nanmean(buckets[k]) for k in sorted(buckets.keys())}


def plot_metric_three_axis_summary(
    *,
    rows_in: List[Dict[str, Any]],
    metric_key: str,
    output_filename: str,
    y_label: str,
    suptitle: str,
    plots_dir: Path,
) -> Optional[Path]:
    """cfg_scale / cut_npe / cut_firsttime 각각에 대해 단일 곡선(전역 평균) 1x3 subplot."""
    by_cfg = aggregate_metric_by_single_axis(rows_in, "cfg_scale", metric_key)
    by_npe = aggregate_metric_by_single_axis(rows_in, "cut_npe", metric_key)
    by_ft = aggregate_metric_by_single_axis(rows_in, "cut_firsttime", metric_key)

    finite_y: List[float] = []
    for series in (by_cfg, by_npe, by_ft):
        finite_y.extend(v for v in series.values() if np.isfinite(v))
    if not finite_y:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels: List[Tuple[Dict[float, float], str, str]] = [
        (by_cfg, "cfg_scale", "vs cfg_scale"),
        (by_npe, "cut_npe", "vs cut_npe"),
        (by_ft, "cut_firsttime", "vs cut_firsttime"),
    ]
    for ax, (series, xlabel, ptitle) in zip(axes, panels):
        xs = sorted(series.keys())
        ys = [series[x] for x in xs]
        ax.plot(xs, ys, marker="o", color="tab:blue")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(y_label)
        ax.set_title(ptitle)
        ax.grid(True, alpha=0.3)

    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path = plots_dir / output_filename
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def regenerate_event_summary_and_plots(rows: List[Dict[str, str]], summary_dir: Path) -> None:
    if not rows:
        return
    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = summary_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cfg_cut_groups = group_rows_by_cfg_cut(rows)
    cfg_cut_summary_rows = []
    for cfg_scale, cut_npe, cut_firsttime in sorted(cfg_cut_groups.keys()):
        group = cfg_cut_groups[(cfg_scale, cut_npe, cut_firsttime)]
        g = compute_cfg_global_metrics(group)
        term_means = compute_event_ers_term_means(group)
        cfg_cut_summary_rows.append({
            "cfg_scale": cfg_scale,
            "cut_npe": cut_npe,
            "cut_firsttime": cut_firsttime,
            "n_runs": len(group),
            "mean_ers_score": g["mean_ers_score"],
            "mean_npe_log_mae": mean_of(group, "npe_log_mae"),
            "mean_time_mae": mean_of(group, "time_mae"),
            "mean_active_dom_iou": mean_of(group, "active_dom_iou"),
            "fpd_like": g["fpd_like"],
            "cov_err": g["cov_err"],
            "sep_power_npe": g["sep_power_npe"],
            "sep_power_time": g["sep_power_time"],
            **term_means,
        })

    cfg_cut_summary_csv = summary_dir / "summary_by_cfg_cut.csv"
    with cfg_cut_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cfg_cut_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cfg_cut_summary_rows)

    cut_pairs = sorted({
        (row["cut_npe"], row["cut_firsttime"]) for row in cfg_cut_summary_rows
    })
    if cfg_cut_summary_rows:
        metric_plot_specs = [
            ("mean_ers_score", "mean_ers_vs_cfg_npecut_firsttimecut.png", "mean ERS", "Mean Event Reconstruction Score"),
            ("mean_ers_term_npe_log", "ers_term_npe_log_vs_cfg_npecut_firsttimecut.png", "mean_ers_term_npe_log", "ERS term: npe_log_mae"),
            ("mean_ers_term_time_mae_norm", "ers_term_time_mae_norm_vs_cfg_npecut_firsttimecut.png", "mean_ers_term_time_mae_norm", "ERS term: normalized time_mae"),
            ("mean_ers_term_iou_penalty", "ers_term_iou_penalty_vs_cfg_npecut_firsttimecut.png", "mean_ers_term_iou_penalty", "ERS term: IoU penalty"),
            ("mean_ers_term_npe_w1_norm", "ers_term_npe_w1_norm_vs_cfg_npecut_firsttimecut.png", "mean_ers_term_npe_w1_norm", "ERS term: normalized npe_wasserstein"),
            ("mean_ers_term_time_w1_norm", "ers_term_time_w1_norm_vs_cfg_npecut_firsttimecut.png", "mean_ers_term_time_w1_norm", "ERS term: normalized time_wasserstein"),
        ]
        saved_plot_paths: List[Path] = []
        for key, filename, y_label, suptitle in metric_plot_specs:
            out = plot_metric_three_axis_summary(
                rows_in=cfg_cut_summary_rows,
                metric_key=key,
                output_filename=filename,
                y_label=y_label,
                suptitle=suptitle,
                plots_dir=plots_dir,
            )
            if out is not None:
                saved_plot_paths.append(out)

        if saved_plot_paths:
            print("[INFO] Saved event 1x3 summary plots (cfg / cut_npe / cut_firsttime):")
            for p in saved_plot_paths:
                print(f"       - {p}")

        if len(cut_pairs) == 1:
            cfg_summary_csv = summary_dir / "summary_by_cfg.csv"
            with cfg_summary_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[k for k in cfg_cut_summary_rows[0].keys() if k not in {"cut_npe", "cut_firsttime"}])
                writer.writeheader()
                writer.writerows([{k: v for k, v in row.items() if k not in {"cut_npe", "cut_firsttime"}} for row in cfg_cut_summary_rows])

    sortable_rows = []
    for row in rows:
        score = _to_float(row.get("ers_score", np.nan))
        if np.isfinite(score):
            sortable_rows.append((score, row))
    sortable_rows.sort(key=lambda x: x[0])
    best_rows = [r for _, r in sortable_rows[:20]]
    best_csv = summary_dir / "best_samples_top20.csv"
    with best_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(best_rows)


# -----------------------------
# Multifield summary helpers
# -----------------------------
def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(as_jsonable(data), f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_line_with_band(x: np.ndarray, mean_true: np.ndarray, lo_true: np.ndarray, hi_true: np.ndarray,
                        mean_gen: np.ndarray, lo_gen: np.ndarray, hi_gen: np.ndarray,
                        xlabel: str, ylabel: str, title: str, save_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(x, mean_true, label="actual", color="black")
    plt.fill_between(x, lo_true, hi_true, color="gray", alpha=0.25)
    plt.plot(x, mean_gen, label="generated", color="tab:blue")
    plt.fill_between(x, lo_gen, hi_gen, color="tab:blue", alpha=0.20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_heatmap(matrix: np.ndarray, title: str, save_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    im = plt.imshow(matrix, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close()


def vector_covariance_chi2(sample_vec: np.ndarray, mu_true: np.ndarray, cov_true: np.ndarray, reg: float = 1e-8) -> float:
    sample_vec = np.asarray(sample_vec, dtype=float)
    mu_true = np.asarray(mu_true, dtype=float)
    cov_true = np.asarray(cov_true, dtype=float)
    valid = np.isfinite(sample_vec) & np.isfinite(mu_true)
    if valid.sum() < 2:
        return np.nan
    s = sample_vec[valid]
    mu = mu_true[valid]
    C = cov_true[np.ix_(valid, valid)]
    C = 0.5 * (C + C.T)
    C = C + np.eye(C.shape[0]) * reg
    try:
        inv = np.linalg.pinv(C)
    except Exception:
        return np.nan
    diff = s - mu
    chi2 = float(diff.T @ inv @ diff) / float(valid.sum())
    return chi2


def correlation_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return np.full((x.shape[1], x.shape[1]), np.nan)
    return np.corrcoef(x, rowvar=False)


def collect_bundle_rows(rows: List[Dict[str, str]]) -> List[Tuple[Dict[str, str], Dict[str, Any]]]:
    out = []
    for row in rows:
        bundle_path = row.get("bundle_path", "").strip()
        if not bundle_path:
            continue
        p = Path(bundle_path)
        if not p.exists():
            continue
        try:
            bundle = load_json(p)
        except Exception:
            continue
        out.append((row, bundle))
    return out


def unique_actual_auto_curves(bundle_rows: List[Tuple[Dict[str, str], Dict[str, Any]]], channel: str) -> np.ndarray:
    by_ref: Dict[int, np.ndarray] = {}
    for row, bundle in bundle_rows:
        ref_idx = _to_int(row.get("ref_idx", 0))
        curve = np.asarray(bundle["auto"]["actual"][channel], dtype=float)
        by_ref[ref_idx] = curve
    if not by_ref:
        return np.empty((0, 0))
    return np.stack([by_ref[k] for k in sorted(by_ref.keys())], axis=0)


def sample_auto_curves(bundle_rows: List[Tuple[Dict[str, str], Dict[str, Any]]], channel: str) -> np.ndarray:
    curves = []
    for _, bundle in bundle_rows:
        curves.append(np.asarray(bundle["auto"]["sample"][channel], dtype=float))
    return np.stack(curves, axis=0) if curves else np.empty((0, 0))


def unique_actual_curve_map(bundle_rows: List[Tuple[Dict[str, str], Dict[str, Any]]], kind: str, key: str) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for row, bundle in bundle_rows:
        ref_idx = _to_int(row.get("ref_idx", 0))
        out[ref_idx] = np.asarray(bundle[kind]["actual"][key], dtype=float)
    return out


def sample_curve_map(bundle_rows: List[Tuple[Dict[str, str], Dict[str, Any]]], kind: str, key: str) -> Dict[int, List[np.ndarray]]:
    out: Dict[int, List[np.ndarray]] = defaultdict(list)
    for row, bundle in bundle_rows:
        ref_idx = _to_int(row.get("ref_idx", 0))
        out[ref_idx].append(np.asarray(bundle[kind]["sample"][key], dtype=float))
    return out


def regenerate_multifield_summary_and_plots(rows: List[Dict[str, str]], summary_dir: Path, args: argparse.Namespace) -> None:
    if not rows:
        return
    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = summary_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _write_csv(path: Path, rows_out: List[Dict[str, Any]]) -> None:
        if not rows_out:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)

    def _safe_nanmean(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(np.mean(arr))

    def _robust_scale_map(value_map: Dict[float, float]) -> Dict[float, float]:
        keys = sorted(value_map.keys())
        vals = np.array([_to_float(value_map[k], np.nan) for k in keys], dtype=float)
        finite = np.isfinite(vals)
        if finite.sum() == 0:
            return {k: np.nan for k in keys}
        med = float(np.nanmedian(vals[finite]))
        q25 = float(np.nanpercentile(vals[finite], 25))
        q75 = float(np.nanpercentile(vals[finite], 75))
        iqr = q75 - q25
        if not np.isfinite(iqr) or iqr < 1e-12:
            mad = float(np.nanmedian(np.abs(vals[finite] - med)))
            scale = mad if np.isfinite(mad) and mad >= 1e-12 else float(np.nanstd(vals[finite]))
            if not np.isfinite(scale) or scale < 1e-12:
                scale = 1.0
        else:
            scale = iqr
        return {k: ((value_map[k] - med) / scale) if np.isfinite(_to_float(value_map[k], np.nan)) else np.nan for k in keys}

    def _weighted_mean_from_dict(score_map: Dict[str, float], weight_map: Dict[str, float]) -> float:
        vals = []
        weights = []
        for key, weight in weight_map.items():
            value = _to_float(score_map.get(key, np.nan), np.nan)
            if np.isfinite(value):
                vals.append(value)
                weights.append(weight)
        if not vals:
            return np.nan
        w = np.asarray(weights, dtype=float)
        v = np.asarray(vals, dtype=float)
        if np.sum(w) <= 0:
            return np.nan
        return float(np.dot(v, w) / np.sum(w))

    def _mean_ers_from_group(group: List[Dict[str, str]]) -> float:
        """구버전 로그는 multifield 행에 ers_score가 비어 있을 수 있어 total_score로 대체한다."""
        v = mean_of(group, "ers_score")
        if np.isfinite(v):
            return v
        return mean_of(group, "total_score")

    cfg_groups = group_rows_by_cfg(rows)
    cfg_cut_groups = group_rows_by_cfg_cut(rows)
    ref_cfg_groups = group_rows_by_ref_cfg(rows)

    cfg_summary_rows = []
    for cfg_scale in sorted(cfg_groups.keys()):
        group = cfg_groups[cfg_scale]
        cfg_summary_rows.append({
            "cfg_scale": cfg_scale,
            "n_runs": len(group),
            "mean_total_score": mean_of(group, "total_score"),
            "mean_ers_score": _mean_ers_from_group(group),
            "mean_pdf_score": mean_of(group, "pdf_score"),
            "mean_auto_score": mean_of(group, "auto_score"),
            "mean_cross_score": mean_of(group, "cross_score"),
            "mean_coherence_score": mean_of(group, "coherence_score"),
            "mean_bispectrum_score": mean_of(group, "bispectrum_score"),
            "parameter_response_consistency": np.nan,
            "cv_score": np.nan,
            "ees_score": np.nan,
            "scaled_mean_ers_score": np.nan,
            "overall_score": np.nan,
        })
    cfg_summary_map = {float(r["cfg_scale"]): r for r in cfg_summary_rows}

    cfg_cut_summary_rows = []
    for cfg_scale, cut_npe, cut_firsttime in sorted(cfg_cut_groups.keys()):
        group = cfg_cut_groups[(cfg_scale, cut_npe, cut_firsttime)]
        cfg_cut_summary_rows.append({
            "cfg_scale": cfg_scale,
            "cut_npe": cut_npe,
            "cut_firsttime": cut_firsttime,
            "n_runs": len(group),
            "mean_total_score": mean_of(group, "total_score"),
            "mean_ers_score": _mean_ers_from_group(group),
            "mean_pdf_score": mean_of(group, "pdf_score"),
            "mean_auto_score": mean_of(group, "auto_score"),
            "mean_cross_score": mean_of(group, "cross_score"),
            "mean_coherence_score": mean_of(group, "coherence_score"),
            "mean_bispectrum_score": mean_of(group, "bispectrum_score"),
            # 5번·6번 지표: cut_npe별 집계 (diversity 계산 후 채워짐)
            "parameter_response_consistency": np.nan,
            "cv_score": np.nan,
        })

    ref_cfg_summary_rows = []
    for (ref_idx, cfg_scale), group in sorted(ref_cfg_groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        ref_cfg_summary_rows.append({
            "ref_idx": ref_idx,
            "cfg_scale": cfg_scale,
            "n_runs": len(group),
            "mean_total_score": mean_of(group, "total_score"),
            "mean_ers_score": _mean_ers_from_group(group),
            "mean_pdf_score": mean_of(group, "pdf_score"),
            "mean_auto_score": mean_of(group, "auto_score"),
            "mean_cross_score": mean_of(group, "cross_score"),
            "mean_coherence_score": mean_of(group, "coherence_score"),
            "mean_bispectrum_score": mean_of(group, "bispectrum_score"),
        })

    bundle_rows = collect_bundle_rows(rows)
    if bundle_rows:
        first_bundle = bundle_rows[0][1]
        channel_names = list(first_bundle.get("channel_names", []))
        pair_names = list(first_bundle.get("cross", {}).get("actual", {}).keys())
        k = np.asarray(first_bundle.get("k_centers", []), dtype=float)
    else:
        channel_names = []
        pair_names = []
        k = np.asarray([], dtype=float)

    diversity_rows = []
    grouped = defaultdict(list)
    for row, bundle in bundle_rows:
        grouped[(_to_float(row.get("cfg_scale", np.nan)), row.get("label_key", ""))].append((row, bundle))

    n_div_plots = 0
    for (cfg_scale, label_key), group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if not label_key:
            continue
        if len({_to_int(r[0].get("ref_idx", 0)) for r in group}) < max(2, args.min_diversity_actual_refs):
            continue
        for ch in channel_names:
            actual_mat = unique_actual_auto_curves(group, ch)
            sample_mat = sample_auto_curves(group, ch)
            if actual_mat.shape[0] < max(2, args.min_diversity_actual_refs) or sample_mat.shape[0] < max(2, args.min_diversity_samples):
                continue
            mu_a = np.nanmean(actual_mat, axis=0)
            mu_s = np.nanmean(sample_mat, axis=0)
            lo_a = np.nanpercentile(actual_mat, 16, axis=0)
            hi_a = np.nanpercentile(actual_mat, 84, axis=0)
            lo_s = np.nanpercentile(sample_mat, 16, axis=0)
            hi_s = np.nanpercentile(sample_mat, 84, axis=0)
            cov_a = np.cov(actual_mat, rowvar=False)
            chi2_vals = [vector_covariance_chi2(v, mu_a, cov_a) for v in sample_mat]
            std_a = np.nanstd(actual_mat, axis=0)
            std_s = np.nanstd(sample_mat, axis=0)
            valid_std = np.isfinite(std_a) & np.isfinite(std_s) & (std_a > 1e-12) & (std_s > 1e-12)
            cv_std_ratio_auto = float(np.mean(np.abs(np.log(std_s[valid_std] / std_a[valid_std])))) if np.any(valid_std) else np.nan
            corr_a = correlation_matrix(actual_mat)
            corr_s = correlation_matrix(sample_mat)
            cv_modecorr_fro_auto = np.nan
            if corr_a.shape == corr_s.shape and corr_a.size > 0:
                diff = corr_a - corr_s
                diff = diff[np.isfinite(diff)]
                if diff.size > 0:
                    cv_modecorr_fro_auto = float(np.linalg.norm(diff.ravel())) / float(max(corr_a.shape[0], 1))
            diversity_rows.append({
                "cfg_scale": cfg_scale,
                "label_key": label_key,
                "channel": ch,
                "n_actual_refs": actual_mat.shape[0],
                "n_generated_samples": sample_mat.shape[0],
                "cv_chi2red_auto": _safe_nanmean(chi2_vals),
                "cv_std_ratio_auto": cv_std_ratio_auto,
                "cv_modecorr_fro_auto": cv_modecorr_fro_auto,
                "mean_std_actual": float(np.nanmean(std_a)),
                "mean_std_generated": float(np.nanmean(std_s)),
            })
            if n_div_plots < args.max_diversity_plot_groups and k.size > 0:
                label_tag = short_hash(label_key)
                plot_line_with_band(
                    k, mu_a, lo_a, hi_a, mu_s, lo_s, hi_s,
                    xlabel="k", ylabel=f"P_{ch}(k)",
                    title=f"Auto-power envelope | cfg={cfg_scale} | {ch} | label={label_tag}",
                    save_path=plots_dir / "diversity" / f"auto_env_cfg_{cfg_scale}_label_{label_tag}_{ch}.png",
                )
                plot_heatmap(corr_a, f"Actual mode corr | cfg={cfg_scale} | {ch} | {label_tag}",
                             plots_dir / "diversity" / f"corr_actual_cfg_{cfg_scale}_label_{label_tag}_{ch}.png")
                plot_heatmap(corr_s, f"Generated mode corr | cfg={cfg_scale} | {ch} | {label_tag}",
                             plots_dir / "diversity" / f"corr_generated_cfg_{cfg_scale}_label_{label_tag}_{ch}.png")
                n_div_plots += 1

    cv_summary_rows = []
    if diversity_rows:
        grouped_cv = defaultdict(list)
        for row in diversity_rows:
            grouped_cv[_to_float(row["cfg_scale"])].append(row)
        for cfg_scale, group in sorted(grouped_cv.items()):
            cv_std_vals = [_to_float(r["cv_std_ratio_auto"]) for r in group]
            cv_mode_vals = [_to_float(r["cv_modecorr_fro_auto"]) for r in group]
            cv_chi2_vals = [_to_float(r["cv_chi2red_auto"]) for r in group]
            cv_score = _safe_nanmean(cv_std_vals + cv_mode_vals + cv_chi2_vals)
            row = {
                "cfg_scale": cfg_scale,
                "n_groups": len(group),
                "mean_cv_std_ratio_auto": _safe_nanmean(cv_std_vals),
                "mean_cv_modecorr_fro_auto": _safe_nanmean(cv_mode_vals),
                "mean_cv_chi2red_auto": _safe_nanmean(cv_chi2_vals),
                "cv_score": cv_score,
            }
            cv_summary_rows.append(row)
            if cfg_scale in cfg_summary_map:
                cfg_summary_map[cfg_scale]["cv_score"] = cv_score
            # cfg_cut_summary_rows에도 동일 cfg의 모든 cut 행에 반영
            for cut_row in cfg_cut_summary_rows:
                if _to_float(cut_row.get("cfg_scale", np.nan), np.nan) == cfg_scale:
                    cut_row["cv_score"] = cv_score

    parameter_response_rows = []
    if bundle_rows and args.response_dim is not None and args.fiducial_ref_idx is not None:
        by_cfg_ref = defaultdict(list)
        ref_label: Dict[int, List[float]] = {}
        for row, bundle in bundle_rows:
            cfg = _to_float(row.get("cfg_scale", np.nan))
            ref = _to_int(row.get("ref_idx", 0))
            by_cfg_ref[(cfg, ref)].append((row, bundle))
            if bundle.get("label_vector") is not None:
                ref_label[ref] = [float(v) for v in bundle.get("label_vector")]
            else:
                raw = row.get("label_json", "")
                if raw:
                    try:
                        ref_label[ref] = [float(v) for v in json.loads(raw)]
                    except Exception:
                        pass
        if args.fiducial_ref_idx in ref_label:
            fid_label = ref_label[args.fiducial_ref_idx]
            dim = int(args.response_dim)
            if 0 <= dim < len(fid_label):
                matching_refs = []
                for ref, label in ref_label.items():
                    if len(label) != len(fid_label):
                        continue
                    ok = True
                    for i, (a, b) in enumerate(zip(label, fid_label)):
                        if i == dim:
                            continue
                        if abs(a - b) > args.response_tol:
                            ok = False
                            break
                    if ok:
                        matching_refs.append(ref)
                matching_refs = sorted(set(matching_refs), key=lambda r: ref_label[r][dim])
                matching_refs = matching_refs[: args.max_response_values]
                for cfg_scale in sorted({k[0] for k in by_cfg_ref.keys()}):
                    if (cfg_scale, args.fiducial_ref_idx) not in by_cfg_ref:
                        continue
                    fid_group = by_cfg_ref[(cfg_scale, args.fiducial_ref_idx)]
                    fid_bundle = fid_group[0][1]
                    for ch in channel_names:
                        fid_actual = np.asarray(fid_bundle["auto"]["actual"][ch], dtype=float)
                        fid_sample_curves = [np.asarray(b[1]["auto"]["sample"][ch], dtype=float) for b in fid_group]
                        fid_sample_mean = np.nanmean(np.stack(fid_sample_curves, axis=0), axis=0)
                        for ref in matching_refs:
                            if ref == args.fiducial_ref_idx or (cfg_scale, ref) not in by_cfg_ref:
                                continue
                            group = by_cfg_ref[(cfg_scale, ref)]
                            rep_bundle = group[0][1]
                            actual_curve = np.asarray(rep_bundle["auto"]["actual"][ch], dtype=float)
                            sample_curves = [np.asarray(g[1]["auto"]["sample"][ch], dtype=float) for g in group]
                            sample_mean = np.nanmean(np.stack(sample_curves, axis=0), axis=0)
                            actual_ratio = np.divide(actual_curve, fid_actual, out=np.full_like(actual_curve, np.nan), where=np.isfinite(fid_actual) & (np.abs(fid_actual) > 1e-12))
                            sample_ratio = np.divide(sample_mean, fid_sample_mean, out=np.full_like(sample_mean, np.nan), where=np.isfinite(fid_sample_mean) & (np.abs(fid_sample_mean) > 1e-12))
                            parameter_response_rows.append({
                                "cfg_scale": cfg_scale,
                                "kind": "auto",
                                "key": ch,
                                "ref_idx": ref,
                                "fiducial_ref_idx": args.fiducial_ref_idx,
                                "response_dim": dim,
                                "response_value": ref_label[ref][dim],
                                "mean_abs_ratio_error": relative_curve_error(actual_ratio, sample_ratio),
                            })
                            if k.size > 0:
                                plt.figure(figsize=(8, 5))
                                plt.plot(k, actual_ratio, label="actual ratio", color="black")
                                plt.plot(k, sample_ratio, label="generated ratio", color="tab:blue")
                                plt.xlabel("k")
                                plt.ylabel("response ratio")
                                plt.title(f"1P auto response | cfg={cfg_scale} | {ch} | ref={ref} vs fid={args.fiducial_ref_idx}")
                                plt.grid(True, alpha=0.3)
                                plt.legend()
                                plt.tight_layout()
                                save_path = plots_dir / "response" / f"response_auto_cfg_{cfg_scale}_{ch}_ref_{ref}_fid_{args.fiducial_ref_idx}.png"
                                save_path.parent.mkdir(parents=True, exist_ok=True)
                                plt.savefig(save_path, dpi=180)
                                plt.close()
                    for kind in ("cross", "coherence"):
                        for key in pair_names:
                            fid_actual = np.asarray(fid_bundle[kind]["actual"][key], dtype=float)
                            fid_sample_curves = [np.asarray(b[1][kind]["sample"][key], dtype=float) for b in fid_group]
                            fid_sample_mean = np.nanmean(np.stack(fid_sample_curves, axis=0), axis=0)
                            for ref in matching_refs:
                                if ref == args.fiducial_ref_idx or (cfg_scale, ref) not in by_cfg_ref:
                                    continue
                                group = by_cfg_ref[(cfg_scale, ref)]
                                rep_bundle = group[0][1]
                                actual_curve = np.asarray(rep_bundle[kind]["actual"][key], dtype=float)
                                sample_curves = [np.asarray(g[1][kind]["sample"][key], dtype=float) for g in group]
                                sample_mean = np.nanmean(np.stack(sample_curves, axis=0), axis=0)
                                actual_ratio = np.divide(actual_curve, fid_actual, out=np.full_like(actual_curve, np.nan), where=np.isfinite(fid_actual) & (np.abs(fid_actual) > 1e-12))
                                sample_ratio = np.divide(sample_mean, fid_sample_mean, out=np.full_like(sample_mean, np.nan), where=np.isfinite(fid_sample_mean) & (np.abs(fid_sample_mean) > 1e-12))
                                parameter_response_rows.append({
                                    "cfg_scale": cfg_scale,
                                    "kind": kind,
                                    "key": key,
                                    "ref_idx": ref,
                                    "fiducial_ref_idx": args.fiducial_ref_idx,
                                    "response_dim": dim,
                                    "response_value": ref_label[ref][dim],
                                    "mean_abs_ratio_error": relative_curve_error(actual_ratio, sample_ratio),
                                })
                                if k.size > 0:
                                    plt.figure(figsize=(8, 5))
                                    plt.plot(k, actual_ratio, label="actual ratio", color="black")
                                    plt.plot(k, sample_ratio, label="generated ratio", color="tab:blue")
                                    plt.xlabel("k")
                                    plt.ylabel("response ratio")
                                    plt.title(f"1P {kind} response | cfg={cfg_scale} | {key} | ref={ref} vs fid={args.fiducial_ref_idx}")
                                    plt.grid(True, alpha=0.3)
                                    plt.legend()
                                    plt.tight_layout()
                                    save_path = plots_dir / "response" / f"response_{kind}_cfg_{cfg_scale}_{key}_ref_{ref}_fid_{args.fiducial_ref_idx}.png"
                                    save_path.parent.mkdir(parents=True, exist_ok=True)
                                    plt.savefig(save_path, dpi=180)
                                    plt.close()
        if parameter_response_rows:
            grouped_pr = defaultdict(list)
            for row in parameter_response_rows:
                grouped_pr[_to_float(row["cfg_scale"])].append(row)
            for cfg_scale, group in sorted(grouped_pr.items()):
                consistency = _safe_nanmean([_to_float(r["mean_abs_ratio_error"]) for r in group])
                if cfg_scale in cfg_summary_map:
                    cfg_summary_map[cfg_scale]["parameter_response_consistency"] = consistency
                # cfg_cut_summary_rows에도 동일 cfg의 모든 cut 행에 반영
                for cut_row in cfg_cut_summary_rows:
                    if _to_float(cut_row.get("cfg_scale", np.nan), np.nan) == cfg_scale:
                        cut_row["parameter_response_consistency"] = consistency

    # ── cfg_cut_summary_rows의 mean_total_score 재계산 ───────────────────────────────
    # 5번(parameter_response_consistency)·6번(cv_score)를 포함하고,
    # 각 지표에 log1p → median ratio normalize를 적용해 스케일 차이로 한 지표가 튀는 문제를 방지한다.
    #
    # 적용 순서:
    #   1. actual-scale 정규화는 per-sample 계산 시 이미 반영됨 (유지)
    #   2. 각 지표에 log1p(x) 적용 → 큰 값의 동적 범위를 압축
    #   3. 전체 cut 행에서 지표별 median을 구해 median ratio로 정규화
    #      → 각 지표가 "median 기준 1.0" 수준으로 통일됨
    #   4. 기존 가중치 적용

    _metric_keys_for_total = [
        "mean_pdf_score", "mean_auto_score", "mean_cross_score",
        "mean_coherence_score", "mean_bispectrum_score",
        "parameter_response_consistency", "cv_score",
    ]
    _weight_keys_no_bis = {
        "mean_pdf_score": 0.15, "mean_auto_score": 0.20,
        "mean_cross_score": 0.15, "mean_coherence_score": 0.15,
        "parameter_response_consistency": 0.20, "cv_score": 0.15,
    }
    _weight_keys_bis = {
        "mean_pdf_score": 0.12, "mean_auto_score": 0.18,
        "mean_cross_score": 0.14, "mean_coherence_score": 0.14,
        "parameter_response_consistency": 0.18, "cv_score": 0.12,
        "mean_bispectrum_score": 0.12,
    }

    # step 1: log1p 변환 (음수 방어: max(0, x))
    def _log1p_safe(x: float) -> float:
        if not np.isfinite(x):
            return np.nan
        return float(np.log1p(max(0.0, x)))

    # step 2: 지표별 median 계산 (log1p 공간에서)
    _log1p_vals: Dict[str, List[float]] = {k: [] for k in _metric_keys_for_total}
    for cut_row in cfg_cut_summary_rows:
        for k in _metric_keys_for_total:
            v = _log1p_safe(_to_float(cut_row.get(k, np.nan), np.nan))
            if np.isfinite(v):
                _log1p_vals[k].append(v)

    _medians: Dict[str, float] = {}
    for k, vals in _log1p_vals.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        _medians[k] = float(np.median(arr)) if arr.size > 0 else np.nan

    # step 3 & 4: 각 행에 대해 정규화 후 가중합으로 mean_total_score 재계산
    any_bis_global = any(
        np.isfinite(_to_float(r.get("mean_bispectrum_score", np.nan), np.nan))
        for r in cfg_cut_summary_rows
    )
    _w_total = _weight_keys_bis if any_bis_global else _weight_keys_no_bis

    for cut_row in cfg_cut_summary_rows:
        weighted_sum = 0.0
        weight_sum = 0.0
        for k, w in _w_total.items():
            raw = _to_float(cut_row.get(k, np.nan), np.nan)
            log_v = _log1p_safe(raw)
            if not np.isfinite(log_v):
                continue
            med = _medians.get(k, np.nan)
            if not np.isfinite(med) or med < 1e-12:
                # median이 0에 가까우면 절댓값 그대로 사용
                normalized = log_v
            else:
                normalized = log_v / med   # median ratio normalize
            weighted_sum += w * normalized
            weight_sum += w
        if weight_sum > 1e-12:
            cut_row["mean_total_score"] = weighted_sum / weight_sum

    # cfg_summary_map의 mean_total_score도 동일한 log1p + median ratio normalize로 재계산
    # (cfg_cut_summary_rows와 _medians를 공유하여 일관된 스케일 적용)
    any_bispectrum = any(np.isfinite(_to_float(r.get("mean_bispectrum_score", np.nan), np.nan)) for r in cfg_summary_rows)
    _w_cfg = _weight_keys_bis if any_bispectrum else _weight_keys_no_bis
    for row in cfg_summary_rows:
        weighted_sum = 0.0
        weight_sum = 0.0
        for k, w in _w_cfg.items():
            raw = _to_float(row.get(k, np.nan), np.nan)
            log_v = _log1p_safe(raw)
            if not np.isfinite(log_v):
                continue
            med = _medians.get(k, np.nan)
            if not np.isfinite(med) or med < 1e-12:
                normalized = log_v
            else:
                normalized = log_v / med
            weighted_sum += w * normalized
            weight_sum += w
        if weight_sum > 1e-12:
            row["mean_total_score"] = weighted_sum / weight_sum
            cfg_summary_map[float(row["cfg_scale"])]["mean_total_score"] = row["mean_total_score"]
    ers_for_scaling = {}
    for cfg_scale, row in cfg_summary_map.items():
        ers_proxy = mean_of(cfg_groups[cfg_scale], "ers_score")
        if not np.isfinite(ers_proxy):
            ers_proxy = _to_float(row.get("mean_total_score", np.nan), np.nan)
        ers_for_scaling[cfg_scale] = ers_proxy
    scaled_ers = _robust_scale_map(ers_for_scaling)

    # EES 계산용: EES key -> _metric_keys_for_total key 매핑 (median 참조용)
    _ees_to_metric = {
        "pdf":        "mean_pdf_score",
        "auto":       "mean_auto_score",
        "cross":      "mean_cross_score",
        "coherence":  "mean_coherence_score",
        "response":   "parameter_response_consistency",
        "cv":         "cv_score",
        "bispectrum": "mean_bispectrum_score",
    }

    def _compute_ees_scaled(raw_score_map, ees_weights_map, medians, ees_to_metric):
        # log1p + median ratio normalize 후 가중평균
        ws = 0.0
        wt = 0.0
        for key, w in ees_weights_map.items():
            v = _to_float(raw_score_map.get(key, np.nan), np.nan)
            log_v = _log1p_safe(v)
            if not np.isfinite(log_v):
                continue
            mkey = ees_to_metric.get(key, key)
            med = medians.get(mkey, np.nan)
            normalized = log_v if (not np.isfinite(med) or med < 1e-12) else log_v / med
            ws += w * normalized
            wt += w
        return ws / wt if wt > 1e-12 else np.nan

    for cfg_scale, row in cfg_summary_map.items():
        score_map = {
            "pdf":       _to_float(row.get("mean_pdf_score", np.nan), np.nan),
            "auto":      _to_float(row.get("mean_auto_score", np.nan), np.nan),
            "cross":     _to_float(row.get("mean_cross_score", np.nan), np.nan),
            "coherence": _to_float(row.get("mean_coherence_score", np.nan), np.nan),
            "response":  _to_float(row.get("parameter_response_consistency", np.nan), np.nan),
            "cv":        _to_float(row.get("cv_score", np.nan), np.nan),
            "bispectrum":_to_float(row.get("mean_bispectrum_score", np.nan), np.nan),
        }
        if any_bispectrum:
            ees_weights = {"pdf": 0.12, "auto": 0.18, "cross": 0.14, "coherence": 0.14,
                           "response": 0.18, "cv": 0.12, "bispectrum": 0.12}
        else:
            ees_weights = {"pdf": 0.15, "auto": 0.20, "cross": 0.15, "coherence": 0.15,
                           "response": 0.20, "cv": 0.15}
        ees_score = _compute_ees_scaled(score_map, ees_weights, _medians, _ees_to_metric)
        row["ees_score"] = ees_score
        row["scaled_mean_ers_score"] = scaled_ers.get(cfg_scale, np.nan)
        row["overall_score"] = (
            0.25 * row["scaled_mean_ers_score"] + 0.75 * ees_score
            if np.isfinite(_to_float(row["scaled_mean_ers_score"], np.nan)) and np.isfinite(_to_float(ees_score, np.nan))
            else ees_score
        )

    # cut_npe 스윕 플롯용 EES: log1p + median ratio normalize 적용
    for row in cfg_cut_summary_rows:
        cfg_scale = _to_float(row.get("cfg_scale", np.nan), np.nan)
        score_map = {
            "pdf":       _to_float(row.get("mean_pdf_score", np.nan), np.nan),
            "auto":      _to_float(row.get("mean_auto_score", np.nan), np.nan),
            "cross":     _to_float(row.get("mean_cross_score", np.nan), np.nan),
            "coherence": _to_float(row.get("mean_coherence_score", np.nan), np.nan),
            "response":  _to_float(row.get("parameter_response_consistency", np.nan), np.nan),
            "cv":        _to_float(row.get("cv_score", np.nan), np.nan),
            "bispectrum":_to_float(row.get("mean_bispectrum_score", np.nan), np.nan),
        }
        if any_bispectrum:
            ees_weights = {"pdf": 0.12, "auto": 0.18, "cross": 0.14, "coherence": 0.14,
                           "response": 0.18, "cv": 0.12, "bispectrum": 0.12}
        else:
            ees_weights = {"pdf": 0.15, "auto": 0.20, "cross": 0.15, "coherence": 0.15,
                           "response": 0.20, "cv": 0.15}
        mean_ees = _compute_ees_scaled(score_map, ees_weights, _medians, _ees_to_metric)
        row["mean_ees_score"] = mean_ees
        cfg_scaled_ers = _to_float(cfg_summary_map.get(cfg_scale, {}).get("scaled_mean_ers_score", np.nan), np.nan)
        row["mean_overall_score"] = (
            0.25 * cfg_scaled_ers + 0.75 * mean_ees
            if np.isfinite(cfg_scaled_ers) and np.isfinite(_to_float(mean_ees, np.nan))
            else mean_ees
        )

    final_cfg_rows = [cfg_summary_map[cfg] for cfg in sorted(cfg_summary_map.keys())]
    _write_csv(summary_dir / "ensemble_summary_by_cfg.csv", final_cfg_rows)
    _write_csv(summary_dir / "summary_by_cfg.csv", final_cfg_rows)
    _write_csv(summary_dir / "summary_by_cfg_cut.csv", cfg_cut_summary_rows)
    _write_csv(summary_dir / "ensemble_summary_by_ref_cfg.csv", ref_cfg_summary_rows)
    _write_csv(summary_dir / "parameter_response_summary.csv", parameter_response_rows)
    _write_csv(summary_dir / "cv_summary.csv", cv_summary_rows)
    _write_csv(summary_dir / "diversity_by_cfg_label_channel.csv", diversity_rows)

    no_cfg_plots = bool(getattr(args, "no_cfg_plots", False))
    no_cut_npe_plots = bool(getattr(args, "no_cut_npe_plots", False))

    xs = [r["cfg_scale"] for r in final_cfg_rows]
    metric_keys = [
        ("mean_total_score", "total_score_vs_cfg.png", "Mean total score vs cfg_scale"),
        ("mean_ers_score", "ers_score_vs_cfg.png", "Mean ERS score vs cfg_scale"),
        ("mean_pdf_score", "pdf_score_vs_cfg.png", "Mean PDF score vs cfg_scale"),
        ("mean_auto_score", "auto_score_vs_cfg.png", "Mean auto-power score vs cfg_scale"),
        ("mean_cross_score", "cross_score_vs_cfg.png", "Mean cross-power score vs cfg_scale"),
        ("mean_coherence_score", "coherence_score_vs_cfg.png", "Mean coherence score vs cfg_scale"),
        ("mean_bispectrum_score", "bispectrum_score_vs_cfg.png", "Mean bispectrum score vs cfg_scale"),
        ("parameter_response_consistency", "parameter_response_consistency_vs_cfg.png", "Parameter-response consistency vs cfg_scale"),
        ("cv_score", "cv_score_vs_cfg.png", "CV score vs cfg_scale"),
        ("ees_score", "ees_vs_cfg.png", "EES vs cfg_scale"),
        ("overall_score", "overall_score_vs_cfg.png", "Overall score vs cfg_scale"),
    ]
    _always_plot_cfg_keys = {"parameter_response_consistency", "cv_score"}
    if not no_cfg_plots:
        for key, filename, title in metric_keys:
            ys = [r.get(key, np.nan) for r in final_cfg_rows]
            has_data = any(np.isfinite(_to_float(y, np.nan)) for y in ys)
            if not has_data and key not in _always_plot_cfg_keys:
                continue
            plt.figure(figsize=(8, 5))
            if has_data:
                plt.plot(xs, ys, marker="o")
            else:
                # 데이터 없음 안내 플롯
                if key == "parameter_response_consistency":
                    msg = ("parameter_response_consistency를 계산하려면\n"
                           "--response_dim 과 --fiducial_ref_idx 를 지정하세요.")
                else:
                    msg = ("cv_score를 계산하려면\n"
                           "동일 label + cfg 조건의 ref가 2개 이상 필요합니다.")
                plt.text(0.5, 0.5, msg,
                         ha="center", va="center", fontsize=11,
                         transform=plt.gca().transAxes,
                         bbox=dict(boxstyle="round,pad=0.5", fc="#fff3cd", ec="#ffc107"))
            plt.xlabel("cfg_scale")
            plt.ylabel(key)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / filename, dpi=180)
            plt.close()

    cut_pairs = sorted({(r["cut_npe"], r["cut_firsttime"]) for r in cfg_cut_summary_rows})
    if (not no_cfg_plots) and cfg_cut_summary_rows and len(cut_pairs) > 1:
        cut_plot_specs = [
            ("mean_total_score", "mean_total_score_vs_cfg_by_cut.png", "Mean total score vs cfg_scale by cut"),
            ("mean_pdf_score", "mean_pdf_score_vs_cfg_by_cut.png", "Mean PDF score vs cfg_scale by cut"),
            ("mean_auto_score", "mean_auto_score_vs_cfg_by_cut.png", "Mean auto score vs cfg_scale by cut"),
        ]
        for key, filename, title in cut_plot_specs:
            plt.figure(figsize=(8, 5))
            plotted = False
            for cut_npe, cut_firsttime in cut_pairs:
                sub = [
                    r for r in cfg_cut_summary_rows
                    if r["cut_npe"] == cut_npe and r["cut_firsttime"] == cut_firsttime
                ]
                sub = sorted(sub, key=lambda r: r["cfg_scale"])
                xs_sub = [r["cfg_scale"] for r in sub]
                ys_sub = [r.get(key, np.nan) for r in sub]
                if not any(np.isfinite(_to_float(y, np.nan)) for y in ys_sub):
                    continue
                plotted = True
                plt.plot(xs_sub, ys_sub, marker="o", label=f"npe={cut_npe}, t={cut_firsttime}")
            if not plotted:
                plt.close()
                continue
            plt.xlabel("cfg_scale")
            plt.ylabel(key)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / filename, dpi=180)
            plt.close()

    # cut_npe 스윕 중심 분석용: x축=cut_npe, 라인=(cfg_scale, cut_firsttime)
    if cfg_cut_summary_rows and not no_cut_npe_plots:
        cut_npe_plot_specs = [
            ("mean_total_score", "mean_total_score_vs_cut_npe.png", "Mean total score vs cut_npe"),
            ("mean_ees_score", "mean_ees_score_vs_cut_npe.png", "Mean EES score vs cut_npe"),
            ("mean_overall_score", "mean_overall_score_vs_cut_npe.png", "Mean overall score vs cut_npe"),
            ("mean_ers_score", "mean_ers_score_vs_cut_npe.png", "Mean ERS score vs cut_npe"),
            ("mean_pdf_score", "mean_pdf_score_vs_cut_npe.png", "Mean PDF score vs cut_npe"),
            ("mean_auto_score", "mean_auto_score_vs_cut_npe.png", "Mean auto score vs cut_npe"),
            ("mean_cross_score", "mean_cross_score_vs_cut_npe.png", "Mean cross score vs cut_npe"),
            ("mean_coherence_score", "mean_coherence_score_vs_cut_npe.png", "Mean coherence score vs cut_npe"),
            ("mean_bispectrum_score", "mean_bispectrum_score_vs_cut_npe.png", "Mean bispectrum score vs cut_npe"),
            # 5번 지표: parameter-response consistency vs cut_npe
            ("parameter_response_consistency", "mean_parameter_response_vs_cut_npe.png",
             "Parameter-response consistency vs cut_npe"),
            # 6번 지표: cv_score (diversity / cosmic-variance fidelity) vs cut_npe
            ("cv_score", "mean_cv_score_vs_cut_npe.png", "CV score (diversity) vs cut_npe"),
        ]
        line_keys = sorted({(r["cfg_scale"], r["cut_firsttime"]) for r in cfg_cut_summary_rows})
        # cv_score / parameter_response_consistency는 조건이 맞지 않으면 NaN이므로,
        # 이 두 지표는 "데이터 없음" 안내 플롯을 항상 저장한다.
        _always_plot_keys = {"parameter_response_consistency", "cv_score"}

        for key, filename, title in cut_npe_plot_specs:
            plt.figure(figsize=(8, 5))
            plotted = False
            for cfg_scale, cut_firsttime in line_keys:
                sub = [
                    r for r in cfg_cut_summary_rows
                    if r["cfg_scale"] == cfg_scale and r["cut_firsttime"] == cut_firsttime
                ]
                sub = sorted(sub, key=lambda r: r["cut_npe"])
                xs_sub = [r["cut_npe"] for r in sub]
                ys_sub = [r.get(key, np.nan) for r in sub]
                if not any(np.isfinite(_to_float(y, np.nan)) for y in ys_sub):
                    continue
                plotted = True
                plt.plot(xs_sub, ys_sub, marker="o", label=f"cfg={cfg_scale}, t={cut_firsttime}")
            if not plotted:
                if key in _always_plot_keys:
                    # 데이터는 없지만 "계산 조건 미충족" 안내 플롯 저장
                    if key == "parameter_response_consistency":
                        msg = "parameter_response_consistency를 계산하려면\n--response_dim 과 --fiducial_ref_idx 를 지정하세요."
                    else:  # cv_score
                        msg = "cv_score를 계산하려면\n동일 label + cfg 조건의 ref가 2개 이상 필요합니다."
                    plt.text(0.5, 0.5, msg,
                             ha="center", va="center", fontsize=11,
                             transform=plt.gca().transAxes,
                             bbox=dict(boxstyle="round,pad=0.5", fc="#fff3cd", ec="#ffc107"))
                    plt.xlabel("cut_npe")
                    plt.ylabel(key)
                    plt.title(title)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(plots_dir / filename, dpi=180)
                plt.close()
                continue
            plt.xlabel("cut_npe")
            plt.ylabel(key)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / filename, dpi=180)
            plt.close()

        # cut_npe heatmap: cfg별로 (y=ref_idx, x=cut_npe)
        heat_specs = [
            ("mean_total_score", "heatmap_cut_npe_total_cfg_{cfg_tag}.png", "Mean total score heatmap vs cut_npe"),
            ("mean_ees_score", "heatmap_cut_npe_ees_cfg_{cfg_tag}.png", "Mean EES score heatmap vs cut_npe"),
            ("mean_overall_score", "heatmap_cut_npe_overall_cfg_{cfg_tag}.png", "Mean overall score heatmap vs cut_npe"),
        ]
        cfg_values = sorted({_to_float(r["cfg_scale"]) for r in cfg_cut_summary_rows})
        for cfg_scale in cfg_values:
            row_cfg = [r for r in rows if _to_float(r.get("cfg_scale", np.nan), np.nan) == cfg_scale]
            if not row_cfg:
                continue
            npe_vals = sorted({_to_float(r.get("cut_npe", np.nan), np.nan) for r in row_cfg})
            ref_vals = sorted({_to_int(r.get("ref_idx", 0)) for r in row_cfg})
            if not npe_vals or not ref_vals:
                continue
            cfg_scaled_ers = _to_float(cfg_summary_map.get(cfg_scale, {}).get("scaled_mean_ers_score", np.nan), np.nan)
            for key, filename_tpl, title in heat_specs:
                heat = np.full((len(ref_vals), len(npe_vals)), np.nan, dtype=float)
                for i, ref_idx in enumerate(ref_vals):
                    for j, npe in enumerate(npe_vals):
                        sub_cell = [
                            r for r in row_cfg
                            if _to_int(r.get("ref_idx", 0)) == ref_idx
                            and _to_float(r.get("cut_npe", np.nan), np.nan) == npe
                        ]
                        if key == "mean_total_score":
                            cell = [_to_float(r.get("total_score", np.nan), np.nan) for r in sub_cell]
                        elif key == "mean_ees_score":
                            cell = []
                            for r in sub_cell:
                                score_map = {
                                    "pdf": _to_float(r.get("pdf_score", np.nan), np.nan),
                                    "auto": _to_float(r.get("auto_score", np.nan), np.nan),
                                    "cross": _to_float(r.get("cross_score", np.nan), np.nan),
                                    "coherence": _to_float(r.get("coherence_score", np.nan), np.nan),
                                    "response": _to_float(r.get("parameter_response_consistency", np.nan), np.nan),
                                    "cv": _to_float(r.get("cv_score", np.nan), np.nan),
                                    "bispectrum": _to_float(r.get("bispectrum_score", np.nan), np.nan),
                                }
                                if any_bispectrum:
                                    ees_weights = {
                                        "pdf": 0.12,
                                        "auto": 0.18,
                                        "cross": 0.14,
                                        "coherence": 0.14,
                                        "response": 0.18,
                                        "cv": 0.12,
                                        "bispectrum": 0.12,
                                    }
                                else:
                                    ees_weights = {
                                        "pdf": 0.15,
                                        "auto": 0.20,
                                        "cross": 0.15,
                                        "coherence": 0.15,
                                        "response": 0.20,
                                        "cv": 0.15,
                                    }
                                cell.append(_weighted_mean_from_dict(score_map, ees_weights))
                        else:
                            cell = []
                            for r in sub_cell:
                                score_map = {
                                    "pdf": _to_float(r.get("pdf_score", np.nan), np.nan),
                                    "auto": _to_float(r.get("auto_score", np.nan), np.nan),
                                    "cross": _to_float(r.get("cross_score", np.nan), np.nan),
                                    "coherence": _to_float(r.get("coherence_score", np.nan), np.nan),
                                    "response": _to_float(r.get("parameter_response_consistency", np.nan), np.nan),
                                    "cv": _to_float(r.get("cv_score", np.nan), np.nan),
                                    "bispectrum": _to_float(r.get("bispectrum_score", np.nan), np.nan),
                                }
                                if any_bispectrum:
                                    ees_weights = {
                                        "pdf": 0.12,
                                        "auto": 0.18,
                                        "cross": 0.14,
                                        "coherence": 0.14,
                                        "response": 0.18,
                                        "cv": 0.12,
                                        "bispectrum": 0.12,
                                    }
                                else:
                                    ees_weights = {
                                        "pdf": 0.15,
                                        "auto": 0.20,
                                        "cross": 0.15,
                                        "coherence": 0.15,
                                        "response": 0.20,
                                        "cv": 0.15,
                                    }
                                ees_val = _weighted_mean_from_dict(score_map, ees_weights)
                                ov = (
                                    0.25 * cfg_scaled_ers + 0.75 * ees_val
                                    if np.isfinite(cfg_scaled_ers) and np.isfinite(_to_float(ees_val, np.nan))
                                    else ees_val
                                )
                                cell.append(ov)
                        cell = [v for v in cell if np.isfinite(_to_float(v, np.nan))]
                        if cell:
                            heat[i, j] = float(np.mean(cell))
                if not np.any(np.isfinite(heat)):
                    continue
                cfg_tag = str(cfg_scale).replace(".", "p").replace("-", "m")
                plt.figure(figsize=(max(8, len(npe_vals) * 1.0), max(4, len(ref_vals) * 0.5)))
                im = plt.imshow(heat, aspect="auto")
                plt.colorbar(im, label=key)
                plt.xticks(range(len(npe_vals)), [str(v) for v in npe_vals], rotation=45)
                plt.yticks(range(len(ref_vals)), [str(v) for v in ref_vals])
                plt.xlabel("cut_npe")
                plt.ylabel("ref_idx")
                plt.title(f"{title} | cfg={cfg_scale}")
                plt.tight_layout()
                plt.savefig(plots_dir / filename_tpl.format(cfg_tag=cfg_tag), dpi=180)
                plt.close()

    refs = sorted({k[0] for k in ref_cfg_groups.keys()})
    cfgs = sorted({k[1] for k in ref_cfg_groups.keys()})
    if (not no_cfg_plots) and refs and cfgs:
        heat = np.full((len(refs), len(cfgs)), np.nan, dtype=float)
        for i, ref_idx in enumerate(refs):
            for j, cfg_scale in enumerate(cfgs):
                group = ref_cfg_groups.get((ref_idx, cfg_scale), [])
                heat[i, j] = mean_of(group, "total_score") if group else np.nan
        plt.figure(figsize=(max(8, len(cfgs) * 1.2), max(5, len(refs) * 0.5)))
        im = plt.imshow(heat, aspect="auto")
        plt.colorbar(im, label="mean total score")
        plt.xticks(range(len(cfgs)), [str(c) for c in cfgs], rotation=45)
        plt.yticks(range(len(refs)), [str(r) for r in refs])
        plt.xlabel("cfg_scale")
        plt.ylabel("ref_idx")
        plt.title("Heatmap of mean total score")
        plt.tight_layout()
        plt.savefig(plots_dir / "heatmap_ref_cfg_mean_total_score.png", dpi=180)
        plt.close()


# -----------------------------
# sample runner
# -----------------------------
def run_one(
    *,
    python_exec: str,
    sample_script_path: Path,
    sample_runner: str,
    checkpoint_path: Path,
    output_dir: Path,
    num_samples: int,
    ref_idx: int,
    gpu: int,
    histogram: bool,
    cut_npe: float,
    cut_firsttime: float,
    cfg_scale: float,
    h5_path: Path,
    flow_mode: str,
    sampling_method: Optional[str],
    sampling_steps: Optional[int],
    dry_run: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if sample_runner == "sample_cfg":
        cmd = [
            python_exec,
            str(sample_script_path),
            "--checkpoint", str(checkpoint_path),
            "--output_dir", str(output_dir),
            "--num_samples", str(num_samples),
            "--ref_idx", str(ref_idx),
            "--gpu", str(gpu),
            "--cut_npe", str(cut_npe),
            "--cut_firsttime", str(cut_firsttime),
            "--cfg_scale", str(cfg_scale),
        ]
        if histogram:
            cmd.append("--histogram")
    elif sample_runner == "rectified_flow_jointzero":
        cmd = [
            python_exec,
            str(sample_script_path),
            "--checkpoint", str(checkpoint_path),
            "--out_dir", str(output_dir),
            "--num_samples", str(num_samples),
            "--ref_idx", str(ref_idx),
            "--gpu", str(gpu),
            "--cfg_scale", str(cfg_scale),
            "--h5_path", str(h5_path),
        ]
        if flow_mode:
            cmd.extend(["--flow_mode", str(flow_mode)])
        if sampling_method:
            cmd.extend(["--sampling_method", str(sampling_method)])
        if sampling_steps is not None:
            cmd.extend(["--sampling_steps", str(int(sampling_steps))])
        if not histogram:
            cmd.append("--no_hist")
    else:
        raise ValueError(f"Unsupported sample_runner: {sample_runner}")

    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def collect_sample_files(output_dir: Path, ref_idx: int) -> List[Path]:
    candidates = sorted(output_dir.glob(f"sampled_event_{ref_idx}_sample_*.npy"))
    return [p for p in candidates if not p.stem.endswith("_norm")]


# -----------------------------
# Summary dispatcher
# -----------------------------
def default_summary_plot_args(**overrides: Any) -> argparse.Namespace:
    """
    `regenerate_multifield_summary_and_plots`에 넘길 기본 인자.
    CSV만으로 그래프만 다시 그릴 때 `regenerate_plots_from_log.py` 등에서 사용한다.
    """
    base: Dict[str, Any] = {
        "min_diversity_actual_refs": 2,
        "min_diversity_samples": 4,
        "max_diversity_plot_groups": 6,
        "response_dim": None,
        "fiducial_ref_idx": None,
        "response_tol": 1e-8,
        "max_response_values": 12,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def regenerate_summary_and_plots(csv_path: Path, summary_dir: Path, args: argparse.Namespace) -> None:
    rows = read_csv_rows(csv_path)
    if not rows:
        return
    event_rows = [r for r in rows if r.get("eval_mode", "event") == "event"]
    multifield_rows = [r for r in rows if r.get("eval_mode", "") == "multifield_map"]
    if event_rows:
        regenerate_event_summary_and_plots(event_rows, summary_dir / "event")
    if multifield_rows:
        regenerate_multifield_summary_and_plots(multifield_rows, summary_dir / "multifield", args)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_tasks_root = script_dir
    default_sample_script = script_dir / "sample_rectified_flow_0413_jointzero.py"
    default_h5_path = script_dir / "GENESIS-data" / "22644_0921_time_shift.h5"
    default_checkpoint = (
        script_dir
        / "tasks"
        / "rectified_flow_0413_jointzero_transformer"
        / "models"
        / "best_checkpoint_epoch_040_val_loss_0.055823.pt"
    )

    p = argparse.ArgumentParser(
        description="Run sampling sweeps, evaluate samples, append logs, and generate cumulative summaries."
    )
    p.add_argument("--python_exec", default="python")
    p.add_argument(
        "--sample_cfg",
        default=str(default_sample_script),
        help="Path to the sampling script. Kept as --sample_cfg for backward compatibility.",
    )
    p.add_argument(
        "--sample_runner",
        choices=["sample_cfg", "rectified_flow_jointzero"],
        default="rectified_flow_jointzero",
    )
    p.add_argument("--tasks_root", default=str(default_tasks_root))
    p.add_argument(
        "--checkpoint_path",
        default=str(default_checkpoint),
        help="Explicit checkpoint path. If empty, fall back to legacy --checkpoint_choice behavior.",
    )
    p.add_argument("--checkpoint_choice", choices=["best", "final"], default="best")
    p.add_argument("--base_output_dir", default="")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--histogram", action="store_true", default=True)
    p.add_argument("--no_histogram", action="store_true")
    p.add_argument("--gen_cut_npe", type=float, default=1.0)
    p.add_argument("--gen_cut_firsttime", type=float, default=0.1)
    p.add_argument("--eval_cut_npe_start", type=float, default=1.0)
    p.add_argument("--eval_cut_npe_end", type=float, default=1.0)
    p.add_argument("--eval_cut_npe_step", type=float, default=1.0)
    p.add_argument("--eval_cut_firsttime_start", type=float, default=0.1)
    p.add_argument("--eval_cut_firsttime_end", type=float, default=0.1)
    p.add_argument("--eval_cut_firsttime_step", type=float, default=0.1)
    p.add_argument("--ref_start", type=int, required=True)
    p.add_argument("--ref_end", type=int, required=True)
    p.add_argument("--ref_step", type=int, required=True)
    p.add_argument("--cfg_start", type=float, required=True)
    p.add_argument("--cfg_end", type=float, required=True)
    p.add_argument("--cfg_step", type=float, required=True)
    p.add_argument(
        "--folder_template",
        default="{ckpt_tag}/ref_{ref_idx}/cfg_{cfg_scale}",
        help='Subfolder template under output/. Example: "{ckpt_tag}/ref_{ref_idx}/cfg_{cfg_scale}"',
    )
    p.add_argument("--h5_path", default=str(default_h5_path))
    p.add_argument("--log_csv_name", default="experiment_log.csv")
    p.add_argument("--recompute_csv_name", default="recompute_log.csv")
    p.add_argument("--summary_dir_name", default="summary")
    p.add_argument("--skip_existing_logged", action="store_true", default=True)
    p.add_argument("--no_skip_existing_logged", action="store_true", default=False)
    p.add_argument("--no_cfg_plots", action="store_true", default=False)
    p.add_argument("--no_cut_npe_plots", action="store_true", default=False,
                   help="cut_npe sweep 관련 그래프를 그리지 않음 (cfg sweep만 할 때 사용)")
    p.add_argument("--force_recompute_existing", action="store_true", default=False)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--sample_flow_mode", choices=["checkpoint", "rectified_flow", "ot"], default="ot")
    p.add_argument("--sample_sampling_method", default=None)
    p.add_argument("--sample_sampling_steps", type=int, default=None)

    # evaluation mode
    p.add_argument("--eval_mode", choices=["event", "multifield_map"], default="event")

    # multifield config
    p.add_argument("--channel_names", default="Mcdm,Mgas,T")
    p.add_argument("--channel_types", default="density,density,temperature")
    p.add_argument("--pdf_eps", default="1e-6,1e-6,1e-6")
    p.add_argument("--pdf_bins", type=int, default=64)
    p.add_argument("--map_box_size", type=float, default=25.0)
    p.add_argument("--k_bins", type=int, default=35)
    p.add_argument("--log_k_bins", action="store_true", default=True)
    p.add_argument("--no_log_k_bins", action="store_true")
    p.add_argument("--enable_bispectrum", action="store_true", default=False)
    p.add_argument(
        "--bispectrum_triplets",
        default="",
        help="Semicolon-separated k-bin triplets, e.g. '2,2,2;2,6,6;3,5,7'",
    )

    # diversity / response controls
    p.add_argument("--min_diversity_actual_refs", type=int, default=2)
    p.add_argument("--min_diversity_samples", type=int, default=4)
    p.add_argument("--max_diversity_plot_groups", type=int, default=6)
    p.add_argument("--response_dim", type=int, default=None)
    p.add_argument("--fiducial_ref_idx", type=int, default=None)
    p.add_argument("--response_tol", type=float, default=1e-8)
    p.add_argument("--max_response_values", type=int, default=12)

    args = p.parse_args()
    if args.no_skip_existing_logged:
        args.skip_existing_logged = False

    if args.no_histogram:
        args.histogram = False
    if args.no_log_k_bins:
        args.log_k_bins = False

    multifield_config: Optional[MultifieldEvalConfig] = None
    if args.eval_mode == "multifield_map":
        multifield_config = parse_multifield_config(args)

    tasks_root = Path(args.tasks_root).resolve()
    sample_script_path = Path(args.sample_cfg).resolve()

    if str(args.checkpoint_path).strip():
        ckpt = Path(args.checkpoint_path).expanduser().resolve()
        ckpt_tag = ckpt.stem
    else:
        models_dir = tasks_root
        if args.checkpoint_choice == "best":
            ckpt = models_dir / "tasks_yeji" / "models" / "best_checkpoint_epoch_012_val_loss_0.006395.pt"
            ckpt_tag = "best"
        else:
            ckpt = models_dir / "tasks_yeji" / "models" / "model_checkpoint_final.pt"
            ckpt_tag = "final"

    if str(args.base_output_dir).strip():
        base_output = Path(args.base_output_dir).expanduser().resolve()
    elif args.sample_runner == "rectified_flow_jointzero":
        base_output = ckpt.parent.parent / "eval_output"
    else:
        base_output = tasks_root / "tasks_yeji" / "output"

    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if not sample_script_path.exists():
        raise FileNotFoundError(f"sample script not found: {sample_script_path}")

    args.h5_path = str(Path(args.h5_path).resolve())
    synced_h5_path = Path(args.h5_path)

    H5Dataset = setup_project_imports(tasks_root)
    dataset = H5Dataset(h5_path=args.h5_path)

    ref_list = irange(args.ref_start, args.ref_end, args.ref_step)
    cfg_list = frange(args.cfg_start, args.cfg_end, args.cfg_step)
    eval_cut_npe_list, eval_cut_firsttime_list = make_eval_cut_lists(args)

    if args.eval_mode == "multifield_map" and multifield_config is not None and ref_list:
        try:
            probe_item = dataset[ref_list[0]]
            probe_sig = to_numpy(probe_item[0])
            inferred_channels = infer_channel_count_from_signal(probe_sig)
            configured_channels = len(multifield_config.channel_names)
            if inferred_channels is not None and inferred_channels != configured_channels:
                print(
                    "[WARN] multifield channel count mismatch; "
                    f"configured={configured_channels}, inferred_from_data={inferred_channels}. "
                    "Auto-adjusting channel config to data."
                )
                multifield_config.channel_names = _resize_with_fallback(
                    multifield_config.channel_names, inferred_channels, "channel", "channel"
                )
                multifield_config.channel_types = _resize_with_fallback(
                    multifield_config.channel_types, inferred_channels, "type", "generic"
                )
                multifield_config.pdf_eps = _resize_with_fallback(
                    multifield_config.pdf_eps, inferred_channels, "eps", 1e-6
                )
        except Exception as e:
            print(f"[WARN] Failed to infer multifield channel count from data: {e}")

    log_csv_path = base_output / args.log_csv_name
    recompute_csv_path = base_output / args.recompute_csv_name
    summary_dir = base_output / args.summary_dir_name

    print(f"Checkpoint tag: {ckpt_tag}")
    print(f"Checkpoint: {ckpt}")
    print(f"Sample runner: {args.sample_runner}")
    print(f"Sample script: {sample_script_path}")
    print(f"Output base: {base_output}")
    print(f"H5 path: {args.h5_path}")
    print(f"Sampler H5 path: {synced_h5_path}")
    if args.sample_runner != "sample_cfg":
        print(f"Sampler flow mode: {args.sample_flow_mode}")
        if args.sample_sampling_method:
            print(f"Sampler method override: {args.sample_sampling_method}")
        if args.sample_sampling_steps is not None:
            print(f"Sampler step override: {args.sample_sampling_steps}")
    print(f"Evaluation mode: {args.eval_mode}")
    print(
        "Generation cut_npe source: "
        "eval_cut_npe sweep values (legacy --gen_cut_npe no longer drives sampling)"
    )
    print(f"Generation cut_firsttime: {args.gen_cut_firsttime}")
    if args.eval_mode == "event":
        print(f"Evaluation cut_npe sweep: {eval_cut_npe_list}")
        print(f"Evaluation cut_firsttime sweep: {eval_cut_firsttime_list}")
    print(f"Main log CSV: {log_csv_path}")
    if args.force_recompute_existing:
        print(f"Recompute CSV: {recompute_csv_path}")

    sweep_total = len(ref_list) * len(cfg_list)
    sweep_iter = tqdm(
        itertools.product(ref_list, cfg_list),
        total=sweep_total,
        desc="Ref/Cfg sweep",
        file=sys.stdout,
    )
    for ref_idx, cfg_scale in sweep_iter:
        cfg_tag = str(cfg_scale).replace(".", "p").replace("-", "m")
        sub = args.folder_template.format(ckpt_tag=ckpt_tag, ref_idx=ref_idx, cfg_scale=cfg_tag)
        out_dir_base = base_output / sub
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        logged_eval_keys = load_logged_eval_keys(log_csv_path) if args.skip_existing_logged else set()

        sig_ref_raw, geo_raw, label_raw = dataset[ref_idx]
        actual_sig = to_numpy(sig_ref_raw)
        # dataset 반환 순서가 (sig, geo, label)이므로 label만 사용
        label_vector = extract_label_vector(label_raw, None)
        label_json = json.dumps(as_jsonable(label_vector)) if label_vector is not None else ""
        label_key = stable_label_key(label_vector) if label_vector is not None else ""

        appended_count = 0
        recomputed_count = 0

        def evaluate_and_log(
            sample_path: Path,
            mode_name: str,
            target_csv: Path,
            gen_cut_npe: float,
            eval_cut_pairs: List[Tuple[float, float]],
            eval_pbar: Optional[tqdm],
            out_dir_for_row: Path,
        ) -> None:
            nonlocal appended_count, recomputed_count
            try:
                sample_idx = int(sample_path.stem.split("_")[-1])
            except Exception:
                sample_idx = -1

            sample_sig = np.load(sample_path)
            for eval_cut_npe, eval_cut_firsttime in eval_cut_pairs:
                try:
                    eval_key = (
                        str(sample_path.resolve()),
                        float(eval_cut_npe),
                        float(eval_cut_firsttime),
                        str(args.eval_mode),
                    )
                    if target_csv == log_csv_path and args.skip_existing_logged and eval_key in logged_eval_keys:
                        continue

                    row: Dict[str, Any] = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "run_id": run_id,
                        "mode": mode_name,
                        "eval_mode": args.eval_mode,
                        "checkpoint_tag": ckpt_tag,
                        "checkpoint_path": str(ckpt),
                        "ref_idx": ref_idx,
                        "cfg_scale": cfg_scale,
                        "cfg_tag": cfg_tag,
                        "sample_idx": sample_idx,
                        "num_samples_requested": args.num_samples,
                        "gen_cut_npe": gen_cut_npe,
                        "gen_cut_firsttime": args.gen_cut_firsttime,
                        "cut_npe": eval_cut_npe,
                        "cut_firsttime": eval_cut_firsttime,
                        "output_dir": str(out_dir_for_row),
                        "sample_path": str(sample_path),
                        "bundle_path": "",
                        "label_json": label_json,
                        "label_key": label_key,
                    }
                    if args.eval_mode == "event":
                        metrics = evaluate_event_sample_vs_actual(
                            actual_sig_raw=actual_sig,
                            sample_sig_raw=sample_sig,
                            cut_npe=eval_cut_npe,
                            cut_firsttime=eval_cut_firsttime,
                        )
                        row.update(metrics)
                    else:
                        assert multifield_config is not None
                        actual_cut = apply_event_style_eval_cuts(
                            actual_sig, cut_npe=eval_cut_npe, cut_firsttime=eval_cut_firsttime
                        )
                        sample_cut = apply_event_style_eval_cuts(
                            sample_sig, cut_npe=eval_cut_npe, cut_firsttime=eval_cut_firsttime
                        )
                        actual_map = prepare_multifield_map(actual_cut, len(multifield_config.channel_names))
                        sample_map = prepare_multifield_map(sample_cut, len(multifield_config.channel_names))
                        metrics, bundle = evaluate_multifield_sample_vs_actual(
                            actual_map=actual_map,
                            sample_map=sample_map,
                            config=multifield_config,
                            label_vector=label_vector,
                        )
                        bundle.update({
                            "ref_idx": ref_idx,
                            "cfg_scale": cfg_scale,
                            "sample_idx": sample_idx,
                            "sample_path": str(sample_path),
                            "checkpoint_tag": ckpt_tag,
                            "gen_cut_npe": gen_cut_npe,
                            "gen_cut_firsttime": args.gen_cut_firsttime,
                            "cut_npe": eval_cut_npe,
                            "cut_firsttime": eval_cut_firsttime,
                        })
                        bundle_path = (
                            out_dir_for_row
                            / "eval_bundles"
                            / f"bundle_ref_{ref_idx}_cfg_{cfg_tag}_sample_{sample_idx}.json"
                        )
                        save_json(bundle_path, bundle)
                        row["bundle_path"] = str(bundle_path)
                        row.update(metrics)

                    append_row_to_csv(target_csv, row)
                    if target_csv == log_csv_path:
                        logged_eval_keys.add(eval_key)
                        appended_count += 1
                    else:
                        recomputed_count += 1
                finally:
                    if eval_pbar is not None:
                        eval_pbar.update(1)

        for gen_cut_npe in eval_cut_npe_list:
            cut_tag = str(gen_cut_npe).replace(".", "p").replace("-", "m")
            out_dir = out_dir_base / f"gen_cut_npe_{cut_tag}"

            pre_existing_files = {str(p.resolve()) for p in collect_sample_files(out_dir, ref_idx)}
            run_one(
                python_exec=args.python_exec,
                sample_script_path=sample_script_path,
                sample_runner=args.sample_runner,
                checkpoint_path=ckpt,
                output_dir=out_dir,
                num_samples=args.num_samples,
                ref_idx=ref_idx,
                gpu=args.gpu,
                histogram=args.histogram,
                cut_npe=gen_cut_npe,
                cut_firsttime=args.gen_cut_firsttime,
                cfg_scale=cfg_scale,
                h5_path=Path(args.h5_path),
                flow_mode=args.sample_flow_mode,
                sampling_method=args.sample_sampling_method,
                sampling_steps=args.sample_sampling_steps,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                continue

            post_files = [p.resolve() for p in collect_sample_files(out_dir, ref_idx)]
            new_files = [p for p in post_files if str(p) not in pre_existing_files]
            normal_candidates = new_files if new_files else post_files
            recompute_candidates = post_files if args.force_recompute_existing else []
            eval_cut_pairs = [(gen_cut_npe, eval_cut_firsttime) for eval_cut_firsttime in eval_cut_firsttime_list]

            eval_work_total = (len(normal_candidates) + len(recompute_candidates)) * len(eval_cut_pairs)
            eval_pbar = tqdm(
                total=eval_work_total,
                desc=f"Eval ref={ref_idx} cfg={cfg_scale} gen_cut_npe={gen_cut_npe}",
                file=sys.stdout,
                leave=False,
            ) if eval_work_total > 0 else None

            try:
                for sample_path in sorted(normal_candidates):
                    evaluate_and_log(
                        sample_path,
                        "normal",
                        log_csv_path,
                        gen_cut_npe,
                        eval_cut_pairs,
                        eval_pbar,
                        out_dir,
                    )

                for sample_path in sorted(recompute_candidates):
                    evaluate_and_log(
                        sample_path,
                        "recompute",
                        recompute_csv_path,
                        gen_cut_npe,
                        eval_cut_pairs,
                        eval_pbar,
                        out_dir,
                    )
            finally:
                if eval_pbar is not None:
                    eval_pbar.close()

        if appended_count == 0 and recomputed_count == 0:
            print(f"[INFO] No work items for ref_idx={ref_idx}, cfg_scale={cfg_scale}")
        else:
            print(f"[INFO] Appended {appended_count} new rows to main log for ref_idx={ref_idx}, cfg_scale={cfg_scale}")
            if args.force_recompute_existing:
                print(f"[INFO] Recomputed {recomputed_count} logged rows into separate CSV for ref_idx={ref_idx}, cfg_scale={cfg_scale}")

        regenerate_summary_and_plots(log_csv_path, summary_dir, args)
        print(f"[INFO] Updated cumulative summary/plots: {summary_dir}")

    print("All sweeps completed.")


if __name__ == "__main__":
    main()