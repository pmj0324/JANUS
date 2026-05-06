"""Fast nPE ranking utilities for large HDF5 signal datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import numpy as np

SortMode = Literal["active_then_sum", "sum_then_active"]


def _load_cache(cache_npz: Path | None) -> dict | None:
    if cache_npz is None or not cache_npz.is_file():
        return None
    try:
        data = np.load(cache_npz)
        return {
            "ref_idx": data["ref_idx"],
            "active_npe_count": data["active_npe_count"],
            "npe_sum": data["npe_sum"],
            "npe_max": data["npe_max"],
            "npe_mean_active": data["npe_mean_active"],
        }
    except Exception:
        return None


def _save_cache(cache_npz: Path | None, stats: dict) -> None:
    if cache_npz is None:
        return
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_npz,
        ref_idx=stats["ref_idx"],
        active_npe_count=stats["active_npe_count"],
        npe_sum=stats["npe_sum"],
        npe_max=stats["npe_max"],
        npe_mean_active=stats["npe_mean_active"],
    )


def compute_npe_stats_fast(
    h5_path: str,
    *,
    signal_key: str = "input",
    chunk_events: int = 4096,
    cache_npz: Path | None = None,
    use_cache: bool = True,
) -> dict:
    """Compute per-event nPE stats using chunked vectorized HDF5 reads."""
    if use_cache:
        cached = _load_cache(cache_npz)
        if cached is not None:
            return cached

    h5_path = str(Path(h5_path).expanduser().resolve())
    chunk_events = max(256, int(chunk_events))

    with h5py.File(h5_path, "r") as f:
        ds = f[signal_key]
        num_events = int(ds.shape[0])

        ref_idx = np.arange(num_events, dtype=np.int64)
        active_npe_count = np.zeros(num_events, dtype=np.int32)
        npe_sum = np.zeros(num_events, dtype=np.float64)
        npe_max = np.zeros(num_events, dtype=np.float32)

        for start in range(0, num_events, chunk_events):
            end = min(start + chunk_events, num_events)
            block = np.asarray(ds[start:end, 0, :], dtype=np.float32)  # (B, L), nPE channel only

            finite = np.isfinite(block)
            positive = finite & (block > 0.0)

            act = np.sum(positive, axis=1, dtype=np.int32)
            summed = np.sum(np.where(positive, block, 0.0), axis=1, dtype=np.float64)
            mx = np.max(np.where(positive, block, -np.inf), axis=1)
            mx = np.where(np.isfinite(mx), mx, 0.0).astype(np.float32)

            active_npe_count[start:end] = act
            npe_sum[start:end] = summed
            npe_max[start:end] = mx

    npe_mean_active = np.divide(
        npe_sum,
        active_npe_count,
        out=np.zeros_like(npe_sum, dtype=np.float64),
        where=active_npe_count > 0,
    )

    stats = {
        "ref_idx": ref_idx,
        "active_npe_count": active_npe_count,
        "npe_sum": npe_sum,
        "npe_max": npe_max,
        "npe_mean_active": npe_mean_active,
    }
    _save_cache(cache_npz, stats)
    return stats


def topk_rows_from_stats(stats: dict, top_k: int, *, sort_mode: SortMode) -> list[dict]:
    """Select top-k rows from precomputed stats with configurable ranking priority."""
    k = max(1, int(top_k))

    ref_idx = stats["ref_idx"]
    active_npe_count = stats["active_npe_count"]
    npe_sum = stats["npe_sum"]
    npe_max = stats["npe_max"]
    npe_mean_active = stats["npe_mean_active"]

    if sort_mode == "active_then_sum":
        order = np.lexsort((-npe_sum, -active_npe_count))
    elif sort_mode == "sum_then_active":
        order = np.lexsort((-npe_max, -active_npe_count, -npe_sum))
    else:
        raise ValueError(f"Unknown sort_mode: {sort_mode}")

    top = order[: min(k, order.size)]

    rows: list[dict] = []
    for i in top.tolist():
        rows.append(
            {
                "ref_idx": int(ref_idx[i]),
                "active_npe_count": int(active_npe_count[i]),
                "npe_sum": float(npe_sum[i]),
                "npe_max": float(npe_max[i]),
                "npe_mean_active": float(npe_mean_active[i]),
            }
        )
    return rows
