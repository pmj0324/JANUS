#!/usr/bin/env python3
"""Find PMTs that are never active in an H5 dataset and visualize them."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def _set_3d_transparent(ax) -> None:
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.tick_params(axis="both", which="both", length=0, labelsize=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_alpha(0.0)
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        except Exception:
            pass
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))


def _draw_detector_hull(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    edge_string_idx = [1, 6, 50, 74, 73, 78, 75, 31]
    top_xy, bottom_xy = [], []
    for i in edge_string_idx:
        top_i = (i - 1) * 60
        bot_i = top_i + 59
        if top_i < len(x):
            top_xy.append([x[top_i], y[top_i]])
        if bot_i < len(x):
            bottom_xy.append([x[bot_i], y[bot_i]])

    if not top_xy or not bottom_xy:
        return

    top_xy = np.asarray(top_xy, dtype=np.float32)
    bottom_xy = np.asarray(bottom_xy, dtype=np.float32)
    z_top = float(np.nanmax(z))
    z_bottom = float(np.nanmin(z))

    for xy, z_val in ((top_xy, z_top), (bottom_xy, z_bottom)):
        if xy.shape[0] >= 2:
            closed = np.vstack([xy, xy[0]])
            ax.plot(closed[:, 0], closed[:, 1], np.full(closed.shape[0], z_val), color="black", lw=1.4, alpha=0.75)

    if top_xy.shape[0] == bottom_xy.shape[0]:
        for (tx, ty), (bx, by) in zip(top_xy, bottom_xy):
            ax.plot([tx, bx], [ty, by], [z_top, z_bottom], color="black", lw=1.1, alpha=0.75)


def _hit_mask(block: np.ndarray, hit_mode: str, npe_threshold: float, ftime_threshold: float) -> np.ndarray:
    npe = block[:, 0, :]
    ftime = block[:, 1, :]
    npe_hit = np.isfinite(npe) & (npe > npe_threshold)
    ftime_hit = np.isfinite(ftime) & (ftime > ftime_threshold)
    if hit_mode == "npe":
        return npe_hit
    if hit_mode == "ftime":
        return ftime_hit
    if hit_mode == "either":
        return npe_hit | ftime_hit
    if hit_mode == "both":
        return npe_hit & ftime_hit
    raise ValueError(f"Unknown hit_mode: {hit_mode}")


def scan_hit_counts(
    h5_path: Path,
    *,
    signal_key: str,
    chunk_events: int,
    hit_mode: str,
    npe_threshold: float,
    ftime_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        sig = f[signal_key]
        n_events = int(sig.shape[0])
        n_pmts = int(sig.shape[2])
        hit_counts = np.zeros(n_pmts, dtype=np.int64)

        for start in range(0, n_events, chunk_events):
            end = min(start + chunk_events, n_events)
            block = np.asarray(sig[start:end], dtype=np.float32)
            hits = _hit_mask(block, hit_mode, npe_threshold, ftime_threshold)
            hit_counts += hits.sum(axis=0, dtype=np.int64)
            print(f"scanned {end}/{n_events} events", flush=True)

        x = np.asarray(f["xpmt"][:], dtype=np.float32)
        y = np.asarray(f["ypmt"][:], dtype=np.float32)
        z = np.asarray(f["zpmt"][:], dtype=np.float32)

    return hit_counts, x, y, z


def save_summary(
    out_dir: Path,
    hit_counts: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    n_events: int,
) -> None:
    never = hit_counts == 0
    np.savetxt(out_dir / "never_hit_pmt_indices.txt", np.where(never)[0], fmt="%d")
    np.save(out_dir / "hit_counts_by_pmt.npy", hit_counts)
    np.save(out_dir / "never_hit_mask.npy", never)

    with (out_dir / "pmt_hit_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pmt_idx", "x", "y", "z", "hit_count", "hit_fraction", "never_hit"],
        )
        writer.writeheader()
        for i in range(hit_counts.size):
            writer.writerow(
                {
                    "pmt_idx": i,
                    "x": f"{float(x[i]):.6f}",
                    "y": f"{float(y[i]):.6f}",
                    "z": f"{float(z[i]):.6f}",
                    "hit_count": int(hit_counts[i]),
                    "hit_fraction": f"{float(hit_counts[i]) / max(n_events, 1):.10f}",
                    "never_hit": int(never[i]),
                }
            )


def plot_never_hit(
    out_path: Path,
    hit_counts: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    title: str,
    show_hull: bool,
) -> None:
    import matplotlib.pyplot as plt

    never = hit_counts == 0
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_alpha(0.0)

    if show_hull:
        _draw_detector_hull(ax, x, y, z)

    ax.scatter(x[~never], y[~never], z[~never], s=6, c="lightgray", alpha=0.28, depthshade=False, label="Hit at least once")
    if never.any():
        ax.scatter(x[never], y[never], z[never], s=46, c="crimson", alpha=0.95, depthshade=False, label="Never hit")

    ax.set_title(title, pad=18)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper right")
    _set_3d_transparent(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_hit_counts(out_path: Path, hit_counts: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, *, title: str, show_hull: bool) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_alpha(0.0)

    if show_hull:
        _draw_detector_hull(ax, x, y, z)

    color_values = np.log10(hit_counts.astype(np.float64) + 1.0)
    sc = ax.scatter(x, y, z, s=9, c=color_values, cmap="viridis", alpha=0.9, depthshade=False)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.62, aspect=20, pad=0.08)
    cbar.set_label("log10(hit_count + 1)", rotation=270, labelpad=18)

    ax.set_title(title, pad=18)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    _set_3d_transparent(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find PMTs that are never active in an H5 dataset and plot them.")
    parser.add_argument("--h5_path", type=str, default="GENESIS-data/22644_0921_time_shift.h5")
    parser.add_argument("--out_dir", type=str, default="tasks/rectified_flow_0413_jointzero_transformer/never_hit_pmts")
    parser.add_argument("--signal_key", type=str, default="input")
    parser.add_argument("--chunk_events", type=int, default=2048)
    parser.add_argument("--hit_mode", choices=["npe", "ftime", "either", "both"], default="npe")
    parser.add_argument("--npe_threshold", type=float, default=0.0)
    parser.add_argument("--ftime_threshold", type=float, default=0.0)
    parser.add_argument("--no_hull", action="store_true")
    args = parser.parse_args()

    h5_path = Path(args.h5_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not h5_path.is_file():
        raise FileNotFoundError(f"H5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        n_events = int(f[args.signal_key].shape[0])
        n_pmts = int(f[args.signal_key].shape[2])

    hit_counts, x, y, z = scan_hit_counts(
        h5_path,
        signal_key=args.signal_key,
        chunk_events=max(1, int(args.chunk_events)),
        hit_mode=args.hit_mode,
        npe_threshold=float(args.npe_threshold),
        ftime_threshold=float(args.ftime_threshold),
    )

    if hit_counts.size != n_pmts:
        raise RuntimeError(f"Expected {n_pmts} PMTs, got {hit_counts.size}")

    save_summary(out_dir, hit_counts, x, y, z, n_events=n_events)

    never_count = int(np.count_nonzero(hit_counts == 0))
    title_suffix = (
        f"H5={h5_path.name} | events={n_events:,} | PMTs={n_pmts:,} | "
        f"mode={args.hit_mode} | never={never_count:,}"
    )
    plot_never_hit(
        out_dir / "never_hit_pmts.png",
        hit_counts,
        x,
        y,
        z,
        title=f"PMTs Never Hit\n{title_suffix}",
        show_hull=not args.no_hull,
    )
    plot_hit_counts(
        out_dir / "hit_counts_by_pmt.png",
        hit_counts,
        x,
        y,
        z,
        title=f"PMT Hit Counts\n{title_suffix}",
        show_hull=not args.no_hull,
    )

    print("done")
    print(f"h5_path={h5_path}")
    print(f"out_dir={out_dir}")
    print(f"n_events={n_events}")
    print(f"n_pmts={n_pmts}")
    print(f"never_hit_count={never_count}")
    if never_count:
        print(f"never_hit_indices={out_dir / 'never_hit_pmt_indices.txt'}")
    print(f"plot_never_hit={out_dir / 'never_hit_pmts.png'}")
    print(f"plot_hit_counts={out_dir / 'hit_counts_by_pmt.png'}")


if __name__ == "__main__":
    main()
