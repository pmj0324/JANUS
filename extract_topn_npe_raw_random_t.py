#!/usr/bin/env python3
"""Extract top-N high-nPE raw events and visualize random intermediate x_t for rectified flow."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import train_exp_rectified_flow_0413_jointzero as m
from utils.io.h5_rank_npe import compute_npe_stats_fast, topk_rows_from_stats


def save_overlay_histograms(
    real_denorm: np.ndarray,
    xt_denorm: np.ndarray,
    output_path: Path,
    *,
    bins: int = 80,
    log_y: bool = True,
) -> None:
    real_denorm = np.asarray(real_denorm, dtype=np.float32)
    xt_denorm = np.asarray(xt_denorm, dtype=np.float32)

    real_npe = real_denorm[0].ravel()
    real_ftime = real_denorm[1].ravel()
    xt_npe = xt_denorm[0].ravel()
    xt_ftime = xt_denorm[1].ravel()

    def _finite(arr: np.ndarray) -> np.ndarray:
        return arr[np.isfinite(arr)]

    real_npe = _finite(real_npe)
    real_ftime = _finite(real_ftime)
    xt_npe = _finite(xt_npe)
    xt_ftime = _finite(xt_ftime)

    def _range(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        merged = np.concatenate([a, b]) if (a.size or b.size) else np.array([0.0], dtype=np.float32)
        lo = float(np.min(merged))
        hi = float(np.max(merged))
        if lo == hi:
            hi = lo + 1.0
        return lo, hi

    npe_min, npe_max = _range(real_npe, xt_npe)
    ftime_min, ftime_max = _range(real_ftime, xt_ftime)

    import matplotlib.pyplot as plt

    def _draw(save_path: Path, xlog: bool) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        panels = [
            (axes[0], real_npe, xt_npe, "nPE (denorm)", npe_min, npe_max),
            (axes[1], real_ftime, xt_ftime, "FirstTime (denorm)", ftime_min, ftime_max),
        ]

        for ax, raw_arr, xt_arr, title, x_min, x_max in panels:
            arr_raw = raw_arr
            arr_xt = xt_arr
            if xlog:
                arr_raw = arr_raw[arr_raw > 0]
                arr_xt = arr_xt[arr_xt > 0]
                positive = np.concatenate([arr_raw, arr_xt]) if (arr_raw.size or arr_xt.size) else np.array([1.0], dtype=np.float32)
                x_min = float(np.min(positive))
                x_max = float(np.max(positive))
                if x_min == x_max:
                    x_max = x_min * 1.1 if x_min > 0 else 1.0

            ax.hist(
                arr_raw,
                bins=bins,
                range=(x_min, x_max),
                density=True,
                color="tab:blue",
                alpha=0.45,
                label=f"Raw x0 (N={arr_raw.size})",
            )
            ax.hist(
                arr_xt,
                bins=bins,
                range=(x_min, x_max),
                density=True,
                color="tab:orange",
                alpha=0.45,
                label=f"x_t (N={arr_xt.size})",
            )
            ax.set_yscale("log")
            if xlog:
                ax.set_xscale("log")
            ax.set_title(title + (" | x-log" if xlog else " | x-linear"))
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _draw(output_path, xlog=False)
    _draw(output_path.with_name(output_path.stem + "_xlog" + output_path.suffix), xlog=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract top high-nPE raw events and save random intermediate t visualizations",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/work/icecube_janus/JANUS/tasks/rectified_flow_0413_jointzero_transformer/top30_raw_random_t",
        help="Output root directory",
    )
    parser.add_argument("--top_k", type=int, default=30, help="Number of events to extract")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--h5_path", type=str, default=None, help="Override H5 dataset path")
    parser.add_argument("--t_min", type=float, default=0.05, help="Min random t")
    parser.add_argument("--t_max", type=float, default=0.95, help="Max random t")
    parser.add_argument("--hist_bins", type=int, default=80, help="Histogram bins")
    parser.add_argument("--chunk_events", type=int, default=4096, help="H5 scan chunk size (events)")
    parser.add_argument("--no_stats_cache", action="store_true", help="Disable cached nPE stats")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    args = parser.parse_args()

    if not (0.0 <= args.t_min < args.t_max <= 1.0):
        raise ValueError("Require 0 <= t_min < t_max <= 1")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Use --cpu to run on CPU.")
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)

    h5_path = args.h5_path if args.h5_path is not None else m.h5_path
    dataset = m.H5Dataset(
        h5_path=h5_path,
        angle_conversion=m.data_angle_conversion,
        num_workers=0,
        shuffle=False,
    )

    print(f"Dataset length: {len(dataset)}")
    print("Finding top high-nPE events (fast chunked H5 scan)...")
    stats_cache = out_dir / "_cache_npe_stats.npz"
    stats = compute_npe_stats_fast(
        h5_path=h5_path,
        signal_key="input",
        chunk_events=args.chunk_events,
        cache_npz=stats_cache,
        use_cache=(not args.no_stats_cache),
    )
    top_rows = topk_rows_from_stats(
        stats,
        top_k=args.top_k,
        sort_mode="sum_then_active",
    )
    print(f"Selected top-{len(top_rows)} events by npe_sum")

    summary_csv = out_dir / "top_events_random_t_summary.csv"
    summary_fields = [
        "rank",
        "ref_idx",
        "npe_sum",
        "active_npe_count",
        "npe_max",
        "random_t",
        "event_dir",
    ]

    flow = m.RectifiedFlow()

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()

        for rank, row in enumerate(top_rows, 1):
            ref_idx = int(row["ref_idx"])
            event_dir = out_dir / f"rank_{rank:02d}_ref_{ref_idx:05d}"
            event_dir.mkdir(parents=True, exist_ok=True)

            sig_raw, geo_raw, label_raw = dataset[ref_idx]
            sig_raw_np = np.asarray(sig_raw, dtype=np.float32)
            geo_np = np.asarray(geo_raw, dtype=np.float32)
            label_np = np.asarray(label_raw, dtype=np.float32)

            # Save raw tensors/numpy
            np.save(event_dir / "raw_sig.npy", sig_raw_np)
            np.save(event_dir / "raw_geo.npy", geo_np)
            np.save(event_dir / "raw_label.npy", label_np)

            # x0 normalization with the exact training pipeline
            sig_t = sig_raw.unsqueeze(0).to(device, non_blocking=True)
            label_t = label_raw.unsqueeze(0).to(device, non_blocking=True)
            sig_clamp = m._clamp_sig(sig_t)
            x0_norm, _ = m.prepare_batch(sig_clamp, label_t, verbose=False)

            # Random intermediate t and x_t path
            t_val = float(rng.uniform(args.t_min, args.t_max))
            t = torch.tensor([t_val], dtype=torch.float32, device=device)
            x1 = torch.randn_like(x0_norm)
            x_t_norm = flow.compute_path(x0_norm, x1, t)

            # Denormalize for visualization
            x0_denorm = m.denormalize_sig(x0_norm)[0].detach().cpu().numpy()
            x_t_denorm = m.denormalize_sig(x_t_norm)[0].detach().cpu().numpy()

            # Save intermediates
            np.save(event_dir / "x0_norm.npy", x0_norm[0].detach().cpu().numpy())
            np.save(event_dir / "x1_noise.npy", x1[0].detach().cpu().numpy())
            np.save(event_dir / "x_t_norm.npy", x_t_norm[0].detach().cpu().numpy())
            np.save(event_dir / "x_t_denorm.npy", x_t_denorm)
            np.save(event_dir / "x0_denorm.npy", x0_denorm)

            # Save raw / x_t plots
            m.show_event_dual_plot(
                sig=x0_denorm,
                geo=geo_np,
                label=label_np,
                output_path=str(event_dir / "raw_event.png"),
                figure_size=(18, 8),
                marker_size=8.0,
                show_detector_hull=True,
                show=False,
                title_prefix=f"TopN nPE Raw | rank={rank} ref={ref_idx}",
                firsttime_title="FirstTime (raw x0)",
                npe_title="nPE (raw x0)",
            )

            m.show_event_dual_plot(
                sig=x_t_denorm,
                geo=geo_np,
                label=label_np,
                output_path=str(event_dir / "random_t_event.png"),
                figure_size=(18, 8),
                marker_size=8.0,
                show_detector_hull=True,
                show=False,
                title_prefix=f"Random intermediate x_t | rank={rank} ref={ref_idx} | t={t_val:.4f}",
                firsttime_title="FirstTime (x_t denorm)",
                npe_title="nPE (x_t denorm)",
            )

            m.save_epoch_comparison_plot(
                real_sig=x0_denorm,
                sampled_sig=x_t_denorm,
                geo=geo_np,
                label=label_np,
                output_path=event_dir / "compare_raw_vs_random_t.png",
                title_prefix=f"Raw x0 vs random x_t | rank={rank} ref={ref_idx} | t={t_val:.4f}",
                figure_size=(18, 8),
                marker_size=10.0,
            )

            save_overlay_histograms(
                real_denorm=x0_denorm,
                xt_denorm=x_t_denorm,
                output_path=event_dir / "hist_raw_vs_random_t.png",
                bins=args.hist_bins,
                log_y=True,
            )

            writer.writerow(
                {
                    "rank": rank,
                    "ref_idx": ref_idx,
                    "npe_sum": f"{row['npe_sum']:.6f}",
                    "active_npe_count": row["active_npe_count"],
                    "npe_max": f"{row['npe_max']:.6f}",
                    "random_t": f"{t_val:.8f}",
                    "event_dir": str(event_dir),
                }
            )

            print(
                f"[{rank:02d}/{len(top_rows)}] ref={ref_idx} "
                f"npe_sum={row['npe_sum']:.2f} active={row['active_npe_count']} t={t_val:.4f}"
            )

    print("done")
    print(f"device={device}")
    print(f"out_dir={out_dir}")
    print(f"summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
