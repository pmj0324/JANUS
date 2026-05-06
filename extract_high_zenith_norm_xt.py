#!/usr/bin/env python3
"""Pick high-zenith events and visualize x0/x1/x_t in normalized space."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch

import train_exp_rectified_flow_0413_jointzero as m


def top_high_zenith_indices(h5_path: str, top_k: int) -> list[dict]:
    with h5py.File(h5_path, "r") as f:
        labels = np.asarray(f["label"][:, :], dtype=np.float32)  # (N, 6)
    zenith_rad = labels[:, 1]
    order = np.argsort(-zenith_rad)  # descending
    sel = order[: max(1, int(top_k))]

    rows = []
    for rank, idx in enumerate(sel.tolist(), 1):
        z = float(zenith_rad[idx])
        rows.append(
            {
                "rank": rank,
                "ref_idx": int(idx),
                "zenith_rad": z,
                "zenith_deg": float(np.degrees(z)),
            }
        )
    return rows


def save_norm_hist_overlay(x0: np.ndarray, x1: np.ndarray, xt: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    x0 = np.asarray(x0, dtype=np.float32)
    x1 = np.asarray(x1, dtype=np.float32)
    xt = np.asarray(xt, dtype=np.float32)

    panels = [
        (0, "nPE (normalized)"),
        (1, "FirstTime (normalized)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, ch, title in [(axes[0], 0, "nPE (normalized)"), (axes[1], 1, "FirstTime (normalized)")]:
        a = x0[ch].ravel()
        b = x1[ch].ravel()
        c = xt[ch].ravel()
        merged = np.concatenate([a, b, c])
        merged = merged[np.isfinite(merged)]
        if merged.size == 0:
            lo, hi = -1.0, 1.0
        else:
            lo, hi = float(np.min(merged)), float(np.max(merged))
            if lo == hi:
                hi = lo + 1.0

        ax.hist(a, bins=80, range=(lo, hi), density=True, alpha=0.35, color="tab:blue", label="x0")
        ax.hist(b, bins=80, range=(lo, hi), density=True, alpha=0.35, color="tab:green", label="x1")
        ax.hist(c, bins=80, range=(lo, hi), density=True, alpha=0.35, color="tab:orange", label="x_t")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Normalized value")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_norm(sig_norm: np.ndarray, geo_np: np.ndarray, label_np: np.ndarray, out_path: Path, title: str) -> None:
    m.show_event_dual_plot(
        sig=sig_norm,
        geo=geo_np,
        label=label_np,
        output_path=str(out_path),
        figure_size=(18, 8),
        marker_size=8.0,
        show_detector_hull=True,
        show=False,
        title_prefix=title,
        firsttime_title="FirstTime (norm)",
        npe_title="nPE (norm)",
        firsttime_cbar_label="FirstTime (normalized)",
        npe_cbar_label="nPE (normalized)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="High-zenith event extraction with normalized x0/x1/x_t plots")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/work/icecube_janus/JANUS/tasks/rectified_flow_0413_jointzero_transformer/high_zenith_norm_xt_0789",
    )
    parser.add_argument("--top_k", type=int, default=30, help="How many high-zenith events to process")
    parser.add_argument("--t_value", type=float, default=0.789, help="Intermediate t in [0,1]")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--h5_path", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not (0.0 <= args.t_value <= 1.0):
        raise ValueError("t_value must be in [0,1]")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Use --cpu to run on CPU.")
        device = torch.device("cuda")

    h5_path = args.h5_path if args.h5_path is not None else m.h5_path

    top_rows = top_high_zenith_indices(h5_path, top_k=args.top_k)

    dataset = m.H5Dataset(
        h5_path=h5_path,
        angle_conversion=m.data_angle_conversion,
        num_workers=0,
        shuffle=False,
    )

    flow = m.RectifiedFlow()

    summary_csv = out_dir / "high_zenith_norm_xt_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "ref_idx", "zenith_rad", "zenith_deg", "t_value", "event_dir"],
        )
        writer.writeheader()

        for row in top_rows:
            rank = int(row["rank"])
            ref_idx = int(row["ref_idx"])
            event_dir = out_dir / f"rank_{rank:02d}_ref_{ref_idx:05d}"
            event_dir.mkdir(parents=True, exist_ok=True)

            sig_raw, geo_raw, label_raw = dataset[ref_idx]
            geo_np = np.asarray(geo_raw, dtype=np.float32)
            label_np = np.asarray(label_raw, dtype=np.float32)

            sig = sig_raw.unsqueeze(0).to(device, non_blocking=True)
            label = label_raw.unsqueeze(0).to(device, non_blocking=True)

            sig_clamp = m._clamp_sig(sig)
            x0_norm, _ = m.prepare_batch(sig_clamp, label, verbose=False)

            # x1 in normalized space (Gaussian noise)
            x1_norm = torch.randn_like(x0_norm)
            t = torch.tensor([float(args.t_value)], device=device, dtype=torch.float32)
            xt_norm = flow.compute_path(x0_norm, x1_norm, t)

            x0_np = x0_norm[0].detach().cpu().numpy()
            x1_np = x1_norm[0].detach().cpu().numpy()
            xt_np = xt_norm[0].detach().cpu().numpy()

            np.save(event_dir / "x0_norm.npy", x0_np)
            np.save(event_dir / "x1_norm.npy", x1_np)
            np.save(event_dir / "xt_norm.npy", xt_np)
            np.save(event_dir / "raw_sig.npy", np.asarray(sig_raw, dtype=np.float32))
            np.save(event_dir / "raw_label.npy", label_np)
            np.save(event_dir / "raw_geo.npy", geo_np)

            plot_norm(
                x0_np,
                geo_np,
                label_np,
                event_dir / "x0_norm.png",
                title=f"High Zenith rank={rank} ref={ref_idx} | x0 (normalized)",
            )
            plot_norm(
                x1_np,
                geo_np,
                label_np,
                event_dir / "x1_norm.png",
                title=f"High Zenith rank={rank} ref={ref_idx} | x1 (normalized noise)",
            )
            plot_norm(
                xt_np,
                geo_np,
                label_np,
                event_dir / f"xt_norm_t_{args.t_value:.3f}.png",
                title=f"High Zenith rank={rank} ref={ref_idx} | x_t (normalized) t={args.t_value:.3f}",
            )

            m.save_epoch_comparison_plot(
                real_sig=x0_np,
                sampled_sig=xt_np,
                geo=geo_np,
                label=label_np,
                output_path=event_dir / f"compare_x0_vs_xt_norm_t_{args.t_value:.3f}.png",
                title_prefix=f"x0 vs x_t in normalized space | rank={rank} ref={ref_idx} t={args.t_value:.3f}",
                figure_size=(18, 8),
                marker_size=10.0,
            )

            save_norm_hist_overlay(
                x0=x0_np,
                x1=x1_np,
                xt=xt_np,
                output_path=event_dir / f"hist_x0_x1_xt_norm_t_{args.t_value:.3f}.png",
            )

            writer.writerow(
                {
                    "rank": rank,
                    "ref_idx": ref_idx,
                    "zenith_rad": f"{row['zenith_rad']:.8f}",
                    "zenith_deg": f"{row['zenith_deg']:.6f}",
                    "t_value": f"{args.t_value:.6f}",
                    "event_dir": str(event_dir),
                }
            )
            print(
                f"[{rank:02d}/{len(top_rows)}] ref={ref_idx} zenith={row['zenith_deg']:.2f}deg t={args.t_value:.3f}"
            )

    print("done")
    print(f"device={device}")
    print(f"out_dir={out_dir}")
    print(f"summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
