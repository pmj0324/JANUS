#!/usr/bin/env python3
"""Sample top-N high-active-nPE events using jointzero rectified-flow checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import random_split

import train_exp_rectified_flow_0413_jointzero as m
from sample_rectified_flow_0413_jointzero import resolve_default_h5_path, save_overlay_histograms
from utils.io.h5_rank_npe import compute_npe_stats_fast
import utils.vis.event_show as event_show


DEFAULT_CKPT = "/home/work/icecube_janus/JANUS/tasks/rectified_flow_0413_jointzero_transformer/models/best_checkpoint_epoch_040_val_loss_0.055823.pt"


def _format_label_line(label_np: np.ndarray, row: dict | None = None) -> str:
    e, ux, uy, x, y, z = [float(v) for v in np.asarray(label_np).tolist()]
    base = f"Label: E={e:.3f}PeV, ux={ux:.3f}, uy={uy:.3f}, X={x:.1f}, Y={y:.1f}, Z={z:.1f}"
    if row is None or "zenith_deg" not in row:
        return base
    return (
        f"{base}, Zenith={float(row['zenith_deg']):.3f}deg, "
        f"Azimuth={float(row['azimuth_deg']):.3f}deg"
    )


def _apply_plot_cut_denorm(sig_denorm: np.ndarray, cut_value: float) -> np.ndarray:
    out = np.asarray(sig_denorm, dtype=np.float32).copy()
    out = np.where(out < float(cut_value), 0.0, out)
    return out


def _make_3d_panes_transparent(ax) -> None:
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
            axis.line.set_color((1.0, 1.0, 1.0, 0.0))
        except Exception:
            pass
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    try:
        ax.figure.patch.set_alpha(0.0)
    except Exception:
        pass


def _build_split_indices(dataset, subset: str, val_ratio: float, split_seed: int) -> np.ndarray:
    n = len(dataset)
    if subset == "all":
        return np.arange(n, dtype=np.int64)

    val_size = int(n * float(val_ratio))
    train_size = n - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(int(split_seed)),
    )
    if subset == "val":
        return np.asarray(val_ds.indices, dtype=np.int64)
    if subset == "train":
        return np.asarray(train_ds.indices, dtype=np.int64)
    raise ValueError(f"Unknown subset: {subset}")


def _load_label_stats(h5_path: str, *, label_key: str = "label") -> dict:
    """Load raw H5 labels and derive direction columns for event selection."""
    with h5py.File(str(Path(h5_path).expanduser().resolve()), "r") as f:
        labels = np.asarray(f[label_key][:, :], dtype=np.float32)

    if labels.ndim != 2 or labels.shape[1] < 6:
        raise ValueError(f"Expected H5 label dataset shape (N, 6), got {labels.shape}")

    energy_pev = labels[:, 0].astype(np.float64) / 1e6
    zenith_rad = labels[:, 1].astype(np.float64)
    azimuth_rad = labels[:, 2].astype(np.float64)
    sin_zenith = np.sin(zenith_rad)

    return {
        "energy_pev": energy_pev,
        "zenith_rad": zenith_rad,
        "zenith_deg": np.degrees(zenith_rad),
        "azimuth_rad": azimuth_rad,
        "azimuth_deg": np.degrees(azimuth_rad),
        "ux": sin_zenith * np.cos(azimuth_rad),
        "uy": sin_zenith * np.sin(azimuth_rad),
        "uz": np.cos(zenith_rad),
        "cos_zenith": np.cos(zenith_rad),
    }


def _row_from_index(stats: dict, label_stats: dict, idx: int) -> dict:
    row = {
        "ref_idx": int(idx),
        "active_npe_count": int(stats["active_npe_count"][idx]),
        "npe_sum": float(stats["npe_sum"][idx]),
        "npe_max": float(stats["npe_max"][idx]),
        "npe_mean_active": float(stats["npe_mean_active"][idx]),
    }
    for key, values in label_stats.items():
        row[key] = float(values[idx])
    return row


def _parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        return None
    return [float(p) for p in parts]


def _make_zenith_bin_edges(
    zenith_values: np.ndarray,
    *,
    zenith_min: float | None,
    zenith_max: float | None,
    zenith_num_bins: int,
    zenith_bin_edges: str | None,
) -> np.ndarray:
    explicit_edges = _parse_float_list(zenith_bin_edges)
    if explicit_edges is not None:
        edges = np.asarray(explicit_edges, dtype=np.float64)
        if edges.size < 2:
            raise ValueError("--zenith_bin_edges needs at least two comma-separated values")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("--zenith_bin_edges must be strictly increasing")
        return edges

    num_bins = int(zenith_num_bins)
    if num_bins <= 0:
        raise ValueError("--zenith_num_bins must be > 0 when using --select_by zenith_bin_active_npe")

    finite = zenith_values[np.isfinite(zenith_values)]
    if finite.size == 0:
        raise ValueError("No finite zenith values available for binning")
    lo = float(zenith_min) if zenith_min is not None else float(np.min(finite))
    hi = float(zenith_max) if zenith_max is not None else float(np.max(finite))
    if not hi > lo:
        raise ValueError(f"Invalid zenith bin range: min={lo}, max={hi}")
    return np.linspace(lo, hi, num_bins + 1, dtype=np.float64)


def _rank_active_pmt_within_indices(stats: dict, indices: np.ndarray) -> np.ndarray:
    return np.lexsort((-stats["npe_sum"][indices], -stats["active_npe_count"][indices]))


def _select_zenith_binned_rows(
    stats: dict,
    label_stats: dict,
    candidate_indices: np.ndarray,
    top_k: int,
    *,
    zenith_min: float | None,
    zenith_max: float | None,
    zenith_sort: str,
    zenith_num_bins: int,
    zenith_bin_edges: str | None,
    per_zenith_bin: int | None,
) -> list[dict]:
    zenith_values = label_stats["zenith_deg"][candidate_indices]
    edges = _make_zenith_bin_edges(
        zenith_values,
        zenith_min=zenith_min,
        zenith_max=zenith_max,
        zenith_num_bins=zenith_num_bins,
        zenith_bin_edges=zenith_bin_edges,
    )
    num_bins = edges.size - 1
    per_bin = int(per_zenith_bin) if per_zenith_bin is not None else int(np.ceil(max(1, int(top_k)) / num_bins))
    per_bin = max(1, per_bin)

    bin_ids = range(num_bins) if zenith_sort == "asc" else range(num_bins - 1, -1, -1)
    rows: list[dict] = []
    for bin_id in bin_ids:
        lo = float(edges[bin_id])
        hi = float(edges[bin_id + 1])
        if bin_id == num_bins - 1:
            in_bin = (zenith_values >= lo) & (zenith_values <= hi)
        else:
            in_bin = (zenith_values >= lo) & (zenith_values < hi)
        bin_indices = candidate_indices[in_bin]
        if bin_indices.size == 0:
            continue

        order_local = _rank_active_pmt_within_indices(stats, bin_indices)
        chosen = bin_indices[order_local[: min(per_bin, order_local.size)]]
        for rank_in_bin, idx in enumerate(chosen.tolist(), 1):
            row = _row_from_index(stats, label_stats, int(idx))
            row["zenith_bin_id"] = int(bin_id)
            row["zenith_bin_min_deg"] = lo
            row["zenith_bin_max_deg"] = hi
            row["rank_in_zenith_bin"] = int(rank_in_bin)
            rows.append(row)

    return rows[: max(1, int(top_k))]


def _select_rows(
    stats: dict,
    label_stats: dict,
    candidate_indices: np.ndarray,
    top_k: int,
    *,
    select_by: str,
    zenith_min: float | None,
    zenith_max: float | None,
    zenith_sort: str,
    zenith_num_bins: int,
    zenith_bin_edges: str | None,
    per_zenith_bin: int | None,
) -> list[dict]:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if candidate_indices.size == 0:
        return []

    if zenith_min is not None:
        candidate_indices = candidate_indices[label_stats["zenith_deg"][candidate_indices] >= float(zenith_min)]
    if zenith_max is not None:
        candidate_indices = candidate_indices[label_stats["zenith_deg"][candidate_indices] <= float(zenith_max)]
    if candidate_indices.size == 0:
        return []

    if select_by == "zenith_bin_active_npe":
        return _select_zenith_binned_rows(
            stats,
            label_stats,
            candidate_indices,
            top_k,
            zenith_min=zenith_min,
            zenith_max=zenith_max,
            zenith_sort=zenith_sort,
            zenith_num_bins=zenith_num_bins,
            zenith_bin_edges=zenith_bin_edges,
            per_zenith_bin=per_zenith_bin,
        )

    if select_by == "active_npe":
        order_local = _rank_active_pmt_within_indices(stats, candidate_indices)
    elif select_by == "npe_sum":
        order_local = np.lexsort((-stats["npe_max"][candidate_indices], -stats["npe_sum"][candidate_indices]))
    elif select_by == "npe_max":
        order_local = np.lexsort((-stats["npe_sum"][candidate_indices], -stats["npe_max"][candidate_indices]))
    elif select_by == "npe_mean_active":
        order_local = np.lexsort((-stats["npe_sum"][candidate_indices], -stats["npe_mean_active"][candidate_indices]))
    elif select_by in {"zenith", "cos_zenith"}:
        values = label_stats["zenith_deg" if select_by == "zenith" else "cos_zenith"][candidate_indices]
        primary = values if zenith_sort == "asc" else -values
        order_local = np.lexsort((-stats["npe_sum"][candidate_indices], primary))
    else:
        raise ValueError(f"Unknown select_by: {select_by}")

    take = order_local[: min(max(1, int(top_k)), order_local.size)]
    chosen = candidate_indices[take]

    return [_row_from_index(stats, label_stats, int(idx)) for idx in chosen.tolist()]


def _write_selected_events_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "rank",
        "ref_idx",
        "active_npe_count",
        "npe_sum",
        "npe_max",
        "npe_mean_active",
        "energy_pev",
        "zenith_rad",
        "zenith_deg",
        "azimuth_rad",
        "azimuth_deg",
        "ux",
        "uy",
        "uz",
        "cos_zenith",
        "zenith_bin_id",
        "zenith_bin_min_deg",
        "zenith_bin_max_deg",
        "rank_in_zenith_bin",
    ]
    int_fields = {"rank", "ref_idx", "active_npe_count", "zenith_bin_id", "rank_in_zenith_bin"}
    float_fields = set(fieldnames) - int_fields
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, row in enumerate(rows, 1):
            out = {"rank": rank}
            for key in fieldnames:
                if key == "rank":
                    continue
                value = row.get(key, "")
                if value == "":
                    out[key] = ""
                else:
                    out[key] = f"{float(value):.8f}" if key in float_fields else int(value)
            w.writerow(out)

def main() -> None:
    parser = argparse.ArgumentParser(description="Top-N active nPE conditioned sampling for jointzero model")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/work/icecube_janus/JANUS/tasks/rectified_flow_0413_jointzero_transformer/samples_topn_active_npe_rk4_100",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Alias of --out_dir")
    parser.add_argument("--top_k", type=int, default=10, help="Number of high-active-nPE events to sample")
    parser.add_argument(
        "--select_by",
        type=str,
        choices=[
            "active_npe",
            "npe_sum",
            "npe_max",
            "npe_mean_active",
            "zenith",
            "cos_zenith",
            "zenith_bin_active_npe",
        ],
        default="active_npe",
        help="Event ranking key. zenith_bin_active_npe bins zenith first, then ranks by active PMTs within each bin.",
    )
    parser.add_argument("--zenith_min", type=float, default=None, help="Minimum raw H5 zenith in degrees")
    parser.add_argument("--zenith_max", type=float, default=None, help="Maximum raw H5 zenith in degrees")
    parser.add_argument(
        "--zenith_sort",
        type=str,
        choices=["asc", "desc"],
        default="desc",
        help="Sort order for --select_by zenith or cos_zenith",
    )
    parser.add_argument(
        "--zenith_num_bins",
        type=int,
        default=0,
        help="Number of equal-width zenith-degree bins for --select_by zenith_bin_active_npe",
    )
    parser.add_argument(
        "--zenith_bin_edges",
        type=str,
        default=None,
        help="Comma-separated zenith-degree bin edges, e.g. '0,30,60,90,120,150,180'",
    )
    parser.add_argument(
        "--per_zenith_bin",
        type=int,
        default=None,
        help="Events to take from each zenith bin. Default: ceil(top_k / number_of_bins).",
    )
    parser.add_argument("--num_samples", type=int, default=1, help="Samples per selected event")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling_method", type=str, default="rk4")
    parser.add_argument(
        "--sampling_methods",
        type=str,
        default=None,
        help="Comma-separated methods to run in one go (e.g., rk4,euler,dopri5). Overrides --sampling_method.",
    )
    parser.add_argument("--sampling_steps", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=None, help="Override cfg scale")
    parser.add_argument("--force_no_cfg", action="store_true")
    parser.add_argument("--h5_path", type=str, default=None)
    parser.add_argument("--hist_bins", type=int, default=80)
    parser.add_argument("--plot_cut_value", type=float, default=1.0, help="For plots only: values < cut become 0 in denormalized space")
    parser.add_argument("--subset", type=str, choices=["all", "train", "val"], default="all")
    parser.add_argument("--val_ratio", type=float, default=float(m.val_ratio), help="Split ratio to reproduce train/val split")
    parser.add_argument("--split_seed", type=int, default=int(m.seed), help="Seed used for train/val split")
    parser.add_argument("--no_3d_grid", action="store_true", help="Disable grid lines in 3D event plots")
    parser.add_argument("--chunk_events", type=int, default=4096, help="H5 scan chunk size (events)")
    parser.add_argument("--no_stats_cache", action="store_true", help="Disable cached nPE stats")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.output_dir is not None:
        args.out_dir = args.output_dir

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

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

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    h5_path = args.h5_path if args.h5_path is not None else resolve_default_h5_path()
    dataset = m.H5Dataset(
        h5_path=h5_path,
        angle_conversion=m.data_angle_conversion,
        num_workers=0,
        shuffle=False,
    )

    print(f"Dataset length: {len(dataset)}")
    print("Computing nPE stats (fast chunked H5 scan)...")
    stats_cache = out_dir / "_cache_npe_stats.npz"
    stats = compute_npe_stats_fast(
        h5_path=h5_path,
        signal_key="input",
        chunk_events=args.chunk_events,
        cache_npz=stats_cache,
        use_cache=(not args.no_stats_cache),
    )

    print("Loading raw labels for zenith-aware selection...")
    label_stats = _load_label_stats(h5_path)
    candidate_indices = _build_split_indices(dataset, args.subset, args.val_ratio, args.split_seed)
    if args.subset != "all":
        print(f"Subset={args.subset} size={candidate_indices.size} (val_ratio={args.val_ratio}, split_seed={args.split_seed})")
    top_rows = _select_rows(
        stats,
        label_stats,
        candidate_indices,
        top_k=max(1, int(args.top_k)),
        select_by=args.select_by,
        zenith_min=args.zenith_min,
        zenith_max=args.zenith_max,
        zenith_sort=args.zenith_sort,
        zenith_num_bins=args.zenith_num_bins,
        zenith_bin_edges=args.zenith_bin_edges,
        per_zenith_bin=args.per_zenith_bin,
    )
    if not top_rows:
        raise RuntimeError(
            "No events selected. Check --subset, --zenith_min/--zenith_max, and --top_k."
        )

    summary_csv = out_dir / "selected_events.csv"
    _write_selected_events_csv(summary_csv, top_rows)

    sig0, geo0, _ = dataset[0]
    geo_norm = m.apply_minmax_geo(geo0, m.geo_min, m.geo_max, feature_range=(0, 1))

    model = m.FlowDiTTransformer(
        geo=geo_norm,
        d_model=int(checkpoint.get("d_model", m.model_d_model)),
        nhead=int(checkpoint.get("nhead", m.model_nhead)),
        depth=int(checkpoint.get("depth", m.model_depth)),
        mlp_ratio=float(checkpoint.get("mlp_ratio", m.model_mlp_ratio)),
        dropout=float(checkpoint.get("dropout", m.model_dropout)),
        label_dim=int(checkpoint.get("label_dim", m.model_label_dim)),
        attention_type=str(checkpoint.get("attention_type", m.attention_type)),
        linformer_k=int(checkpoint.get("linformer_k", m.linformer_k)),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    if args.no_3d_grid:
        def _style_axes_no_grid(ax):
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
            ax.grid(False)
            _make_3d_panes_transparent(ax)
        event_show._style_axes = _style_axes_no_grid
        m.event_vis._style_axes = _style_axes_no_grid
    else:
        def _style_axes_with_grid(ax):
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
            ax.grid(True, alpha=0.25)
            _make_3d_panes_transparent(ax)
        event_show._style_axes = _style_axes_with_grid
        m.event_vis._style_axes = _style_axes_with_grid

    if args.sampling_methods:
        method_list = [s.strip().lower() for s in args.sampling_methods.split(",") if s.strip()]
    else:
        method_list = [str(args.sampling_method).lower()]
    valid_methods = {"euler", "heun", "rk4", "dopri5"}
    for method in method_list:
        if method not in valid_methods:
            raise ValueError(f"Unsupported sampling method: {method}. Choose from {sorted(valid_methods)}")

    sampling_steps = int(args.sampling_steps)

    use_cfg = bool(checkpoint.get("use_cfg", m.use_cfg)) and (not args.force_no_cfg)
    cfg_scale = float(checkpoint.get("cfg_scale", m.cfg_scale))
    if args.cfg_scale is not None:
        cfg_scale = float(args.cfg_scale)

    flow_matching = m.RectifiedFlow()

    print("Events selected:")
    for rank, row in enumerate(top_rows, 1):
        bin_text = ""
        if "zenith_bin_id" in row:
            bin_text = (
                f" bin={row['zenith_bin_id']}[{row['zenith_bin_min_deg']:.2f},"
                f"{row['zenith_bin_max_deg']:.2f}] rank_in_bin={row['rank_in_zenith_bin']}"
            )
        print(
            f"  #{rank:02d} ref_idx={row['ref_idx']} active_npe_count={row['active_npe_count']} "
            f"npe_sum={row['npe_sum']:.3f} zenith={row['zenith_deg']:.3f}deg{bin_text}"
        )

    for sampling_method in method_list:
        method_out_dir = out_dir / f"method_{sampling_method}"
        method_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nRunning sampling method: {sampling_method}")

        for rank, row in enumerate(top_rows, 1):
            ref_idx = int(row["ref_idx"])
            event_dir = method_out_dir / f"rank_{rank:02d}_ref_{ref_idx:04d}"
            event_dir.mkdir(parents=True, exist_ok=True)

            sig_ref_raw, geo_ref_raw, label_ref_raw = dataset[ref_idx]
            sig_ref = sig_ref_raw.unsqueeze(0).to(device, non_blocking=True)
            label_ref = label_ref_raw.unsqueeze(0).to(device, non_blocking=True)

            sig_ref_clamp = m._clamp_sig(sig_ref)
            sig_ref_norm, label_ref_norm = m.prepare_batch(sig_ref_clamp, label_ref, verbose=False)

            with torch.inference_mode():
                x1 = torch.randn(args.num_samples, 2, model.L, device=device)
                if use_cfg:
                    x_uncond = m._sample_flow_matching(flow_matching, sampling_method, model, x1, sampling_steps, None, device)
                    label_cond = label_ref_norm.repeat(args.num_samples, 1)
                    x_cond = m._sample_flow_matching(flow_matching, sampling_method, model, x1, sampling_steps, label_cond, device)
                    x_norm = x_uncond + cfg_scale * (x_cond - x_uncond)
                else:
                    label_cond = label_ref_norm.repeat(args.num_samples, 1)
                    x_norm = m._sample_flow_matching(flow_matching, sampling_method, model, x1, sampling_steps, label_cond, device)

            samples_denorm = m.denormalize_sig(x_norm).detach().cpu().numpy()
            samples_norm = x_norm.detach().cpu().numpy()

            real_sig_denorm = m.denormalize_sig(sig_ref_norm)[0].detach().cpu().numpy()
            real_sig_denorm_plot = _apply_plot_cut_denorm(real_sig_denorm, args.plot_cut_value)
            geo_ref_np = geo_ref_raw.detach().cpu().numpy()
            label_ref_np = label_ref_raw.detach().cpu().numpy()
            label_line = _format_label_line(label_ref_np, row)
            title_kind = f"{args.select_by} Rank {rank}"
            if "zenith_bin_id" in row:
                title_kind += (
                    f" | zbin {row['zenith_bin_id']} "
                    f"[{row['zenith_bin_min_deg']:.1f},{row['zenith_bin_max_deg']:.1f}]"
                )

            np.save(event_dir / f"ref_{ref_idx:04d}_real_denorm.npy", real_sig_denorm)

            real_png_path = event_dir / f"ref_{ref_idx:04d}_real.png"
            m.show_event_dual_plot(
                sig=real_sig_denorm_plot,
                geo=geo_ref_np,
                label=label_ref_np,
                output_path=str(real_png_path),
                figure_size=(18, 8),
                marker_size=8.0,
                show_detector_hull=True,
                show=False,
                title_prefix=f"{title_kind} | ref={ref_idx} | Real\n{label_line}",
                firsttime_title="FirstTime (real)",
                npe_title="nPE (real)",
            )

            for i in range(args.num_samples):
                idx = i + 1
                sample_denorm = samples_denorm[i]
                sample_norm = samples_norm[i]
                sample_denorm_plot = _apply_plot_cut_denorm(sample_denorm, args.plot_cut_value)
                sample_prefix = f"sample_{idx:03d}_ref_{ref_idx:04d}"

                denorm_npy = event_dir / f"{sample_prefix}_denorm.npy"
                norm_npy = event_dir / f"{sample_prefix}_norm.npy"
                np.save(denorm_npy, sample_denorm)
                np.save(norm_npy, sample_norm)

                png_path = event_dir / f"{sample_prefix}_event.png"
                m.show_event_dual_plot(
                    sig=sample_denorm_plot,
                    geo=geo_ref_np,
                    label=label_ref_np,
                    output_path=str(png_path),
                    figure_size=(18, 8),
                    marker_size=8.0,
                    show_detector_hull=True,
                    show=False,
                    title_prefix=(
                        f"{title_kind} | ref={ref_idx} | sample={idx} | "
                        f"{sampling_method}/{sampling_steps} | cfg={cfg_scale if use_cfg else 'off'}\n"
                        f"{label_line}"
                    ),
                    firsttime_title="FirstTime (sampled)",
                    npe_title="nPE (sampled)",
                )

                compare_png_path = event_dir / f"{sample_prefix}_compare_real_vs_sample.png"
                m.save_epoch_comparison_plot(
                    real_sig=real_sig_denorm_plot,
                    sampled_sig=sample_denorm_plot,
                    geo=geo_ref_np,
                    label=label_ref_np,
                    output_path=compare_png_path,
                    title_prefix=(
                        f"{title_kind} | ref={ref_idx} | sample={idx} | "
                        f"{sampling_method}/{sampling_steps} | cfg={cfg_scale if use_cfg else 'off'}\n"
                        f"{label_line}"
                    ),
                    figure_size=(18, 8),
                    marker_size=10.0,
                )

                hist_path = event_dir / f"{sample_prefix}_hist_overlay.png"
                save_overlay_histograms(
                    real_denorm=real_sig_denorm_plot,
                    sampled_denorm=sample_denorm_plot,
                    output_path=hist_path,
                    bins=args.hist_bins,
                    log_y=True,
                )

                print(
                    f"[{sampling_method} | rank {rank}/{len(top_rows)} | sample {idx}/{args.num_samples}] "
                    f"saved: {denorm_npy.name}"
                )

    print("done")
    print(f"device={device}")
    print(f"checkpoint={ckpt_path}")
    print(f"out_dir={out_dir}")
    print(f"top_k={len(top_rows)}, num_samples_per_event={args.num_samples}")
    print(f"sampling_methods={method_list}, sampling_steps={sampling_steps}")
    print(f"use_cfg={use_cfg}, cfg_scale={cfg_scale}")
    print(f"selected list csv={summary_csv}")
    print(f"subset={args.subset}")
    print(f"select_by={args.select_by}, zenith_min={args.zenith_min}, zenith_max={args.zenith_max}")
    if args.select_by == "zenith_bin_active_npe":
        print(
            f"zenith_num_bins={args.zenith_num_bins}, zenith_bin_edges={args.zenith_bin_edges}, "
            f"per_zenith_bin={args.per_zenith_bin}"
        )


if __name__ == "__main__":
    main()
