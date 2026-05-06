#!/usr/bin/env python3
"""Sampling script for checkpoints produced by train_exp_rectified_flow_0413_jointzero.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import train_exp_rectified_flow_0413_jointzero as m
from flow.optimal_transport import OptimalTransportFlow


def build_flow_matching(flow_mode: str):
    mode = str(flow_mode).strip().lower()
    if mode in {"rectified_flow", "rectified", "rf"}:
        return m.RectifiedFlow(), "rectified_flow"
    if mode in {"ot", "optimal_transport", "optimal-transport"}:
        return OptimalTransportFlow(), "ot"
    raise ValueError(
        f"Unsupported flow_mode='{flow_mode}'. Choose from: rectified_flow, ot"
    )


def resolve_default_h5_path() -> str:
    raw = Path(m.h5_path)
    if raw.is_absolute():
        return str(raw)
    return str((Path(m.__file__).resolve().parent / raw).resolve())


def save_overlay_histograms(
    real_denorm: np.ndarray,
    sampled_denorm: np.ndarray,
    output_path: Path,
    *,
    bins: int = 80,
    log_y: bool = True,
) -> None:
    real_denorm = np.asarray(real_denorm, dtype=np.float32)
    sampled_denorm = np.asarray(sampled_denorm, dtype=np.float32)

    real_npe = real_denorm[0].ravel()
    real_ftime = real_denorm[1].ravel()
    sampled_npe = sampled_denorm[0].ravel()
    sampled_ftime = sampled_denorm[1].ravel()

    def _finite(arr: np.ndarray) -> np.ndarray:
        return arr[np.isfinite(arr)]

    real_npe = _finite(real_npe)
    real_ftime = _finite(real_ftime)
    sampled_npe = _finite(sampled_npe)
    sampled_ftime = _finite(sampled_ftime)

    def _range(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        merged = np.concatenate([a, b]) if (a.size or b.size) else np.array([0.0], dtype=np.float32)
        lo = float(np.min(merged))
        hi = float(np.max(merged))
        if lo == hi:
            hi = lo + 1.0
        return lo, hi

    npe_min, npe_max = _range(real_npe, sampled_npe)
    ftime_min, ftime_max = _range(real_ftime, sampled_ftime)

    def _draw(save_path: Path, xlog: bool) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        panels = [
            (axes[0], real_npe, sampled_npe, "nPE", npe_min, npe_max),
            (axes[1], real_ftime, sampled_ftime, "Hit Time", ftime_min, ftime_max),
        ]

        for ax, true_arr, gen_arr, xlabel, x_min, x_max in panels:
            arr_true = true_arr
            arr_gen = gen_arr
            if xlog:
                arr_true = arr_true[arr_true > 0]
                arr_gen = arr_gen[arr_gen > 0]
                positive = np.concatenate([arr_true, arr_gen]) if (arr_true.size or arr_gen.size) else np.array([1.0], dtype=np.float32)
                x_min = float(np.min(positive))
                x_max = float(np.max(positive))
                if x_min == x_max:
                    x_max = x_min * 1.1 if x_min > 0 else 1.0

            ax.hist(
                arr_true,
                bins=bins,
                range=(x_min, x_max),
                density=True,
                color="tab:blue",
                alpha=0.45,
                label="IceCube simulation",
            )
            ax.hist(
                arr_gen,
                bins=bins,
                range=(x_min, x_max),
                density=True,
                color="tab:orange",
                alpha=0.45,
                label="JANUS generated",
            )
            ax.set_yscale("log")
            if xlog:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel)
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
        description="Sample from rectified-flow jointzero checkpoint with training-matched preprocessing.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--ref_idx", type=int, default=0, help="Reference event index for condition label")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sampling_method", type=str, default=None, help="Override sampler: euler/heun/rk4/dopri5")
    parser.add_argument("--sampling_steps", type=int, default=None, help="Override ODE steps")
    parser.add_argument("--cfg_scale", type=float, default=None, help="Override CFG scale")
    parser.add_argument("--force_no_cfg", action="store_true", help="Disable CFG even if checkpoint has use_cfg=True")
    parser.add_argument(
        "--flow_mode",
        type=str,
        default="checkpoint",
        help="Flow implementation to use: checkpoint/rectified_flow/ot",
    )
    parser.add_argument("--h5_path", type=str, default=None, help="Override dataset path")
    parser.add_argument("--no_png", action="store_true", help="Do not save event PNGs")
    parser.add_argument("--no_compare_png", action="store_true", help="Do not save side-by-side real vs sample PNGs")
    parser.add_argument("--no_hist", action="store_true", help="Do not save overlay histogram PNGs")
    parser.add_argument("--hist_bins", type=int, default=80, help="Histogram bins")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    args = parser.parse_args()

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
        if args.gpu is None:
            device = torch.device("cuda")
        else:
            if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
                raise ValueError(
                    f"GPU index {args.gpu} is invalid; available CUDA devices: {torch.cuda.device_count()}"
                )
            device = torch.device(f"cuda:{args.gpu}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    h5_path = args.h5_path if args.h5_path is not None else resolve_default_h5_path()
    dataset = m.H5Dataset(
        h5_path=h5_path,
        angle_conversion=m.data_angle_conversion,
        num_workers=0,
        shuffle=False,
    )

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

    sampling_method = str(checkpoint.get("sampling_method", m.sampling_method))
    if args.sampling_method is not None:
        sampling_method = args.sampling_method

    sampling_steps = int(checkpoint.get("sampling_steps", m.sampling_steps))
    if args.sampling_steps is not None:
        sampling_steps = int(args.sampling_steps)

    use_cfg = bool(checkpoint.get("use_cfg", m.use_cfg)) and (not args.force_no_cfg)
    cfg_scale = float(checkpoint.get("cfg_scale", m.cfg_scale))
    if args.cfg_scale is not None:
        cfg_scale = float(args.cfg_scale)

    checkpoint_flow_mode = str(checkpoint.get("flow_name", getattr(m, "flow_name", "rectified_flow")))
    requested_flow_mode = checkpoint_flow_mode if args.flow_mode == "checkpoint" else args.flow_mode
    flow_matching, resolved_flow_mode = build_flow_matching(requested_flow_mode)

    ref_idx = int(args.ref_idx)
    sig_ref_raw, geo_ref_raw, label_ref_raw = dataset[ref_idx]

    sig_ref = sig_ref_raw.unsqueeze(0).to(device, non_blocking=True)
    label_ref = label_ref_raw.unsqueeze(0).to(device, non_blocking=True)
    sig_ref_clamp = m._clamp_sig(sig_ref)
    sig_ref_norm, label_ref_norm = m.prepare_batch(sig_ref_clamp, label_ref, verbose=False)

    with torch.inference_mode():
        x1 = torch.randn(args.num_samples, 2, model.L, device=device)

        if use_cfg:
            x_uncond = m._sample_flow_matching(
                flow_matching, sampling_method, model, x1, sampling_steps, None, device
            )
            label_cond = label_ref_norm.repeat(args.num_samples, 1)
            x_cond = m._sample_flow_matching(
                flow_matching, sampling_method, model, x1, sampling_steps, label_cond, device
            )
            x_norm = x_uncond + cfg_scale * (x_cond - x_uncond)
        else:
            label_cond = label_ref_norm.repeat(args.num_samples, 1)
            x_norm = m._sample_flow_matching(
                flow_matching, sampling_method, model, x1, sampling_steps, label_cond, device
            )

    samples_denorm = m.denormalize_sig(x_norm).detach().cpu().numpy()
    samples_norm = x_norm.detach().cpu().numpy()

    real_sig_denorm = m.denormalize_sig(sig_ref_norm)[0].detach().cpu().numpy()
    geo_ref_np = geo_ref_raw.detach().cpu().numpy()
    label_ref_np = label_ref_raw.detach().cpu().numpy()

    np.save(out_dir / f"real_event_{ref_idx:04d}_denorm.npy", real_sig_denorm)

    for i in range(args.num_samples):
        idx = i + 1
        sample_denorm = samples_denorm[i]
        sample_norm = samples_norm[i]

        denorm_npy = out_dir / f"sampled_event_{ref_idx}_sample_{idx:03d}.npy"
        norm_npy = out_dir / f"sampled_event_{ref_idx}_sample_{idx:03d}_norm.npy"
        np.save(denorm_npy, sample_denorm)
        np.save(norm_npy, sample_norm)

        if not args.no_png:
            real_png_path = out_dir / f"real_event_{ref_idx:04d}.png"
            m.show_event_dual_plot(
                sig=real_sig_denorm,
                geo=geo_ref_np,
                label=label_ref_np,
                output_path=str(real_png_path),
                figure_size=(18, 8),
                marker_size=8.0,
                show_detector_hull=True,
                show=False,
                title_prefix=f"Rectified Flow JointZero | ref={ref_idx} | Real event",
                firsttime_title="FirstTime (real)",
                npe_title="nPE (real)",
            )

            png_path = out_dir / f"sampled_event_{ref_idx}_sample_{idx:03d}.png"
            m.show_event_dual_plot(
                sig=sample_denorm,
                geo=geo_ref_np,
                label=label_ref_np,
                output_path=str(png_path),
                figure_size=(18, 8),
                marker_size=8.0,
                show_detector_hull=True,
                show=False,
                title_prefix=(
                    f"Rectified Flow JointZero | ref={ref_idx} | sample={idx} | "
                    f"{sampling_method}/{sampling_steps} | cfg={cfg_scale if use_cfg else 'off'}"
                ),
                firsttime_title="FirstTime (sampled)",
                npe_title="nPE (sampled)",
            )

        if not args.no_compare_png:
            compare_png_path = out_dir / f"compare_real_vs_sample_ref_{ref_idx}_sample_{idx:03d}.png"
            m.save_epoch_comparison_plot(
                real_sig=real_sig_denorm,
                sampled_sig=sample_denorm,
                geo=geo_ref_np,
                label=label_ref_np,
                output_path=compare_png_path,
                title_prefix=(
                    f"Real vs Generated | ref={ref_idx} | sample={idx} | "
                    f"{sampling_method}/{sampling_steps} | cfg={cfg_scale if use_cfg else 'off'}"
                ),
                figure_size=(18, 8),
                marker_size=10.0,
            )

        if not args.no_hist:
            hist_path = out_dir / f"sampled_event_{ref_idx}_sample_{idx:03d}_hist_overlay.png"
            save_overlay_histograms(
                real_denorm=real_sig_denorm,
                sampled_denorm=sample_denorm,
                output_path=hist_path,
                bins=args.hist_bins,
                log_y=True,
            )

        print(f"[{idx}/{args.num_samples}] saved: {denorm_npy.name}")

    print("done")
    print(f"device={device}")
    print(f"checkpoint={ckpt_path}")
    print(f"out_dir={out_dir}")
    print(f"ref_idx={ref_idx}, num_samples={args.num_samples}")
    print(f"flow_mode={resolved_flow_mode}")
    print(f"sampling_method={sampling_method}, sampling_steps={sampling_steps}")
    print(f"use_cfg={use_cfg}, cfg_scale={cfg_scale}")


if __name__ == "__main__":
    main()
