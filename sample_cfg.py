#!/usr/bin/env python3
"""
Sampling script for GENESIS diffusion model trained with train_exp_cfg.py.
Supports Classifier-Free Guidance (CFG): loads use_cfg/cfg_scale from checkpoint and applies
  eps_hat = eps_uncond + cfg_scale * (eps_cond - eps_uncond) when sampling.
Compatible with checkpoints from both train_exp.py and train_exp_cfg.py (CFG optional).
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from dataloader.h5 import H5Dataset
from diffusion.schedules import sigmoid_beta_schedule, compute_alpha_schedule
from utils.normalize import normalize, denormalize_log_minmax, denormalize_minmax, apply_minmax_geo
from utils.vis.event_show import show_event_dual_plot
from utils.device import get_default_device
from models import DiffusionDiTTransformer


def get_device(gpu_id: int = None) -> torch.device:
    if torch.cuda.is_available():
        if gpu_id is not None:
            if gpu_id >= torch.cuda.device_count():
                raise ValueError(f"GPU {gpu_id} not available.")
            return torch.device(f"cuda:{gpu_id}")
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_reserved(i) / 1024**3
            if mem < 1.0:
                return torch.device(f"cuda:{i}")
        return torch.device("cuda:0")
    return get_default_device()


# ---- 정규화 (train_exp_cfg / sample.py와 동일) ----
npe_clip = 1000.0
ftime_clip = 21000.0
log_min = 0.0
ftime_log_max = float(np.log1p(ftime_clip))
_feature_range = (-1, 1)
_label_methods = ["log_minmax", "identity", "identity", "minmax", "minmax", "minmax"]
_label_feature_ranges = [_feature_range] * 6
_ENERGY_PEV_MINMAX = {"min": 1.0, "max": 100.0}
energy_log_min = float(np.log1p(_ENERGY_PEV_MINMAX["min"]))
energy_log_max = float(np.log1p(_ENERGY_PEV_MINMAX["max"]))
_LABEL_XYZ_MINMAX = [
    {"min": -570.9000244140625, "max": 576.3699951171875},
    {"min": -521.0800170898438, "max": 509.5},
    {"min": -509.8599853515625, "max": 506.0566711425781},
]
_GEO_XYZ_MINMAX = [
    {"min": -570.9000244140625, "max": 576.3699951171875},
    {"min": -521.0800170898438, "max": 509.5},
    {"min": -509.8599853515625, "max": 506.0566711425781},
]
geo_min = np.array([_GEO_XYZ_MINMAX[j]["min"] for j in range(3)], dtype=np.float32)
geo_max = np.array([_GEO_XYZ_MINMAX[j]["max"] for j in range(3)], dtype=np.float32)
_label_stats = [
    {"log_min": energy_log_min, "log_max": energy_log_max},
    {},
    {},
    _LABEL_XYZ_MINMAX[0],
    _LABEL_XYZ_MINMAX[1],
    _LABEL_XYZ_MINMAX[2],
]
_channel_stats = [
    {"min": 0.0, "max": npe_clip},
    {"log_min": log_min, "log_max": ftime_log_max},
]
LABEL_NAMES = ["Energy (PeV)", "ux", "uy", "X", "Y", "Z"]


@normalize(
    channel_methods=["minmax", "log_minmax"],
    feature_ranges=[_feature_range, _feature_range],
    channel_stats=_channel_stats,
    arg_index=0,
    label_arg_index=1,
    label_methods=_label_methods,
    label_feature_ranges=_label_feature_ranges,
    label_stats=_label_stats,
)
def prepare_batch(sig: torch.Tensor, label: torch.Tensor, *, verbose: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    if verbose:
        print("prepare_batch: label", label)
    return (sig, label)


def _clamp_sig(sig: torch.Tensor) -> torch.Tensor:
    s = sig.clone()
    s[:, 0] = torch.clamp(s[:, 0], min=0.0, max=npe_clip)
    s[:, 1] = torch.clamp(s[:, 1], min=0.0, max=ftime_clip)
    return s


def denormalize_sig(sig: torch.Tensor) -> torch.Tensor:
    out = sig.clone()
    if sig.dim() == 3:
        out[:, 0, :] = denormalize_minmax(sig[:, 0, :], 0.0, npe_clip, _feature_range)
        out[:, 1, :] = denormalize_log_minmax(sig[:, 1, :], log_min, ftime_log_max, _feature_range)
    else:
        out[0, :] = denormalize_minmax(sig[0, :], 0.0, npe_clip, _feature_range)
        out[1, :] = denormalize_log_minmax(sig[1, :], log_min, ftime_log_max, _feature_range)
    return out


def _apply_cuts(sig: np.ndarray, cut_npe: float, cut_firsttime: float) -> np.ndarray:
    out = sig.copy()
    if cut_npe > 0:
        out[0] = np.where(out[0] <= cut_npe, 0.0, out[0])
    if cut_firsttime > 0:
        out[1] = np.where(out[1] <= cut_firsttime, 0.0, out[1])
    return out


def plot_histogram(
    sig: np.ndarray,
    output_path: Path,
    title_suffix: str = "",
    cut_npe: float = 0.0,
    cut_firsttime: float = 0.0,
):
    npe = sig[0]
    ftime = sig[1]
    # inf/nan 제외 (샘플에서 FirstTime 등이 inf 나올 수 있음)
    npe_plot = npe[(npe > cut_npe) & np.isfinite(npe)]
    ftime_plot = ftime[(ftime > cut_firsttime) & np.isfinite(ftime)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    if len(npe_plot) > 0:
        ax1.hist(npe_plot, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('nPE')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'nPE Distribution{title_suffix}')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(npe_plot.mean(), color='red', linestyle='--', label=f'Mean: {npe_plot.mean():.2f}')
        ax1.axvline(np.median(npe_plot), color='green', linestyle='--', label=f'Median: {np.median(npe_plot):.2f}')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No nPE above cut', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'nPE Distribution{title_suffix} (empty)')
    ax2 = axes[1]
    if len(ftime_plot) > 0:
        ax2.hist(ftime_plot, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('FirstTime')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'FirstTime Distribution{title_suffix}')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(ftime_plot.mean(), color='red', linestyle='--', label=f'Mean: {ftime_plot.mean():.2f}')
        ax2.axvline(np.median(ftime_plot), color='green', linestyle='--', label=f'Median: {np.median(ftime_plot):.2f}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No FirstTime above cut', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'FirstTime Distribution{title_suffix} (empty)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved histogram: {output_path}")


def remove_orig_mod_prefix(state_dict: dict) -> dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_state_dict[key[len("_orig_mod."):]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_model(checkpoint_path: str, device: torch.device):
    """train_exp_cfg / train_exp 체크포인트 로드. use_cfg, cfg_scale 있으면 반환."""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    T = checkpoint.get("T", 1000)
    beta_start = checkpoint.get("beta_start", 1e-4)
    beta_end = checkpoint.get("beta_end", 2e-2)
    use_cfg = checkpoint.get("use_cfg", False)
    cfg_scale = float(checkpoint.get("cfg_scale", 2.0))
    d_model = checkpoint.get("d_model", 256)
    nhead = checkpoint.get("nhead", 8)
    depth = checkpoint.get("depth", 6)
    mlp_ratio = checkpoint.get("mlp_ratio", 4.0)
    label_dim = checkpoint.get("label_dim", 6)

    print(f"Model config: T={T}, d_model={d_model}, nhead={nhead}, depth={depth}")
    print(f"CFG: use_cfg={use_cfg}, cfg_scale={cfg_scale}")

    h5_path = "./GENESIS-data/22644_0921_time_shift.h5"
    dataset = H5Dataset(h5_path=h5_path)
    geo_raw = dataset[0][1]
    geo = apply_minmax_geo(geo_raw, geo_min, geo_max, feature_range=(0, 1))

    model = DiffusionDiTTransformer(
        geo=geo,
        d_model=d_model,
        nhead=nhead,
        depth=depth,
        mlp_ratio=mlp_ratio,
        dropout=0.0,
        label_dim=label_dim,
    ).to(device)

    state_dict = checkpoint["model_state_dict"]
    state_dict = remove_orig_mod_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if "betas" in checkpoint and "alphas_cumprod" in checkpoint:
        betas = checkpoint["betas"].to(device)
        alphas = checkpoint["alphas"].to(device)
        alphas_cumprod = checkpoint["alphas_cumprod"].to(device)
    else:
        betas = sigmoid_beta_schedule(timesteps=T, beta_start=beta_start, beta_end=beta_end).to(device)
        alpha_schedule = compute_alpha_schedule(betas)
        alphas = alpha_schedule["alphas"]
        alphas_cumprod = alpha_schedule["alphas_cumprod"]

    return model, betas, alphas, alphas_cumprod, T, dataset, use_cfg, cfg_scale


def sample(
    model: nn.Module,
    label: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    T: int,
    num_samples: int = 1,
    device: torch.device = None,
    scaler: GradScaler = None,
    output_dir: Path = None,
    ref_idx: int = 0,
    geo_np: np.ndarray = None,
    label_np: np.ndarray = None,
    save_histogram: bool = False,
    cut_npe: float = 0.0,
    cut_firsttime: float = 0.0,
    use_cfg: bool = False,
    cfg_scale: float = 2.0,
    null_label_norm: torch.Tensor = None,
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    B = num_samples

    with torch.no_grad():
        dummy_sig = torch.zeros(1, 2, model.L, device=device)
        _, label_norm = prepare_batch(dummy_sig, label.unsqueeze(0) if label.dim() == 1 else label, verbose=False)
        if label_norm.dim() == 1:
            label_norm = label_norm.unsqueeze(0)
        del dummy_sig

    x = torch.randn(B, 2, model.L, device=device)
    print("Running reverse diffusion (T -> 1)%s..." % (" with CFG" if (use_cfg and cfg_scale != 1.0 and null_label_norm is not None) else ""))
    pbar = tqdm(reversed(range(1, T + 1)), total=T, desc="Sampling")

    with torch.no_grad():
        for t_val in pbar:
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            with autocast(device.type, enabled=(scaler is not None)):
                if use_cfg and cfg_scale != 1.0 and null_label_norm is not None:
                    eps_uncond = model(x, t_batch, null_label_norm.expand(B, -1))
                    eps_cond = model(x, t_batch, label_norm)
                    eps_hat = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
                else:
                    eps_hat = model(x, t_batch, label_norm)

            idx = t_val - 1
            alpha_t = alphas[idx]
            alpha_bar_t = alphas_cumprod[idx]
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (betas[idx] / torch.sqrt(1.0 - alpha_bar_t)) * eps_hat
            )
            if t_val > 1:
                alpha_bar_prev = alphas_cumprod[idx - 1] if idx > 0 else torch.tensor(1.0, device=device)
                posterior_variance = betas[idx] * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                var = torch.sqrt(posterior_variance)
                x = mean + var * torch.randn_like(x)
            else:
                x = mean
            pbar.set_postfix({"t": t_val, "mean": f"{x.mean().item():.4f}", "std": f"{x.std().item():.4f}"})

    samples_denorm = denormalize_sig(x)
    samples_np = samples_denorm.detach().cpu().numpy()
    del x, samples_denorm, label_norm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and getattr(torch.mps, "empty_cache", None):
        torch.mps.empty_cache()

    print("Sampling completed!")
    print(f"Generated {num_samples} sample(s)")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    saved_samples = []
    for i in range(num_samples):
        sample_np = samples_np[i]
        print(f"\nSample {i+1}/{num_samples}:")
        print(f"  Shape: {sample_np.shape}")
        print(f"  nPE range: [{sample_np[0].min():.2f}, {sample_np[0].max():.2f}]")
        print(f"  FirstTime range: [{sample_np[1].min():.2f}, {sample_np[1].max():.2f}]")
        if output_dir:
            np_output_path = output_dir / f"sampled_event_{ref_idx}_sample_{i+1:03d}.npy"
            np.save(np_output_path, sample_np)
            print(f"  Saved numpy: {np_output_path}")
            if geo_np is not None and label_np is not None:
                sig_vis = _apply_cuts(sample_np, cut_npe, cut_firsttime)
                img_output_path = output_dir / f"sampled_event_{ref_idx}_sample_{i+1:03d}.png"
                show_event_dual_plot(
                    sig=sig_vis,
                    geo=geo_np,
                    label=label_np,
                    output_path=str(img_output_path),
                    figure_size=(18, 8),
                    marker_size=8.0,
                    show_detector_hull=True,
                    show=False,
                    title_prefix=f"sample_cfg.py | CFG={cfg_scale} | Sampled #{i+1} | event {ref_idx}",
                    firsttime_title="FirstTime (sampled)",
                    npe_title="nPE (sampled)",
                )
                print(f"  Saved image: {img_output_path}")
            if save_histogram:
                hist_output_path = output_dir / f"sampled_event_{ref_idx}_sample_{i+1:03d}_histogram.png"
                plot_histogram(sample_np, hist_output_path, title_suffix=f" (Sample #{i+1})", cut_npe=cut_npe, cut_firsttime=cut_firsttime)
        saved_samples.append(sample_np)

    return saved_samples[0] if len(saved_samples) == 1 else saved_samples


def main():
    parser = argparse.ArgumentParser(description="Sample from GENESIS (train_exp_cfg checkpoints, with CFG support)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt from train_exp_cfg or train_exp)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples")
    parser.add_argument("--ref_idx", type=int, default=0, help="Dataset index for label")
    parser.add_argument("--label", type=str, default=None, help="Custom label: Energy,ux,uy,X,Y,Z")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID")
    parser.add_argument("--histogram", action="store_true", help="Save histograms")
    parser.add_argument("--cut_npe", type=float, default=0.0, help="nPE cut (hide <= value)")
    parser.add_argument("--cut_firsttime", type=float, default=0.0, help="FirstTime cut (hide <= value)")
    parser.add_argument("--cfg_scale", type=float, default=None, help="Override CFG scale from checkpoint (default: use checkpoint)")
    args = parser.parse_args()

    device = get_device(args.gpu)
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir.absolute()}")

    model, betas, alphas, alphas_cumprod, T, dataset, use_cfg, cfg_scale = load_model(args.checkpoint, device)
    if args.cfg_scale is not None:
        cfg_scale = args.cfg_scale
        print(f"Overriding cfg_scale to {cfg_scale}")

    scaler = None
    geo_np = None
    
    # 모델용 dataset: angle_conversion=True (기본값) -> label은 [Energy (PeV), ux, uy, X, Y, Z] 형식
    # 시각화용 dataset: angle_conversion=False -> label은 [Energy (MeV), Zenith (rad), Azimuth (rad), X, Y, Z] 형식
    h5_path = "./GENESIS-data/22644_0921_time_shift.h5"
    dataset_orig = H5Dataset(h5_path=h5_path, angle_conversion=False, energy_in_pev=False)
    
    if args.label:
        # 사용자 지정 label: [Energy, ux, uy, X, Y, Z] 형식 (모델 입력 형식)
        label_values = [float(x.strip()) for x in args.label.split(",")]
        if len(label_values) != 6:
            raise ValueError("Label must have 6 values: Energy,ux,uy,X,Y,Z")
        label = torch.tensor(label_values, device=device, dtype=torch.float32)
        label_np = label.detach().cpu().numpy()
        ref_idx = 0
        sig_ref_raw, geo_ref_raw, _ = dataset[0]
        geo_np = geo_ref_raw.detach().cpu().numpy()
        # 시각화용: 원본 label 가져오기 (Energy는 MeV, 각도는 라디안)
        _, _, label_orig_raw = dataset_orig[ref_idx]
        label_orig_np = label_orig_raw.detach().cpu().numpy()
    else:
        # 데이터셋에서 가져오기
        ref_idx = args.ref_idx
        sig_ref_raw, geo_ref_raw, label_ref_raw = dataset[ref_idx]
        # 모델용 label: [Energy (PeV), ux, uy, X, Y, Z] 형식 (angle_conversion=True)
        label = label_ref_raw.to(device)
        label_np = label_ref_raw.detach().cpu().numpy()
        geo_np = geo_ref_raw.detach().cpu().numpy()
        # 시각화용: 원본 label 가져오기 (Energy는 MeV, 각도는 라디안)
        _, _, label_orig_raw = dataset_orig[ref_idx]
        label_orig_np = label_orig_raw.detach().cpu().numpy()

    print(f"Using label (for model, [Energy(PeV), ux, uy, X, Y, Z]): {label_np}")
    print(f"Using original label (for visualization, [Energy(MeV), Zenith(rad), Azimuth(rad), X, Y, Z]): {label_orig_np}")

    null_label_norm = None
    if use_cfg and cfg_scale != 1.0:
        with torch.no_grad():
            dummy_sig = torch.zeros(1, 2, model.L, device=device)
            null_label_raw = torch.zeros(1, 6, device=device)
            _, null_label_norm = prepare_batch(dummy_sig, null_label_raw, verbose=False)

    if not args.label:
        print(f"\nPlotting actual data from index {ref_idx}...")
        sig_ref_clamp = _clamp_sig(sig_ref_raw.unsqueeze(0).to(device))
        sig_ref_denorm = sig_ref_clamp[0].detach().cpu().numpy()
        # 오리지널(actual)에는 cut 없음 (0, 0). cut은 샘플에만 적용.
        sig_actual_vis = _apply_cuts(sig_ref_denorm, 0.0, 0.0)
        actual_output_path = output_dir / f"actual_event_{ref_idx}.png"
        # 시각화: 원본 label 사용 (Energy는 MeV, 각도는 라디안)
        # show_event_dual_plot가 자동으로 형식을 감지하고 변환함
        show_event_dual_plot(
            sig=sig_actual_vis,
            geo=geo_np,
            label=label_orig_np,
            output_path=str(actual_output_path),
            figure_size=(18, 8),
            marker_size=8.0,
            show_detector_hull=True,
            show=False,
            title_prefix=f"sample_cfg.py | Actual | event {ref_idx}",
            firsttime_title="FirstTime (actual)",
            npe_title="nPE (actual)",
        )
        print(f"Actual data saved to: {actual_output_path}")
        if args.histogram:
            actual_hist_path = output_dir / f"actual_event_{ref_idx}_histogram.png"
            plot_histogram(sig_ref_denorm, actual_hist_path, title_suffix=" (Actual)", cut_npe=0.0, cut_firsttime=0.0)

    sample(
        model=model,
        label=label,
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        T=T,
        num_samples=args.num_samples,
        device=device,
        scaler=scaler,
        output_dir=output_dir,
        ref_idx=ref_idx,
        geo_np=geo_np,
        label_np=label_orig_np,  # 시각화용 원본 label 사용
        save_histogram=args.histogram,
        cut_npe=args.cut_npe,
        cut_firsttime=args.cut_firsttime,
        use_cfg=use_cfg,
        cfg_scale=cfg_scale,
        null_label_norm=null_label_norm,
    )
    print("Done!")


if __name__ == "__main__":
    main()
