#!/usr/bin/env python3
"""
Training script for GENESIS diffusion model.
Uses config YAML and project modules: dataloader, diffusion, utils.normalize,
utils.device, utils.vis, models.dit.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

from dataloader.h5 import H5Dataset
from diffusion.schedules import get_noise_schedule, compute_alpha_schedule
from diffusion.forward import apply_forward_diffusion
from models import DiffusionDiTTransformer
from utils.device import get_default_device
from utils.normalize import (
    normalize,
    denormalize_minmax,
    denormalize_log_minmax,
    apply_minmax_geo,
)
from utils.vis.event_show import show_event_dual_plot

# ---- Normalization defaults (train_exp behavior) ----
NPE_CLIP = 1000.0
FTIME_CLIP = 21000.0
LOG_MIN = 0.0
FEATURE_RANGE = (-1, 1)
LABEL_NAMES = ["Energy (PeV)", "ux", "uy", "X", "Y", "Z"]
LABEL_METHODS = ["log_minmax", "identity", "identity", "minmax", "minmax", "minmax"]
LABEL_FEATURE_RANGES = [FEATURE_RANGE] * 6
ENERGY_PEV_MINMAX = {"min": 1.0, "max": 100.0}
LABEL_XYZ_MINMAX = [
    {"min": -570.9000244140625, "max": 576.3699951171875},
    {"min": -521.0800170898438, "max": 509.5},
    {"min": -509.8599853515625, "max": 506.0566711425781},
]
GEO_XYZ_MINMAX = [
    {"min": -570.9000244140625, "max": 576.3699951171875},
    {"min": -521.0800170898438, "max": 509.5},
    {"min": -509.8599853515625, "max": 506.0566711425781},
]


# Module-level normalize config (same structure as train_exp.py)
_FTIME_LOG_MAX = float(np.log1p(FTIME_CLIP))
_ENERGY_LOG_MIN = float(np.log1p(ENERGY_PEV_MINMAX["min"]))
_ENERGY_LOG_MAX = float(np.log1p(ENERGY_PEV_MINMAX["max"]))
_CHANNEL_STATS = [
    {"min": 0.0, "max": NPE_CLIP},
    {"log_min": LOG_MIN, "log_max": _FTIME_LOG_MAX},
]
_LABEL_STATS = [
    {"log_min": _ENERGY_LOG_MIN, "log_max": _ENERGY_LOG_MAX},
    {},
    {},
    LABEL_XYZ_MINMAX[0],
    LABEL_XYZ_MINMAX[1],
    LABEL_XYZ_MINMAX[2],
]


@normalize(
    channel_methods=["minmax", "log_minmax"],
    feature_ranges=[FEATURE_RANGE, FEATURE_RANGE],
    channel_stats=_CHANNEL_STATS,
    arg_index=0,
    label_arg_index=1,
    label_methods=LABEL_METHODS,
    label_feature_ranges=LABEL_FEATURE_RANGES,
    label_stats=_LABEL_STATS,
)
def prepare_batch(sig: torch.Tensor, label: torch.Tensor, *, verbose: bool = False):
    if verbose:
        print("prepare_batch: label", label)
    return (sig, label)


def _print_label_normalize_config():
    print("label normalize (per column):")
    for j, name in enumerate(LABEL_NAMES):
        m = LABEL_METHODS[j]
        fr = LABEL_FEATURE_RANGES[j]
        st = _LABEL_STATS[j] if j < len(_LABEL_STATS) else {}
        if m == "identity":
            detail = "identity (no transform)"
        elif m == "log_minmax":
            detail = f"log_minmax -> {fr}  stats={st}"
        elif m == "minmax":
            detail = f"minmax -> {fr}  stats={st} (dataset min/max)" if st else f"minmax -> {fr}  stats={st}"
        else:
            detail = f"{m} -> {fr}  stats={st}"
        print(f"  [{j}] {name}: {detail}")


def denormalize_sig(sig: torch.Tensor) -> torch.Tensor:
    """정규화된 sig를 원 스케일로 역정규화."""
    out = sig.clone()
    if sig.dim() == 3:
        out[:, 0, :] = denormalize_minmax(sig[:, 0, :], 0.0, NPE_CLIP, FEATURE_RANGE)
        out[:, 1, :] = denormalize_log_minmax(
            sig[:, 1, :], LOG_MIN, float(np.log1p(FTIME_CLIP)), FEATURE_RANGE
        )
    else:
        out[0, :] = denormalize_minmax(sig[0, :], 0.0, NPE_CLIP, FEATURE_RANGE)
        out[1, :] = denormalize_log_minmax(
            sig[1, :], LOG_MIN, float(np.log1p(FTIME_CLIP)), FEATURE_RANGE
        )
    return out


def clamp_sig(sig: torch.Tensor) -> torch.Tensor:
    """Clamp npe/ftime before normalize."""
    s = sig.clone()
    if s.dim() == 3:
        s[:, 0, :] = torch.clamp(s[:, 0, :], min=0.0, max=NPE_CLIP)
        s[:, 1, :] = torch.clamp(s[:, 1, :], min=0.0, max=FTIME_CLIP)
    else:
        s[0, :] = torch.clamp(s[0, :], min=0.0, max=NPE_CLIP)
        s[1, :] = torch.clamp(s[1, :], min=0.0, max=FTIME_CLIP)
    return s


def sample_timesteps(batch: int, T: int, device: torch.device) -> torch.Tensor:
    """t in [1, T] (t=0 제외)."""
    return torch.randint(low=1, high=T + 1, size=(batch,), device=device, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Device: use config if requested and available, else fall back to best available
    requested = config.get("device")
    if requested:
        try:
            dev = torch.device(requested)
            if dev.type == "cuda" and not torch.cuda.is_available():
                raise AssertionError("CUDA not available")
            if dev.type == "mps" and (not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available()):
                raise AssertionError("MPS not available")
            device = dev
        except (AssertionError, RuntimeError) as e:
            device = get_default_device()
            print(f"Requested device '{requested}' not available ({e}), using: {device}")
    else:
        device = get_default_device()
    print("device:", device)

    # Output dir
    output_dir = Path(config.get("path", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output directory:", output_dir.absolute())

    # Data
    data_cfg = config["data"]
    loader_type = data_cfg.get("loader", "h5")
    if loader_type in ["h5", "hdf5"]:
        dataset = H5Dataset(
            h5_path=data_cfg["h5_path"],
            angle_conversion=data_cfg.get("angle_conversion", False),
            num_workers=data_cfg.get("num_workers"),
            shuffle=data_cfg.get("shuffle"),
        )
    else:
        raise ValueError(f"Unsupported loader: {loader_type}")

    batch_size = data_cfg.get("bsz", 256)
    num_workers = data_cfg.get("num_workers", 0) or 0
    shuffle = dataset.shuffle if dataset.shuffle is not None else data_cfg.get("shuffle", True)
    pin_memory = data_cfg.get("pin_memory", device.type == "cuda")

    # train_exp.py와 동일: multiprocessing_context 미설정 → Linux 기본 fork 사용 (데이터셋 pickle 없음)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    print("dataset length:", len(dataset))
    _print_label_normalize_config()

    # Diffusion schedule
    diff_cfg = config.get("diffusion", {})
    schedule_cfg = diff_cfg.get("schedule", {})
    schedule_type = schedule_cfg.get("type", "sigmoid")
    timesteps = schedule_cfg.get("timesteps", 1000)
    beta_start = schedule_cfg.get("beta_start", 1e-4)
    beta_end = schedule_cfg.get("beta_end", 2e-2)

    betas = get_noise_schedule(
        schedule_type,
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        **{k: v for k, v in schedule_cfg.items() if k not in ("type", "timesteps", "beta_start", "beta_end")}
    ).to(device)
    alpha_schedule = compute_alpha_schedule(betas)
    alphas = alpha_schedule["alphas"]
    alphas_cumprod = alpha_schedule["alphas_cumprod"]
    T = timesteps
    print("schedule:", schedule_type, "timesteps:", T, "betas range:", betas.min().item(), betas.max().item())

    geo_min = np.array([GEO_XYZ_MINMAX[j]["min"] for j in range(3)], dtype=np.float32)
    geo_max = np.array([GEO_XYZ_MINMAX[j]["max"] for j in range(3)], dtype=np.float32)

    # Model (DiT)
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "dit")
    if model_type != "dit":
        raise ValueError(f"Only model type 'dit' is supported; got '{model_type}'")
    opts = model_cfg.get("options", {})
    d_model = opts.get("d_model", 256)
    nhead = opts.get("nhead", 8)
    depth = opts.get("depth", 6)
    mlp_ratio = opts.get("mlp_ratio", 4.0)
    dropout = opts.get("dropout", 0.0)
    label_dim = opts.get("label_dim", 6)

    # train_exp.py와 동일: dataset[0][1]로 geo 사용 (fork 시 pickle 없음)
    geo_raw = dataset[0][1]
    geo_norm = apply_minmax_geo(geo_raw, geo_min, geo_max, feature_range=(0, 1))
    use_checkpointing = opts.get("use_checkpointing", True)
    model = DiffusionDiTTransformer(
        geo=geo_norm,
        d_model=d_model,
        nhead=nhead,
        depth=depth,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        label_dim=label_dim,
        use_checkpointing=use_checkpointing,
    ).to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"[GPU] after model: {torch.cuda.memory_allocated() / 1e9:.3f} GB")

    # Optimizer
    train_cfg = config.get("training", {})
    lr = train_cfg.get("lr", 3e-4)
    num_epochs = train_cfg.get("nepochs", 20)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    # AMP
    try:
        scaler = GradScaler(device.type) if device.type in ("cuda", "mps") else None
    except (TypeError, ValueError):
        scaler = GradScaler() if device.type == "cuda" else None
    print("AMP enabled:", scaler is not None)

    # Optional torch.compile
    if config.get("compile_model", False) and hasattr(torch, "compile"):
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.suppress_errors = True
            print("Compiling model with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compilation successful!")
        except Exception as e:
            print("Model compilation failed (continuing without compile):", e)
    print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    # Training loop
    model.train()
    loss_hist = []
    steps_per_epoch = len(loader)
    total_steps = num_epochs * steps_per_epoch
    best_loss = float("inf")
    best_checkpoint_path = None
    print_every = config.get("print_every", 50)

    print(f"Training for {num_epochs} epochs ({steps_per_epoch} batches/epoch, {total_steps} total steps)")
    if device.type == "cuda" and batch_size >= 256:
        print("(If GPU OOM, reduce data.bsz in config, e.g. 128 or 64)")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        epoch_losses = []
        pbar = tqdm(enumerate(loader, 1), total=steps_per_epoch, desc=f"Epoch {epoch}/{num_epochs}")

        for batch_idx, (sig, geo, label) in pbar:
            sig = sig.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            sig_clamp = clamp_sig(sig)
            x0, label_norm = prepare_batch(sig_clamp, label, verbose=(epoch == 1 and batch_idx == 1))

            if device.type == "cuda" and epoch == 1 and batch_idx == 1:
                print(f"[GPU] after first batch (before step): {torch.cuda.memory_allocated() / 1e9:.3f} GB")

            B = x0.shape[0]
            t = sample_timesteps(B, T, device)
            noise = torch.randn_like(x0)
            x_t = apply_forward_diffusion(x0=x0, betas=betas, timesteps=t, noise=noise)

            with autocast(device.type, enabled=(scaler is not None)):
                eps_hat = model(x_t, t, label_norm)
                loss = F.mse_loss(eps_hat, noise)

            optim.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            loss_val = float(loss.item())
            if device.type == "cuda" and epoch == 1 and batch_idx == 1:
                print(f"[GPU] after first step: {torch.cuda.memory_allocated() / 1e9:.3f} GB (peak: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB)")
            loss_hist.append(loss_val)
            epoch_losses.append(loss_val)
            current_step = (epoch - 1) * steps_per_epoch + batch_idx

            if loss_val < best_loss:
                best_loss = loss_val
                if best_checkpoint_path is not None and best_checkpoint_path.exists():
                    best_checkpoint_path.unlink()
                best_checkpoint_path = output_dir / (
                    f"best_checkpoint_epoch_{epoch:03d}_batch_{batch_idx:05d}_step_{current_step:05d}_loss_{best_loss:.6f}.pt"
                )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "step": current_step,
                        "loss": loss_val,
                        "best_loss": best_loss,
                        "betas": betas.cpu(),
                        "alphas": alphas.cpu(),
                        "alphas_cumprod": alphas_cumprod.cpu(),
                        "T": T,
                        "beta_start": beta_start,
                        "beta_end": beta_end,
                        "d_model": model.d_model,
                        "nhead": nhead,
                        "depth": depth,
                        "mlp_ratio": mlp_ratio,
                        "label_dim": label_dim,
                    },
                    best_checkpoint_path,
                )

            pbar.set_postfix(
                loss=f"{np.mean(epoch_losses):.6f}",
                step=current_step,
                best=f"{best_loss:.6f}",
            )

        epoch_avg = np.mean(epoch_losses)
        print(f"\nepoch {epoch:3d}/{num_epochs} completed | avg loss: {epoch_avg:.6f} | best loss: {best_loss:.6f}")
        if best_checkpoint_path:
            print("Best model saved:", best_checkpoint_path.name)
        print("-" * 60)

    print("Training done!")

    # Quick visualization: one event, different t
    model.eval()
    event_idx = 0
    sig_raw, geo_raw, label_raw = dataset[event_idx]
    geo_np = geo_raw.detach().cpu().numpy()
    label_np = label_raw.detach().cpu().numpy()
    sig = sig_raw.unsqueeze(0).to(device)
    label = label_raw.unsqueeze(0).to(device)
    sig_clamp = clamp_sig(sig)
    x0, _ = prepare_batch(sig_clamp, label, verbose=False)

    for t_val in [0, 250, 500, 750, 1000]:
        t = torch.tensor([t_val], device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = apply_forward_diffusion(x0=x0, betas=betas, timesteps=t, noise=noise)
        x_t_denorm = denormalize_sig(x_t)[0].detach().cpu().numpy()
        out_path = output_dir / f"event_{event_idx}_t_{t_val}.png"
        show_event_dual_plot(
            sig=x_t_denorm,
            geo=geo_np,
            label=label_np,
            output_path=str(out_path),
            figure_size=(18, 8),
            marker_size=8.0,
            show_detector_hull=True,
            show=False,
            title_prefix=f"train.py | {schedule_type} | event {event_idx} | t={t_val}",
            firsttime_title="FirstTime (x_t, denorm)",
            npe_title="nPE (x_t, denorm)",
        )

    # Sampling (DDPM reverse)
    print("\n" + "=" * 60)
    print("Sampling from trained model...")
    print("=" * 60)
    model.eval()
    ref_idx = 0
    with torch.no_grad():
        sig_ref_raw, geo_ref_raw, label_ref_raw = dataset[ref_idx]
        sig_ref_clamp = clamp_sig(sig_ref_raw.unsqueeze(0).to(device))
        sig_ref_denorm = sig_ref_clamp[0].detach().cpu().numpy()
        geo_ref_np = geo_ref_raw.detach().cpu().numpy()
        label_ref_np = label_ref_raw.detach().cpu().numpy()
        actual_path = output_dir / f"actual_event_{ref_idx}.png"
        show_event_dual_plot(
            sig=sig_ref_denorm,
            geo=geo_ref_np,
            label=label_ref_np,
            output_path=str(actual_path),
            figure_size=(18, 8),
            marker_size=8.0,
            show_detector_hull=True,
            show=False,
            title_prefix=f"train.py | Actual data | event {ref_idx}",
            firsttime_title="FirstTime (actual)",
            npe_title="nPE (actual)",
        )
        _, label_ref_norm = prepare_batch(sig_ref_clamp, label_ref_raw.unsqueeze(0).to(device), verbose=False)
        B = 1
        x = torch.randn(B, 2, model.L, device=device)
        for t_val in reversed(range(1, T + 1)):
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            with autocast(device.type, enabled=(scaler is not None)):
                eps_hat = model(x, t_batch, label_ref_norm)
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
        samples_denorm = denormalize_sig(x)
        sample_np = samples_denorm[0].detach().cpu().numpy()
        sampled_path = output_dir / f"sampled_event_{ref_idx}.png"
        show_event_dual_plot(
            sig=sample_np,
            geo=geo_ref_np,
            label=label_ref_np,
            output_path=str(sampled_path),
            figure_size=(18, 8),
            marker_size=8.0,
            show_detector_hull=True,
            show=False,
            title_prefix=f"train.py | Sampled | event {ref_idx}",
            firsttime_title="FirstTime (sampled)",
            npe_title="nPE (sampled)",
        )
    print("Sampling completed!")

    # Save final checkpoint
    print("\n" + "=" * 60)
    print("Saving final model...")
    print("=" * 60)
    final_path = output_dir / "model_checkpoint_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "epoch": num_epochs,
            "final_loss": loss_hist[-1] if loss_hist else None,
            "best_loss": best_loss,
            "betas": betas.cpu(),
            "alphas": alphas.cpu(),
            "alphas_cumprod": alphas_cumprod.cpu(),
            "T": T,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "d_model": model.d_model,
            "nhead": nhead,
            "depth": depth,
            "mlp_ratio": mlp_ratio,
            "label_dim": label_dim,
        },
        final_path,
    )
    print("Final model saved to:", final_path)
    if best_checkpoint_path:
        print("Best model (loss={:.6f}):".format(best_loss), best_checkpoint_path.name)
    print("Done!")


if __name__ == "__main__":
    main()
