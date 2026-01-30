#!/usr/bin/env python3
"""
Training script for GENESIS diffusion model with:
- YAML config
- Seed for reproducibility
- Train/val split and validation loss
- Classifier-Free Guidance (CFG): label dropout at train, guidance scale at sampling
Reuses: diffusion.schedules, diffusion.forward, utils.normalize, utils.vis, models.dit, dataloader.h5
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, random_split

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from tqdm import tqdm
import yaml

# Add project root to path (same as train_exp)
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from dataloader.h5 import H5Dataset
from diffusion.schedules import get_noise_schedule, compute_alpha_schedule
from diffusion.forward import apply_forward_diffusion
from utils.normalize import normalize, denormalize_log_minmax, denormalize_minmax, apply_minmax_geo
from utils.vis.event_show import show_event_dual_plot
from utils.device import get_default_device
from models import DiffusionDiTTransformer


def set_seed(seed: int):
    """재현성을 위한 시드 설정."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cuDNN 비결정론 비활성화 (선택)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---- 정규화 상수 (train_exp.py와 동일) ----
npe_clip = 1000.0
ftime_clip = 21000.0
log_min = 0.0
npe_log_max = float(np.log1p(npe_clip))
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


def denormalize_sig(sig: torch.Tensor) -> torch.Tensor:
    out = sig.clone()
    if sig.dim() == 3:
        out[:, 0, :] = denormalize_minmax(sig[:, 0, :], 0.0, npe_clip, _feature_range)
        out[:, 1, :] = denormalize_log_minmax(sig[:, 1, :], log_min, ftime_log_max, _feature_range)
    else:
        out[0, :] = denormalize_minmax(sig[0, :], 0.0, npe_clip, _feature_range)
        out[1, :] = denormalize_log_minmax(sig[1, :], log_min, ftime_log_max, _feature_range)
    return out


def _clamp_sig(sig: torch.Tensor) -> torch.Tensor:
    s = sig.clone()
    s[:, 0] = torch.clamp(s[:, 0], min=0.0, max=npe_clip)
    s[:, 1] = torch.clamp(s[:, 1], min=0.0, max=ftime_clip)
    return s


def sample_timesteps(batch: int, T: int, device: torch.device) -> torch.Tensor:
    return torch.randint(low=1, high=T + 1, size=(batch,), device=device, dtype=torch.long)


def _compute_val_loss(model, val_loader, betas, T, device, prepare_batch, _clamp_sig, scaler):
    """Validation loss (no grad, no CFG dropout for eval)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for sig, geo, label in val_loader:
            sig = sig.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            sig_clamp = _clamp_sig(sig)
            x0, label_norm = prepare_batch(sig_clamp, label, verbose=False)
            B = x0.shape[0]
            t = sample_timesteps(B, T, device)
            noise = torch.randn_like(x0)
            x_t = apply_forward_diffusion(x0=x0, betas=betas, timesteps=t, noise=noise)
            with autocast(device.type, enabled=(scaler is not None)):
                eps_hat = model(x_t, t, label_norm)
                loss = F.mse_loss(eps_hat, noise)
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def main():
    # Explicitly import torch at function start to avoid UnboundLocalError
    import torch
    
    parser = argparse.ArgumentParser(description="Train GENESIS with CFG, val loss, seed, YAML config")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Seed: {seed}")

    device = config.get("device")
    if device:
        try:
            dev = torch.device(device)
            if dev.type == "cuda" and not torch.cuda.is_available():
                dev = get_default_device()
            elif dev.type == "mps" and (not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available()):
                dev = get_default_device()
        except Exception:
            dev = get_default_device()
    else:
        dev = get_default_device()
    device = dev
    print("device:", device)

    output_dir = Path(config.get("path", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Output directory:", output_dir.absolute())

    # Data
    data_cfg = config.get("data", {})
    h5_path = data_cfg.get("h5_path", "./GENESIS-data/22644_0921_time_shift.h5")
    dataset = H5Dataset(
        h5_path=h5_path,
        angle_conversion=data_cfg.get("angle_conversion", True),
    )
    n_total = len(dataset)
    val_ratio = config.get("val_ratio", 0.05)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    # Use __import__ to avoid UnboundLocalError if 'torch' is shadowed elsewhere in main()
    gen = __import__("torch").Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    print(f"Train size: {n_train}, Val size: {n_val}")

    train_cfg = config.get("training", {})
    bsz = train_cfg.get("bsz", 256)
    num_workers = data_cfg.get("num_workers", 0) or 0
    pin_memory = data_cfg.get("pin_memory", device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=bsz,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bsz,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # Diffusion schedule
    diff_cfg = config.get("diffusion", {})
    sched_cfg = diff_cfg.get("schedule", {})
    schedule_type = sched_cfg.get("type", "sigmoid")
    timesteps = sched_cfg.get("timesteps", 1000)
    beta_start = sched_cfg.get("beta_start", 1e-4)
    beta_end = sched_cfg.get("beta_end", 2e-2)
    betas = get_noise_schedule(
        schedule_type,
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        **{k: v for k, v in sched_cfg.items() if k not in ("type", "timesteps", "beta_start", "beta_end")},
    ).to(device)
    alpha_schedule = compute_alpha_schedule(betas)
    alphas = alpha_schedule["alphas"]
    alphas_cumprod = alpha_schedule["alphas_cumprod"]
    T = timesteps
    print("schedule:", schedule_type, "T:", T)

    # Model (geo from first sample)
    geo_raw = dataset[0][1]
    geo_norm = apply_minmax_geo(geo_raw, geo_min, geo_max, feature_range=(0, 1))
    model_cfg = config.get("model", {})
    opts = model_cfg.get("options", {})
    d_model = opts.get("d_model", 256)
    nhead = opts.get("nhead", 8)
    depth = opts.get("depth", 6)
    mlp_ratio = opts.get("mlp_ratio", 4.0)
    dropout = opts.get("dropout", 0.0)
    label_dim = opts.get("label_dim", 6)
    model = DiffusionDiTTransformer(
        geo=geo_norm,
        d_model=d_model,
        nhead=nhead,
        depth=depth,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        label_dim=label_dim,
    ).to(device)

    # Null label for CFG (unconditional)
    cfg_cfg = config.get("cfg", {})
    use_cfg = cfg_cfg.get("use_cfg", False)
    cfg_scale = cfg_cfg.get("cfg_scale", 2.0)
    cfg_dropout = cfg_cfg.get("cfg_dropout", 0.1)
    null_label_norm = None
    if use_cfg:
        with torch.no_grad():
            dummy_sig = torch.zeros(1, 2, model.L, device=device)
            null_label_raw = torch.zeros(1, 6, device=device)
            _, null_label_norm = prepare_batch(dummy_sig, null_label_raw, verbose=False)
        print(f"CFG: use_cfg=True, cfg_scale={cfg_scale}, cfg_dropout={cfg_dropout}")
    else:
        print("CFG: use_cfg=False")

    lr = train_cfg.get("lr", 3e-4)
    num_epochs = train_cfg.get("nepochs", 20)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    try:
        scaler = GradScaler(device.type) if device.type in ("cuda", "mps") else None
    except (TypeError, ValueError):
        scaler = GradScaler() if device.type == "cuda" else None
    print("AMP enabled:", scaler is not None)

    if config.get("compile_model", False) and hasattr(torch, "compile"):
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled.")
        except Exception as e:
            print("Compile failed:", e)
    print("params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    val_every = config.get("val_every", 1)
    steps_per_epoch = len(train_loader)
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_checkpoint_path = None
    loss_hist = []

    print(f"Training {num_epochs} epochs, {steps_per_epoch} batches/epoch")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        epoch_losses = []
        pbar = tqdm(enumerate(train_loader, 1), total=steps_per_epoch, desc=f"Epoch {epoch}/{num_epochs}")

        for batch_idx, (sig, geo, label) in pbar:
            sig = sig.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            sig_clamp = _clamp_sig(sig)
            x0, label_norm = prepare_batch(sig_clamp, label, verbose=(epoch == 1 and batch_idx == 1))

            # CFG: with prob cfg_dropout use null label (unconditional)
            if use_cfg and cfg_dropout > 0 and null_label_norm is not None:
                B = label_norm.shape[0]
                mask = torch.rand(B, 1, device=device) < cfg_dropout
                label_norm = torch.where(mask, null_label_norm.expand(B, -1), label_norm)

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
            loss_hist.append(loss_val)
            epoch_losses.append(loss_val)
            current_step = (epoch - 1) * steps_per_epoch + batch_idx

            # Save best by train loss only when we have no val (otherwise best is by val_loss)
            if n_val == 0 and loss_val < best_train_loss:
                best_train_loss = loss_val
                if best_checkpoint_path is not None and best_checkpoint_path.exists():
                    best_checkpoint_path.unlink()
                best_checkpoint_path = output_dir / (
                    f"best_checkpoint_epoch_{epoch:03d}_batch_{batch_idx:05d}_step_{current_step:05d}_loss_{best_train_loss:.6f}.pt"
                )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "epoch": epoch,
                        "step": current_step,
                        "loss": loss_val,
                        "best_loss": best_train_loss,
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
                        "use_cfg": use_cfg,
                        "cfg_scale": cfg_scale,
                    },
                    best_checkpoint_path,
                )

            pbar.set_postfix(loss=f"{np.mean(epoch_losses):.6f}", best=f"{best_val_loss if n_val else best_train_loss:.6f}")

        epoch_avg = np.mean(epoch_losses)
        # Validation loss
        if val_every > 0 and epoch % val_every == 0 and n_val > 0:
            val_loss = _compute_val_loss(
                model, val_loader, betas, T, device, prepare_batch, _clamp_sig, scaler
            )
            print(f"\nepoch {epoch:3d}/{num_epochs} | train_loss: {epoch_avg:.6f} | val_loss: {val_loss:.6f} | best_val: {best_val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if best_checkpoint_path and best_checkpoint_path.exists():
                    best_checkpoint_path.unlink()
                best_checkpoint_path = output_dir / (
                    f"best_checkpoint_epoch_{epoch:03d}_val_loss_{best_val_loss:.6f}.pt"
                )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "best_loss": best_val_loss,
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
                        "use_cfg": use_cfg,
                        "cfg_scale": cfg_scale,
                    },
                    best_checkpoint_path,
                )
                print(f"  Best model (by val_loss) saved: {best_checkpoint_path.name}")
        else:
            best_display = best_val_loss if n_val > 0 else best_train_loss
            print(f"\nepoch {epoch:3d}/{num_epochs} | train_loss: {epoch_avg:.6f} | best: {best_display:.6f}")
        if best_checkpoint_path:
            print("  Best checkpoint:", best_checkpoint_path.name)
        print("-" * 60)

    print("Training done!")

    # Sampling with CFG
    model.eval()
    ref_idx = 0
    with torch.no_grad():
        sig_ref_raw, geo_ref_raw, label_ref_raw = dataset[ref_idx]
        sig_ref_clamp = _clamp_sig(sig_ref_raw.unsqueeze(0).to(device))
        sig_ref_denorm = sig_ref_clamp[0].detach().cpu().numpy()
        geo_ref_np = geo_ref_raw.detach().cpu().numpy()
        label_ref_np = label_ref_raw.detach().cpu().numpy()
        label_ref = label_ref_raw.unsqueeze(0).to(device)
        _, label_ref_norm = prepare_batch(sig_ref_clamp, label_ref, verbose=False)

        num_samples = 1
        B = num_samples
        x = torch.randn(B, 2, model.L, device=device)
        print("Sampling (CFG scale=%s)..." % (cfg_scale if use_cfg else "N/A"))

        for t_val in reversed(range(1, T + 1)):
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            with autocast(device.type, enabled=(scaler is not None)):
                if use_cfg and cfg_scale != 1.0 and null_label_norm is not None:
                    eps_uncond = model(x, t_batch, null_label_norm.expand(B, -1))
                    eps_cond = model(x, t_batch, label_ref_norm.expand(B, -1))
                    eps_hat = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
                else:
                    eps_hat = model(x, t_batch, label_ref_norm.expand(B, -1))

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

        sample_np = denormalize_sig(x)[0].detach().cpu().numpy()
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
            title_prefix=f"train_exp_cfg | CFG={cfg_scale} | event {ref_idx}",
            firsttime_title="FirstTime (sampled)",
            npe_title="nPE (sampled)",
        )
        print("Sampled saved:", sampled_path)

    model_save_path = output_dir / "model_checkpoint_final.pt"
    best_final = best_val_loss if n_val > 0 else best_train_loss
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "epoch": num_epochs,
            "best_loss": best_final,
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
            "use_cfg": use_cfg,
            "cfg_scale": cfg_scale,
        },
        model_save_path,
    )
    print("Final model saved:", model_save_path)
    if best_checkpoint_path:
        print("Best checkpoint:", best_checkpoint_path.name)
    print("Done!")


if __name__ == "__main__":
    main()
