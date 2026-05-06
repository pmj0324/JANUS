#!/usr/bin/env python3
"""Likelihood estimation for jointzero rectified-flow checkpoints.

Computes conditional log-likelihood in normalized signal space via change-of-variables:
  x_0 --(dx/dt = v_theta)--> x_1,  t: 0 -> 1
  log p(x_0 | c) = log p_1(x_1) + ∫_0^1 div_x v_theta(x_t, t, c) dt

The divergence term is estimated with Hutchinson's trace estimator.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

import train_exp_rectified_flow_0413_jointzero as m
from utils.io.h5_rank_npe import compute_npe_stats_fast, topk_rows_from_stats


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


def _select_top_rows_on_indices(stats: dict, candidate_indices: np.ndarray, top_k: int, sort_mode: str) -> list[dict]:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if candidate_indices.size == 0:
        return []

    active = stats["active_npe_count"][candidate_indices]
    npe_sum = stats["npe_sum"][candidate_indices]
    npe_max = stats["npe_max"][candidate_indices]
    npe_mean = stats["npe_mean_active"][candidate_indices]

    if sort_mode == "active_then_sum":
        order_local = np.lexsort((-npe_sum, -active))
    elif sort_mode == "sum_then_active":
        order_local = np.lexsort((-npe_max, -active, -npe_sum))
    else:
        raise ValueError(f"Unknown sort_mode: {sort_mode}")

    take = order_local[: min(max(1, int(top_k)), order_local.size)]
    chosen = candidate_indices[take]
    rows = []
    for idx in chosen.tolist():
        rows.append(
            {
                "ref_idx": int(idx),
                "active_npe_count": int(stats["active_npe_count"][idx]),
                "npe_sum": float(stats["npe_sum"][idx]),
                "npe_max": float(stats["npe_max"][idx]),
                "npe_mean_active": float(stats["npe_mean_active"][idx]),
            }
        )
    return rows


def _parse_ref_indices(raw: str | None) -> list[int]:
    if raw is None or not raw.strip():
        return []
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def _hutchinson_divergence(
    model: torch.nn.Module,
    x: torch.Tensor,
    t_batch: torch.Tensor,
    label_norm: torch.Tensor,
    num_samples: int,
    noise_type: str,
) -> torch.Tensor:
    """
    Estimate divergence div_x v(x,t,c) per batch item using Hutchinson.
    Returns shape (B,).
    """
    B = x.shape[0]
    div = torch.zeros(B, device=x.device, dtype=x.dtype)

    # x must require grad for divergence wrt x
    x_req = x.detach().requires_grad_(True)
    v = model(x_req, t_batch, label_norm)

    for i in range(max(1, int(num_samples))):
        if noise_type == "rademacher":
            eps = torch.randint_like(x_req, low=0, high=2, dtype=torch.int64).to(x_req.dtype)
            eps = eps * 2.0 - 1.0
        elif noise_type == "gaussian":
            eps = torch.randn_like(x_req)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        proj = (v * eps).sum()
        grad = torch.autograd.grad(
            proj,
            x_req,
            create_graph=False,
            retain_graph=(i < max(1, int(num_samples)) - 1),
            only_inputs=True,
        )[0]
        div = div + (grad * eps).sum(dim=(1, 2))

    div = div / float(max(1, int(num_samples)))
    return div


def estimate_log_likelihood(
    model: torch.nn.Module,
    x0_norm: torch.Tensor,
    label_norm: torch.Tensor,
    *,
    num_steps: int,
    hutchinson_samples: int,
    noise_type: str,
) -> tuple[float, float, float]:
    """
    Returns (logp_x0, logp_prior_x1, integral_div).
    """
    device = x0_norm.device
    dtype = x0_norm.dtype

    dt = 1.0 / float(max(1, int(num_steps)))
    x = x0_norm.detach()
    int_div = torch.zeros(x.shape[0], device=device, dtype=dtype)

    for k in range(max(1, int(num_steps))):
        t_val = (k + 0.5) * dt  # midpoint for better stability
        t_batch = torch.full((x.shape[0],), t_val, device=device, dtype=dtype)

        # divergence estimate at current x
        div = _hutchinson_divergence(
            model,
            x,
            t_batch,
            label_norm,
            num_samples=hutchinson_samples,
            noise_type=noise_type,
        )

        # Euler step for forward dynamics x_{t+dt} = x_t + dt * v(x_t,t)
        with torch.no_grad():
            v = model(x, t_batch, label_norm)
            x = x + dt * v
            int_div = int_div + dt * div

    # Prior at t=1 in normalized space: standard Gaussian
    D = x.shape[1] * x.shape[2]
    logp_prior = -0.5 * (x.pow(2).sum(dim=(1, 2)) + D * math.log(2.0 * math.pi))
    logp_x0 = logp_prior + int_div

    return float(logp_x0.item()), float(logp_prior.item()), float(int_div.item())


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate conditional log-likelihood for rectified-flow jointzero checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--h5_path", type=str, default=None)
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--subset", type=str, choices=["all", "train", "val"], default="val")
    p.add_argument("--val_ratio", type=float, default=float(m.val_ratio))
    p.add_argument("--split_seed", type=int, default=int(m.seed))
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--sort_mode", type=str, choices=["active_then_sum", "sum_then_active"], default="active_then_sum")
    p.add_argument("--ref_indices", type=str, default=None, help="Comma-separated ref indices; overrides top_k selection.")
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--hutchinson_samples", type=int, default=1)
    p.add_argument("--noise_type", type=str, choices=["rademacher", "gaussian"], default="rademacher")
    p.add_argument("--chunk_events", type=int, default=4096)
    p.add_argument("--no_stats_cache", action="store_true")
    p.add_argument("--no_clamp", action="store_true", help="Disable clamp before normalization (default: clamp ON)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else (
        ckpt_path.parent.parent / "likelihood_results.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

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
    h5_path = args.h5_path if args.h5_path is not None else m.h5_path

    dataset = m.H5Dataset(
        h5_path=h5_path,
        angle_conversion=m.data_angle_conversion,
        num_workers=0,
        shuffle=False,
    )

    # Build model exactly as training checkpoint expects
    _, geo0, _ = dataset[0]
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

    explicit_refs = _parse_ref_indices(args.ref_indices)
    if explicit_refs:
        target_rows = [{"ref_idx": int(i)} for i in explicit_refs]
    else:
        stats_cache = out_csv.parent / "_cache_npe_stats_likelihood.npz"
        stats = compute_npe_stats_fast(
            h5_path=h5_path,
            signal_key="input",
            chunk_events=args.chunk_events,
            cache_npz=stats_cache,
            use_cache=(not args.no_stats_cache),
        )
        if args.subset == "all":
            target_rows = topk_rows_from_stats(stats, top_k=max(1, int(args.top_k)), sort_mode=args.sort_mode)
        else:
            cand = _build_split_indices(dataset, args.subset, args.val_ratio, args.split_seed)
            target_rows = _select_top_rows_on_indices(stats, cand, top_k=max(1, int(args.top_k)), sort_mode=args.sort_mode)

    fields = [
        "ref_idx",
        "logp_x0",
        "logp_prior_x1",
        "int_div",
        "bits_per_dim",
        "num_steps",
        "hutchinson_samples",
        "noise_type",
        "subset",
        "clamp_used",
        "npe_sum",
        "active_npe_count",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for i, row in enumerate(target_rows, 1):
            ref_idx = int(row["ref_idx"])
            sig_raw, _, label_raw = dataset[ref_idx]

            sig = sig_raw.unsqueeze(0).to(device, non_blocking=True)
            label = label_raw.unsqueeze(0).to(device, non_blocking=True)
            sig_in = m._clamp_sig(sig) if (not args.no_clamp) else sig
            x0_norm, label_norm = m.prepare_batch(sig_in, label, verbose=False)

            logp_x0, logp_prior, int_div = estimate_log_likelihood(
                model,
                x0_norm,
                label_norm,
                num_steps=args.num_steps,
                hutchinson_samples=args.hutchinson_samples,
                noise_type=args.noise_type,
            )
            D = x0_norm.shape[1] * x0_norm.shape[2]
            bits_per_dim = -logp_x0 / (math.log(2.0) * D)

            npe = np.asarray(sig_raw[0], dtype=np.float32)
            finite = np.isfinite(npe)
            active = finite & (npe > 0.0)
            npe_sum = float(np.sum(npe[active])) if active.any() else 0.0
            active_count = int(active.sum())

            rec = {
                "ref_idx": ref_idx,
                "logp_x0": f"{logp_x0:.8f}",
                "logp_prior_x1": f"{logp_prior:.8f}",
                "int_div": f"{int_div:.8f}",
                "bits_per_dim": f"{bits_per_dim:.10f}",
                "num_steps": int(args.num_steps),
                "hutchinson_samples": int(args.hutchinson_samples),
                "noise_type": args.noise_type,
                "subset": args.subset,
                "clamp_used": (not args.no_clamp),
                "npe_sum": f"{npe_sum:.6f}",
                "active_npe_count": active_count,
            }
            w.writerow(rec)
            print(
                f"[{i}/{len(target_rows)}] ref={ref_idx} logp={logp_x0:.3f} bpd={bits_per_dim:.5f} "
                f"(npe_sum={npe_sum:.1f}, active={active_count})"
            )

    print("done")
    print(f"device={device}")
    print(f"checkpoint={ckpt_path}")
    print(f"out_csv={out_csv}")
    print(f"targets={len(target_rows)}")


if __name__ == "__main__":
    main()

