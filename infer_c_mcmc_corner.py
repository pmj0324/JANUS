#!/usr/bin/env python3
"""Infer c=(E,ux,uy) with MCMC and save corner plot.

Assumptions:
- xyz are fixed to the observed event's true values.
- Posterior is proportional to p(x | c) with optional simple priors.
- log p(x|c) is computed via change-of-variables likelihood in normalized space.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from tqdm import trange
except Exception:  # pragma: no cover
    trange = None
try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None

# Ensure local imports work regardless of launch cwd.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_exp_rectified_flow_0413_jointzero as m
from likelihood_rectified_flow_jointzero import estimate_log_likelihood


def _build_model(checkpoint: dict, dataset, device: torch.device) -> torch.nn.Module:
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
    return model


def _log_prior(theta: np.ndarray, e_min: float, e_max: float) -> float:
    """Uniform prior in ranges + disk constraint ux^2+uy^2<=1."""
    E, ux, uy = float(theta[0]), float(theta[1]), float(theta[2])
    if not (e_min <= E <= e_max):
        return -np.inf
    if not (-1.0 <= ux <= 1.0 and -1.0 <= uy <= 1.0):
        return -np.inf
    if ux * ux + uy * uy > 1.0:
        return -np.inf
    return 0.0


def _make_label_with_fixed_xyz(theta: np.ndarray, xyz_fixed: np.ndarray, device: torch.device) -> torch.Tensor:
    E, ux, uy = [float(v) for v in theta]
    X, Y, Z = [float(v) for v in xyz_fixed]
    return torch.tensor([[E, ux, uy, X, Y, Z]], dtype=torch.float32, device=device)


def _minmax_to_range(x: torch.Tensor, data_min: float, data_max: float, feature_range: tuple[float, float]) -> torch.Tensor:
    fr_min, fr_max = float(feature_range[0]), float(feature_range[1])
    denom = max(float(data_max) - float(data_min), 1e-12)
    return ((x - float(data_min)) / denom) * (fr_max - fr_min) + fr_min


def _normalize_label_only(label: torch.Tensor) -> torch.Tensor:
    """Fast label-only normalization matching train_exp_rectified_flow_0413_jointzero.py."""
    out = label.clone()
    out[:, 0] = _minmax_to_range(
        torch.log1p(out[:, 0]),
        float(m.energy_log_min),
        float(np.log1p(m._ENERGY_PEV_MINMAX["max"])),
        m._feature_range,
    )
    # ux, uy are identity
    out[:, 3] = _minmax_to_range(out[:, 3], m._LABEL_XYZ_MINMAX[0]["min"], m._LABEL_XYZ_MINMAX[0]["max"], m._feature_range)
    out[:, 4] = _minmax_to_range(out[:, 4], m._LABEL_XYZ_MINMAX[1]["min"], m._LABEL_XYZ_MINMAX[1]["max"], m._feature_range)
    out[:, 5] = _minmax_to_range(out[:, 5], m._LABEL_XYZ_MINMAX[2]["min"], m._LABEL_XYZ_MINMAX[2]["max"], m._feature_range)
    return out


def _log_posterior(
    theta: np.ndarray,
    *,
    model: torch.nn.Module,
    x_obs_norm: torch.Tensor,
    xyz_fixed: np.ndarray,
    e_min: float,
    e_max: float,
    num_steps: int,
    hutchinson_samples: int,
    noise_type: str,
) -> float:
    lp = _log_prior(theta, e_min=e_min, e_max=e_max)
    if not np.isfinite(lp):
        return -np.inf

    label = _make_label_with_fixed_xyz(theta, xyz_fixed, x_obs_norm.device)
    label_norm = _normalize_label_only(label)
    logp_x, _, _ = estimate_log_likelihood(
        model,
        x_obs_norm,
        label_norm,
        num_steps=num_steps,
        hutchinson_samples=hutchinson_samples,
        noise_type=noise_type,
    )
    return float(lp + logp_x)


def run_mh_mcmc(
    init_theta: np.ndarray,
    *,
    n_steps: int,
    burn_in: int,
    thin: int,
    prop_scales: np.ndarray,
    logpost_fn,
    rng: np.random.Generator,
    show_progress: bool = True,
    progress_every: int = 20,
) -> tuple[np.ndarray, np.ndarray, float]:
    theta = np.asarray(init_theta, dtype=np.float64).copy()
    cur_lp = float(logpost_fn(theta))
    chain = np.zeros((n_steps, 3), dtype=np.float64)
    logp = np.full(n_steps, -np.inf, dtype=np.float64)
    accepted = 0

    iterator = range(n_steps)
    pbar = None
    if show_progress and trange is not None:
        pbar = trange(n_steps, desc="MCMC", leave=True)
        iterator = pbar

    for i in iterator:
        prop = theta + rng.normal(0.0, prop_scales, size=3)
        prop_lp = float(logpost_fn(prop))
        if np.isfinite(prop_lp):
            log_alpha = prop_lp - cur_lp
            if log_alpha >= 0 or np.log(rng.uniform()) < log_alpha:
                theta = prop
                cur_lp = prop_lp
                accepted += 1
        chain[i] = theta
        logp[i] = cur_lp

        if pbar is not None and ((i + 1) % max(1, progress_every) == 0 or i == n_steps - 1):
            pbar.set_postfix(
                acc=f"{accepted / float(i + 1):.3f}",
                lp=f"{cur_lp:.2f}",
                E=f"{theta[0]:.2f}",
                ux=f"{theta[1]:.3f}",
                uy=f"{theta[2]:.3f}",
            )

    if pbar is not None:
        pbar.close()

    burn = int(max(0, burn_in))
    t = int(max(1, thin))
    samples = chain[burn::t]
    samples_logp = logp[burn::t]
    acc_rate = accepted / float(n_steps)
    return samples, samples_logp, acc_rate


def save_corner_plot(samples: np.ndarray, out_path: Path) -> None:
    labels = ["E (PeV)", "ux", "uy"]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for r in range(3):
        for c in range(3):
            ax = axes[r, c]
            if r == c:
                ax.hist(samples[:, c], bins=60, density=True, color="tab:blue", alpha=0.7)
                ax.set_ylabel("Density")
            elif r > c:
                ax.hist2d(samples[:, c], samples[:, r], bins=60, cmap="Blues")
            else:
                ax.axis("off")
                continue

            if r == 2:
                ax.set_xlabel(labels[c])
            if c == 0 and r > 0:
                ax.set_ylabel(labels[r])
            ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _credible_level_thresholds(pdf2d: np.ndarray, masses=(0.5, 0.9)) -> list[float]:
    """Return density thresholds whose super-level sets contain given posterior masses."""
    p = np.asarray(pdf2d, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 0.0)
    s = p.sum()
    if s <= 0:
        return [0.0 for _ in masses]
    p = p / s
    flat = p.ravel()
    order = np.argsort(flat)[::-1]
    sorted_p = flat[order]
    cdf = np.cumsum(sorted_p)
    thresholds = []
    for m in masses:
        idx = np.searchsorted(cdf, float(m), side="left")
        idx = min(max(idx, 0), len(sorted_p) - 1)
        thresholds.append(float(sorted_p[idx]))
    return thresholds


def save_posterior_contour_plot(
    samples: np.ndarray,
    true_theta: np.ndarray,
    best_theta: np.ndarray,
    out_path: Path,
    *,
    pair: str = "ux,uy",
    bins: int = 120,
    smooth_sigma: float = 1.2,
) -> None:
    """
    Save 2D posterior heatmap + 50/90% credible contours + true/best markers.
    pair options: 'ux,uy', 'E,ux', 'E,uy'
    """
    name_to_idx = {"E": 0, "ux": 1, "uy": 2}
    parts = [p.strip() for p in pair.split(",")]
    if len(parts) != 2 or parts[0] not in name_to_idx or parts[1] not in name_to_idx:
        raise ValueError(f"Unsupported pair '{pair}'. Use one of ux,uy / E,ux / E,uy")
    ix, iy = name_to_idx[parts[0]], name_to_idx[parts[1]]

    x = samples[:, ix]
    y = samples[:, iy]
    x_true, y_true = float(true_theta[ix]), float(true_theta[iy])
    x_best, y_best = float(best_theta[ix]), float(best_theta[iy])

    # Robust plotting range
    def _rng(a: np.ndarray, pad=0.08):
        lo, hi = np.percentile(a, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.min(a)), float(np.max(a))
        if lo == hi:
            hi = lo + 1.0
        d = hi - lo
        return lo - pad * d, hi + pad * d

    xlo, xhi = _rng(x)
    ylo, yhi = _rng(y)

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[xlo, xhi], [ylo, yhi]], density=False)
    if gaussian_filter is not None and smooth_sigma > 0:
        H = gaussian_filter(H, sigma=float(smooth_sigma))

    # Convert to "pdf-like" grid for contour mass thresholds
    H = np.asarray(H, dtype=np.float64)
    if H.sum() > 0:
        H_pdf = H / H.sum()
    else:
        H_pdf = H
    thr50, thr90 = _credible_level_thresholds(H_pdf, masses=(0.5, 0.9))

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    XX, YY = np.meshgrid(xc, yc, indexing="xy")
    Z = H_pdf.T  # (y, x) for plotting

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(xedges, yedges, H_pdf.T, shading="auto", cmap="Oranges")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Posterior density (a.u.)")

    # 90% outer, 50% inner
    levels = sorted(set([thr90, thr50]))
    cs = ax.contour(XX, YY, Z, levels=levels, colors=["red", "gray"], linewidths=[1.8, 1.8])
    if len(cs.levels) == 2:
        labels = ["90%", "50%"] if cs.levels[0] == thr90 else ["50%", "90%"]
        for i, coll in enumerate(cs.collections):
            coll.set_label(f"Credible {labels[i]}")

    ax.scatter([x_true], [y_true], marker="*", s=150, c="black", edgecolors="white", linewidths=0.8, label="True value")
    ax.scatter([x_best], [y_best], marker="s", s=45, c="black", label="Best posterior sample")

    ax.set_xlabel(parts[0])
    ax.set_ylabel(parts[1])
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Infer c=(E,ux,uy) by MCMC and save corner plot")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--ref_idx", type=int, required=True)
    p.add_argument("--h5_path", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--num_steps_likelihood", type=int, default=60)
    p.add_argument("--hutchinson_samples", type=int, default=1)
    p.add_argument("--noise_type", type=str, choices=["rademacher", "gaussian"], default="rademacher")
    p.add_argument("--n_mcmc_steps", type=int, default=1200)
    p.add_argument("--burn_in", type=int, default=300)
    p.add_argument("--thin", type=int, default=3)
    p.add_argument("--prop_e", type=float, default=0.5, help="Proposal std for E (PeV)")
    p.add_argument("--prop_u", type=float, default=0.03, help="Proposal std for ux/uy")
    p.add_argument("--e_min", type=float, default=1.0)
    p.add_argument("--e_max", type=float, default=100.0)
    p.add_argument("--no_clamp", action="store_true")
    p.add_argument("--no_tqdm", action="store_true")
    p.add_argument("--contour_pair", type=str, default="ux,uy", help="2D pair: ux,uy or E,ux or E,uy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (ckpt_path.parent.parent / "mcmc_corner")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
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
    model = _build_model(checkpoint, dataset, device)

    sig_raw, _, label_raw = dataset[int(args.ref_idx)]
    sig = sig_raw.unsqueeze(0).to(device, non_blocking=True)
    label = label_raw.unsqueeze(0).to(device, non_blocking=True)
    sig_in = m._clamp_sig(sig) if (not args.no_clamp) else sig
    x_obs_norm, _ = m.prepare_batch(sig_in, label, verbose=False)

    label_np = np.asarray(label_raw, dtype=np.float32)
    init_theta = np.array([label_np[0], label_np[1], label_np[2]], dtype=np.float64)
    xyz_fixed = np.array([label_np[3], label_np[4], label_np[5]], dtype=np.float64)

    def logpost(theta):
        return _log_posterior(
            theta,
            model=model,
            x_obs_norm=x_obs_norm,
            xyz_fixed=xyz_fixed,
            e_min=float(args.e_min),
            e_max=float(args.e_max),
            num_steps=int(args.num_steps_likelihood),
            hutchinson_samples=int(args.hutchinson_samples),
            noise_type=args.noise_type,
        )

    t0 = time.time()
    samples, samples_logp, acc_rate = run_mh_mcmc(
        init_theta=init_theta,
        n_steps=int(args.n_mcmc_steps),
        burn_in=int(args.burn_in),
        thin=int(args.thin),
        prop_scales=np.array([args.prop_e, args.prop_u, args.prop_u], dtype=np.float64),
        logpost_fn=logpost,
        rng=rng,
        show_progress=(not args.no_tqdm),
    )
    elapsed = time.time() - t0

    best_idx = int(np.argmax(samples_logp))
    best = samples[best_idx]
    med = np.median(samples, axis=0)

    corner_png = out_dir / f"corner_ref_{args.ref_idx}.png"
    save_corner_plot(samples, corner_png)
    contour_png = out_dir / f"contour_ref_{args.ref_idx}_{args.contour_pair.replace(',', '_')}.png"
    save_posterior_contour_plot(
        samples=samples,
        true_theta=label_np[:3],
        best_theta=best,
        out_path=contour_png,
        pair=args.contour_pair,
    )

    summary_csv = out_dir / f"summary_ref_{args.ref_idx}.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ref_idx", "acceptance_rate", "n_samples", "best_logpost",
            "best_E", "best_ux", "best_uy", "median_E", "median_ux", "median_uy",
            "true_E", "true_ux", "true_uy", "X", "Y", "Z",
        ])
        w.writerow([
            int(args.ref_idx),
            f"{acc_rate:.6f}",
            int(samples.shape[0]),
            f"{samples_logp[best_idx]:.8f}",
            f"{best[0]:.8f}",
            f"{best[1]:.8f}",
            f"{best[2]:.8f}",
            f"{med[0]:.8f}",
            f"{med[1]:.8f}",
            f"{med[2]:.8f}",
            f"{label_np[0]:.8f}",
            f"{label_np[1]:.8f}",
            f"{label_np[2]:.8f}",
            f"{xyz_fixed[0]:.8f}",
            f"{xyz_fixed[1]:.8f}",
            f"{xyz_fixed[2]:.8f}",
        ])

    np.save(out_dir / f"samples_ref_{args.ref_idx}.npy", samples)
    np.save(out_dir / f"samples_logpost_ref_{args.ref_idx}.npy", samples_logp)

    print("done")
    print(f"device={device}")
    print(f"ref_idx={args.ref_idx}")
    print(f"acceptance_rate={acc_rate:.4f}")
    print(f"corner_png={corner_png}")
    print(f"contour_png={contour_png}")
    print(f"summary_csv={summary_csv}")
    print(f"best=(E,ux,uy)=({best[0]:.4f}, {best[1]:.4f}, {best[2]:.4f})")
    print(f"elapsed_sec={elapsed:.2f}")


if __name__ == "__main__":
    main()
