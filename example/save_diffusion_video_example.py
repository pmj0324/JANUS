#!/usr/bin/env python3
"""
Save Diffusion Process as Video (MP4 / GIF)
============================================

Renders forward diffusion (noise added step by step) and saves as MP4 and/or GIF.
Uses the same event visualization as visualize_forward_diffusion.

Usage:
    # From GENESIS root, with real data (requires H5 and config):
    python example/save_diffusion_video_example.py \\
        --config example/dit_linear_cosine.yaml \\
        --event-index 0 \\
        --output ./diffusion_video_out \\
        --step 50

    # Synthetic data (no H5 needed, for quick test):
    python example/save_diffusion_video_example.py \\
        --synthetic \\
        --num-timesteps 200 \\
        --step 10 \\
        --output ./diffusion_video_synthetic

Dependencies:
    pip install imageio
    pip install imageio-ffmpeg   # optional, for MP4
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from types import SimpleNamespace

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from utils.vis.diffusion_video import save_diffusion_video
from utils.normalize.log_minmax import apply_log_minmax


def load_config(config_path: str) -> SimpleNamespace:
    """Load YAML config as SimpleNamespace."""
    p = Path(config_path)
    if not p.is_absolute():
        for base in [Path.cwd(), script_dir, project_root]:
            cand = base / p
            if cand.exists():
                p = cand
                break
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)

    def to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: to_ns(v) for k, v in d.items()})
        if isinstance(d, list):
            return [to_ns(x) for x in d]
        return d

    return to_ns(cfg)


def normalize_signal_log_minmax(sig: torch.Tensor) -> torch.Tensor:
    """
    Normalize signal (2, L) with log_minmax to [0, 1] using fixed dataset bounds.
    Same convention as visualize_forward_diffusion: npe [0, 225], firstTime [0, 20676].
    """
    out = sig.clone()
    out[0] = apply_log_minmax(
        sig[0],
        feature_range=(0, 1),
        data_min=0.0,
        data_max=np.log1p(225.0),
    )
    out[1] = apply_log_minmax(
        sig[1],
        feature_range=(0, 1),
        data_min=0.0,
        data_max=np.log1p(20676.0),
    )
    return out


def load_event_from_h5(config_path: str, event_index: int):
    """Load one event (sig, geom, label) and normalize signal."""
    from dataloader.h5 import H5Dataset

    config = load_config(config_path)
    h5_path = getattr(config.data, "h5_path", None)
    if not h5_path:
        raise ValueError("Config data.h5_path is required")
    if not Path(h5_path).is_absolute():
        h5_path = str(project_root / h5_path)
    dataset = H5Dataset(h5_path=h5_path)
    if event_index < 0 or event_index >= len(dataset):
        raise IndexError(f"event_index {event_index} out of range [0, {len(dataset)})")
    sig_raw, geo_raw, label_raw = dataset[event_index]
    sig_norm = normalize_signal_log_minmax(sig_raw)
    x0_sig = sig_norm.unsqueeze(0)
    geom = geo_raw.unsqueeze(0)
    label = label_raw.unsqueeze(0)
    return x0_sig, geom, label


def synthetic_event(L: int = 5160, seed: int = 42):
    """Create minimal synthetic (1,2,L), (1,3,L), (1,6) for testing video without H5."""
    rng = np.random.default_rng(seed)
    # Normalized [0,1] signal: sparse npe and firstTime
    npe = np.zeros(L, dtype=np.float32)
    ftime = np.zeros(L, dtype=np.float32)
    n_hit = max(1, L // 20)
    idx = rng.choice(L, size=n_hit, replace=False)
    npe[idx] = rng.uniform(0.1, 1.0, size=n_hit).astype(np.float32)
    ftime[idx] = rng.uniform(0.1, 1.0, size=n_hit).astype(np.float32)
    sig = np.stack([npe, ftime], axis=0)
    # Simple 3D geometry (line or grid)
    x = np.linspace(-1, 1, L, dtype=np.float32)
    y = np.linspace(-1, 1, L, dtype=np.float32)
    z = np.zeros(L, dtype=np.float32)
    geo = np.stack([x, y, z], axis=0)
    label = np.array([1e6, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    x0_sig = torch.from_numpy(sig).unsqueeze(0)
    geom = torch.from_numpy(geo).unsqueeze(0)
    label = torch.from_numpy(label).unsqueeze(0)
    return x0_sig, geom, label


def main():
    parser = argparse.ArgumentParser(
        description="Save forward diffusion process as MP4/GIF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=str(script_dir / "dit_linear_cosine.yaml"),
        help="Path to YAML config (used when not --synthetic)",
    )
    parser.add_argument(
        "-e", "--event-index",
        type=int,
        default=0,
        help="Event index in dataset (when not --synthetic)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (no H5); L=5160 by default",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=5160,
        help="Signal length L for --synthetic",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./diffusion_video_out",
        help="Output path: file (with/without ext) or directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["mp4", "gif", "both"],
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        choices=["linear", "cosine", "quadratic", "sigmoid"],
        default="linear",
        help="Noise schedule",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=1000,
        help="Total diffusion timesteps T",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=50,
        help="Render every this many steps (0, step, 2*step, ...)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10,
        help="Frames per second",
    )
    parser.add_argument(
        "--cosine-s",
        type=float,
        default=0.008,
        help="Cosine schedule parameter s",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Do not fix random seed (different noise each run)",
    )
    args = parser.parse_args()

    if args.synthetic:
        x0_sig, geom, label = synthetic_event(L=args.length)
        print("Using synthetic event (no H5).")
    else:
        x0_sig, geom, label = load_event_from_h5(args.config, args.event_index)
        print(f"Loaded event index {args.event_index} from config.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    x0_sig = x0_sig.to(device)
    geom = geom.to(device)
    label = label.to(device)

    schedule_kwargs = {}
    if args.schedule == "cosine":
        schedule_kwargs["s"] = args.cosine_s

    path_mp4, path_gif = save_diffusion_video(
        x0_sig=x0_sig,
        geom=geom,
        label=label,
        schedule_name=args.schedule,
        schedule_kwargs=schedule_kwargs or None,
        num_timesteps=args.num_timesteps,
        step=args.step,
        output_path=args.output,
        output_format=args.format,
        fps=args.fps,
        fixed_noise=not args.no_seed,
        seed=None if args.no_seed else 42,
    )
    print("Done.")
    if path_mp4:
        print(f"  MP4: {path_mp4}")
    if path_gif:
        print(f"  GIF: {path_gif}")


if __name__ == "__main__":
    main()
