#!/usr/bin/env python3
"""
Forward Diffusion Visualization Script
======================================

Visualize forward diffusion process with multiple noise schedules.
Shows how noise is added at different timesteps with:
- 3D plots (like example/Visualize)
- Histograms for firstTime and nPE

Usage:
    # Run from GENESIS root directory:
    cd /path/to/GENESIS
    python example/visualize_forward_diffusion.py \
        --event-index 0 \
        --timesteps 0 100 500 999 \
        --schedules linear cosine \
        --output-dir ./forward_viz
    
    # Or specify a custom config file:
    python example/visualize_forward_diffusion.py \
        --config example/dit_linear_cosine.yaml \
        --event-index 0
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import numpy as np
import yaml
from types import SimpleNamespace

# Add parent directory to path for imports
# This allows importing from GENESIS root (dataloader, utils, etc.)
# Calculate project root: if script is in example/, go up one level
script_path = Path(__file__).resolve()
if script_path.parent.name == "example":
    project_root = script_path.parent.parent
else:
    # If script is run from root, parent is already root
    project_root = script_path.parent

# Add project root to Python path BEFORE importing
sys.path.insert(0, str(project_root))


def load_config_from_file(config_path: str):
    """
    Load YAML config file and return as SimpleNamespace object.
    
    Args:
        config_path: Path to YAML config file (can be relative or absolute)
    
    Returns:
        SimpleNamespace object with config attributes
    """
    config_path = Path(config_path)
    
    # If relative path, try relative to current directory first, then script directory
    if not config_path.is_absolute():
        # Try current directory
        if not config_path.exists():
            # Try relative to script directory
            script_dir = Path(__file__).parent
            alt_path = script_dir / config_path
            if alt_path.exists():
                config_path = alt_path
            # Try relative to project root
            else:
                alt_path = project_root / config_path
                if alt_path.exists():
                    config_path = alt_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert nested dicts to SimpleNamespace recursively
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d
    
    return dict_to_namespace(config_dict)

try:
    from dataloader.h5 import H5Dataset
except ImportError as e:
    print(f"Error: Could not import 'dataloader.h5': {e}")
    sys.exit(1)

# Denormalization function (implemented locally since utils.denormalization may not exist)
def denormalize_signal(
    sig_norm: torch.Tensor,
    affine_offsets: list,
    affine_scales: list,
    time_transform: str = "ln",
    channels: str = "signal"
) -> torch.Tensor:
    """
    Denormalize signal tensor.
    
    Args:
        sig_norm: Normalized signal tensor (B, 2, L) or (2, L)
        affine_offsets: List of offsets [charge, time, ...]
        affine_scales: List of scales [charge, time, ...]
        time_transform: "ln" or "log10"
        channels: "signal" to process signal channels only
    
    Returns:
        Denormalized signal tensor in physical units
    """
    sig = sig_norm.clone()
    
    # Step 1: Reverse affine normalization: x = (x_norm * scale) + offset
    for i in range(2):  # For signal channels (nPE, firstTime)
        sig[i, :] = (sig[i, :] * affine_scales[i]) + affine_offsets[i]
    
    # Step 2: Reverse time transform (only for time channel, index 1)
    if time_transform == "ln":
        # Inverse of ln(1+x) is exp(x) - 1
        sig[1, :] = torch.exp(sig[1, :]) - 1.0
    elif time_transform == "log10":
        # Inverse of log10(1+x) is 10^x - 1
        sig[1, :] = torch.pow(10.0, sig[1, :]) - 1.0
    
    # Step 3: Clamp to prevent overflow
    sig[0, :] = torch.clamp(sig[0, :], min=0.0, max=1e10)  # nPE
    sig[1, :] = torch.clamp(sig[1, :], min=0.0, max=1e8)   # time
    
    return sig

from utils.vis.visualize_forward_diffusion import visualize_forward_diffusion
from diffusion.schedules import get_noise_schedule, compute_alpha_schedule
from diffusion.forward import apply_forward_diffusion


def load_event_and_config(config_path: str, event_index: int):
    """Load event from dataloader."""
    print(f"\n Loading configuration from: {config_path}")
    config = load_config_from_file(config_path)
    
    # Extract normalization parameters
    affine_offsets = list(config.data.affine_offsets)
    affine_scales = list(config.data.affine_scales)
    label_offsets = list(config.data.label_offsets)
    label_scales = list(config.data.label_scales)
    
    # Handle h5_path: if relative, make it relative to project root
    h5_path = config.data.h5_path
    if not os.path.isabs(h5_path):
        # Make path relative to project root
        script_path = Path(__file__).resolve()
        if script_path.parent.name == "example":
            project_root = script_path.parent.parent
        else:
            project_root = script_path.parent
        h5_path = project_root / h5_path
        h5_path = str(h5_path.resolve())
    
    print(f" Loading dataset from: {h5_path}")
    dataset = H5Dataset(h5_path=h5_path)
    
    print(f" Dataset loaded: {len(dataset)} events total")
    
    # Check if index is valid
    if event_index < 0 or event_index >= len(dataset):
        print(f" Invalid event index: {event_index} (dataset size: {len(dataset)})")
        sys.exit(1)
    
    # Load event (H5Dataset returns tuple: (sig, geo, label))
    sig_raw, geo_raw, label_raw = dataset[event_index]
    # sig_raw: (2, L) - 원본 데이터 (정규화되지 않음)
    # geo_raw: (3, L)
    # label_raw: (6,)
    
    # visualize_forward_diffusion 함수의 데코레이터가 자동으로 정규화하므로
    # 원본 데이터를 전달합니다 (minmax 정규화는 데코레이터가 처리)
    # Add batch dimension
    x_sig = sig_raw.unsqueeze(0)    # (1, 2, 5160) - 원본 데이터
    geom = geo_raw.unsqueeze(0)     # (1, 3, 5160)
    labels = label_raw.unsqueeze(0)   # (1, 6)
    
    print(f" Event {event_index} loaded:")
    print(f"   Signal shape: {x_sig.shape}")
    print(f"   Signal range - npe: [{x_sig[0, 0, :].min():.2f}, {x_sig[0, 0, :].max():.2f}]")
    print(f"   Signal range - firstTime: [{x_sig[0, 1, :].min():.2f}, {x_sig[0, 1, :].max():.2f}]")
    print(f"   Geometry shape: {geom.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Note: Signal will be normalized by visualize_forward_diffusion decorator")
    
    return x_sig, geom, labels, config


def create_denormalize_fn(config):
    """Create denormalization function."""
    def denormalize(sig_np):
        """Denormalize signal array."""
        sig_tensor = torch.from_numpy(sig_np).float()
        # Remove batch dimension if present (2, L) -> (2, L)
        if sig_tensor.ndim == 3:
            sig_tensor = sig_tensor[0]
        sig_denorm = denormalize_signal(
            sig_tensor,
            list(config.data.affine_offsets),
            list(config.data.affine_scales),
            time_transform=config.data.time_transform,
            channels="signal"
        )
        return sig_denorm.numpy()
    return denormalize


def main():
    parser = argparse.ArgumentParser(
        description="Visualize forward diffusion with multiple noise schedules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Default config file in example folder
    default_config = Path(__file__).parent / "dit_linear_cosine.yaml"
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=str(default_config),
        help=f"Path to YAML config file (default: {default_config})"
    )
    
    parser.add_argument(
        "-e", "--event-index",
        type=int,
        default=0,
        help="Event index to visualize"
    )
    
    parser.add_argument(
        "-t", "--timesteps",
        type=int,
        nargs="+",
        default=[0, 100, 500, 1000],
        help="Timesteps to visualize (must include 0 and final timestep)"
    )
    
    parser.add_argument(
        "-s", "--schedules",
        type=str,
        nargs="+",
        default=["linear", "cosine"],
        choices=["linear", "cosine", "quadratic", "sigmoid"],
        help="Noise schedules to use"
    )
    
    parser.add_argument(
        "--cosine-s",
        type=float,
        default=0.008,
        help="Cosine schedule parameter s"
    )
    
    parser.add_argument(
        "--beta-start",
        type=float,
        default=1e-4,
        help="Starting beta value (for linear, quadratic, sigmoid)"
    )
    
    parser.add_argument(
        "--beta-end",
        type=float,
        default=2e-2,
        help="Ending beta value (for linear, quadratic, sigmoid)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./forward_visualization",
        help="Output directory for saved files"
    )
    
    parser.add_argument(
        "-d", "--detector-csv",
        type=str,
        default=None,
        help="Path to detector geometry CSV file (optional)"
    )
    
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="Don't save 3D plots"
    )
    
    parser.add_argument(
        "--no-histograms",
        action="store_true",
        help="Don't save histograms"
    )

    parser.add_argument(
        "--denormalize",
        action="store_true",
        help="Visualize in original (denormalized) units. Default is to visualize normalized values.",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" Forward Diffusion Visualization")
    print("="*80)
    
    # Load event and config
    x_sig, geom, labels, config = load_event_and_config(
        args.config, args.event_index
    )
    
    # Get total timesteps from config
    if hasattr(config, 'diffusion') and hasattr(config.diffusion, 'timesteps'):
        timesteps_total = config.diffusion.timesteps
    else:
        timesteps_total = 1000  # Default
    
    # Validate timesteps
    timesteps = sorted(set(args.timesteps))
    if timesteps[0] != 0:
        print("  Warning: Adding t=0 to timesteps (original data)")
        timesteps.insert(0, 0)
    if timesteps[-1] > timesteps_total:
        print(f"  Warning: Clamping timesteps to max {timesteps_total}")
        timesteps = [t for t in timesteps if t <= timesteps_total]
        if timesteps[-1] != timesteps_total:
            timesteps.append(timesteps_total)
    
    # Prepare schedules
    schedules = []
    for schedule_name in args.schedules:
        schedule_kwargs = {}
        if schedule_name == "cosine":
            schedule_kwargs["s"] = args.cosine_s
        else:
            schedule_kwargs["beta_start"] = args.beta_start
            schedule_kwargs["beta_end"] = args.beta_end
        schedules.append((schedule_name, schedule_kwargs))
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_sig = x_sig.to(device)
    geom = geom.to(device)
    
    print(f"\n Visualization settings:")
    print(f"   Timesteps: {timesteps}")
    print(f"   Schedules: {[s[0] for s in schedules]}")
    print(f"   Save 3D plots: {not args.no_3d}")
    print(f"   Save histograms: {not args.no_histograms}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Visualize denormalized: {args.denormalize}")
    print(f"   Note: Signal normalization handled by visualize_forward_diffusion decorator")
    
    # Visualize
    # denormalize는 visualize_forward_diffusion 내부 옵션으로 제어합니다.
    visualize_forward_diffusion(
        x0_sig=x_sig,
        geom=geom,
        label=labels,
        schedules=schedules,
        timesteps=timesteps,
        output_dir=args.output_dir,
        detector_csv=args.detector_csv,
        save_3d=not args.no_3d,
        save_histograms=not args.no_histograms,
        denormalize=args.denormalize,
        denormalize_fn=None,
    )
    
    print("\n" + "="*80)
    print(" Visualization complete!")
    print("="*80)


if __name__ == "__main__":
    main()
