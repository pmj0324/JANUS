#!/usr/bin/env python3
"""
visualize_event.py

데이터로더를 이용해 특정 인덱스의 이벤트를 3D로 시각화하는 유틸리티.
- config.yaml 파일에서 설정을 읽어옴
- PMTSignalsH5를 통해 정규화된 데이터를 로드
- 역정규화를 적용하여 원본 스케일로 복원
- NPZ 파일로 저장 후 3D 시각화 생성
"""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config_from_file
from dataloader.pmt_dataloader import PMTSignalsH5
from utils.event_visualization.event_fast import plot_event_fast


def denormalize_signal(
    x_sig_norm: np.ndarray,
    charge_offset: float,
    charge_scale: float,
    time_offset: float,
    time_scale: float,
    time_transform: str,
) -> np.ndarray:
    """
    역정규화를 적용하여 원본 스케일로 복원.
    
    Args:
        x_sig_norm: (2, L) 정규화된 배열 [charge, time]
        charge_offset, charge_scale: charge 정규화 파라미터
        time_offset, time_scale: time 정규화 파라미터
        time_transform: 'ln' 또는 'log10'
    
    Returns:
        (2, L) 역정규화된 배열
    """
    result = x_sig_norm.copy()
    
    # Step 1: Reverse affine normalization for both channels
    # x_orig = (x_norm * scale) + offset
    result[0] = (result[0] * charge_scale) + charge_offset
    result[1] = (result[1] * time_scale) + time_offset
    
    # Step 2: Reverse log transformation for time
    # Original transformation was: log(1+x)
    # Reverse: x = inverse_log(y) - 1
    if time_transform == 'ln':
        # Reverse: ln(1+x) -> x = exp(y) - 1
        result[1] = np.exp(result[1]) - 1.0
    elif time_transform == 'log10':
        # Reverse: log10(1+x) -> x = 10^y - 1
        result[1] = np.power(10.0, result[1]) - 1.0
    else:
        raise ValueError(f"Unknown time_transform: {time_transform}")
    
    # Clamp to prevent overflow/underflow
    result[0] = np.clip(result[0], 0.0, 1e10)  # charge
    result[1] = np.clip(result[1], 0.0, 1e10)  # time
    
    return result


def denormalize_labels(
    labels_norm: np.ndarray,
    label_offsets: np.ndarray,
    label_scales: np.ndarray,
) -> np.ndarray:
    """
    역정규화를 적용하여 레이블을 원본 스케일로 복원.
    
    Args:
        labels_norm: (6,) 정규화된 레이블
        label_offsets: (6,) offset 파라미터
        label_scales: (6,) scale 파라미터
    
    Returns:
        (6,) 역정규화된 레이블
    """
    return (labels_norm * label_scales) + label_offsets


def visualize_event_from_dataloader(
    config_path: str,
    event_index: int,
    output_dir: str = "./event_visualization",
    detector_csv: str = "./configs/detector_geometry.csv",
):
    """
    데이터로더를 통해 특정 인덱스의 이벤트를 로드하고 시각화.
    
    Args:
        config_path: YAML 설정 파일 경로
        event_index: 시각화할 이벤트의 인덱스
        output_dir: 출력 디렉토리
        detector_csv: detector geometry CSV 파일 경로
    """
    print("\n" + "="*80)
    print("📊 Event Visualization from DataLoader")
    print("="*80)
    
    # Load configuration
    print(f"\n📂 Loading configuration from: {config_path}")
    config = load_config_from_file(config_path)
    
    # Extract normalization parameters from config
    # affine_offsets/scales: [charge, time, x, y, z]
    # label_offsets/scales: [Energy, Zenith, Azimuth, X, Y, Z]
    affine_offsets = list(config.data.affine_offsets)
    affine_scales = list(config.data.affine_scales)
    label_offsets = list(config.data.label_offsets)
    label_scales = list(config.data.label_scales)
    
    # Create dataset
    print(f"📂 Loading dataset from: {config.data.h5_path}")
    print(f"   Time transform: {config.data.time_transform}")
    
    dataset = PMTSignalsH5(
        h5_path=config.data.h5_path,
        replace_time_inf_with=0.0,
        channel_first=True,
        time_transform=config.data.time_transform,
        affine_offsets=tuple(affine_offsets),
        affine_scales=tuple(affine_scales),
        label_offsets=tuple(label_offsets),
        label_scales=tuple(label_scales),
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} events total")
    
    # Check if index is valid
    if event_index < 0 or event_index >= len(dataset):
        print(f"❌ Invalid event index: {event_index} (dataset size: {len(dataset)})")
        sys.exit(1)
    
    # Load event from dataset
    print(f"\n🔍 Loading event at index: {event_index}")
    x_sig, geom, labels, real_idx = dataset[event_index]
    
    # Convert to numpy
    x_sig_norm = x_sig.numpy()    # (2, 5160) - normalized
    geom_norm = geom.numpy()      # (3, 5160) - normalized geometry
    labels_norm = labels.numpy()  # (6,) - normalized labels
    
    print(f"   Real HDF5 index: {real_idx}")
    print(f"   Signal shape: {x_sig_norm.shape}")
    print(f"   Geometry shape: {geom_norm.shape}")
    print(f"   Labels shape: {labels_norm.shape}")
    
    # Denormalize signal data
    print(f"\n🔄 Denormalizing data...")
    print(f"   Charge: offset={affine_offsets[0]}, scale={affine_scales[0]}")
    print(f"   Time: offset={affine_offsets[1]}, scale={affine_scales[1]}, transform={config.data.time_transform}")
    
    x_sig_denorm = denormalize_signal(
        x_sig_norm,
        charge_offset=affine_offsets[0],
        charge_scale=affine_scales[0],
        time_offset=affine_offsets[1],
        time_scale=affine_scales[1],
        time_transform=config.data.time_transform,
    )
    
    # Denormalize labels
    labels_denorm = denormalize_labels(
        labels_norm,
        label_offsets=np.array(label_offsets, dtype=np.float32),
        label_scales=np.array(label_scales, dtype=np.float32),
    )
    
    # Print event information
    label_names = ['energy', 'zenith', 'azimuth', 'x', 'y', 'z']
    print(f"\n📊 Event Information:")
    for i, name in enumerate(label_names):
        print(f"   {name}: {labels_denorm[i]:.3f}")
    
    # Statistics
    print(f"\n📈 Signal Statistics (denormalized):")
    charge = x_sig_denorm[0]
    time = x_sig_denorm[1]
    
    nonzero_charge = charge[charge > 0]
    nonzero_time = time[charge > 0]  # Time where charge exists
    
    print(f"   Charge (NPE):")
    print(f"      Non-zero count: {len(nonzero_charge)} / {len(charge)} ({100*len(nonzero_charge)/len(charge):.1f}%)")
    if len(nonzero_charge) > 0:
        print(f"      Mean: {nonzero_charge.mean():.2f}")
        print(f"      Std: {nonzero_charge.std():.2f}")
        print(f"      Min/Max: {nonzero_charge.min():.2f} / {nonzero_charge.max():.2f}")
    
    print(f"   Time (ns) where charge > 0:")
    if len(nonzero_time) > 0:
        finite_time = nonzero_time[np.isfinite(nonzero_time)]
        if len(finite_time) > 0:
            print(f"      Count: {len(finite_time)}")
            print(f"      Mean: {finite_time.mean():.2f}")
            print(f"      Std: {finite_time.std():.2f}")
            print(f"      Min/Max: {finite_time.min():.2f} / {finite_time.max():.2f}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as NPZ
    npz_path = output_path / f"event_{event_index}.npz"
    print(f"\n💾 Saving NPZ file to: {npz_path}")
    
    np.savez(
        npz_path,
        input=x_sig_denorm,  # (2, 5160)
        label=labels_denorm,  # (6,)
        info=labels_denorm,   # Same as label for compatibility
    )
    
    # Create fast 3D visualization
    png_path = output_path / f"event_{event_index}.png"
    print(f"🎨 Creating fast 3D visualization...")
    
    try:
        # Load geometry from CSV
        geometry = np.loadtxt(detector_csv, delimiter=',', skiprows=1, usecols=(1,2,3))
        
        # Direct fast plotting
        fig, axes = plot_event_fast(
            charge_data=x_sig_denorm[0],
            time_data=x_sig_denorm[1],
            geometry=geometry,
            labels=labels_denorm,
            output_path=str(png_path),
            plot_type="both",
            figure_size=(16, 8),
            show_detector_hull=True,
            show_background=True,
            sphere_size=2.0,
            alpha=0.8
        )
        print(f"✅ Fast 3D visualization saved to: {png_path}")
    except Exception as e:
        print(f"⚠️  3D visualization failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"   NPZ file is still available at: {npz_path}")
    
    print("\n" + "="*80)
    print("✅ Event visualization complete!")
    print("="*80)
    print(f"\n📁 Output files:")
    print(f"   NPZ: {npz_path}")
    print(f"   PNG: {png_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize events from dataloader",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Event index to visualize"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./event_visualization",
        help="Output directory for NPZ and PNG files"
    )
    
    parser.add_argument(
        "--detector-csv",
        type=str,
        default="./configs/detector_geometry.csv",
        help="Path to detector geometry CSV file"
    )
    
    args = parser.parse_args()
    
    visualize_event_from_dataloader(
        config_path=args.config,
        event_index=args.index,
        output_dir=args.output_dir,
        detector_csv=args.detector_csv,
    )


if __name__ == "__main__":
    main()
