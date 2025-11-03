#!/usr/bin/env python3
"""
NPZ-style 3D Visualization for Generated Samples
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional

from utils.denormalization import denormalize_signal


def visualize_sample_as_npz(
    diffusion,
    real_x_sig: torch.Tensor,
    real_geom: torch.Tensor,
    real_label: torch.Tensor,
    save_path: Optional[str] = None,
    detector_csv: str = "csv/detector_geometry.csv"
):
    """
    Generate one sample and visualize it in NPZ format (3D sphere plot).
    
    Args:
        diffusion: Trained diffusion model
        real_x_sig: Real signal (1, 2, L) - NORMALIZED (for comparison)
        real_geom: Real geometry (1, 3, L) - NORMALIZED
        real_label: Real label (1, 6) - NORMALIZED
        save_path: Path to save the generated sample visualization
        detector_csv: Path to detector geometry CSV
    """
    from utils.event_visualization.event_show import show_event_from_npz
    
    device = next(diffusion.parameters()).device
    
    # Get normalization parameters from model
    norm_params = diffusion.model.get_normalization_params()
    affine_offsets = norm_params['affine_offsets'][:2]  # [charge, time]
    affine_scales = norm_params['affine_scales'][:2]
    label_offsets = norm_params['label_offsets']
    label_scales = norm_params['label_scales']
    time_transform = norm_params['time_transform']
    
    print("\n" + "="*70)
    print("🎨 Generating Sample for NPZ Visualization")
    print("="*70)
    
    # Move to device
    real_geom = real_geom.to(device)
    real_label = real_label.to(device)
    
    # Generate sample using the model's label and geometry
    print("  🔄 Generating sample with diffusion model...")
    with torch.no_grad():
        generated = diffusion.sample(
            label=real_label,
            geom=real_geom,
            shape=(1, 2, real_geom.shape[-1])
        )
    
    # Denormalize generated sample
    def denorm_sample(x_norm, label_norm):
        """Denormalize signal and label"""
        # Signal denormalization
        x = x_norm.cpu().numpy()[0]  # (2, L)
        
        # Affine inverse
        offsets = np.array(affine_offsets).reshape(2, 1)
        scales = np.array(affine_scales).reshape(2, 1)
        x = (x * scales) + offsets
        
        # Time inverse transform
        if time_transform == "ln":
            x[1, :] = np.exp(x[1, :]) - 1.0
        elif time_transform == "log10":
            x[1, :] = np.power(10.0, x[1, :]) - 1.0
        
        # Clamp time
        x[1, :] = np.clip(x[1, :], 0, 1e8)
        
        # Label denormalization
        label = label_norm.cpu().numpy()[0]  # (6,)
        label_offsets_np = np.array(label_offsets)
        label_scales_np = np.array(label_scales)
        label = (label * label_scales_np) + label_offsets_np
        
        return x, label
    
    generated_denorm, label_denorm = denorm_sample(generated, real_label)
    
    # Print statistics
    charge = generated_denorm[0, :]
    time = generated_denorm[1, :]
    nonzero_mask = charge > 0
    
    print(f"  📊 Generated Sample Statistics:")
    print(f"     Charge (NPE): mean={charge[nonzero_mask].mean():.2f}, "
          f"max={charge[nonzero_mask].max():.2f}, "
          f"n_hits={nonzero_mask.sum()}")
    print(f"     Time (ns):    mean={time[nonzero_mask].mean():.2f}, "
          f"max={time[nonzero_mask].max():.2f}")
    print(f"  🏷️  Label: Energy={label_denorm[0]:.2e}, "
          f"Zenith={label_denorm[1]:.2f}, "
          f"Azimuth={label_denorm[2]:.2f}")
    print(f"           Position=({label_denorm[3]:.1f}, {label_denorm[4]:.1f}, {label_denorm[5]:.1f})")
    
    # Save as NPZ for visualization
    if save_path:
        save_path = Path(save_path)
        npz_dir = save_path.parent / "npz_samples"
        npz_dir.mkdir(parents=True, exist_ok=True)
        
        npz_path = npz_dir / "generated_sample.npz"
        np.savez(
            npz_path,
            input=generated_denorm,  # (2, L)
            label=label_denorm,      # (6,)
            info=label_denorm        # Placeholder
        )
        
        print(f"  💾 Saved NPZ to: {npz_path}")
        
        # Create 3D visualization
        try:
            print(f"  🎨 Creating 3D visualization...")
            fig, ax = show_event_from_npz(
                npz_path=str(npz_path),
                detector_csv=detector_csv,
                output_path=str(save_path),
                figure_size=(15, 10)
            )
            print(f"  ✅ Saved 3D visualization to: {save_path}")
        except Exception as e:
            print(f"  ⚠️  3D visualization failed: {e}")
            print(f"     NPZ file is still available at: {npz_path}")
    
    print("="*70 + "\n")
    
    return generated_denorm, label_denorm

