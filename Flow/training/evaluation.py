#!/usr/bin/env python3
"""
Post-training Evaluation
========================

Automatic evaluation after training completion:
- Generate samples with trained model
- Compare with real data
- Visualize side-by-side comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

from utils.denormalization import denormalize_signal


def compare_generated_vs_real(
    diffusion,
    real_x_sig: torch.Tensor,
    real_geom: torch.Tensor,
    real_label: torch.Tensor,
    num_samples: int = 4,
    save_dir: Optional[str] = "evaluation",
    affine_offsets: Optional[tuple] = None,
    affine_scales: Optional[tuple] = None,
    time_transform: Optional[str] = None
):
    """
    Compare generated samples with real samples.
    
    Args:
        diffusion: Trained diffusion model
        real_x_sig: Real signals (N, 2, L)
        real_geom: Real geometry (N, 3, L) or (3, L)
        real_label: Real labels (N, 6)
        num_samples: Number of samples to compare
        save_dir: Directory to save comparison plots
        affine_offsets: Normalization offsets (if None, get from model)
        affine_scales: Normalization scales (if None, get from model)
        time_transform: Time transformation type (if None, get from model)
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    device = next(diffusion.parameters()).device
    
    # Get normalization parameters from model if not provided
    if affine_offsets is None or affine_scales is None or time_transform is None:
        # Get model from diffusion wrapper
        if hasattr(diffusion, 'model'):
            model = diffusion.model
        elif hasattr(diffusion, 'get_normalization_params'):
            model = diffusion
        else:
            raise ValueError("Cannot find model with normalization params")
        
        norm_params = model.get_normalization_params()
        if affine_offsets is None:
            affine_offsets = tuple(norm_params['affine_offsets'])
        if affine_scales is None:
            affine_scales = tuple(norm_params['affine_scales'])
        if time_transform is None:
            time_transform = norm_params['time_transform']
    
    print(f"\n📊 Using normalization parameters from model:")
    print(f"  Offsets: {affine_offsets}")
    print(f"  Scales: {affine_scales}")
    print(f"  Time transform: {time_transform}")
    N = min(num_samples, real_x_sig.size(0))
    
    # Prepare real samples
    real_x_sig = real_x_sig[:N].to(device)
    real_label = real_label[:N].to(device)
    
    # Handle geometry
    if real_geom.ndim == 2:  # (3, L)
        real_geom = real_geom.unsqueeze(0).expand(N, -1, -1)
    real_geom = real_geom[:N].to(device)
    
    print(f"\n{'='*70}")
    print("🎨 Generating Samples for Comparison")
    print(f"{'='*70}")
    print(f"Using {N} samples with classifier-free guidance (scale={diffusion.cfg.cfg_scale})")
    
    # Generate samples with same conditions as real data
    with torch.no_grad():
        generated_x_sig = diffusion.sample(
            label=real_label,
            geom=real_geom,
            shape=(N, 2, 5160)
        )
    
    print(f"✅ Generated {N} samples")
    
    # =================================================================
    # DENORMALIZATION
    # =================================================================
    # Both real and generated data come normalized from dataloader/model.
    # Apply: affine inverse + time inverse transformation.
    # =================================================================
    
    # Denormalize real data: Full denormalization (affine + time transform)
    real_denorm = denormalize_signal(
        real_x_sig, affine_offsets, affine_scales,
        time_transform=time_transform, channels="signal"
    )
    
    # Denormalize generated data: Full denormalization (affine + time transform)
    generated_denorm = denormalize_signal(
        generated_x_sig, affine_offsets, affine_scales,
        time_transform=time_transform, channels="signal"
    )
    
    # Create comparison plot
    fig, axes = plt.subplots(N, 2, figsize=(14, 3*N))
    if N == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(N):
        # Real sample
        ax = axes[i, 0]
        real_charge = real_denorm[i, 0, :].cpu().numpy()
        real_time = real_denorm[i, 1, :].cpu().numpy()
        
        mask_real = real_charge > 0
        ax.scatter(real_time[mask_real], real_charge[mask_real], 
                  s=2, alpha=0.6, c='blue', label='Real')
        ax.set_title(f'Sample {i+1}: Real Data')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Charge (NPE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add label info
        energy = real_label[i, 0].item()
        zenith = real_label[i, 1].item()
        azimuth = real_label[i, 2].item()
        ax.text(0.02, 0.98, 
                f'E={energy:.2e}\nθ={zenith:.2f}\nφ={azimuth:.2f}',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Generated sample
        ax = axes[i, 1]
        gen_charge = generated_denorm[i, 0, :].cpu().numpy()
        gen_time = generated_denorm[i, 1, :].cpu().numpy()
        
        mask_gen = gen_charge > 0
        ax.scatter(gen_time[mask_gen], gen_charge[mask_gen], 
                  s=2, alpha=0.6, c='red', label='Generated')
        ax.set_title(f'Sample {i+1}: Generated (CFG scale={diffusion.cfg.cfg_scale})')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Charge (NPE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add same label info
        ax.text(0.02, 0.98, 
                f'E={energy:.2e}\nθ={zenith:.2f}\nφ={azimuth:.2f}',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / 'generated_vs_real.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Comparison saved to: {save_path}")
    
    plt.close()
    
    # Compute and print statistics
    print(f"\n{'='*70}")
    print("📊 Statistics Comparison")
    print(f"{'='*70}")
    
    # Dataset info
    print(f"\nDataset info:")
    print(f"  Shape: {real_denorm.shape} (batch, channels[charge/time], PMTs)")
    print(f"  Total elements: {real_denorm.shape[0]} samples × {real_denorm.shape[2]} PMTs = {real_denorm.shape[0] * real_denorm.shape[2]} PMT readings")
    
    real_charge_nonzero = real_denorm[:, 0, :][real_denorm[:, 0, :] > 0]
    gen_charge_nonzero = generated_denorm[:, 0, :][generated_denorm[:, 0, :] > 0]
    
    print(f"\nCharge (NPE) - Non-zero values:")
    print(f"  Real:      mean={real_charge_nonzero.mean():.4f} std={real_charge_nonzero.std():.4f} (n={len(real_charge_nonzero)} non-zero)")
    print(f"  Generated: mean={gen_charge_nonzero.mean():.4f} std={gen_charge_nonzero.std():.4f} (n={len(gen_charge_nonzero)} non-zero)")
    print(f"  Difference: {abs(real_charge_nonzero.mean() - gen_charge_nonzero.mean()):.4f}")
    print(f"  Sparsity:  Real={100*(1-len(real_charge_nonzero)/(real_denorm.shape[0]*real_denorm.shape[2])):.1f}% zeros, Gen={100*(1-len(gen_charge_nonzero)/(generated_denorm.shape[0]*generated_denorm.shape[2])):.1f}% zeros")
    
    real_time_nonzero = real_denorm[:, 1, :][real_denorm[:, 0, :] > 0]
    gen_time_nonzero = generated_denorm[:, 1, :][generated_denorm[:, 0, :] > 0]
    
    # Filter out inf and nan values for time
    real_time_valid = real_time_nonzero[torch.isfinite(real_time_nonzero)]
    gen_time_valid = gen_time_nonzero[torch.isfinite(gen_time_nonzero)]
    
    print(f"\nTime (ns) - Where charge > 0 (finite values only):")
    if len(real_time_valid) > 0 and len(gen_time_valid) > 0:
        print(f"  Real:      mean={real_time_valid.mean():.4f} std={real_time_valid.std():.4f} (n={len(real_time_valid)} valid)")
        print(f"  Generated: mean={gen_time_valid.mean():.4f} std={gen_time_valid.std():.4f} (n={len(gen_time_valid)} valid)")
        print(f"  Difference: {abs(real_time_valid.mean() - gen_time_valid.mean()):.4f}")
    else:
        print(f"  Warning: No valid time values found (all inf/nan)")
        print(f"  Real valid values: {len(real_time_valid)}/{len(real_time_nonzero)}")
        print(f"  Generated valid values: {len(gen_time_valid)}/{len(gen_time_nonzero)}")
    
    print(f"\n{'='*70}\n")
    
    return {
        'real_charge_mean': real_charge_nonzero.mean().item(),
        'gen_charge_mean': gen_charge_nonzero.mean().item(),
        'real_time_mean': real_time_valid.mean().item() if len(real_time_valid) > 0 else float('nan'),
        'gen_time_mean': gen_time_valid.mean().item() if len(gen_time_valid) > 0 else float('nan'),
    }

