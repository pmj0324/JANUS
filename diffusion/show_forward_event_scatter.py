#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize diffusion process: original vs noisy signals
"""

import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

from config import load_config_from_file, get_default_config
from dataloader.pmt_dataloader import make_dataloader
from models.factory import ModelFactory
from utils.denormalization import denormalize_signal


def visualize_diffusion_process(config_path: str = None, num_samples: int = 4, output_path: str = None):
    """
    Visualize original and noisy signals at different timesteps.
    
    Args:
        config_path: Path to config file (if None, uses default)
        num_samples: Number of samples to visualize
        output_path: Output directory path (if None, uses default)
    """
    # Load config
    if config_path:
        config = load_config_from_file(config_path)
    else:
        config = get_default_config()
    
    # Create dataloader with normalization parameters
    dataloader = make_dataloader(
        config.data.h5_path,
        batch_size=num_samples,
        shuffle=True,
        num_workers=0,
        replace_time_inf_with=config.data.replace_time_inf_with,
        channel_first=config.data.channel_first,
        time_transform=config.model.time_transform,
        affine_offsets=tuple(config.model.affine_offsets),
        affine_scales=tuple(config.model.affine_scales),
        label_offsets=tuple(config.model.label_offsets),
        label_scales=tuple(config.model.label_scales)
    )
    
    # Get one batch
    x_sig, geom, label, idx = next(iter(dataloader))
    print(f"Loaded batch: x_sig shape={x_sig.shape}, label shape={label.shape}")
    
    # Expand geom if needed
    if geom.ndim == 2:
        geom = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)
    
    # Create model and diffusion wrapper
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model,
        config.diffusion,
        device=torch.device('cpu')
    )
    
    # Timesteps to visualize (t=0 is original, then show noise steps from t=1 to t=T)
    timesteps = [0, 1, 250, 500, 1000]
    T = diffusion.cfg.timesteps
    print(f"Note: t=0 is original data, t=1 is first noise step, t={T} is final timestep")
    
    # Create figure with 4 rows: NPE norm, NPE denorm, Time norm, Time denorm
    fig, axes = plt.subplots(num_samples * 4, len(timesteps) + 1, figsize=(20, 16 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(4, -1)
    
    for sample_idx in range(num_samples):
        x0_sig_single = x_sig[sample_idx:sample_idx+1]  # (1, 2, L)
        label_single = label[sample_idx:sample_idx+1]    # (1, 6)
        
        # Denormalize original signal for display
        x0_denorm = denormalize_signal(
            x0_sig_single, 
            config.model.affine_offsets, 
            config.model.affine_scales,
            time_transform=config.model.time_transform,
            channels="signal"
        )
        
        # ========== ROW 0: NPE NORMALIZED ==========
        # Plot original NPE signal (normalized)
        ax_npe_norm = axes[sample_idx * 4, 0]
        charge_norm = x0_sig_single[0, 0, :].numpy()
        
        # Plot all charges including zeros
        pmt_indices = np.arange(len(charge_norm))
        ax_npe_norm.scatter(pmt_indices, charge_norm, s=1, alpha=0.5, color='blue')
        
        # Add standard Gaussian envelope
        try:
            x_range = np.linspace(pmt_indices.min(), pmt_indices.max(), 100)
            charge_gauss = np.exp(-0.5 * ((x_range - pmt_indices.mean()) / pmt_indices.std())**2)
            ax_npe_norm.plot(x_range, charge_gauss, 'r-', linewidth=2, alpha=0.8, label='Standard Gaussian (μ=0, σ=1)')
            ax_npe_norm.legend()
        except:
            pass
        ax_npe_norm.set_title(f'Sample {sample_idx}: NPE Original (Normalized)')
        ax_npe_norm.set_xlabel('PMT Index')
        ax_npe_norm.set_ylabel('Charge (normalized)')
        ax_npe_norm.set_ylim(0, charge_norm.max() * 1.1 if charge_norm.max() > 0 else 1)

        # ========== ROW 1: NPE DENORMALIZED ==========
        # Plot original NPE signal (denormalized)
        ax_npe_denorm = axes[sample_idx * 4 + 1, 0]
        charge = x0_denorm[0, 0, :].numpy()
        
        # Plot all charges including zeros
        pmt_indices = np.arange(len(charge))
        ax_npe_denorm.scatter(pmt_indices, charge, s=1, alpha=0.5, color='red')
        
        # Add scaled Gaussian envelope
        try:
            charge_std = charge.std()
            x_range = np.linspace(pmt_indices.min(), pmt_indices.max(), 100)
            charge_gauss = charge_std * np.exp(-0.5 * ((x_range - pmt_indices.mean()) / pmt_indices.std())**2)
            ax_npe_denorm.plot(x_range, charge_gauss, 'b-', linewidth=2, alpha=0.8, label=f'Scaled Gaussian (σ={charge_std:.1f})')
            ax_npe_denorm.legend()
        except:
            pass
        ax_npe_denorm.set_title(f'Sample {sample_idx}: NPE Original (Denormalized)')
        ax_npe_denorm.set_xlabel('PMT Index')
        ax_npe_denorm.set_ylabel('Charge (NPE)')
        ax_npe_denorm.set_ylim(0, charge.max() * 1.1 if charge.max() > 0 else 1)

        # ========== ROW 2: TIME NORMALIZED ==========
        # Plot original Time signal (normalized)
        ax_time_norm = axes[sample_idx * 4 + 2, 0]
        time_norm = x0_sig_single[0, 1, :].numpy()
        
        # Plot all times including zeros
        pmt_indices = np.arange(len(time_norm))
        ax_time_norm.scatter(pmt_indices, time_norm, s=1, alpha=0.5, color='blue')
        
        # Add standard Gaussian envelope
        try:
            x_range = np.linspace(pmt_indices.min(), pmt_indices.max(), 100)
            time_gauss = np.exp(-0.5 * ((x_range - pmt_indices.mean()) / pmt_indices.std())**2)
            ax_time_norm.plot(x_range, time_gauss, 'r-', linewidth=2, alpha=0.8, label='Standard Gaussian (μ=0, σ=1)')
            ax_time_norm.legend()
        except:
            pass
        ax_time_norm.set_title(f'Sample {sample_idx}: Time Original (Normalized)')
        ax_time_norm.set_xlabel('PMT Index')
        ax_time_norm.set_ylabel('Time (normalized)')
        ax_time_norm.set_ylim(0, time_norm.max() * 1.1 if time_norm.max() > 0 else 1)

        # ========== ROW 3: TIME DENORMALIZED ==========
        # Plot original Time signal (denormalized)
        ax_time_denorm = axes[sample_idx * 4 + 3, 0]
        time = x0_denorm[0, 1, :].numpy()
        
        # Plot all times including zeros
        pmt_indices = np.arange(len(time))
        ax_time_denorm.scatter(pmt_indices, time, s=1, alpha=0.5, color='red')
        
        # Add scaled Gaussian envelope
        try:
            time_std = time.std()
            x_range = np.linspace(pmt_indices.min(), pmt_indices.max(), 100)
            time_gauss = time_std * np.exp(-0.5 * ((x_range - pmt_indices.mean()) / pmt_indices.std())**2)
            ax_time_denorm.plot(x_range, time_gauss, 'b-', linewidth=2, alpha=0.8, label=f'Scaled Gaussian (σ={time_std:.1f})')
            ax_time_denorm.legend()
        except:
            pass
        ax_time_denorm.set_title(f'Sample {sample_idx}: Time Original (Denormalized)')
        ax_time_denorm.set_xlabel('PMT Index')
        ax_time_denorm.set_ylabel('Time (ns)')
        ax_time_denorm.set_ylim(0, time.max() * 1.1 if time.max() > 0 else 1)
        
        # Add label info to the first row
        energy = label_single[0, 0].item()
        zenith = label_single[0, 1].item()
        ax_npe_norm.text(0.02, 0.98, f'E={energy:.1e}\nθ={zenith:.2f}', 
                         transform=ax_npe_norm.transAxes, va='top', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Plot noisy signals at different timesteps
        for t_idx, t_value in enumerate(timesteps):
            # Add noise
            t_tensor = torch.tensor([t_value], dtype=torch.long)
            x_sig_t = diffusion.q_sample(x0_sig_single, t_tensor)
            
            # Denormalize noisy signal for display
            x_sig_t_denorm = denormalize_signal(
                x_sig_t, 
                config.model.affine_offsets, 
                config.model.affine_scales,
                time_transform=config.model.time_transform,
                channels="signal"
            )
            
            # ========== ROW 0: NPE NORMALIZED NOISY ==========
            ax_npe_norm = axes[sample_idx * 4, t_idx + 1]
            charge_noisy_norm = x_sig_t[0, 0, :].numpy()
            
            # Plot all normalized noisy NPE including zeros
            pmt_indices = np.arange(len(charge_noisy_norm))
            ax_npe_norm.scatter(pmt_indices, charge_noisy_norm, s=1, alpha=0.5, color='blue')
            
            # Add standard Gaussian envelope
            try:
                x_range = np.linspace(pmt_indices.min(), pmt_indices.max(), 100)
                charge_gauss = np.exp(-0.5 * ((x_range - pmt_indices.mean()) / pmt_indices.std())**2)
                ax_npe_norm.plot(x_range, charge_gauss, 'r-', linewidth=2, alpha=0.8, label='Standard Gaussian (μ=0, σ=1)')
                ax_npe_norm.legend()
            except:
                pass
            ax_npe_norm.set_title(f't={t_value} NPE (Normalized)')
            ax_npe_norm.set_xlabel('PMT Index')
            ax_npe_norm.set_ylabel('Charge (normalized)')
            
            # Show noise statistics (in normalized scale)
            noise_std_norm = (x_sig_t - x0_sig_single).std().item()
            signal_std_norm = x0_sig_single.std().item()
            snr_norm = signal_std_norm / (noise_std_norm + 1e-8)
            ax_npe_norm.text(0.02, 0.98, f'SNR={snr_norm:.2f}\nσ_n={noise_std_norm:.3f}', 
                             transform=ax_npe_norm.transAxes, va='top', fontsize=8,
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            # ========== ROW 1: NPE DENORMALIZED NOISY ==========
            ax_npe_denorm = axes[sample_idx * 4 + 1, t_idx + 1]
            charge_noisy = x_sig_t_denorm[0, 0, :].numpy()
            
            # Plot all denormalized noisy NPE including zeros
            pmt_indices = np.arange(len(charge_noisy))
            ax_npe_denorm.scatter(pmt_indices, charge_noisy, s=1, alpha=0.5, color='red')
            
            # Add scaled Gaussian envelope
            try:
                charge_std = charge_noisy.std()
                x_range = np.linspace(pmt_indices.min(), pmt_indices.max(), 100)
                charge_gauss = charge_std * np.exp(-0.5 * ((x_range - pmt_indices.mean()) / pmt_indices.std())**2)
                ax_npe_denorm.plot(x_range, charge_gauss, 'b-', linewidth=2, alpha=0.8, label=f'Scaled Gaussian (σ={charge_std:.1f})')
                ax_npe_denorm.legend()
            except:
                pass
            ax_npe_denorm.set_title(f't={t_value} NPE (Denormalized)')
            ax_npe_denorm.set_xlabel('PMT Index')
            ax_npe_denorm.set_ylabel('Charge (NPE)')
            
            # Show noise statistics (in original scale)
            noise_std = (x_sig_t_denorm - x0_denorm).std().item()
            signal_std = x0_denorm.std().item()
            snr = signal_std / (noise_std + 1e-8)
            ax_npe_denorm.text(0.02, 0.98, f'SNR={snr:.2f}\nσ_n={noise_std:.2f}', 
                               transform=ax_npe_denorm.transAxes, va='top', fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # ========== ROW 2: TIME NORMALIZED NOISY ==========
            ax_time_norm = axes[sample_idx * 4 + 2, t_idx + 1]
            time_noisy_norm = x_sig_t[0, 1, :].numpy()
            
            # Plot all normalized noisy Time including zeros
            pmt_indices = np.arange(len(time_noisy_norm))
            ax_time_norm.scatter(pmt_indices, time_noisy_norm, s=1, alpha=0.5, color='blue')
            
            # Add standard Gaussian envelope
            try:
                x_range = np.linspace(pmt_indices.min(), pmt_indices.max(), 100)
                time_gauss = np.exp(-0.5 * ((x_range - pmt_indices.mean()) / pmt_indices.std())**2)
                ax_time_norm.plot(x_range, time_gauss, 'r-', linewidth=2, alpha=0.8, label='Standard Gaussian (μ=0, σ=1)')
                ax_time_norm.legend()
            except:
                pass
            ax_time_norm.set_title(f't={t_value} Time (Normalized)')
            ax_time_norm.set_xlabel('PMT Index')
            ax_time_norm.set_ylabel('Time (normalized)')

            # ========== ROW 3: TIME DENORMALIZED NOISY ==========
            ax_time_denorm = axes[sample_idx * 4 + 3, t_idx + 1]
            time_noisy = x_sig_t_denorm[0, 1, :].numpy()
            
            # Plot all denormalized noisy Time including zeros
            pmt_indices = np.arange(len(time_noisy))
            ax_time_denorm.scatter(pmt_indices, time_noisy, s=1, alpha=0.5, color='red')
            
            # Add scaled Gaussian envelope
            try:
                time_std = time_noisy.std()
                x_range = np.linspace(pmt_indices.min(), pmt_indices.max(), 100)
                time_gauss = time_std * np.exp(-0.5 * ((x_range - pmt_indices.mean()) / pmt_indices.std())**2)
                ax_time_denorm.plot(x_range, time_gauss, 'b-', linewidth=2, alpha=0.8, label=f'Scaled Gaussian (σ={time_std:.1f})')
                ax_time_denorm.legend()
            except:
                pass
            ax_time_denorm.set_title(f't={t_value} Time (Denormalized)')
            ax_time_denorm.set_xlabel('PMT Index')
            ax_time_denorm.set_ylabel('Time (ns)')
    
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_dir = Path('visualizations')
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'diffusion_process_normalized_vs_denormalized.png'
    else:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'diffusion_process_normalized_vs_denormalized.png'
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_file}")
    print(f"📁 Full path: {output_file.absolute()}")
    print(f"📊 Figure size: {len(timesteps) + 1} columns × {num_samples * 4} rows")
    print("📊 Row 0: NPE Normalized (training space)")
    print("📊 Row 1: NPE Denormalized (physical units)")
    print("📊 Row 2: Time Normalized (training space)")
    print("📊 Row 3: Time Denormalized (physical units)")
    print(f"📊 Timesteps: {timesteps}")
    
    # Close figure to free memory
    plt.close()
    
    # Print statistics (in original scale)
    print("\n" + "="*70)
    print("📊 Signal Statistics (Original Scale):")
    print("="*70)
    
    # Denormalize all signals for statistics
    x_sig_denorm = denormalize_signal(
        x_sig, 
        config.model.affine_offsets, 
        config.model.affine_scales,
        time_transform=config.model.time_transform,
        channels="signal"
    )
    
    print(f"Original signal (denormalized):")
    print(f"  Charge range: [{x_sig_denorm[:, 0, :].min():.4f}, {x_sig_denorm[:, 0, :].max():.4f}]")
    print(f"  Charge mean±std: {x_sig_denorm[:, 0, :].mean():.4f}±{x_sig_denorm[:, 0, :].std():.4f}")
    print(f"  Time range: [{x_sig_denorm[:, 1, :].min():.4f}, {x_sig_denorm[:, 1, :].max():.4f}]")
    print(f"  Time mean±std: {x_sig_denorm[:, 1, :].mean():.4f}±{x_sig_denorm[:, 1, :].std():.4f}")
    
    print(f"\nNormalized signal (for training):")
    print(f"  Charge range: [{x_sig[:, 0, :].min():.4f}, {x_sig[:, 0, :].max():.4f}]")
    print(f"  Time range: [{x_sig[:, 1, :].min():.4f}, {x_sig[:, 1, :].max():.4f}]")
    
    print(f"\nLabel statistics:")
    for i, name in enumerate(['Energy', 'Zenith', 'Azimuth', 'X', 'Y', 'Z']):
        vals = label[:, i]
        print(f"  {name:8s}: [{vals.min():.4f}, {vals.max():.4f}], mean={vals.mean():.4f}")
    
    print(f"\nNormalization parameters:")
    print(f"  Charge: (x - {config.model.affine_offsets[0]}) / {config.model.affine_scales[0]}")
    print(f"  Time:   (x - {config.model.affine_offsets[1]}) / {config.model.affine_scales[1]}")
    print(f"  Positions: x / {config.model.affine_scales[2]}")
    
    print(f"\nDiffusion parameters:")
    print(f"  Timesteps: {config.diffusion.timesteps}")
    print(f"  Beta: [{config.diffusion.beta_start}, {config.diffusion.beta_end}]")
    print(f"  Objective: {config.diffusion.objective}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize diffusion process")
    parser.add_argument("-c", "--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("-n", "--num-samples", type=int, default=4,
                        help="Number of samples to visualize")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory path")
    
    args = parser.parse_args()
    
    visualize_diffusion_process(args.config, args.num_samples, args.output)

