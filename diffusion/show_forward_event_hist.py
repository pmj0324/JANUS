#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize diffusion process: normalized vs denormalized histograms
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
from typing import List

from config import load_config_from_file, get_default_config
from dataloader.pmt_dataloader import make_dataloader
from models.factory import ModelFactory
from utils.denormalization import denormalize_signal


def visualize_diffusion_histograms(config_path: str = None, num_samples: int = 4, output_path: str = None, timesteps: List[int] = None):
    """
    Visualize normalized and denormalized signal histograms at different timesteps.
    
    Args:
        config_path: Path to config file (if None, uses default)
        num_samples: Number of samples to visualize
        output_path: Output file path (if None, uses default)
        timesteps: List of timesteps to visualize (if None, uses default)
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
    
    # Timesteps to visualize (t=0 is original, then show noise steps from t=1 to t=T-1)
    if timesteps is None:
        timesteps = [0, 1, 250, 500, 999]  # Default timesteps
    else:
        # Ensure timesteps are within valid range (0 to T)
        T = diffusion.cfg.timesteps
        timesteps = [max(0, min(t, T)) for t in timesteps]
        timesteps = sorted(list(set(timesteps)))  # Remove duplicates and sort
    
    T = diffusion.cfg.timesteps
    print(f"Note: t=0 is x0 (original data), t=1 is x1 (first noise step), t={T} is xT (final timestep, completely noisy)")
    print(f"Visualizing timesteps: {timesteps}")
    
    # Create figure with 4 rows: NPE normalized, NPE denormalized, Time normalized, Time denormalized
    fig, axes = plt.subplots(4, len(timesteps) + 1, figsize=(20, 16))
    
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
        ax_npe_norm = axes[0, 0]
        charge_norm = x0_sig_single[0, 0, :].numpy()
        
        # Plot all charges including zeros
        mask_norm = charge_norm >= 0  # Include zeros
        if mask_norm.sum() > 1:
            try:
                n, bins, patches = ax_npe_norm.hist(charge_norm[mask_norm], bins=20, alpha=0.7, color='blue', density=True)
                
                # Add standard Gaussian (μ=0, σ=1) for normalized data
                x_gauss = np.linspace(charge_norm[mask_norm].min(), charge_norm[mask_norm].max(), 100)
                gauss_fit = stats.norm.pdf(x_gauss, 0, 1)
                ax_npe_norm.plot(x_gauss, gauss_fit, 'r-', linewidth=2, label='Standard Gaussian (μ=0, σ=1)')
                ax_npe_norm.legend()
            except:
                ax_npe_norm.text(0.5, 0.5, 'Histogram error', ha='center', va='center', transform=ax_npe_norm.transAxes)
        else:
            ax_npe_norm.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_npe_norm.transAxes)
        ax_npe_norm.set_title(f'Sample {sample_idx}: NPE Original (Normalized)')
        ax_npe_norm.set_xlabel('Charge (normalized)')
        ax_npe_norm.set_ylabel('Density')
        
        # Add statistics
        mean_norm = charge_norm[mask_norm].mean()
        std_norm = charge_norm[mask_norm].std()
        ax_npe_norm.axvline(mean_norm, color='red', linestyle='--', alpha=0.8, label=f'μ={mean_norm:.3f}')
        ax_npe_norm.legend()
        
        # ========== ROW 1: NPE DENORMALIZED ==========
        # Plot original NPE signal (denormalized)
        ax_npe_denorm = axes[1, 0]
        charge = x0_denorm[0, 0, :].numpy()
        
        # Plot all charges including zeros
        mask = charge >= 0  # Include zeros
        if mask.sum() > 1:
            try:
                n, bins, patches = ax_npe_denorm.hist(charge[mask], bins=20, alpha=0.7, color='red', density=True)
                
                # Add scaled Gaussian for denormalized data
                # Scale standard Gaussian (μ=0, σ=1) to match data scale
                data_std = charge[mask].std()
                x_gauss = np.linspace(charge[mask].min(), charge[mask].max(), 100)
                gauss_fit = stats.norm.pdf(x_gauss, 0, data_std)
                ax_npe_denorm.plot(x_gauss, gauss_fit, 'b-', linewidth=2, label=f'Scaled Gaussian (σ={data_std:.1f})')
                ax_npe_denorm.legend()
            except:
                ax_npe_denorm.text(0.5, 0.5, 'Histogram error', ha='center', va='center', transform=ax_npe_denorm.transAxes)
        else:
            ax_npe_denorm.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_npe_denorm.transAxes)
        ax_npe_denorm.set_title(f'Sample {sample_idx}: NPE Original (Denormalized)')
        ax_npe_denorm.set_xlabel('Charge (NPE)')
        ax_npe_denorm.set_ylabel('Density')
        
        # Add statistics
        mean_denorm = charge[mask].mean()
        std_denorm = charge[mask].std()
        ax_npe_denorm.axvline(mean_denorm, color='blue', linestyle='--', alpha=0.8, label=f'μ={mean_denorm:.1f}')
        ax_npe_denorm.legend()
        
        # ========== ROW 2: TIME NORMALIZED ==========
        # Plot original Time signal (normalized)
        ax_time_norm = axes[2, 0]
        time_norm = x0_sig_single[0, 1, :].numpy()
        
        # Plot all times including zeros
        mask_time_norm = time_norm >= 0  # Include zeros
        if mask_time_norm.sum() > 1:
            try:
                n, bins, patches = ax_time_norm.hist(time_norm[mask_time_norm], bins=20, alpha=0.7, color='blue', density=True)
                
                # Add standard Gaussian (μ=0, σ=1) for normalized data
                x_gauss = np.linspace(time_norm[mask_time_norm].min(), time_norm[mask_time_norm].max(), 100)
                gauss_fit = stats.norm.pdf(x_gauss, 0, 1)
                ax_time_norm.plot(x_gauss, gauss_fit, 'r-', linewidth=2, label='Standard Gaussian (μ=0, σ=1)')
                ax_time_norm.legend()
            except:
                ax_time_norm.text(0.5, 0.5, 'Histogram error', ha='center', va='center', transform=ax_time_norm.transAxes)
        else:
            ax_time_norm.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_time_norm.transAxes)
        ax_time_norm.set_title(f'Sample {sample_idx}: Time Original (Normalized)')
        ax_time_norm.set_xlabel('Time (normalized)')
        ax_time_norm.set_ylabel('Density')
        
        # Add statistics
        mean_time_norm = time_norm[mask_time_norm].mean()
        std_time_norm = time_norm[mask_time_norm].std()
        ax_time_norm.axvline(mean_time_norm, color='red', linestyle='--', alpha=0.8, label=f'μ={mean_time_norm:.3f}')
        ax_time_norm.legend()
        
        # ========== ROW 3: TIME DENORMALIZED ==========
        # Plot original Time signal (denormalized)
        ax_time_denorm = axes[3, 0]
        time = x0_denorm[0, 1, :].numpy()
        
        # Plot all times including zeros
        mask_time = time >= 0  # Include zeros
        if mask_time.sum() > 1:
            try:
                n, bins, patches = ax_time_denorm.hist(time[mask_time], bins=20, alpha=0.7, color='red', density=True)
                
                # Add scaled Gaussian for denormalized data
                # Scale standard Gaussian (μ=0, σ=1) to match data scale
                data_std = time[mask_time].std()
                x_gauss = np.linspace(time[mask_time].min(), time[mask_time].max(), 100)
                gauss_fit = stats.norm.pdf(x_gauss, 0, data_std)
                ax_time_denorm.plot(x_gauss, gauss_fit, 'b-', linewidth=2, label=f'Scaled Gaussian (σ={data_std:.1f})')
                ax_time_denorm.legend()
            except:
                ax_time_denorm.text(0.5, 0.5, 'Histogram error', ha='center', va='center', transform=ax_time_denorm.transAxes)
        else:
            ax_time_denorm.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_time_denorm.transAxes)
        ax_time_denorm.set_title(f'Sample {sample_idx}: Time Original (Denormalized)')
        ax_time_denorm.set_xlabel('Time (ns)')
        ax_time_denorm.set_ylabel('Density')
        
        # Add statistics
        mean_time_denorm = time[mask_time].mean()
        std_time_denorm = time[mask_time].std()
        ax_time_denorm.axvline(mean_time_denorm, color='blue', linestyle='--', alpha=0.8, label=f'μ={mean_time_denorm:.1e}')
        ax_time_denorm.legend()
        
        # Add label info to first subplot
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
            ax_npe_norm = axes[0, t_idx + 1]
            charge_noisy_norm = x_sig_t[0, 0, :].numpy()
            
            # Plot all normalized noisy NPE including zeros
            if mask_norm.sum() > 1:
                try:
                    n, bins, patches = ax_npe_norm.hist(charge_noisy_norm[mask_norm], bins=20, alpha=0.7, color='blue', density=True)
                    
                    # Add standard Gaussian (μ=0, σ=1) for normalized data
                    x_gauss = np.linspace(charge_noisy_norm[mask_norm].min(), charge_noisy_norm[mask_norm].max(), 100)
                    gauss_fit = stats.norm.pdf(x_gauss, 0, 1)
                    ax_npe_norm.plot(x_gauss, gauss_fit, 'r-', linewidth=2, label='Standard Gaussian (μ=0, σ=1)')
                    ax_npe_norm.legend()
                except:
                    ax_npe_norm.text(0.5, 0.5, 'Histogram error', ha='center', va='center', transform=ax_npe_norm.transAxes)
            else:
                ax_npe_norm.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_npe_norm.transAxes)
            ax_npe_norm.set_title(f't={t_value} NPE (Normalized)')
            ax_npe_norm.set_xlabel('Charge (normalized)')
            ax_npe_norm.set_ylabel('Density')
            
            # Show noise statistics (in normalized scale)
            noise_std_norm = (x_sig_t - x0_sig_single)[0, 0, mask_norm].std().item()
            signal_std_norm = x0_sig_single[0, 0, mask_norm].std().item()
            snr_norm = signal_std_norm / (noise_std_norm + 1e-8)
            ax_npe_norm.text(0.02, 0.98, f'SNR={snr_norm:.2f}\nσ_n={noise_std_norm:.3f}', 
                             transform=ax_npe_norm.transAxes, va='top', fontsize=8,
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            # ========== ROW 1: NPE DENORMALIZED NOISY ==========
            ax_npe_denorm = axes[1, t_idx + 1]
            charge_noisy = x_sig_t_denorm[0, 0, :].numpy()
            
            # Plot denormalized noisy NPE
            if mask.sum() > 1:
                try:
                    n, bins, patches = ax_npe_denorm.hist(charge_noisy[mask], bins=20, alpha=0.7, color='red', density=True)
                    
                    # Add scaled Gaussian for denormalized data
                    data_std = charge_noisy[mask].std()
                    x_gauss = np.linspace(charge_noisy[mask].min(), charge_noisy[mask].max(), 100)
                    gauss_fit = stats.norm.pdf(x_gauss, 0, data_std)
                    ax_npe_denorm.plot(x_gauss, gauss_fit, 'b-', linewidth=2, label=f'Scaled Gaussian (σ={data_std:.1f})')
                    ax_npe_denorm.legend()
                except:
                    ax_npe_denorm.text(0.5, 0.5, 'Histogram error', ha='center', va='center', transform=ax_npe_denorm.transAxes)
            else:
                ax_npe_denorm.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_npe_denorm.transAxes)
            ax_npe_denorm.set_title(f't={t_value} NPE (Denormalized)')
            ax_npe_denorm.set_xlabel('Charge (NPE)')
            ax_npe_denorm.set_ylabel('Density')
            
            # Show noise statistics (in original scale)
            noise_std = (x_sig_t_denorm - x0_denorm)[0, 0, mask].std().item()
            signal_std = x0_denorm[0, 0, mask].std().item()
            snr = signal_std / (noise_std + 1e-8)
            ax_npe_denorm.text(0.02, 0.98, f'SNR={snr:.2f}\nσ_n={noise_std:.2f}', 
                               transform=ax_npe_denorm.transAxes, va='top', fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # ========== ROW 2: TIME NORMALIZED NOISY ==========
            ax_time_norm = axes[2, t_idx + 1]
            time_noisy_norm = x_sig_t[0, 1, :].numpy()
            
            # Plot normalized noisy Time
            if mask_time_norm.sum() > 1:
                try:
                    n, bins, patches = ax_time_norm.hist(time_noisy_norm[mask_time_norm], bins=20, alpha=0.7, color='blue', density=True)
                    
                    # Add standard Gaussian (μ=0, σ=1) for normalized data
                    x_gauss = np.linspace(time_noisy_norm[mask_time_norm].min(), time_noisy_norm[mask_time_norm].max(), 100)
                    gauss_fit = stats.norm.pdf(x_gauss, 0, 1)
                    ax_time_norm.plot(x_gauss, gauss_fit, 'r-', linewidth=2, label='Standard Gaussian (μ=0, σ=1)')
                    ax_time_norm.legend()
                except:
                    ax_time_norm.text(0.5, 0.5, 'Histogram error', ha='center', va='center', transform=ax_time_norm.transAxes)
            else:
                ax_time_norm.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_time_norm.transAxes)
            ax_time_norm.set_title(f't={t_value} Time (Normalized)')
            ax_time_norm.set_xlabel('Time (normalized)')
            ax_time_norm.set_ylabel('Density')
            
            # Show noise statistics (in normalized scale)
            noise_std_time_norm = (x_sig_t - x0_sig_single)[0, 1, mask_time_norm].std().item()
            signal_std_time_norm = x0_sig_single[0, 1, mask_time_norm].std().item()
            snr_time_norm = signal_std_time_norm / (noise_std_time_norm + 1e-8)
            ax_time_norm.text(0.02, 0.98, f'SNR={snr_time_norm:.2f}\nσ_n={noise_std_time_norm:.3f}', 
                              transform=ax_time_norm.transAxes, va='top', fontsize=8,
                              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            # ========== ROW 3: TIME DENORMALIZED NOISY ==========
            ax_time_denorm = axes[3, t_idx + 1]
            time_noisy = x_sig_t_denorm[0, 1, :].numpy()
            
            # Plot denormalized noisy Time
            if mask_time.sum() > 1:
                try:
                    n, bins, patches = ax_time_denorm.hist(time_noisy[mask_time], bins=20, alpha=0.7, color='red', density=True)
                    
                    # Add scaled Gaussian for denormalized data
                    data_std = time_noisy[mask_time].std()
                    x_gauss = np.linspace(time_noisy[mask_time].min(), time_noisy[mask_time].max(), 100)
                    gauss_fit = stats.norm.pdf(x_gauss, 0, data_std)
                    ax_time_denorm.plot(x_gauss, gauss_fit, 'b-', linewidth=2, label=f'Scaled Gaussian (σ={data_std:.1f})')
                    ax_time_denorm.legend()
                except:
                    ax_time_denorm.text(0.5, 0.5, 'Histogram error', ha='center', va='center', transform=ax_time_denorm.transAxes)
            else:
                ax_time_denorm.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_time_denorm.transAxes)
            ax_time_denorm.set_title(f't={t_value} Time (Denormalized)')
            ax_time_denorm.set_xlabel('Time (ns)')
            ax_time_denorm.set_ylabel('Density')
            
            # Show noise statistics (in original scale)
            noise_std_time = (x_sig_t_denorm - x0_denorm)[0, 1, mask_time].std().item()
            signal_std_time = x0_denorm[0, 1, mask_time].std().item()
            snr_time = signal_std_time / (noise_std_time + 1e-8)
            ax_time_denorm.text(0.02, 0.98, f'SNR={snr_time:.2f}\nσ_n={noise_std_time:.2e}', 
                                transform=ax_time_denorm.transAxes, va='top', fontsize=8,
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_dir = Path('visualizations')
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / 'diffusion_process_histograms_normalized_vs_denormalized.png'
    else:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'diffusion_process_histograms_normalized_vs_denormalized.png'
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Histogram visualization saved to: {output_file}")
    print(f"📁 Full path: {output_file.absolute()}")
    print(f"📊 Figure size: {len(timesteps) + 1} columns × 4 rows")
    print("📊 Row 0: NPE Normalized (training space)")
    print("📊 Row 1: NPE Denormalized (physical units)")
    print("📊 Row 2: Time Normalized (training space)")
    print("📊 Row 3: Time Denormalized (physical units)")
    print(f"📊 Timesteps: {timesteps}")
    
    # Close figure to free memory
    plt.close()
    
    # Print detailed denormalization information
    print("\n" + "="*80)
    print("🔄 DENORMALIZATION PROCESS EXPLAINED:")
    print("="*80)
    
    print("\n1️⃣ TIME DENORMALIZATION:")
    print(f"   Original time transform: {config.model.time_transform}")
    if config.model.time_transform == "ln":
        print("   Formula: time_raw = exp(time_norm * scale + offset) - 1")
        # Use actual data example
        actual_time_max = x_sig[:, 1].max().item()
        actual_time_raw = torch.exp(torch.tensor(actual_time_max * config.model.affine_scales[1] + config.model.affine_offsets[1])) - 1
        print(f"   Example: exp({actual_time_max:.2f} * {config.model.affine_scales[1]} + {config.model.affine_offsets[1]}) - 1 ≈ {actual_time_raw:.1f} ns")
    elif config.model.time_transform == "log10":
        print("   Formula: time_raw = 10^(time_norm * scale + offset) - 1")
        # Use actual data example
        actual_time_max = x_sig[:, 1].max().item()
        actual_time_raw = 10**(actual_time_max * config.model.affine_scales[1] + config.model.affine_offsets[1]) - 1
        print(f"   Example: 10^({actual_time_max:.2f} * {config.model.affine_scales[1]} + {config.model.affine_offsets[1]}) - 1 ≈ {actual_time_raw:.1f} ns")
    
    print(f"\n2️⃣ CHARGE DENORMALIZATION:")
    print("   Formula: charge_raw = charge_norm * scale + offset")
    print(f"   Parameters: scale={config.model.affine_scales[0]}, offset={config.model.affine_offsets[0]}")
    # Use actual data example
    actual_charge_max = x_sig[:, 0].max().item()
    actual_charge_raw = actual_charge_max * config.model.affine_scales[0] + config.model.affine_offsets[0]
    print(f"   Example: {actual_charge_max:.2f} * {config.model.affine_scales[0]} + {config.model.affine_offsets[0]} = {actual_charge_raw:.1f} NPE")
    
    print(f"\n3️⃣ GEOMETRY DENORMALIZATION:")
    print("   Formula: pos_raw = pos_norm * scale + offset")
    print(f"   Parameters: scale={config.model.affine_scales[2]}, offset={config.model.affine_offsets[2]}")
    # Use actual geometry example
    actual_geom_max = geom[:, 0].max().item()  # X coordinate
    actual_geom_raw = actual_geom_max * config.model.affine_scales[2] + config.model.affine_offsets[2]
    print(f"   Example: {actual_geom_max:.2f} * {config.model.affine_scales[2]} + {config.model.affine_offsets[2]} = {actual_geom_raw:.1f} m")
    
    print(f"\n4️⃣ LABEL DENORMALIZATION:")
    print("   Formula: label_raw = label_norm * scale + offset")
    for i, name in enumerate(['Energy', 'Zenith', 'Azimuth', 'X', 'Y', 'Z']):
        scale = config.model.label_scales[i]
        offset = config.model.label_offsets[i]
        print(f"   {name:8s}: {name}_raw = {name}_norm * {scale} + {offset}")
    
    # Print statistics (in original scale)
    print("\n" + "="*80)
    print("📊 SIGNAL STATISTICS:")
    print("="*80)
    
    # Denormalize all signals for statistics
    x_sig_denorm = denormalize_signal(
        x_sig, 
        config.model.affine_offsets, 
        config.model.affine_scales,
        time_transform=config.model.time_transform,
        channels="signal"
    )
    
    print(f"Original signal (denormalized):")
    print(f"  Charge range: [{x_sig_denorm[:, 0, :].min():.4f}, {x_sig_denorm[:, 0, :].max():.4f}] NPE")
    print(f"  Charge mean±std: {x_sig_denorm[:, 0, :].mean():.4f}±{x_sig_denorm[:, 0, :].std():.4f} NPE")
    print(f"  Time range: [{x_sig_denorm[:, 1, :].min():.4f}, {x_sig_denorm[:, 1, :].max():.4f}] ns")
    print(f"  Time mean±std: {x_sig_denorm[:, 1, :].mean():.4f}±{x_sig_denorm[:, 1, :].std():.4f} ns")
    
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
    print(f"  Schedule: {config.diffusion.schedule}")
    if hasattr(config.diffusion, 'cosine_s'):
        print(f"  Cosine_s: {config.diffusion.cosine_s}")
    print(f"  Beta: [{config.diffusion.beta_start}, {config.diffusion.beta_end}]")
    print(f"  Objective: {config.diffusion.objective}")
    
    # Verify noise schedule
    print(f"\n🔍 NOISE SCHEDULE VERIFICATION:")
    print(f"  Actual beta range: [{diffusion.betas.min():.6f}, {diffusion.betas.max():.6f}]")
    print(f"  First 5 betas: {diffusion.betas[:5].tolist()}")
    print(f"  Last 5 betas: {diffusion.betas[-5:].tolist()}")
    
    # Test q_sample with different timesteps
    print(f"\n🧪 Q_SAMPLE TEST:")
    x0_test = torch.randn(1, 2, 5160)
    test_timesteps = [0, 1, 100, 500, 999]
    for t in test_timesteps:
        if t == 0:
            print(f"  t={t} (x0): No noise added (original data)")
            continue
        t_tensor = torch.tensor([t], dtype=torch.long)
        x_t = diffusion.q_sample(x0_test, t_tensor)
        noise_added = x_t - x0_test
        noise_std = noise_added.std().item()
        theoretical_std = diffusion.sqrt_one_minus_alphas_cumprod[t].item()
        print(f"  t={t}: noise_std={noise_std:.4f}, theoretical_std={theoretical_std:.4f}")
    
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize diffusion process with histograms")
    parser.add_argument("-c", "--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("-n", "--num-samples", type=int, default=4,
                        help="Number of samples to visualize")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory path")
    parser.add_argument("-t", "--timesteps", type=int, nargs="+", default=None,
                        help="Timesteps to visualize (e.g., -t 0 1 100 500 999)")
    
    args = parser.parse_args()
    visualize_diffusion_histograms(args.config, args.num_samples, args.output, args.timesteps)
