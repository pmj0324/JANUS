#!/usr/bin/env python3
"""
Diffusion Process Visualization
================================

Visualize forward and reverse diffusion processes step-by-step.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from utils.denormalization import denormalize_signal


def visualize_diffusion_steps(
    diffusion,
    real_x_sig: torch.Tensor,
    real_geom: torch.Tensor,
    real_label: torch.Tensor,
    num_steps: int = 200,
    save_path: Optional[str] = None,
    # Optional: also output NPZ-style 3D plots (per-timestep) using detector CSV
    generate_npz_3d: bool = False,
    detector_csv: str = "csv/detector_geometry.csv",
    npz_out_dir: Optional[str] = None,
    # If True, save NPZ/PNG for every timestep (forward + reverse)
    npz_all_timesteps: bool = False
):
    """
    Visualize forward and reverse diffusion process step-by-step for one sample.
    
    Args:
        diffusion: Trained diffusion model
        real_x_sig: Real signal (1, 2, L) - NORMALIZED
        real_geom: Real geometry (1, 3, L) - NORMALIZED
        real_label: Real label (1, 6) - NORMALIZED
        num_steps: Number of steps to visualize
        save_path: Path to save the visualization
    """
    device = next(diffusion.parameters()).device
    
    # Get normalization parameters from model
    norm_params = diffusion.model.get_normalization_params()
    affine_offsets = norm_params['affine_offsets'][:2]  # [charge, time]
    affine_scales = norm_params['affine_scales'][:2]
    time_transform = norm_params['time_transform']
    
    print("\n" + "="*70)
    print("🎬 Diffusion Process Visualization - Single Sample")
    print("="*70)
    
    # Move to device
    real_x_sig = real_x_sig.to(device)
    real_geom = real_geom.to(device)
    real_label = real_label.to(device)
    
    # =========================================================================
    # PART 1: Forward Diffusion (Original → Noise)
    # =========================================================================
    print("\n📤 Forward Diffusion: Original → Noise")
    print("-"*70)
    
    total_timesteps = diffusion.cfg.timesteps
    step_indices = np.linspace(0, total_timesteps - 1, num_steps, dtype=int)
    
    forward_samples = []
    forward_samples.append(real_x_sig.clone())  # t=0 (original, normalized)
    
    # Prepare NPZ helper if requested (forward)
    if generate_npz_3d:
        try:
            from utils.event_visualization.event_show import show_event_from_npz  # local import
            base_out_dir = Path(npz_out_dir) if npz_out_dir else (Path(save_path).parent if save_path else Path("outputs/plots"))
            npz_forward_dir = base_out_dir / "npz_per_timestep" / "forward"
            npz_reverse_dir = base_out_dir / "npz_per_timestep" / "reverse"
            npz_forward_dir.mkdir(parents=True, exist_ok=True)
            npz_reverse_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If visualization cannot be prepared, disable NPZ generation
            generate_npz_3d = False

    def _save_npz_event(sample_denorm: np.ndarray, step_name: str, out_dir: Path):
        npz_path = out_dir / f"event_{step_name}.npz"
        np.savez(npz_path, input=sample_denorm, label=np.zeros(6, dtype=np.float32), info=np.zeros(6, dtype=np.float32))
        png_path = out_dir / f"event_{step_name}.png"
        try:
            show_event_from_npz(npz_path=str(npz_path), detector_csv=detector_csv, output_path=str(png_path), figure_size=(12, 8))
            print(f"  ✅ NPZ plot saved: {png_path}")
        except Exception as e:
            print(f"  ⚠️  NPZ plot failed ({step_name}): {e}")

    for i, t in enumerate(step_indices[1:], 1):
        # Add noise according to forward process
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        
        # Get noise schedule values
        sqrt_alphas_cumprod = diffusion._extract(diffusion.sqrt_alphas_cumprod, t_tensor, real_x_sig.shape)
        sqrt_one_minus_alphas_cumprod = diffusion._extract(
            diffusion.sqrt_one_minus_alphas_cumprod, t_tensor, real_x_sig.shape
        )
        
        # Forward process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        noise = torch.randn_like(real_x_sig)
        x_t = sqrt_alphas_cumprod * real_x_sig + sqrt_one_minus_alphas_cumprod * noise
        
        forward_samples.append(x_t)
        
        # Optionally save every timestep for forward process
        if generate_npz_3d and npz_all_timesteps:
            den = denorm_sample(x_t)
            _save_npz_event(den, f"forward_t{int(t)}", npz_forward_dir)
        
        # Print statistics (normalized space)
        charge_mean = x_t[0, 0, :].mean().item()
        charge_std = x_t[0, 0, :].std().item()
        time_mean = x_t[0, 1, :].mean().item()
        time_std = x_t[0, 1, :].std().item()
        
        print(f"  Step {i:2d} (t={t:4d}): "
              f"Charge [{x_t[0, 0, :].min():.4f}, {x_t[0, 0, :].max():.4f}] "
              f"mean={charge_mean:.4f} std={charge_std:.4f} | "
              f"Time [{x_t[0, 1, :].min():.4f}, {x_t[0, 1, :].max():.4f}] "
              f"mean={time_mean:.4f} std={time_std:.4f}")
    
    # =========================================================================
    # PART 2: Reverse Diffusion (Noise → Reconstructed)
    # =========================================================================
    print("\n📥 Reverse Diffusion: Noise → Reconstructed")
    print("-"*70)
    
    # Start from pure noise
    shape = real_x_sig.shape
    x = torch.randn(shape, device=device)
    
    reverse_samples = []
    reverse_t_values = list(reversed(range(0, total_timesteps)))
    reverse_step_indices = [reverse_t_values[i] for i in np.linspace(0, len(reverse_t_values) - 1, num_steps, dtype=int)]
    
    # Track which steps to save
    steps_to_save = set(reverse_step_indices)
    save_counter = 0
    
    for t in reversed(range(0, total_timesteps)):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
        
        # Model prediction (with CFG if enabled)
        if diffusion.use_cfg:
            # Conditional prediction
            with torch.no_grad():
                eps_cond = diffusion.model(x, t_tensor, real_label, real_geom)
            
            # Unconditional prediction (zero labels)
            uncond_label = torch.zeros_like(real_label)
            with torch.no_grad():
                eps_uncond = diffusion.model(x, t_tensor, uncond_label, real_geom)
            
            # Classifier-free guidance
            eps = eps_uncond + diffusion.cfg_scale * (eps_cond - eps_uncond)
        else:
            with torch.no_grad():
                eps = diffusion.model(x, t_tensor, real_label, real_geom)
        
        # DDPM reverse step
        betas_t = diffusion._extract(diffusion.betas, t_tensor, x.shape)
        sqrt_one_minus_alphas_cumprod_t = diffusion._extract(
            diffusion.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape
        )
        sqrt_recip_alphas_t = diffusion._extract(diffusion.sqrt_recip_alphas, t_tensor, x.shape)
        
        # Mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t > 0:
            posterior_variance_t = diffusion._extract(diffusion.posterior_variance, t_tensor, x.shape)
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            x = model_mean
        
        # Save at specific steps
        if t in steps_to_save:
            reverse_samples.append(x.clone())
            
            # Print statistics (still normalized)
            charge_mean = x[0, 0, :].mean().item()
            charge_std = x[0, 0, :].std().item()
            time_mean = x[0, 1, :].mean().item()
            time_std = x[0, 1, :].std().item()
            
            print(f"  Step {save_counter:2d} (t={t:4d}): "
                  f"Charge [{x[0, 0, :].min():.4f}, {x[0, 0, :].max():.4f}] "
                  f"mean={charge_mean:.4f} std={charge_std:.4f} | "
                  f"Time [{x[0, 1, :].min():.4f}, {x[0, 1, :].max():.4f}] "
                  f"mean={time_mean:.4f} std={time_mean:.4f}")
            save_counter += 1
        
        # Optionally save every timestep for reverse process
        if generate_npz_3d and npz_all_timesteps:
            den = denorm_sample(x)
            _save_npz_event(den, f"reverse_t{int(t)}", npz_reverse_dir)
    
    # =========================================================================
    # PART 3: Denormalization & Visualization
    # =========================================================================
    print("\n🎨 Creating Visualization...")
    print("-"*70)
    
    # Denormalize samples
    def denorm_sample(x_norm):
        """Denormalize: (x_norm * scale) + offset, then inverse time transform"""
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
        
        return x
    
    # Denormalize all samples
    forward_denorm = [denorm_sample(s) for s in forward_samples]
    reverse_denorm = [denorm_sample(s) for s in reverse_samples]
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Calculate PMT-wise statistics
    def calc_stats(sample):
        """Calculate statistics for visualization"""
        charge = sample[0, :]
        time = sample[1, :]
        
        # Only non-zero charges
        nonzero_mask = charge > 0
        charge_nonzero = charge[nonzero_mask] if nonzero_mask.any() else np.array([0])
        time_nonzero = time[nonzero_mask] if nonzero_mask.any() else np.array([0])
        
        return {
            'charge_mean': charge_nonzero.mean(),
            'charge_max': charge_nonzero.max(),
            'time_mean': time_nonzero.mean(),
            'time_max': time_nonzero.max(),
            'n_hit': nonzero_mask.sum()
        }
    
    # Forward diffusion plots
    for i, sample in enumerate(forward_denorm):
        stats = calc_stats(sample)
        
        # Charge
        ax1 = plt.subplot(4, num_steps, i + 1)
        charge = sample[0, :]
        ax1.hist(charge[charge > 0], bins=50, alpha=0.7, color='blue')
        ax1.set_title(f'Forward t={step_indices[i] if i > 0 else 0}\nHits={stats["n_hit"]}', fontsize=8)
        ax1.set_xlabel('Charge (NPE)', fontsize=7)
        ax1.set_ylabel('Count', fontsize=7)
        ax1.tick_params(labelsize=6)
        ax1.set_yscale('log')
        
        # Time
        ax2 = plt.subplot(4, num_steps, num_steps + i + 1)
        time = sample[1, :]
        ax2.hist(time[charge > 0], bins=50, alpha=0.7, color='green')
        ax2.set_xlabel('Time (ns)', fontsize=7)
        ax2.set_ylabel('Count', fontsize=7)
        ax2.tick_params(labelsize=6)
        ax2.set_yscale('log')
    
    # Reverse diffusion plots
    for i, sample in enumerate(reverse_denorm):
        stats = calc_stats(sample)
        
        # Charge
        ax3 = plt.subplot(4, num_steps, 2*num_steps + i + 1)
        charge = sample[0, :]
        ax3.hist(charge[charge > 0], bins=50, alpha=0.7, color='red')
        ax3.set_title(f'Reverse t={reverse_step_indices[i]}\nHits={stats["n_hit"]}', fontsize=8)
        ax3.set_xlabel('Charge (NPE)', fontsize=7)
        ax3.set_ylabel('Count', fontsize=7)
        ax3.tick_params(labelsize=6)
        ax3.set_yscale('log')
        
        # Time
        ax4 = plt.subplot(4, num_steps, 3*num_steps + i + 1)
        time = sample[1, :]
        ax4.hist(time[charge > 0], bins=50, alpha=0.7, color='orange')
        ax4.set_xlabel('Time (ns)', fontsize=7)
        ax4.set_ylabel('Count', fontsize=7)
        ax4.tick_params(labelsize=6)
        ax4.set_yscale('log')
    
    plt.suptitle('Diffusion Process Visualization: Forward (Top 2 rows) vs Reverse (Bottom 2 rows)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {save_path}")
    
    plt.close()
    
    print("="*70)
    print("✅ Diffusion process visualization complete!")
    print("="*70 + "\n")
    
    # =========================================================================
    # PART 4 (Optional): NPZ-style 3D visualization per timestep (from CSV)
    # =========================================================================
    if generate_npz_3d and not npz_all_timesteps:
        try:
            from utils.event_visualization.event_show import show_event_from_npz
            print("\n🧩 Creating NPZ-style 3D visualizations (per selected timestep)...")
            print("-"*70)
            
            # Output directory
            base_out_dir = Path(npz_out_dir) if npz_out_dir else (Path(save_path).parent if save_path else Path("outputs/plots"))
            out_dir = base_out_dir / "npz_per_timestep"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Helper to save one event
            def save_npz_event(sample_denorm: np.ndarray, step_name: str):
                npz_path = out_dir / f"event_{step_name}.npz"
                np.savez(
                    npz_path,
                    input=sample_denorm,  # (2, L)
                    label=np.zeros(6, dtype=np.float32),
                    info=np.zeros(6, dtype=np.float32)
                )
                png_path = out_dir / f"event_{step_name}.png"
                try:
                    show_event_from_npz(
                        npz_path=str(npz_path),
                        detector_csv=detector_csv,
                        output_path=str(png_path),
                        figure_size=(12, 8)
                    )
                    print(f"  ✅ Saved 3D NPZ plot: {png_path}")
                except Exception as e:
                    print(f"  ⚠️  Failed NPZ 3D plot ({step_name}): {e}")
            
            # Forward selected timesteps (first = original)
            for i, sample in enumerate(forward_denorm):
                step_name = f"forward_t{int(step_indices[i]) if i > 0 else 0}"
                save_npz_event(sample, step_name)
            
            # Reverse selected timesteps
            for i, sample in enumerate(reverse_denorm):
                step_name = f"reverse_t{int(reverse_step_indices[i])}"
                save_npz_event(sample, step_name)
            
            print(f"🗂️  NPZ outputs: {out_dir}")
        except Exception as e:
            print(f"⚠️  NPZ per-timestep visualization failed: {e}")
    
    return forward_denorm, reverse_denorm

