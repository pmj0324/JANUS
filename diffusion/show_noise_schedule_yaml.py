#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YAML-based Noise Schedule Visualizer
====================================

Visualize noise schedules based on YAML configuration files.
Shows only the configured schedule (no comparisons).

Author: Minje Park
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from config import load_config_from_file
from diffusion.noise_schedules import (
    linear_beta_schedule, 
    cosine_beta_schedule, 
    quadratic_beta_schedule, 
    sigmoid_beta_schedule
)


def compute_alpha_schedule(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute alpha values from beta schedule."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return {
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod)
    }


class NoiseScheduleVisualizer:
    """Visualize noise schedules from YAML configuration."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = load_config_from_file(config_path)
        
        # Extract diffusion parameters
        self.timesteps = self.config.diffusion.timesteps
        self.beta_start = getattr(self.config.diffusion, 'beta_start', 1e-4)
        self.beta_end = getattr(self.config.diffusion, 'beta_end', 2e-2)
        self.schedule_type = getattr(self.config.diffusion, 'schedule', 'linear')
        
        # Initialize storage
        self.schedules = {}
        self.alpha_schedules = {}
        self.figsize = (15, 12)
        
        print(f"📂 Loaded configuration from: {config_path}")
        print(f"   Timesteps: {self.timesteps}")
        print(f"   Beta range: [{self.beta_start}, {self.beta_end}]")
        print(f"   Schedule type: {self.schedule_type}")
    
    def generate_schedules_from_config(self) -> Dict[str, torch.Tensor]:
        """Generate noise schedule based on YAML configuration."""
        # Generate configured schedule
        if self.schedule_type.lower() == "linear":
            configured_betas = linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
            schedule_name = f'{self.schedule_type.title()} (Configured)'
        elif self.schedule_type.lower() == "cosine":
            cosine_s = getattr(self.config.diffusion, 'cosine_s', 0.008)
            configured_betas = cosine_beta_schedule(self.timesteps, cosine_s)
            schedule_name = f'{self.schedule_type.title()} (Configured)'
        elif self.schedule_type.lower() == "quadratic":
            configured_betas = quadratic_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
            schedule_name = f'{self.schedule_type.title()} (Configured)'
        elif self.schedule_type.lower() == "sigmoid":
            configured_betas = sigmoid_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
            schedule_name = f'{self.schedule_type.title()} (Configured)'
        else:
            # Default to linear if unknown
            configured_betas = linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
            schedule_name = f'{self.schedule_type.title()} (Configured)'
        
        # Store only the configured schedule
        self.schedules = {schedule_name: configured_betas}
        
        # Compute alpha schedules
        for name, betas in self.schedules.items():
            self.alpha_schedules[name] = compute_alpha_schedule(betas)
            
        return self.schedules
    
    def plot_beta_schedules(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot configured beta schedule.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.schedules:
            self.generate_schedules_from_config()
            
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Get the configured schedule
        schedule_name, betas = next(iter(self.schedules.items()))
        
        # Main beta plot
        ax = axes[0, 0]
        ax.plot(range(self.timesteps), betas, color='red', linewidth=3, label=schedule_name)
        ax.set_title(f'Beta Schedule: {schedule_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('β_t')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Zoomed beta plot (first 100 timesteps)
        ax = axes[0, 1]
        ax.plot(range(100), betas[:100], color='red', linewidth=3, label=schedule_name)
        ax.set_title('Beta Schedule (First 100 Steps)', fontsize=14)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('β_t')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Beta distribution histogram
        ax = axes[1, 0]
        ax.hist(betas.numpy(), bins=50, alpha=0.8, color='red', label=schedule_name)
        ax.set_title('Beta Value Distribution', fontsize=14)
        ax.set_xlabel('β_t Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Beta statistics
        ax = axes[1, 1]
        stats = {
            'min': betas.min().item(),
            'max': betas.max().item(),
            'mean': betas.mean().item(),
            'std': betas.std().item()
        }
        
        categories = ['Min', 'Max', 'Mean', 'Std']
        values = [stats['min'], stats['max'], stats['mean'], stats['std']]
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        ax.set_title('Beta Statistics', fontsize=14)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{value:.6f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Beta schedule plot saved to: {save_path}")
            
        return fig
    
    def plot_alpha_schedules(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot configured alpha schedule.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.alpha_schedules:
            self.generate_schedules_from_config()
            
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Get the configured schedule
        schedule_name, alphas = next(iter(self.alpha_schedules.items()))
        
        # Alpha cumprod plot
        ax = axes[0, 0]
        ax.plot(range(self.timesteps), alphas['alphas_cumprod'], 
               color='red', linewidth=3, label=schedule_name)
        ax.set_title(f'ᾱ_t (Alpha Cumprod): {schedule_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('ᾱ_t')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sqrt alpha cumprod plot
        ax = axes[0, 1]
        ax.plot(range(self.timesteps), alphas['sqrt_alphas_cumprod'], 
               color='red', linewidth=3, label=schedule_name)
        ax.set_title('√ᾱ_t (Sqrt Alpha Cumprod)', fontsize=14)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('√ᾱ_t')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sqrt one minus alpha cumprod plot
        ax = axes[1, 0]
        ax.plot(range(self.timesteps), alphas['sqrt_one_minus_alphas_cumprod'], 
               color='red', linewidth=3, label=schedule_name)
        ax.set_title('√(1-ᾱ_t)', fontsize=14)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('√(1-ᾱ_t)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Alpha statistics
        ax = axes[1, 1]
        final_alpha = alphas['alphas_cumprod'][-1].item()
        final_sqrt_alpha = alphas['sqrt_alphas_cumprod'][-1].item()
        final_sqrt_one_minus = alphas['sqrt_one_minus_alphas_cumprod'][-1].item()
        
        categories = ['Final ᾱ_t', 'Final √ᾱ_t', 'Final √(1-ᾱ_t)']
        values = [final_alpha, final_sqrt_alpha, final_sqrt_one_minus]
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        ax.set_title('Final Alpha Values', fontsize=14)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   f'{value:.6f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Alpha schedule plot saved to: {save_path}")
            
        return fig
    
    def plot_forward_diffusion_effects(self, 
                                     sample_data: torch.Tensor,
                                     test_timesteps: List[int] = [0, 1, 100, 250, 500, 750, 1000],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot forward diffusion effects on sample data.
        
        Args:
            sample_data: Sample data tensor (B, 2, L)
            test_timesteps: List of timesteps to test
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.alpha_schedules:
            self.generate_schedules_from_config()
        
        # Use first sample
        sample = sample_data[0:1]  # Keep batch dimension
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Get the configured schedule
        schedule_name, alphas = next(iter(self.alpha_schedules.items()))
        
        # Apply forward diffusion for configured schedule
        schedule_results = []
        for t in test_timesteps:
            # Apply forward diffusion
            sqrt_alphas_cumprod_t = alphas['sqrt_alphas_cumprod'][t].item()
            sqrt_one_minus_alphas_cumprod_t = alphas['sqrt_one_minus_alphas_cumprod'][t].item()
            
            noise = torch.randn_like(sample)
            noisy_sample = sqrt_alphas_cumprod_t * sample + sqrt_one_minus_alphas_cumprod_t * noise
            
            schedule_results.append(noisy_sample)
        
        results = {schedule_name: schedule_results}
        
        # Plot charge distributions
        ax = axes[0, 0]
        final_sample = schedule_results[-1]  # Use final timestep
        charge_data = final_sample[0, 0].cpu().numpy()  # First channel (charge)
        ax.hist(charge_data, bins=50, alpha=0.8, color='red', 
               label=f'{schedule_name} (t={test_timesteps[-1]})')
        ax.set_title(f'Charge Distribution (Final Timestep): {schedule_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Charge Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot time distributions
        ax = axes[0, 1]
        time_data = final_sample[0, 1].cpu().numpy()  # Second channel (time)
        ax.hist(time_data, bins=50, alpha=0.8, color='red', 
               label=f'{schedule_name} (t={test_timesteps[-1]})')
        ax.set_title(f'Time Distribution (Final Timestep): {schedule_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot signal coefficient evolution
        ax = axes[1, 0]
        signal_coeffs = [alphas['sqrt_alphas_cumprod'][t].item() for t in test_timesteps]
        ax.plot(test_timesteps, signal_coeffs, 'o-', color='red', linewidth=3, 
               markersize=6, label=schedule_name)
        ax.set_title(f'Signal Coefficient Evolution: {schedule_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('√ᾱ_t (Signal Coefficient)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot noise coefficient evolution
        ax = axes[1, 1]
        noise_coeffs = [alphas['sqrt_one_minus_alphas_cumprod'][t].item() for t in test_timesteps]
        ax.plot(test_timesteps, noise_coeffs, 'o-', color='red', linewidth=3, 
               markersize=6, label=schedule_name)
        ax.set_title(f'Noise Coefficient Evolution: {schedule_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('√(1-ᾱ_t) (Noise Coefficient)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Forward diffusion effects plot saved to: {save_path}")
            
        return fig
    
    def plot_snr_evolution(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Signal-to-Noise Ratio evolution across timesteps for configured schedule.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.alpha_schedules:
            self.generate_schedules_from_config()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Get the configured schedule
        schedule_name, alphas = next(iter(self.alpha_schedules.items()))
        
        timesteps = range(self.timesteps)
        
        # Calculate SNR = sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod
        snr_values = []
        for t in timesteps:
            sqrt_alpha = alphas['sqrt_alphas_cumprod'][t].item()
            sqrt_one_minus = alphas['sqrt_one_minus_alphas_cumprod'][t].item()
            snr = sqrt_alpha / sqrt_one_minus if sqrt_one_minus > 0 else float('inf')
            snr_values.append(snr)
        
        ax.plot(timesteps, snr_values, color='red', linewidth=3, label=schedule_name)
        
        ax.set_title(f'Signal-to-Noise Ratio Evolution: {schedule_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('SNR = √ᾱ_t / √(1-ᾱ_t)')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at SNR = 1
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='SNR = 1')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 SNR evolution plot saved to: {save_path}")
            
        return fig
    
    def generate_comprehensive_report(self, 
                                    sample_data: Optional[torch.Tensor] = None,
                                    output_dir: str = "./noise_schedule_output") -> None:
        """
        Generate comprehensive noise schedule visualization report.
        
        Args:
            sample_data: Optional sample data for forward diffusion effects
            output_dir: Output directory for plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🎨 Generating comprehensive noise schedule visualization report...")
        print(f"📁 Output directory: {output_path}")
        
        # Generate schedules if not already done
        if not self.schedules:
            self.generate_schedules_from_config()
        
        # Plot 1: Beta schedules
        beta_plot_path = output_path / "yaml_noise_schedule_beta_comparison.png"
        self.plot_beta_schedules(str(beta_plot_path))
        
        # Plot 2: Alpha schedules
        alpha_plot_path = output_path / "yaml_noise_schedule_alpha_comparison.png"
        self.plot_alpha_schedules(str(alpha_plot_path))
        
        # Plot 3: SNR evolution
        snr_plot_path = output_path / "yaml_noise_schedule_snr_evolution.png"
        self.plot_snr_evolution(str(snr_plot_path))
        
        # Plot 4: Forward diffusion effects (if sample data provided)
        if sample_data is not None:
            diffusion_plot_path = output_path / "yaml_noise_schedule_forward_diffusion.png"
            self.plot_forward_diffusion_effects(sample_data, test_timesteps=[0, 1, 100, 250, 500, 750, 999], save_path=str(diffusion_plot_path))
        
        # Print summary statistics
        print("\n📊 Noise Schedule Summary:")
        print("=" * 80)
        print(f"Configuration: {self.config_path}")
        print(f"Schedule Type: {self.schedule_type}")
        print(f"Timesteps: {self.timesteps}")
        
        # Show actual calculated beta range instead of YAML values
        if self.schedules:
            schedule_name, betas = next(iter(self.schedules.items()))
            print(f"YAML Beta Range: [{self.beta_start}, {self.beta_end}] (for reference)")
            print(f"Actual Beta Range: [{betas.min():.6f}, {betas.max():.6f}] (calculated)")
            if self.schedule_type.lower() == 'cosine':
                cosine_s = getattr(self.config.diffusion, 'cosine_s', 0.008)
                print(f"Note: Cosine schedule uses cosine_s={cosine_s}, ignores beta_start/beta_end")
        else:
            print(f"Beta Range: [{self.beta_start}, {self.beta_end}]")
        
        print("=" * 80)
        
        for name, betas in self.schedules.items():
            print(f"\n🔸 {name.upper()}:")
            print(f"  Beta range: [{betas.min():.6f}, {betas.max():.6f}]")
            print(f"  Beta mean: {betas.mean():.6f}")
            print(f"  Beta std: {betas.std():.6f}")
            
            # Alpha statistics
            alphas = self.alpha_schedules[name]
            print(f"  Final ᾱ_t: {alphas['alphas_cumprod'][-1]:.6f}")
            print(f"  Final √ᾱ_t: {alphas['sqrt_alphas_cumprod'][-1]:.6f}")
            print(f"  Final √(1-ᾱ_t): {alphas['sqrt_one_minus_alphas_cumprod'][-1]:.6f}")
            
            # SNR
            final_snr = alphas['sqrt_alphas_cumprod'][-1] / alphas['sqrt_one_minus_alphas_cumprod'][-1]
            print(f"  Final SNR: {final_snr:.6f}")
        
        print(f"\n✅ Visualization report generated in: {output_path}")


def load_sample_data(data_path: str, num_samples: int = 4) -> Optional[torch.Tensor]:
    """Load sample data from H5 file."""
    try:
        import h5py
        with h5py.File(data_path, 'r') as f:
            if 'input' in f:
                data = f['input'][:num_samples]
                return torch.tensor(data, dtype=torch.float32)
            else:
                print(f"⚠️  No 'input' dataset found in {data_path}")
                return None
    except Exception as e:
        print(f"⚠️  Could not load sample data from {data_path}: {e}")
        return None


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize noise schedule from YAML config")
    parser.add_argument("--config", "-c", type=str, required=True,
                       help="Path to YAML configuration file")
    parser.add_argument("--sample-data-path", type=str, default=None,
                       help="Path to H5 file for sample data (optional)")
    parser.add_argument("--output-dir", "-o", type=str, default="./noise_schedule_output",
                       help="Output directory for plots")
    parser.add_argument("--num-samples", type=int, default=4,
                       help="Number of samples to load (if sample data provided)")
    
    args = parser.parse_args()
    
    # Load sample data if provided
    sample_data = None
    if args.sample_data_path:
        print(f"📁 Loading {args.num_samples} samples from: {args.sample_data_path}")
        sample_data = load_sample_data(args.sample_data_path, args.num_samples)
        if sample_data is not None:
            print(f"   Signal shape: {sample_data.shape}")
        else:
            print("📊 Generating plots without forward diffusion effects...")
    
    # Create visualizer
    visualizer = NoiseScheduleVisualizer(args.config)
    
    # Generate comprehensive report
    visualizer.generate_comprehensive_report(
        sample_data=sample_data,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()





