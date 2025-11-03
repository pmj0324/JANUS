#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noise Schedule Visualizer for GENESIS

This module provides comprehensive visualization tools for different noise schedules
used in diffusion models, including beta values, alpha values, and forward diffusion
effects on sample data.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple

from noise_schedules import (
    linear_beta_schedule,
    cosine_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
    compute_alpha_schedule
)


class NoiseScheduleVisualizer:
    """Visualizer for noise schedules and their effects."""
    
    def __init__(self, timesteps: int = 1000, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize the visualizer.
        
        Args:
            timesteps: Number of diffusion timesteps
            figsize: Figure size for plots
        """
        self.timesteps = timesteps
        self.figsize = figsize
        self.schedules = {}
        self.alpha_schedules = {}
        
    def generate_schedules(self, 
                          beta_start: float = 1e-4, 
                          beta_end: float = 2e-2, 
                          cosine_s: float = 0.008) -> Dict[str, torch.Tensor]:
        """
        Generate all noise schedules.
        
        Args:
            beta_start: Starting noise level
            beta_end: Ending noise level  
            cosine_s: Small offset for cosine schedule
            
        Returns:
            Dictionary of schedule names to beta tensors
        """
        self.schedules = {
            'Linear': linear_beta_schedule(self.timesteps, beta_start, beta_end),
            'Cosine': cosine_beta_schedule(self.timesteps, cosine_s),
            'Quadratic': quadratic_beta_schedule(self.timesteps, beta_start, beta_end),
            'Sigmoid': sigmoid_beta_schedule(self.timesteps, beta_start, beta_end)
        }
        
        # Compute alpha schedules
        for name, betas in self.schedules.items():
            self.alpha_schedules[name] = compute_alpha_schedule(betas)
            
        return self.schedules
    
    def plot_beta_schedules(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot beta schedules comparison.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.schedules:
            self.generate_schedules()
            
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        colors = ['blue', 'red', 'green', 'orange']
        
        # Main beta plot
        ax = axes[0, 0]
        for i, (name, betas) in enumerate(self.schedules.items()):
            ax.plot(range(self.timesteps), betas, label=name, color=colors[i], linewidth=2)
        ax.set_title('Beta Schedules Comparison', fontsize=14)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('β_t')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Zoomed beta plot (first 100 timesteps)
        ax = axes[0, 1]
        for i, (name, betas) in enumerate(self.schedules.items()):
            ax.plot(range(100), betas[:100], label=name, color=colors[i], linewidth=2)
        ax.set_title('Beta Schedules (First 100 Steps)', fontsize=14)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('β_t')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Beta distribution histograms
        ax = axes[1, 0]
        for i, (name, betas) in enumerate(self.schedules.items()):
            ax.hist(betas.numpy(), bins=50, alpha=0.6, label=name, color=colors[i])
        ax.set_title('Beta Value Distributions', fontsize=14)
        ax.set_xlabel('β_t Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Beta statistics
        ax = axes[1, 1]
        schedule_stats = []
        for name, betas in self.schedules.items():
            stats = {
                'name': name,
                'min': betas.min().item(),
                'max': betas.max().item(),
                'mean': betas.mean().item(),
                'std': betas.std().item()
            }
            schedule_stats.append(stats)
        
        names = [s['name'] for s in schedule_stats]
        means = [s['mean'] for s in schedule_stats]
        stds = [s['std'] for s in schedule_stats]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, means, width, label='Mean', color='skyblue')
        ax.bar(x + width/2, stds, width, label='Std', color='lightcoral')
        
        ax.set_title('Beta Statistics Comparison', fontsize=14)
        ax.set_xlabel('Schedule')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Beta schedules plot saved to: {save_path}")
            
        return fig
    
    def plot_alpha_schedules(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot alpha schedules comparison.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.alpha_schedules:
            self.generate_schedules()
            
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        colors = ['blue', 'red', 'green', 'orange']
        
        # Alpha cumprod plot
        ax = axes[0, 0]
        for i, (name, alphas) in enumerate(self.alpha_schedules.items()):
            ax.plot(range(self.timesteps), alphas['alphas_cumprod'], 
                   label=name, color=colors[i], linewidth=2)
        ax.set_title('ᾱ_t (Alpha Cumprod)', fontsize=14)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('ᾱ_t')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sqrt alpha cumprod plot
        ax = axes[0, 1]
        for i, (name, alphas) in enumerate(self.alpha_schedules.items()):
            ax.plot(range(self.timesteps), alphas['sqrt_alphas_cumprod'], 
                   label=name, color=colors[i], linewidth=2)
        ax.set_title('√ᾱ_t (Sqrt Alpha Cumprod)', fontsize=14)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('√ᾱ_t')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Sqrt one minus alpha cumprod plot
        ax = axes[1, 0]
        for i, (name, alphas) in enumerate(self.alpha_schedules.items()):
            ax.plot(range(self.timesteps), alphas['sqrt_one_minus_alphas_cumprod'], 
                   label=name, color=colors[i], linewidth=2)
        ax.set_title('√(1-ᾱ_t)', fontsize=14)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('√(1-ᾱ_t)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Final alpha cumprod comparison
        ax = axes[1, 1]
        final_alphas = [alphas['alphas_cumprod'][-1].item() 
                       for alphas in self.alpha_schedules.values()]
        names = list(self.alpha_schedules.keys())
        
        bars = ax.bar(names, final_alphas, color=colors[:len(names)])
        ax.set_title('Final ᾱ_t Values', fontsize=14)
        ax.set_ylabel('Final ᾱ_t')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_alphas):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Alpha schedules plot saved to: {save_path}")
            
        return fig
    
    def plot_forward_diffusion_effects(self, 
                                     sample_data: torch.Tensor,
                                     test_timesteps: List[int] = [0, 100, 250, 500, 750, 999],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot forward diffusion effects on sample data.
        
        Args:
            sample_data: Sample data tensor [batch_size, channels, seq_len]
            test_timesteps: Timesteps to test
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.alpha_schedules:
            self.generate_schedules()
            
        # Use first sample for visualization
        sample = sample_data[0:1]  # Keep batch dimension
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        colors = ['blue', 'red', 'green', 'orange']
        
        # Apply forward diffusion for each schedule
        results = {}
        for name, alphas in self.alpha_schedules.items():
            schedule_results = []
            for t in test_timesteps:
                # Apply forward diffusion
                sqrt_alphas_cumprod_t = alphas['sqrt_alphas_cumprod'][t]
                sqrt_one_minus_alphas_cumprod_t = alphas['sqrt_one_minus_alphas_cumprod'][t]
                
                noise = torch.randn_like(sample)
                noisy_sample = sqrt_alphas_cumprod_t * sample + sqrt_one_minus_alphas_cumprod_t * noise
                
                schedule_results.append(noisy_sample)
            results[name] = schedule_results
        
        # Plot charge distributions
        ax = axes[0, 0]
        for i, (name, noisy_samples) in enumerate(results.items()):
            final_sample = noisy_samples[-1]  # Use final timestep
            charge_data = final_sample[0, 0].cpu().numpy()  # First channel (charge)
            ax.hist(charge_data, bins=50, alpha=0.6, label=f'{name} (t={test_timesteps[-1]})', 
                   color=colors[i])
        ax.set_title('Charge Distributions (Final Timestep)', fontsize=14)
        ax.set_xlabel('Charge Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot time distributions
        ax = axes[0, 1]
        for i, (name, noisy_samples) in enumerate(results.items()):
            final_sample = noisy_samples[-1]  # Use final timestep
            time_data = final_sample[0, 1].cpu().numpy()  # Second channel (time)
            ax.hist(time_data, bins=50, alpha=0.6, label=f'{name} (t={test_timesteps[-1]})', 
                   color=colors[i])
        ax.set_title('Time Distributions (Final Timestep)', fontsize=14)
        ax.set_xlabel('Time Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot signal coefficient evolution
        ax = axes[1, 0]
        for i, (name, alphas) in enumerate(self.alpha_schedules.items()):
            signal_coeffs = [alphas['sqrt_alphas_cumprod'][t].item() for t in test_timesteps]
            ax.plot(test_timesteps, signal_coeffs, 'o-', label=name, 
                   color=colors[i], linewidth=2, markersize=6)
        ax.set_title('Signal Coefficient Evolution', fontsize=14)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('√ᾱ_t (Signal Coefficient)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot noise coefficient evolution
        ax = axes[1, 1]
        for i, (name, alphas) in enumerate(self.alpha_schedules.items()):
            noise_coeffs = [alphas['sqrt_one_minus_alphas_cumprod'][t].item() for t in test_timesteps]
            ax.plot(test_timesteps, noise_coeffs, 'o-', label=name, 
                   color=colors[i], linewidth=2, markersize=6)
        ax.set_title('Noise Coefficient Evolution', fontsize=14)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('√(1-ᾱ_t) (Noise Coefficient)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Forward diffusion effects plot saved to: {save_path}")
            
        return fig
    
    def generate_comprehensive_report(self, 
                                    sample_data: Optional[torch.Tensor] = None,
                                    output_dir: str = "./outputs") -> None:
        """
        Generate comprehensive visualization report.
        
        Args:
            sample_data: Sample data for forward diffusion testing
            output_dir: Output directory for plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("🎨 Generating comprehensive noise schedule visualization report...")
        
        # Generate schedules if not already done
        if not self.schedules:
            self.generate_schedules()
        
        # Plot 1: Beta schedules
        beta_plot_path = output_path / "noise_schedule_beta_comparison.png"
        self.plot_beta_schedules(str(beta_plot_path))
        
        # Plot 2: Alpha schedules
        alpha_plot_path = output_path / "noise_schedule_alpha_comparison.png"
        self.plot_alpha_schedules(str(alpha_plot_path))
        
        # Plot 3: Forward diffusion effects (if sample data provided)
        if sample_data is not None:
            diffusion_plot_path = output_path / "noise_schedule_forward_diffusion.png"
            self.plot_forward_diffusion_effects(sample_data, str(diffusion_plot_path))
        
        # Print summary statistics
        print("\n📊 Noise Schedule Summary:")
        print("=" * 60)
        for name, betas in self.schedules.items():
            print(f"\n🔸 {name.upper()} SCHEDULE:")
            print(f"  Beta range: [{betas.min():.6f}, {betas.max():.6f}]")
            print(f"  Beta mean: {betas.mean():.6f}")
            print(f"  Beta std: {betas.std():.6f}")
            
            if name in self.alpha_schedules:
                alphas = self.alpha_schedules[name]
                print(f"  Final ᾱ_t: {alphas['alphas_cumprod'][-1]:.6f}")
                print(f"  Final √ᾱ_t: {alphas['sqrt_alphas_cumprod'][-1]:.6f}")
                print(f"  Final √(1-ᾱ_t): {alphas['sqrt_one_minus_alphas_cumprod'][-1]:.6f}")
        
        print(f"\n✅ Visualization report generated in: {output_dir}")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Noise Schedule Visualizer")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of timesteps")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--sample-data-path", type=str, help="Path to sample data (optional)")
    parser.add_argument("--beta-start", type=float, default=1e-4, help="Beta start value")
    parser.add_argument("--beta-end", type=float, default=2e-2, help="Beta end value")
    parser.add_argument("--cosine-s", type=float, default=0.008, help="Cosine schedule offset")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = NoiseScheduleVisualizer(timesteps=args.timesteps)
    
    # Load sample data if provided
    sample_data = None
    if args.sample_data_path:
        try:
            import h5py
            with h5py.File(args.sample_data_path, 'r') as f:
                # Load a small batch for testing
                signals = torch.tensor(f['signals'][:4])  # First 4 samples
                sample_data = signals
            print(f"📁 Loaded sample data from: {args.sample_data_path}")
        except Exception as e:
            print(f"⚠️ Could not load sample data: {e}")
            print("📊 Generating plots without forward diffusion effects...")
    
    # Generate comprehensive report
    visualizer.generate_comprehensive_report(
        sample_data=sample_data,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
