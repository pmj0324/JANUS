#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian Diffusion Module
==========================

DDPM-style diffusion for conditional generation of PMT signals.

Forward process: q(x_t | x_0) = N(x_t; sqrt(α̅_t)x_0, (1-α̅_t)I)
Reverse process: p_θ(x_{t-1} | x_t, c) where θ = model parameters, c = conditions

Supports:
- ε-prediction (predict noise)
- x0-prediction (predict clean signal)
- DDPM sampling
- DDIM sampling (TODO)

Author: Minje Park
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    """
    DDPM-style trainer/sampler for p(x|c) with geometry.
    
    Model predicts ε̂(x_sig_t, t, label, geom) → (B, 2, L)
    - x_sig: PMT signals (charge, time) - noised during diffusion
    - geom: PMT geometry (x, y, z) - kept clean as conditioning
    - label: Event properties (Energy, Zenith, Azimuth, X, Y, Z)
    
    Args:
        model: Neural network model (e.g., PMTDiT)
        cfg: DiffusionConfig object
    """
    
    def __init__(self, model: nn.Module, cfg: DiffusionConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        
        T = cfg.timesteps
        
        # Create noise schedule
        if cfg.schedule == "cosine":
            betas = self._cosine_beta_schedule(T, cfg.cosine_s)
        else:  # linear (default)
            betas = torch.linspace(cfg.beta_start, cfg.beta_end, T)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register all schedule-related tensors as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Posterior variance for DDPM sampling
        # q(x_{t-1} | x_t, x_0) variance
        posterior_variance = betas * (1 - alphas_cumprod.roll(1, 0)) / (1 - alphas_cumprod)
        posterior_variance[0] = betas[0]
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
    
    def get_normalization_params(self):
        """
        Get normalization parameters from the model.
        Returns affine_offset, affine_scale, label_offset, label_scale, time_transform.
        """
        if hasattr(self.model, 'affine_offset'):
            affine_offset = self.model.affine_offset.squeeze().cpu()
            affine_scale = self.model.affine_scale.squeeze().cpu()
            label_offset = self.model.label_offset.cpu() if hasattr(self.model, 'label_offset') else None
            label_scale = self.model.label_scale.cpu() if hasattr(self.model, 'label_scale') else None
            time_transform = self.model.time_transform if hasattr(self.model, 'time_transform') else "ln"
            return affine_offset, affine_scale, label_offset, label_scale, time_transform
        else:
            return None, None, None, None, "ln"
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(
        self, 
        x0_sig: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: sample from q(x_t | x_0) for signals only.
        
        Args:
            x0_sig: Clean signals (B, 2, L) - [charge, time]
            t: Timesteps (B,) - integer values in [0, T]
                • t=0: Original data (no noise, no parameter used)
                • t=1: First noise step (uses betas[0])
                • t=2: Second noise step (uses betas[1])
                • ...
                • t=T: Final timestep (uses betas[T-1], maximum noise)
            noise: Optional pre-generated noise (B, 2, L)
        
        Returns:
            Noised signals x_t (B, 2, L)
        
        Note:
            Parameter indexing: betas[t-1] for t > 0
            This ensures all T noise parameters are used (betas[0] through betas[T-1])
        """
        # Special case: t=0 returns original data (no noise)
        mask_t0 = (t == 0)
        if mask_t0.all():
            return x0_sig.clone()
        
        if noise is None:
            noise = torch.randn_like(x0_sig)
        
        # For t > 0, use parameter at index t-1
        # t=1 → index 0, t=2 → index 1, ..., t=T → index T-1
        t_idx = torch.where(t > 0, t - 1, torch.zeros_like(t))
        
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t_idx][:, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t_idx][:, None, None]
        
        # Compute noised version
        x_t = sqrt_alpha_bar * x0_sig + sqrt_one_minus_alpha_bar * noise
        
        # Replace t=0 samples with original (if any in batch)
        if mask_t0.any():
            x_t[mask_t0] = x0_sig[mask_t0].clone()
        
        return x_t
    
    def predict_start_from_noise(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise ε.
        
        x_0 = (x_t - sqrt(1-α̅_t) * ε) / sqrt(α̅_t)
        
        Note: For t > 0, use parameter at index t-1
        """
        # For t > 0, use parameter at index t-1
        t_idx = torch.where(t > 0, t - 1, torch.zeros_like(t))
        
        return (
            self.sqrt_recip_alphas_cumprod[t_idx][:, None, None] * x_t -
            self.sqrt_recipm1_alphas_cumprod[t_idx][:, None, None] * noise
        )
    
    def loss(
        self, 
        x0_sig: torch.Tensor, 
        geom: torch.Tensor, 
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute training loss for diffusion model with optional classifier-free guidance.
        
        Args:
            x0_sig: Clean signals (B, 2, L)
            geom: Geometry (B, 3, L) - kept clean
            label: Condition c (B, 6)
        
        Returns:
            MSE loss scalar
        """
        B = x0_sig.size(0)
        device = x0_sig.device
        
        # Sample random timesteps (excluding t=0 which is original data)
        # t=0: original (no noise, skip training)
        # t=1~T: noise steps (train on these, uses all T noise parameters)
        # Note: torch.randint(1, T+1) generates values in [1, T]
        t = torch.randint(1, self.cfg.timesteps + 1, (B,), device=device, dtype=torch.long)
        
        # Sample noise and create noisy signals
        noise = torch.randn_like(x0_sig)
        x_sig_t = self.q_sample(x0_sig, t, noise=noise)
        
        # Classifier-free guidance: randomly drop conditions during training
        if self.cfg.use_cfg and self.training:
            # Create mask for dropping conditions
            drop_mask = torch.rand(B, device=device) < self.cfg.cfg_dropout
            
            # Zero out labels where mask is True (unconditional)
            label_conditioned = label.clone()
            label_conditioned[drop_mask] = 0.0
            
            # Predict with possibly dropped conditions
            pred = self.model(x_sig_t, geom, t, label_conditioned)  # (B, 2, L)
        else:
            # Normal prediction
            pred = self.model(x_sig_t, geom, t, label)  # (B, 2, L)
        
        # Compute loss based on objective
        if self.cfg.objective == "eps":
            target = noise
        elif self.cfg.objective == "x0":
            target = x0_sig
        else:
            raise ValueError(f"Unknown objective: {self.cfg.objective}")
        
        return F.mse_loss(pred, target)
    
    @torch.no_grad()
    def sample(
        self, 
        label: torch.Tensor, 
        geom: torch.Tensor, 
        shape: Tuple[int, int, int],
        return_all_timesteps: bool = False,
        denormalize: bool = False
    ) -> torch.Tensor:
        """
        DDPM sampling: generate samples from p(x|c).
        
        ⚠️ IMPORTANT: Reverse diffusion operates in NORMALIZED space!
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        The reverse diffusion process (T → 0) happens ENTIRELY in normalized space.
        
        Step-by-step process:
        1. x_T ~ N(0, I)                    [normalized space]
        2. for t in T-1 → 0:
               x_t → model → eps_hat        [normalized space]
               x_{t-1} = ddpm_update(x_t)   [normalized space]
        3. Final x_0                        [normalized space]
        4. (Optional) Denormalize           [physical units]
        
        If denormalize=False (default):
            Returns x_0 in NORMALIZED space. You must manually denormalize:
            ```python
            samples_norm = diffusion.sample(label, geom, shape)
            norm_params = diffusion.model.get_normalization_params()
            samples_raw = denormalize_signal(samples_norm, ...)
            ```
        
        If denormalize=True:
            Automatically denormalizes using model's metadata:
            ```python
            samples_raw = diffusion.sample(label, geom, shape, denormalize=True)
            # Already in physical units (NPE, ns)
            ```
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Args:
            label: Conditions (B, 6) - should be NORMALIZED
            geom: Geometry (B, 3, L) - should be NORMALIZED
            shape: Output shape (B, 2, L)
            return_all_timesteps: If True, return all intermediate steps
            denormalize: If True, automatically denormalize using model metadata
        
        Returns:
            Samples x_0 (B, 2, L)
            - If denormalize=False: NORMALIZED space
            - If denormalize=True: Physical units (NPE, ns)
            - If return_all_timesteps=True: list of all timesteps
        """
        B, C, L = shape
        assert C == 2 and geom.shape == (B, 3, L), "shape/geom mismatch"
        
        device = next(self.parameters()).device
        
        # Start from pure noise
        x = torch.randn(B, C, L, device=device)
        
        all_samples = [x] if return_all_timesteps else None
        
        # Reverse diffusion
        # Loop from t=T down to t=1 (stop before t=0)
        # t=0 is original data, so we stop at t=1 and take the mean as x_0
        # Note: range(1, T+1) generates [1, 2, ..., T], reversed = [T, T-1, ..., 2, 1]
        for t_val in reversed(range(1, self.cfg.timesteps + 1)):
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            
            # Classifier-free guidance
            if self.cfg.use_cfg and self.cfg.cfg_scale != 1.0:
                # Predict with condition
                eps_cond = self.model(x, geom, t_batch, label)  # (B, 2, L)
                
                # Predict without condition (unconditional)
                label_uncond = torch.zeros_like(label)
                eps_uncond = self.model(x, geom, t_batch, label_uncond)  # (B, 2, L)
                
                # Combine predictions with guidance scale
                # eps = eps_uncond + scale * (eps_cond - eps_uncond)
                eps_hat = eps_uncond + self.cfg.cfg_scale * (eps_cond - eps_uncond)
            else:
                # Standard prediction without guidance
                eps_hat = self.model(x, geom, t_batch, label)  # (B, 2, L)
            
            # Get schedule values (use t-1 as index)
            idx = t_val - 1
            alpha = self.alphas[idx]
            alpha_bar = self.alphas_cumprod[idx]
            beta = self.betas[idx]
            
            # DDPM mean update (eps-prediction)
            # μ_θ(x_t, t) = (1/sqrt(α_t)) * (x_t - (β_t / sqrt(1-α̅_t)) * ε_θ(x_t, t))
            mean = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_bar)) * eps_hat
            )
            
            # Add noise (except at final step t=1)
            if t_val > 1:
                noise = torch.randn_like(x)
                var = torch.sqrt(self.posterior_variance[idx])
                x = mean + var * noise
            else:
                # t=1: final denoising step, return mean as x_0
                x = mean
            
            if return_all_timesteps:
                all_samples.append(x)
        
        # ═══════════════════════════════════════════════════════════════
        # Denormalization (ONLY at the END, after complete reverse diffusion!)
        # ═══════════════════════════════════════════════════════════════
        if denormalize:
            x = self._denormalize_samples(x)
            if return_all_timesteps:
                # Denormalize all timesteps
                all_samples = [self._denormalize_samples(x_t) for x_t in all_samples]
        
        if return_all_timesteps:
            return all_samples
        return x
    
    def _denormalize_samples(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize samples using model's metadata.
        
        Process:
        1. Get normalization params from model
        2. Affine inverse: x = (x_norm * scale) + offset
        3. Time transform inverse: time = exp(time_norm) - 1 or 10^time_norm - 1
        4. Clamp to prevent overflow
        
        Args:
            x_norm: (B, 2, L) normalized samples
        
        Returns:
            x_raw: (B, 2, L) physical units (NPE, ns)
        """
        # Get normalization parameters from model metadata
        norm_params = self.model.get_normalization_params()
        
        offsets = torch.tensor(
            norm_params['affine_offsets'][:2],  # [charge, time]
            device=x_norm.device,
            dtype=x_norm.dtype
        ).view(1, 2, 1)
        
        scales = torch.tensor(
            norm_params['affine_scales'][:2],  # [charge, time]
            device=x_norm.device,
            dtype=x_norm.dtype
        ).view(1, 2, 1)
        
        time_transform = norm_params['time_transform']
        
        # Step 1: Affine inverse
        # x_physical = (x_norm * scale) + offset
        x = (x_norm * scales) + offsets
        
        # Step 2: Time transform inverse (only for time channel)
        if time_transform == "ln":
            # Inverse of ln(1+x) is exp(x) - 1
            x[:, 1, :] = torch.exp(x[:, 1, :]) - 1.0
        elif time_transform == "log10":
            # Inverse of log10(1+x) is 10^x - 1
            x[:, 1, :] = torch.pow(10.0, x[:, 1, :]) - 1.0
        else:
            raise ValueError(f"Unknown time_transform: {time_transform}")
        
        # Step 3: Clamp time to prevent overflow (exp can produce huge values)
        x[:, 1, :] = torch.clamp(x[:, 1, :], min=0.0, max=1e8)
        
        return x
    
    @torch.no_grad()
    def sample_ddim(
        self,
        label: torch.Tensor,
        geom: torch.Tensor,
        shape: Tuple[int, int, int],
        eta: float = 0.0,
        ddim_steps: int = 50
    ) -> torch.Tensor:
        """
        DDIM sampling: faster sampling with fewer steps.
        
        Args:
            label: Conditions (B, 6)
            geom: Geometry (B, 3, L)
            shape: Output shape (B, 2, L)
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
            ddim_steps: Number of sampling steps (< timesteps for speedup)
        
        Returns:
            Samples x_0 (B, 2, L)
        """
        B, C, L = shape
        device = next(self.parameters()).device
        
        # Create subsequence of timesteps (exclude t=0 which is original)
        # Generate t values: [step_size, 2*step_size, ..., T]
        step_size = self.cfg.timesteps // ddim_steps
        timesteps = list(range(step_size, self.cfg.timesteps + 1, step_size))
        if timesteps[-1] != self.cfg.timesteps:
            timesteps.append(self.cfg.timesteps)  # Ensure final timestep T is included
        timesteps.reverse()  # [T, ..., step_size]
        
        # Start from pure noise
        x = torch.randn(B, C, L, device=device)
        
        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            
            # Predict noise
            eps_hat = self.model(x, geom, t_batch, label)
            
            # Get next timestep (stop at t=1, not t=0)
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
            else:
                t_prev = 1  # Stop at t=1, not t=0
            
            # Get alphas (use t-1 as index for t > 0)
            idx = t_val - 1
            idx_prev = t_prev - 1
            
            # DDIM update (simplified)
            alpha_bar = self.alphas_cumprod[idx]
            alpha_bar_prev = self.alphas_cumprod[idx_prev]
            
            # Predict x_0
            x0_pred = self.predict_start_from_noise(x, t_batch, eps_hat)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_bar_prev - eta ** 2) * eps_hat
            
            # DDIM update
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
            
            if eta > 0 and t_prev_idx > 0:
                noise = torch.randn_like(x)
                x = x + eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * \
                    torch.sqrt(1 - alpha_bar / alpha_bar_prev) * noise
        
        return x

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Extract values from a 1-D tensor for a batch of indices.
        
        Args:
            a: 1-D tensor of values
            t: batch of timestep indices
            x_shape: shape of the tensor to broadcast to
            
        Returns:
            Extracted values broadcast to x_shape
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def create_gaussian_diffusion(
    model: nn.Module,
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    objective: str = "eps",
    schedule: str = "linear"
) -> GaussianDiffusion:
    """
    Factory function to create GaussianDiffusion instance.
    
    Args:
        model: Neural network model
        timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        objective: "eps" or "x0"
        schedule: "linear" or "cosine"
    
    Returns:
        GaussianDiffusion instance
    """
    cfg = DiffusionConfig(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        objective=objective,
        schedule=schedule
    )
    return GaussianDiffusion(model, cfg)

