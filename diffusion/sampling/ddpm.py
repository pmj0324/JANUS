#!/usr/bin/env python3
"""
DDPM Sampling
=============

DDPM (Denoising Diffusion Probabilistic Models) sampling for diffusion models.
"""

import torch
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from diffusion.core import GaussianDiffusion


def predict_start_from_noise(
    diffusion: "GaussianDiffusion",
    x_t: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor
) -> torch.Tensor:
    """
    Predict x_0 from x_t and predicted noise ε.
    
    x_0 = (x_t - sqrt(1-ᾱ_t) * ε) / sqrt(ᾱ_t)
    
    Note: For t > 0, use parameter at index t-1
    """
    # For t > 0, use parameter at index t-1
    t_idx = torch.where(t > 0, t - 1, torch.zeros_like(t))
    
    return (
        diffusion.sqrt_recip_alphas_cumprod[t_idx][:, None, None] * x_t -
        diffusion.sqrt_recipm1_alphas_cumprod[t_idx][:, None, None] * noise
    )


def sample_ddpm(
    diffusion: "GaussianDiffusion",
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
        samples_norm = sample_ddpm(diffusion, label, geom, shape)
        norm_params = diffusion.model.get_normalization_params()
        samples_raw = denormalize_signal(samples_norm, ...)
        ```
    
    If denormalize=True:
        Automatically denormalizes using model's metadata:
        ```python
        samples_raw = sample_ddpm(diffusion, label, geom, shape, denormalize=True)
        # Already in physical units (NPE, ns)
        ```
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Args:
        diffusion: GaussianDiffusion instance
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
    
    # Get device from model or buffers
    device = diffusion.betas.device
    
    # Start from pure noise
    x = torch.randn(B, C, L, device=device)
    
    all_samples = [x] if return_all_timesteps else None
    
    # Reverse diffusion
    # Loop from t=T down to t=1 (stop before t=0)
    # t=0 is original data, so we stop at t=1 and take the mean as x_0
    # Note: range(1, T+1) generates [1, 2, ..., T], reversed = [T, T-1, ..., 2, 1]
    for t_val in reversed(range(1, diffusion.cfg.timesteps + 1)):
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
        
        # Classifier-free guidance
        if diffusion.cfg.use_cfg and diffusion.cfg.cfg_scale != 1.0:
            # Predict with condition
            eps_cond = diffusion.model(x, geom, t_batch, label)  # (B, 2, L)
            
            # Predict without condition (unconditional)
            label_uncond = torch.zeros_like(label)
            eps_uncond = diffusion.model(x, geom, t_batch, label_uncond)  # (B, 2, L)
            
            # Combine predictions with guidance scale
            # eps = eps_uncond + scale * (eps_cond - eps_uncond)
            eps_hat = eps_uncond + diffusion.cfg.cfg_scale * (eps_cond - eps_uncond)
        else:
            # Standard prediction without guidance
            eps_hat = diffusion.model(x, geom, t_batch, label)  # (B, 2, L)
        
        # Get schedule values (use t-1 as index)
        idx = t_val - 1
        alpha = diffusion.alphas[idx]
        alpha_bar = diffusion.alphas_cumprod[idx]
        beta = diffusion.betas[idx]
        
        # DDPM mean update (eps-prediction)
        # μ_θ(x_t, t) = (1/sqrt(α_t)) * (x_t - (β_t / sqrt(1-ᾱ_t)) * ε_θ(x_t, t))
        mean = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * eps_hat
        )
        
        # Add noise (except at final step t=1)
        if t_val > 1:
            noise = torch.randn_like(x)
            var = torch.sqrt(diffusion.posterior_variance[idx])
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
        x = diffusion._denormalize_samples(x)
        if return_all_timesteps:
            # Denormalize all timesteps
            all_samples = [diffusion._denormalize_samples(x_t) for x_t in all_samples]
    
    if return_all_timesteps:
        return all_samples
    return x

