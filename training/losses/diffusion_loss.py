"""
Diffusion-specific Loss with Classifier-Free Guidance
======================================================

Diffusion model training loss with optional CFG support.
"""

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusion.core import GaussianDiffusion

from .mse_loss import mse_loss
from .mae_loss import mae_loss
from .huber_loss import huber_loss
from .smooth_l1 import smooth_l1_loss


def compute_diffusion_loss(
    diffusion: "GaussianDiffusion",
    x0_sig: torch.Tensor,
    geom: torch.Tensor,
    label: torch.Tensor,
    loss_type: str = "mse"
) -> torch.Tensor:
    """
    Compute training loss for diffusion model with optional classifier-free guidance.
    
    Args:
        diffusion: GaussianDiffusion instance
        x0_sig: Clean signals (B, 2, L)
        geom: Geometry (B, 3, L) - kept clean
        label: Condition c (B, 6)
        loss_type: "mse" (L2), "mae" (L1), "huber", or "smooth_l1"
    
    Returns:
        Loss scalar
    """
    B = x0_sig.size(0)
    device = x0_sig.device
    
    # Sample random timesteps (excluding t=0 which is original data)
    # t=0: original (no noise, skip training)
    # t=1~T: noise steps (train on these, uses all T noise parameters)
    # Note: torch.randint(1, T+1) generates values in [1, T]
    t = torch.randint(1, diffusion.cfg.timesteps + 1, (B,), device=device, dtype=torch.long)
    
    # Sample noise and create noisy signals
    noise = torch.randn_like(x0_sig)
    x_sig_t = diffusion.q_sample(x0_sig, t, noise=noise)
    
    # Classifier-free guidance: randomly drop conditions during training
    if diffusion.cfg.use_cfg and diffusion.training:
        # Create mask for dropping conditions
        drop_mask = torch.rand(B, device=device) < diffusion.cfg.cfg_dropout
        
        # Zero out labels where mask is True (unconditional)
        label_conditioned = label.clone()
        label_conditioned[drop_mask] = 0.0
        
        # Predict with possibly dropped conditions
        pred = diffusion.model(x_sig_t, geom, t, label_conditioned)  # (B, 2, L)
    else:
        # Normal prediction
        pred = diffusion.model(x_sig_t, geom, t, label)  # (B, 2, L)
    
    # Compute loss based on objective
    if diffusion.cfg.objective == "eps":
        target = noise
    elif diffusion.cfg.objective == "x0":
        target = x0_sig
    else:
        raise ValueError(f"Unknown objective: {diffusion.cfg.objective}")
    
    # Get loss function based on type
    loss_type = loss_type.lower()
    
    if loss_type in ["mse", "l2"]:
        return mse_loss(pred, target)
    elif loss_type in ["mae", "l1"]:
        return mae_loss(pred, target)
    elif loss_type == "huber":
        return huber_loss(pred, target, delta=1.0)
    elif loss_type == "smooth_l1":
        return smooth_l1_loss(pred, target)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Choose from: mse, mae, huber, smooth_l1"
        )

