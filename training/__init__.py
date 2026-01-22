"""
Training Module for Diffusion Models
====================================

Training-related functions for diffusion models.

Structure:
- losses/: Loss functions (mse, mae, huber, diffusion_loss)
- schedulers/: LR schedulers (constant, linear, cosine, warmup_cosine, plateau, step, exponential)

Usage:
    from training.losses import compute_diffusion_loss, get_loss
    from training.schedulers import get_scheduler
    
    # Loss
    loss = compute_diffusion_loss(diffusion, x0_sig, geom, label, loss_type="mse")
    
    # Scheduler
    scheduler = get_scheduler(
        "warmup_cosine",
        optimizer=optimizer,
        num_training_steps=total_steps,
        warmup_proportion=0.003,
        num_cycles=2
    )
"""

# Loss functions
from .losses import (
    compute_diffusion_loss,
    get_loss,
    mse_loss,
    mae_loss,
    huber_loss,
    smooth_l1_loss,
)

# LR schedulers
from .schedulers import (
    get_scheduler,
    ConstantLRScheduler,
    LinearLRScheduler,
    CosineLRScheduler,
    WarmupCosineLRScheduler,
    PlateauLRScheduler,
    StepLRScheduler,
    ExponentialLRScheduler,
)

# Backward compatibility: compute_loss -> compute_diffusion_loss
from .losses import compute_diffusion_loss as compute_loss

__all__ = [
    # Loss functions
    "compute_loss",  # Backward compatibility
    "compute_diffusion_loss",
    "get_loss",
    "mse_loss",
    "mae_loss",
    "huber_loss",
    "smooth_l1_loss",
    
    # LR schedulers
    "get_scheduler",
    "ConstantLRScheduler",
    "LinearLRScheduler",
    "CosineLRScheduler",
    "WarmupCosineLRScheduler",
    "PlateauLRScheduler",
    "StepLRScheduler",
    "ExponentialLRScheduler",
]
