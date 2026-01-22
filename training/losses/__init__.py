"""
Loss Functions for Diffusion Models
====================================

각 loss function을 개별 파일로 관리:
- mse: L2 loss (Mean Squared Error)
- mae: L1 loss (Mean Absolute Error)
- huber: Huber loss
- smooth_l1: Smooth L1 loss
- diffusion: Diffusion-specific loss with CFG support

Usage:
    from training.losses import get_loss, compute_diffusion_loss
    
    # Get basic loss function
    loss_fn = get_loss("mse")  # or "mae", "huber", etc.
    
    # Diffusion-specific loss
    loss = compute_diffusion_loss(diffusion, x0_sig, geom, label, loss_type="mse")
"""

from .mse_loss import mse_loss
from .mae_loss import mae_loss
from .huber_loss import huber_loss
from .smooth_l1 import smooth_l1_loss
from .diffusion_loss import compute_diffusion_loss

__all__ = [
    # Basic loss functions
    "mse_loss",
    "mae_loss",
    "huber_loss",
    "smooth_l1_loss",
    
    # Diffusion-specific
    "compute_diffusion_loss",
    
    # Factory function
    "get_loss",
]


def get_loss(loss_type: str):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type: "mse", "mae", "huber", or "smooth_l1"
    
    Returns:
        Loss function that takes (pred, target) and returns scalar
    """
    loss_type = loss_type.lower()
    
    if loss_type in ["mse", "l2"]:
        return mse_loss
    elif loss_type in ["mae", "l1"]:
        return mae_loss
    elif loss_type == "huber":
        return huber_loss
    elif loss_type == "smooth_l1":
        return smooth_l1_loss
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Choose from: mse, mae, huber, smooth_l1"
        )

