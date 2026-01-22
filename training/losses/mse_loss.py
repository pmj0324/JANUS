"""
MSE Loss (L2) - Mean Squared Error
===================================

L2 loss: ||pred - target||^2
"""

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error (L2) loss.
    
    Args:
        pred: Predictions (B, C, L) or any shape
        target: Targets (same shape as pred)
    
    Returns:
        MSE loss scalar
    """
    return F.mse_loss(pred, target)

