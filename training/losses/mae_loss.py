"""
MAE Loss (L1) - Mean Absolute Error
=====================================

L1 loss: |pred - target|
"""

import torch
import torch.nn.functional as F


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error (L1) loss.
    
    Args:
        pred: Predictions (B, C, L) or any shape
        target: Targets (same shape as pred)
    
    Returns:
        MAE loss scalar
    """
    return F.l1_loss(pred, target)

