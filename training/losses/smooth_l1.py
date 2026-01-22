"""
Smooth L1 Loss
==============

Smooth L1 loss: similar to Huber loss but with fixed delta=1.0
"""

import torch
import torch.nn.functional as F


def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Smooth L1 loss (Huber loss with delta=1.0).
    
    Args:
        pred: Predictions (B, C, L) or any shape
        target: Targets (same shape as pred)
    
    Returns:
        Smooth L1 loss scalar
    """
    return F.smooth_l1_loss(pred, target)

