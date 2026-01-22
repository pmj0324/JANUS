"""
Huber Loss - Smooth combination of L1 and L2
==============================================

Huber loss: smooth L1/L2 combination
"""

import torch
import torch.nn.functional as F


def huber_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    delta: float = 1.0
) -> torch.Tensor:
    """
    Huber loss: smooth combination of L1 and L2.
    
    For |pred - target| <= delta: uses L2 (smooth)
    For |pred - target| > delta: uses L1 (linear)
    
    Args:
        pred: Predictions (B, C, L) or any shape
        target: Targets (same shape as pred)
        delta: Threshold parameter (default: 1.0)
    
    Returns:
        Huber loss scalar
    """
    return F.huber_loss(pred, target, delta=delta)

