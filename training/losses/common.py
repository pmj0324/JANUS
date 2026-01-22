"""
Common Utilities for Loss Functions
====================================

공통 유틸리티 함수들:
- Weighted loss
- Multi-task loss combination
"""

import torch
from typing import Dict


def weighted_loss(
    loss: torch.Tensor,
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Apply per-sample weights to loss.
    
    Args:
        loss: Per-sample losses (B,)
        weights: Per-sample weights (B,)
    
    Returns:
        Weighted average loss (scalar)
    """
    return (loss * weights).mean()


def multi_task_loss(
    losses: Dict[str, torch.Tensor],
    weights: Dict[str, float]
) -> torch.Tensor:
    """
    Combine multiple losses with weights.
    
    Args:
        losses: Dictionary of loss names to loss values
        weights: Dictionary of loss names to weight values
    
    Returns:
        Combined loss (scalar)
    """
    total = 0.0
    for name, loss in losses.items():
        weight = weights.get(name, 1.0)
        total += weight * loss
    return total

