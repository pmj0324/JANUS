"""
Common Utilities for LR Schedulers
==================================

공통 유틸리티 함수들:
- Warmup schedule computation
- Base scheduler classes
"""

import torch
from typing import List


def get_warmup_schedule(
    num_warmup_steps: int,
    num_training_steps: int
) -> List[float]:
    """
    Compute warmup schedule (linear warmup).
    
    Args:
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
    
    Returns:
        List of warmup multipliers for each step
    """
    if num_warmup_steps == 0:
        return [1.0] * num_training_steps
    
    warmup_schedule = []
    for step in range(num_training_steps):
        if step < num_warmup_steps:
            # Linear warmup: 0 -> 1
            multiplier = float(step) / float(max(1, num_warmup_steps))
        else:
            multiplier = 1.0
        warmup_schedule.append(multiplier)
    
    return warmup_schedule

