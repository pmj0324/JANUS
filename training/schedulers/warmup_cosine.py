"""
Warmup + Cosine Annealing LR Scheduler
=======================================

Warmup followed by cosine annealing (wbcosine).
This is the scheduler used in config as "wbcosine".
"""

import torch.optim.lr_scheduler as lr_scheduler
import math
from .common import get_warmup_schedule


class WarmupCosineLRScheduler(lr_scheduler.LambdaLR):
    """
    Warmup + Cosine annealing learning rate scheduler.
    
    LR increases linearly during warmup, then follows cosine curve.
    """
    
    def __init__(
        self,
        optimizer,
        num_training_steps: int,
        warmup: bool = True,
        warmup_proportion: float = 0.003,
        num_cycles: float = 2.0,
        **kwargs
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            num_training_steps: Total number of training steps
            warmup: Whether to use warmup (default: True)
            warmup_proportion: Proportion of steps for warmup (default: 0.003)
            num_cycles: Number of cosine cycles (default: 2.0)
            **kwargs: Ignored (for compatibility)
        """
        num_warmup_steps = int(warmup_proportion * num_training_steps) if warmup else 0
        
        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                # Linear warmup: 0 -> 1
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                # Cosine annealing
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        
        super().__init__(optimizer, lr_lambda=lr_lambda)

