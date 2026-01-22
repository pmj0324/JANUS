"""
Cosine Annealing LR Scheduler
==============================

Cosine annealing learning rate scheduler.
"""

import torch.optim.lr_scheduler as lr_scheduler
import math


class CosineLRScheduler(lr_scheduler.LambdaLR):
    """
    Cosine annealing learning rate scheduler.
    
    LR follows cosine curve from initial_lr to 0.
    """
    
    def __init__(
        self,
        optimizer,
        num_training_steps: int,
        num_cycles: float = 0.5,
        **kwargs
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            num_training_steps: Total number of training steps
            num_cycles: Number of cosine cycles (default: 0.5 for half cycle)
            **kwargs: Ignored (for compatibility)
        """
        def lr_lambda(current_step: int) -> float:
            if current_step < num_training_steps:
                progress = float(current_step) / float(num_training_steps)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
            else:
                return 0.0
        
        super().__init__(optimizer, lr_lambda=lr_lambda)

