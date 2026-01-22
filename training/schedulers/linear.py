"""
Linear LR Scheduler
====================

Linear decay learning rate scheduler.
"""

import torch.optim.lr_scheduler as lr_scheduler


class LinearLRScheduler(lr_scheduler.LambdaLR):
    """
    Linear decay learning rate scheduler.
    
    LR decreases linearly from initial_lr to 0 over num_training_steps.
    """
    
    def __init__(
        self,
        optimizer,
        num_training_steps: int,
        **kwargs
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            num_training_steps: Total number of training steps
            **kwargs: Ignored (for compatibility)
        """
        def lr_lambda(current_step: int) -> float:
            if current_step < num_training_steps:
                return 1.0 - float(current_step) / float(num_training_steps)
            else:
                return 0.0
        
        super().__init__(optimizer, lr_lambda=lr_lambda)

