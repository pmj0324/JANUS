"""
Constant LR Scheduler
=====================

고정 learning rate (변경 없음)
"""

import torch.optim.lr_scheduler as lr_scheduler


class ConstantLRScheduler(lr_scheduler.LambdaLR):
    """
    Constant learning rate scheduler (no change).
    """
    
    def __init__(self, optimizer, **kwargs):
        """
        Args:
            optimizer: PyTorch optimizer
            **kwargs: Ignored (for compatibility)
        """
        super().__init__(optimizer, lr_lambda=lambda epoch: 1.0)

