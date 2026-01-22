"""
Exponential LR Scheduler
=========================

Exponential decay learning rate scheduler.
"""

import torch.optim.lr_scheduler as lr_scheduler


class ExponentialLRScheduler(lr_scheduler.ExponentialLR):
    """
    Exponential decay learning rate scheduler.
    
    LR is multiplied by gamma every epoch.
    """
    
    def __init__(
        self,
        optimizer,
        gamma: float = 0.95,
        last_epoch: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            gamma: Multiplicative factor of learning rate decay (default: 0.95)
            last_epoch: Index of last epoch (default: -1)
            verbose: Print message when LR is updated (default: False)
            **kwargs: Ignored (for compatibility)
        """
        super().__init__(
            optimizer=optimizer,
            gamma=gamma,
            last_epoch=last_epoch,
            verbose=verbose
        )

