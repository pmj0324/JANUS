"""
Step LR Scheduler
=================

Step decay learning rate scheduler.
"""

import torch.optim.lr_scheduler as lr_scheduler


class StepLRScheduler(lr_scheduler.StepLR):
    """
    Step decay learning rate scheduler.
    
    LR is reduced by gamma every step_size epochs.
    """
    
    def __init__(
        self,
        optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
        **kwargs
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            step_size: Period of learning rate decay (in epochs)
            gamma: Multiplicative factor of learning rate decay (default: 0.1)
            last_epoch: Index of last epoch (default: -1)
            verbose: Print message when LR is updated (default: False)
            **kwargs: Ignored (for compatibility)
        """
        super().__init__(
            optimizer=optimizer,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch,
            verbose=verbose
        )

