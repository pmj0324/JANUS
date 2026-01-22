"""
ReduceLROnPlateau Scheduler
============================

Learning rate reduction when metric plateaus.
"""

import torch.optim.lr_scheduler as lr_scheduler


class PlateauLRScheduler(lr_scheduler.ReduceLROnPlateau):
    """
    Reduce learning rate when a metric has stopped improving.
    
    This scheduler monitors a metric and reduces LR when the metric
    stops improving (plateaus).
    """
    
    def __init__(
        self,
        optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8,
        verbose: bool = False,
        **kwargs
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            mode: "min" (reduce when metric stops decreasing) or 
                  "max" (reduce when metric stops increasing)
            factor: Factor by which LR is reduced (default: 0.1)
            patience: Number of epochs with no improvement before reducing LR (default: 10)
            threshold: Threshold for measuring new optimum (default: 1e-4)
            threshold_mode: "rel" (relative) or "abs" (absolute) (default: "rel")
            cooldown: Number of epochs to wait before resuming normal operation (default: 0)
            min_lr: Lower bound on learning rate (default: 0.0)
            eps: Minimal decay applied to LR (default: 1e-8)
            verbose: Print message when LR is reduced (default: False)
            **kwargs: Ignored (for compatibility)
        """
        super().__init__(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose
        )

