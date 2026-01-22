"""
Learning Rate Schedulers
========================

LR scheduler들을 개별 파일로 관리:
- constant: 고정 LR
- linear: Linear decay
- cosine: Cosine annealing
- warmup_cosine: Warmup + Cosine (wbcosine)
- plateau: ReduceLROnPlateau
- step: Step decay
- exponential: Exponential decay

Usage:
    from training.schedulers import get_scheduler
    
    scheduler = get_scheduler(
        "warmup_cosine",
        optimizer=optimizer,
        num_training_steps=total_steps,
        warmup_proportion=0.003,
        num_cycles=2
    )
"""

from .constant import ConstantLRScheduler
from .linear import LinearLRScheduler
from .cosine import CosineLRScheduler
from .warmup_cosine import WarmupCosineLRScheduler
from .plateau import PlateauLRScheduler
from .step import StepLRScheduler
from .exponential import ExponentialLRScheduler

__all__ = [
    # Scheduler classes
    "ConstantLRScheduler",
    "LinearLRScheduler",
    "CosineLRScheduler",
    "WarmupCosineLRScheduler",
    "PlateauLRScheduler",
    "StepLRScheduler",
    "ExponentialLRScheduler",
    
    # Factory function
    "get_scheduler",
]


def get_scheduler(
    scheduler_type: str,
    optimizer,
    **kwargs
):
    """
    Factory function for creating LR schedulers.
    
    Args:
        scheduler_type: "constant", "linear", "cosine", "warmup_cosine", 
                       "plateau", "step", "exponential"
        optimizer: PyTorch optimizer
        **kwargs: Scheduler-specific parameters
    
    Returns:
        LR scheduler instance
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "constant":
        return ConstantLRScheduler(optimizer, **kwargs)
    elif scheduler_type == "linear":
        return LinearLRScheduler(optimizer, **kwargs)
    elif scheduler_type == "cosine":
        return CosineLRScheduler(optimizer, **kwargs)
    elif scheduler_type in ["warmup_cosine", "wbcosine"]:
        return WarmupCosineLRScheduler(optimizer, **kwargs)
    elif scheduler_type == "plateau":
        return PlateauLRScheduler(optimizer, **kwargs)
    elif scheduler_type == "step":
        return StepLRScheduler(optimizer, **kwargs)
    elif scheduler_type == "exponential":
        return ExponentialLRScheduler(optimizer, **kwargs)
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Choose from: constant, linear, cosine, warmup_cosine, plateau, step, exponential"
        )

