#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning rate schedulers for GENESIS training.

This module provides various learning rate schedulers including:
- Cosine Annealing
- Reduce on Plateau
- Step Decay
- Linear Decay
- Custom warmup schedulers
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import torch
import torch.optim as optim
from config import TrainingConfig


def create_scheduler(optimizer: torch.optim.Optimizer, training_config: TrainingConfig) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create a learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        training_config: Training configuration
        
    Returns:
        Learning rate scheduler or None
    """
    scheduler_name = training_config.scheduler
    
    if scheduler_name is None:
        return None
    
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == "cosine":
        return create_cosine_scheduler(optimizer, training_config)
    elif scheduler_name == "plateau":
        return create_plateau_scheduler(optimizer, training_config)
    elif scheduler_name == "step":
        return create_step_scheduler(optimizer, training_config)
    elif scheduler_name == "linear":
        return create_linear_scheduler(optimizer, training_config)
    elif scheduler_name == "exponential":
        return create_exponential_scheduler(optimizer, training_config)
    elif scheduler_name == "polynomial":
        return create_polynomial_scheduler(optimizer, training_config)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def create_cosine_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> optim.lr_scheduler.CosineAnnealingLR:
    """Create cosine annealing scheduler."""
    t_max = config.cosine_t_max or config.num_epochs
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=0.0
    )


def create_plateau_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> optim.lr_scheduler.ReduceLROnPlateau:
    """Create reduce on plateau scheduler."""
    verbose = getattr(config, 'plateau_verbose', False)
    
    # Build kwargs conditionally (verbose not supported in older PyTorch versions)
    kwargs = {
        'optimizer': optimizer,
        'mode': config.plateau_mode,
        'factor': config.plateau_factor,
        'patience': config.plateau_patience,
        'threshold': config.plateau_threshold,
        'cooldown': config.plateau_cooldown,
    }
    
    # Add verbose only if supported (PyTorch 2.0+)
    import inspect
    if 'verbose' in inspect.signature(optim.lr_scheduler.ReduceLROnPlateau.__init__).parameters:
        kwargs['verbose'] = verbose
    
    return optim.lr_scheduler.ReduceLROnPlateau(**kwargs)


def create_step_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> optim.lr_scheduler.StepLR:
    """Create step decay scheduler."""
    return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.step_size,
        gamma=config.step_gamma
    )


def create_linear_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> optim.lr_scheduler.LinearLR:
    """Create linear decay scheduler."""
    return optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.linear_start_factor,
        end_factor=config.linear_end_factor,
        total_iters=config.num_epochs
    )


def create_exponential_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> optim.lr_scheduler.ExponentialLR:
    """Create exponential decay scheduler."""
    gamma = getattr(config, 'exponential_gamma', 0.95)
    return optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=gamma
    )


def create_polynomial_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> optim.lr_scheduler.PolynomialLR:
    """Create polynomial decay scheduler."""
    power = getattr(config, 'polynomial_power', 2.0)
    return optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=config.num_epochs,
        power=power
    )


def create_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    base_scheduler: torch.optim.lr_scheduler._LRScheduler,
    warmup_steps: int,
    warmup_start_lr: float = 0.0
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a warmup scheduler that combines warmup with a base scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        base_scheduler: Base scheduler to use after warmup
        warmup_steps: Number of warmup steps
        warmup_start_lr: Starting learning rate for warmup
        
    Returns:
        Combined warmup scheduler
    """
    # Create warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_lr / optimizer.param_groups[0]['lr'],
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Create sequential scheduler
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, base_scheduler],
        milestones=[warmup_steps]
    )


def get_scheduler_info() -> Dict[str, Dict[str, Any]]:
    """Get information about available schedulers."""
    return {
        "cosine": {
            "name": "Cosine Annealing",
            "description": "Cosine annealing with optional warm restarts",
            "best_for": "Stable convergence, avoiding local minima",
            "parameters": ["cosine_t_max"],
            "advantages": ["Smooth decay", "Good for fine-tuning", "Avoids local minima"],
            "disadvantages": ["May be slow to converge", "Requires careful tuning"]
        },
        "plateau": {
            "name": "Reduce on Plateau",
            "description": "Reduces learning rate when metric stops improving",
            "best_for": "When you have validation metrics to monitor",
            "parameters": ["plateau_patience", "plateau_factor"],
            "advantages": ["Automatic adaptation", "Good for validation-based training"],
            "disadvantages": ["Requires validation metric", "May be too conservative"]
        },
        "step": {
            "name": "Step Decay",
            "description": "Reduces learning rate by factor at fixed intervals",
            "best_for": "Simple, predictable decay schedule",
            "parameters": ["step_size", "step_gamma"],
            "advantages": ["Simple", "Predictable", "Easy to tune"],
            "disadvantages": ["Discontinuous", "May miss optimal learning rates"]
        },
        "linear": {
            "name": "Linear Decay",
            "description": "Linear decay from start to end factor",
            "best_for": "Smooth, predictable decay",
            "parameters": ["linear_start_factor", "linear_end_factor"],
            "advantages": ["Smooth decay", "Simple", "Predictable"],
            "disadvantages": ["May decay too quickly", "Fixed schedule"]
        },
        "exponential": {
            "name": "Exponential Decay",
            "description": "Exponential decay with fixed gamma",
            "best_for": "Rapid initial decay",
            "parameters": ["exponential_gamma"],
            "advantages": ["Rapid decay", "Simple"],
            "disadvantages": ["May decay too quickly", "Requires careful tuning"]
        },
        "polynomial": {
            "name": "Polynomial Decay",
            "description": "Polynomial decay with specified power",
            "best_for": "Customizable decay curves",
            "parameters": ["polynomial_power"],
            "advantages": ["Flexible", "Smooth decay"],
            "disadvantages": ["More complex", "Requires tuning"]
        }
    }


def print_scheduler_comparison():
    """Print a comparison of all available schedulers."""
    info = get_scheduler_info()
    
    print("Available Learning Rate Schedulers:")
    print("=" * 80)
    
    for scheduler_name, scheduler_info in info.items():
        print(f"\n{scheduler_info['name']} ({scheduler_name.upper()})")
        print(f"Description: {scheduler_info['description']}")
        print(f"Best for: {scheduler_info['best_for']}")
        print(f"Parameters: {', '.join(scheduler_info['parameters'])}")
        print(f"Advantages: {', '.join(scheduler_info['advantages'])}")
        print(f"Disadvantages: {', '.join(scheduler_info['disadvantages'])}")
        print("-" * 80)


def create_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create scheduler from configuration dictionary.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_config: Scheduler configuration dictionary
        
    Returns:
        Learning rate scheduler or None
    """
    scheduler_name = scheduler_config.get('scheduler')
    if scheduler_name is None:
        return None
    
    # Create a temporary training config with scheduler parameters
    class TempConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    temp_config = TempConfig(scheduler_config)
    return create_scheduler(optimizer, temp_config)


if __name__ == "__main__":
    # Test scheduler creation
    import torch.nn as nn
    
    # Create dummy model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Test different schedulers
    schedulers = ["cosine", "plateau", "step", "linear"]
    
    for scheduler_name in schedulers:
        try:
            # Create temporary config
            class TempConfig:
                scheduler = scheduler_name
                num_epochs = 100
                cosine_t_max = 100
                plateau_patience = 10
                plateau_factor = 0.5
                plateau_mode = "min"
                plateau_threshold = 1e-4
                plateau_cooldown = 0
                step_size = 30
                step_gamma = 0.1
                linear_start_factor = 1.0
                linear_end_factor = 0.0
            
            config = TempConfig()
            scheduler = create_scheduler(optimizer, config)
            print(f"{scheduler_name.upper()}: {type(scheduler).__name__} ✓")
            
        except Exception as e:
            print(f"{scheduler_name.upper()}: ERROR - {e}")
    
    print("\nScheduler comparison:")
    print_scheduler_comparison()
