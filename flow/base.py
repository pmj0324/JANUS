"""
Base class for Flow Matching methods.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseFlowMatching(ABC):
    """
    Base class for Flow Matching implementations.
    """
    
    def __init__(self):
        pass
    
    @abstractmethod
    def compute_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute path x_t given x_0, x_1, and time t.
        
        Args:
            x0: Data samples (B, C, L)
            x1: Noise samples (B, C, L)
            t: Time values (B,) in [0, 1]
        
        Returns:
            x_t: Path samples (B, C, L)
        """
        pass
    
    @abstractmethod
    def compute_velocity(self, x0: torch.Tensor, x1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute ground truth velocity u_t given x_0, x_1, x_t, and t.
        
        Args:
            x0: Data samples (B, C, L)
            x1: Noise samples (B, C, L)
            x_t: Path samples (B, C, L)
            t: Time values (B,) in [0, 1]
        
        Returns:
            u_t: Ground truth velocity (B, C, L)
        """
        pass
    
    def compute_loss(
        self,
        v_pred: torch.Tensor,
        v_true: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute loss between predicted and true velocity.
        
        Args:
            v_pred: Predicted velocity (B, C, L)
            v_true: True velocity (B, C, L)
            reduction: Loss reduction method ("mean" or "sum")
        
        Returns:
            loss: Scalar loss value
        """
        if reduction == "mean":
            return torch.mean((v_pred - v_true) ** 2)
        elif reduction == "sum":
            return torch.sum((v_pred - v_true) ** 2)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    @abstractmethod
    def sample_ode_euler(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        num_steps: int = 50,
        label: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Sample using Euler ODE solver.
        
        Args:
            model: Velocity prediction model
            x1: Initial noise (B, C, L)
            num_steps: Number of ODE steps
            label: Conditional label (B, label_dim) or None
            device: Device to run on
        
        Returns:
            x0: Generated samples (B, C, L)
        """
        pass
    
    @abstractmethod
    def sample_ode_rk4(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        num_steps: int = 50,
        label: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Sample using 4th-order Runge-Kutta ODE solver.
        
        Args:
            model: Velocity prediction model
            x1: Initial noise (B, C, L)
            num_steps: Number of ODE steps
            label: Conditional label (B, label_dim) or None
            device: Device to run on
        
        Returns:
            x0: Generated samples (B, C, L)
        """
        pass
