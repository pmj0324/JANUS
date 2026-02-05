"""
Rectified Flow (Straightening Flow) implementation.
Simple and fast flow matching with straight paths.
"""

import torch
import torch.nn as nn
from typing import Optional
from .base import BaseFlowMatching


class RectifiedFlow(BaseFlowMatching):
    """
    Rectified Flow: Straight paths connecting data and noise.
    
    Path: x_t = (1-t) * x_0 + t * x_1
    Velocity: u_t = x_1 - x_0
    """
    
    def __init__(self):
        super().__init__()
    
    def compute_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute straight path: x_t = (1-t) * x_0 + t * x_1
        
        Args:
            x0: Data samples (B, C, L)
            x1: Noise samples (B, C, L)
            t: Time values (B,) in [0, 1]
        
        Returns:
            x_t: Path samples (B, C, L)
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, 1)
        t_reshaped = t.view(-1, 1, 1)
        x_t = (1 - t_reshaped) * x0 + t_reshaped * x1
        return x_t
    
    def compute_velocity(self, x0: torch.Tensor, x1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute ground truth velocity: u_t = x_1 - x_0
        
        Args:
            x0: Data samples (B, C, L)
            x1: Noise samples (B, C, L)
            x_t: Path samples (B, C, L) - not used for rectified flow
            t: Time values (B,) in [0, 1] - not used for rectified flow
        
        Returns:
            u_t: Ground truth velocity (B, C, L)
        """
        return x1 - x0
    
    def sample_ode_euler(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        num_steps: int = 50,
        label: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Sample using Euler ODE solver: dx/dt = -v_θ(x_t, t)
        (Negative because we go from t=1 to t=0)
        
        Args:
            model: Velocity prediction model
            x1: Initial noise (B, C, L)
            num_steps: Number of ODE steps
            label: Conditional label (B, label_dim) or None
            device: Device to run on
        
        Returns:
            x0: Generated samples (B, C, L)
        """
        if device is None:
            device = x1.device
        
        x = x1.clone()
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = 1.0 - (i + 1) * dt  # Go from 1 to 0
            t_batch = torch.full((x.shape[0],), t_val, device=device, dtype=torch.float32)
            
            # Predict velocity
            if label is not None:
                v_pred = model(x, t_batch, label)
            else:
                # Unconditional: use zero label
                zero_label = torch.zeros(x.shape[0], 6, device=device)
                v_pred = model(x, t_batch, zero_label)
            
            # Euler step: x_{t-dt} = x_t - dt * v_θ(x_t, t)
            x = x - dt * v_pred
        
        return x
    
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
        if device is None:
            device = x1.device
        
        x = x1.clone()
        dt = 1.0 / num_steps
        
        def get_velocity(x_t, t_val):
            t_batch = torch.full((x_t.shape[0],), t_val, device=device, dtype=torch.float32)
            if label is not None:
                return model(x_t, t_batch, label)
            else:
                zero_label = torch.zeros(x_t.shape[0], 6, device=device)
                return model(x_t, t_batch, zero_label)
        
        for i in range(num_steps):
            t_val = 1.0 - i * dt  # Current time
            t_next = 1.0 - (i + 1) * dt  # Next time
            
            # RK4 steps
            k1 = get_velocity(x, t_val)
            k2 = get_velocity(x - dt * k1 / 2, (t_val + t_next) / 2)
            k3 = get_velocity(x - dt * k2 / 2, (t_val + t_next) / 2)
            k4 = get_velocity(x - dt * k3, t_next)
            
            # RK4 update: x_{t-dt} = x_t - dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            x = x - dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        return x
