"""
Conditional Flow Matching (CFM) implementation.
More flexible paths using conditional distributions.
"""

import torch
import torch.nn as nn
from typing import Optional
from .base import BaseFlowMatching


class ConditionalFlowMatching(BaseFlowMatching):
    """
    Conditional Flow Matching: Flexible paths using conditional distributions.
    
    Uses Gaussian path: x_t ~ N(μ_t, σ_t²) where
    μ_t = (1-t) * x_0 + t * x_1
    σ_t² = t * (1-t) * σ²
    
    Velocity: u_t = (x_1 - x_0) + (x_t - μ_t) * (2t - 1) / (t(1-t) + ε)
    """
    
    def __init__(self, sigma: float = 0.1, epsilon: float = 1e-5):
        """
        Args:
            sigma: Standard deviation for Gaussian path
            epsilon: Small value to prevent division by zero
        """
        super().__init__()
        self.sigma = sigma
        self.epsilon = epsilon
    
    def compute_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample from Gaussian path: x_t ~ N(μ_t, σ_t²)
        
        Args:
            x0: Data samples (B, C, L)
            x1: Noise samples (B, C, L)
            t: Time values (B,) in [0, 1]
        
        Returns:
            x_t: Path samples (B, C, L)
        """
        # Mean: μ_t = (1-t) * x_0 + t * x_1
        t_reshaped = t.view(-1, 1, 1)
        mu_t = (1 - t_reshaped) * x0 + t_reshaped * x1
        
        # Variance: σ_t² = t * (1-t) * σ²
        sigma_t = self.sigma * torch.sqrt(t * (1 - t) + self.epsilon)
        sigma_t = sigma_t.view(-1, 1, 1)
        
        # Sample: x_t = μ_t + σ_t * ε, where ε ~ N(0, I)
        noise = torch.randn_like(x0)
        x_t = mu_t + sigma_t * noise
        
        return x_t
    
    def compute_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute conditional velocity: u_t = (x_1 - x_0) + (x_t - μ_t) * (2t - 1) / (t(1-t) + ε)
        
        Args:
            x0: Data samples (B, C, L)
            x1: Noise samples (B, C, L)
            x_t: Path samples (B, C, L)
            t: Time values (B,) in [0, 1]
        
        Returns:
            u_t: Ground truth velocity (B, C, L)
        """
        # Mean: μ_t = (1-t) * x_0 + t * x_1
        t_reshaped = t.view(-1, 1, 1)
        mu_t = (1 - t_reshaped) * x0 + t_reshaped * x1
        
        # Velocity: u_t = (x_1 - x_0) + (x_t - μ_t) * (2t - 1) / (t(1-t) + ε)
        base_velocity = x1 - x0
        t_denom = t * (1 - t) + self.epsilon
        correction = (x_t - mu_t) * (2 * t - 1).view(-1, 1, 1) / t_denom.view(-1, 1, 1)
        u_t = base_velocity + correction
        
        return u_t
    
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
        Same as Rectified Flow but with CFM-trained model.
        """
        if device is None:
            device = x1.device
        
        x = x1.clone()
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t_val = 1.0 - (i + 1) * dt
            t_batch = torch.full((x.shape[0],), t_val, device=device, dtype=torch.float32)
            
            if label is not None:
                v_pred = model(x, t_batch, label)
            else:
                zero_label = torch.zeros(x.shape[0], 6, device=device)
                v_pred = model(x, t_batch, zero_label)
            
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
        Same as Rectified Flow but with CFM-trained model.
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
            t_val = 1.0 - i * dt
            t_next = 1.0 - (i + 1) * dt
            
            k1 = get_velocity(x, t_val)
            k2 = get_velocity(x - dt * k1 / 2, (t_val + t_next) / 2)
            k3 = get_velocity(x - dt * k2 / 2, (t_val + t_next) / 2)
            k4 = get_velocity(x - dt * k3, t_next)
            
            x = x - dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        return x
