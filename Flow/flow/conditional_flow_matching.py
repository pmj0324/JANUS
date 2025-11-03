#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional Flow Matching (CFM) Module
========================================

Implementation of Conditional Flow Matching (Lipman et al., 2023)
for generating IceCube PMT signals.

Key differences from Diffusion:
- Direct regression to velocity field v_t(x)
- Straight-line paths between noise and data
- ODE-based sampling (no stochastic component)
- Faster sampling with fewer steps

Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2023)

Author: Minje Park
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowConfig:
    """Configuration for Flow Matching."""
    sigma_min: float = 1e-4
    use_ode_solver: str = "euler"  # "euler", "midpoint", "rk4"
    num_steps: int = 50
    use_cfm: bool = True
    use_cfg: bool = True
    cfg_scale: float = 2.0
    cfg_dropout: float = 0.1


class ConditionalFlowMatching(nn.Module):
    """
    Conditional Flow Matching for p(x|c) with geometry.
    
    Model predicts velocity field v_t(x_sig_t, t, label, geom) → (B, 2, L)
    
    Flow Matching uses:
    - Optimal Transport path: x_t = (1-t)*x_0 + t*x_1 + σ_t*ε
    - Velocity field: v_t = (x_1 - x_0) / (1 - σ_min)
    
    Args:
        model: Neural network model (e.g., PMTDiT)
        cfg: FlowConfig object
    """
    
    def __init__(self, model: nn.Module, cfg: FlowConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        
        # Register sigma_min as buffer
        self.register_buffer("sigma_min", torch.tensor(cfg.sigma_min))
    
    def get_normalization_params(self):
        """Get normalization parameters from the model."""
        if hasattr(self.model, 'affine_offset'):
            affine_offset = self.model.affine_offset.squeeze().cpu()
            affine_scale = self.model.affine_scale.squeeze().cpu()
            label_offset = self.model.label_offset.cpu() if hasattr(self.model, 'label_offset') else None
            label_scale = self.model.label_scale.cpu() if hasattr(self.model, 'label_scale') else None
            time_transform = self.model.time_transform if hasattr(self.model, 'time_transform') else "ln"
            return affine_offset, affine_scale, label_offset, label_scale, time_transform
        else:
            return None, None, None, None, "ln"
    
    def compute_conditional_flow(
        self,
        x0_sig: torch.Tensor,
        x1_sig: torch.Tensor,
        t: torch.Tensor,
        sigma_t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conditional flow path and velocity.
        
        Conditional Flow Matching uses optimal transport path:
        x_t = (1-t)*x_0 + t*x_1 + σ_t*ε
        
        Target velocity:
        v_t = dx_t/dt = x_1 - x_0 + dσ_t/dt * ε
        
        For simplicity, we use constant σ_t = σ_min:
        v_t = x_1 - x_0
        
        Args:
            x0_sig: Source (noise) (B, 2, L)
            x1_sig: Target (data) (B, 2, L)
            t: Time (B,) in [0, 1]
            sigma_t: Optional noise schedule
        
        Returns:
            x_t: Interpolated state (B, 2, L)
            v_t: Target velocity (B, 2, L)
        """
        B = x0_sig.size(0)
        device = x0_sig.device
        
        # Reshape t for broadcasting: (B,) -> (B, 1, 1)
        t_expanded = t.view(B, 1, 1)
        
        # Add small noise for numerical stability
        if sigma_t is None:
            sigma_t = self.sigma_min
        
        # Sample noise
        eps = torch.randn_like(x0_sig)
        
        # Conditional flow path: x_t = (1-t)*x_0 + t*x_1 + σ*ε
        x_t = (1 - t_expanded) * x0_sig + t_expanded * x1_sig + sigma_t * eps
        
        # Target velocity: v_t = x_1 - x_0
        # (For constant sigma, dσ/dt = 0, so velocity is just the difference)
        v_t = x1_sig - x0_sig
        
        return x_t, v_t
    
    def loss(
        self,
        x0_sig: torch.Tensor,
        geom: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Flow Matching training loss.
        
        Loss: E_{t, x_0, x_1} ||v_θ(x_t, t, c) - v_t||^2
        
        where v_t = x_1 - x_0 (target velocity)
        
        Args:
            x0_sig: Clean signals (B, 2, L) - this is x_1 (data)
            geom: Geometry (B, 3, L) - kept clean
            label: Condition c (B, 6)
        
        Returns:
            MSE loss scalar
        """
        B = x0_sig.size(0)
        device = x0_sig.device
        
        # Sample random time t ~ Uniform[0, 1]
        t = torch.rand(B, device=device)
        
        # Sample source noise x_0 ~ N(0, I)
        x1_sig = x0_sig  # Data is the target
        x0_noise = torch.randn_like(x0_sig)  # Source is noise
        
        # Compute conditional flow
        x_t, v_target = self.compute_conditional_flow(x0_noise, x1_sig, t)
        
        # Classifier-free guidance: randomly drop conditions during training
        if self.cfg.use_cfg and self.training:
            drop_mask = torch.rand(B, device=device) < self.cfg.cfg_dropout
            label_conditioned = label.clone()
            label_conditioned[drop_mask] = 0.0
            
            v_pred = self.model(x_t, geom, t, label_conditioned)
        else:
            v_pred = self.model(x_t, geom, t, label)
        
        # Flow matching loss: match predicted velocity to target velocity
        return F.mse_loss(v_pred, v_target)
    
    @torch.no_grad()
    def sample_euler(
        self,
        label: torch.Tensor,
        geom: torch.Tensor,
        shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Sample using Euler ODE solver.
        
        ODE: dx/dt = v_θ(x_t, t, c)
        Euler step: x_{t+dt} = x_t + dt * v_θ(x_t, t, c)
        
        Args:
            label: Condition (N, 6)
            geom: Geometry (N, 3, L)
            shape: (N, 2, L)
        
        Returns:
            Generated samples (N, 2, L)
        """
        N, C, L = shape
        device = label.device
        
        # Start from noise x_0 ~ N(0, I)
        x_t = torch.randn(N, C, L, device=device)
        
        # Time discretization
        dt = 1.0 / self.cfg.num_steps
        
        # Integrate ODE from t=0 to t=1
        for step in range(self.cfg.num_steps):
            t = torch.full((N,), step * dt, device=device)
            
            # Classifier-free guidance
            if self.cfg.use_cfg and self.cfg.cfg_scale != 1.0:
                # Conditional prediction
                v_cond = self.model(x_t, geom, t, label)
                
                # Unconditional prediction (zero labels)
                label_uncond = torch.zeros_like(label)
                v_uncond = self.model(x_t, geom, t, label_uncond)
                
                # CFG: v = v_uncond + scale * (v_cond - v_uncond)
                v_t = v_uncond + self.cfg.cfg_scale * (v_cond - v_uncond)
            else:
                v_t = self.model(x_t, geom, t, label)
            
            # Euler step
            x_t = x_t + dt * v_t
        
        return x_t
    
    @torch.no_grad()
    def sample_midpoint(
        self,
        label: torch.Tensor,
        geom: torch.Tensor,
        shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Sample using Midpoint ODE solver (more accurate than Euler).
        
        Midpoint method:
        k1 = v_θ(x_t, t, c)
        k2 = v_θ(x_t + dt/2 * k1, t + dt/2, c)
        x_{t+dt} = x_t + dt * k2
        """
        N, C, L = shape
        device = label.device
        
        x_t = torch.randn(N, C, L, device=device)
        dt = 1.0 / self.cfg.num_steps
        
        for step in range(self.cfg.num_steps):
            t = torch.full((N,), step * dt, device=device)
            t_mid = torch.full((N,), (step + 0.5) * dt, device=device)
            
            # k1 = v(x_t, t)
            if self.cfg.use_cfg and self.cfg.cfg_scale != 1.0:
                v_cond = self.model(x_t, geom, t, label)
                v_uncond = self.model(x_t, geom, t, torch.zeros_like(label))
                k1 = v_uncond + self.cfg.cfg_scale * (v_cond - v_uncond)
            else:
                k1 = self.model(x_t, geom, t, label)
            
            # k2 = v(x_t + dt/2 * k1, t + dt/2)
            x_mid = x_t + (dt / 2) * k1
            if self.cfg.use_cfg and self.cfg.cfg_scale != 1.0:
                v_cond = self.model(x_mid, geom, t_mid, label)
                v_uncond = self.model(x_mid, geom, t_mid, torch.zeros_like(label))
                k2 = v_uncond + self.cfg.cfg_scale * (v_cond - v_uncond)
            else:
                k2 = self.model(x_mid, geom, t_mid, label)
            
            # Update
            x_t = x_t + dt * k2
        
        return x_t
    
    @torch.no_grad()
    def sample(
        self,
        label: torch.Tensor,
        geom: torch.Tensor,
        shape: Tuple[int, int, int],
        return_all_timesteps: bool = False
    ) -> torch.Tensor:
        """
        Generate samples using Flow Matching ODE.
        
        Args:
            label: Condition (N, 6)
            geom: Geometry (N, 3, L) or (3, L)
            shape: (N, 2, L)
            return_all_timesteps: If True, return all intermediate states
        
        Returns:
            Generated samples (N, 2, L)
        """
        # Ensure geom has batch dimension
        if geom.ndim == 2:
            geom = geom.unsqueeze(0).expand(shape[0], -1, -1)
        
        # Choose ODE solver
        if self.cfg.use_ode_solver == "midpoint":
            samples = self.sample_midpoint(label, geom, shape)
        elif self.cfg.use_ode_solver == "euler":
            samples = self.sample_euler(label, geom, shape)
        else:
            raise ValueError(f"Unknown ODE solver: {self.cfg.use_ode_solver}")
        
        return samples

