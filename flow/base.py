"""
Base class for Flow Matching methods.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


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

    def _make_time_batch(self, batch_size: int, t_value: float, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size,), t_value, device=device, dtype=torch.float32)

    def _make_null_label(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, 6, device=device, dtype=dtype)

    def _rhs(
        self,
        model: nn.Module,
        x: torch.Tensor,
        s: float,
        label: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Forward-time RHS for reverse sampling: dx/ds = -v_theta(x, t=1-s)."""
        t_model = 1.0 - float(s)
        t_batch = self._make_time_batch(x.shape[0], t_model, device)
        if label is None:
            label = self._make_null_label(x.shape[0], device, x.dtype)
        return -model(x, t_batch, label)

    def _step_heun(
        self,
        model: nn.Module,
        x: torch.Tensor,
        s: float,
        h: float,
        label: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        k1 = self._rhs(model, x, s, label, device)
        x_predict = x + h * k1
        k2 = self._rhs(model, x_predict, s + h, label, device)
        return x + 0.5 * h * (k1 + k2)

    def _step_rk4(
        self,
        model: nn.Module,
        x: torch.Tensor,
        s: float,
        h: float,
        label: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        k1 = self._rhs(model, x, s, label, device)
        k2 = self._rhs(model, x + 0.5 * h * k1, s + 0.5 * h, label, device)
        k3 = self._rhs(model, x + 0.5 * h * k2, s + 0.5 * h, label, device)
        k4 = self._rhs(model, x + h * k3, s + h, label, device)
        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def sample_ode_heun(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        num_steps: int = 50,
        label: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Sample using Heun's method (improved Euler / RK2).
        """
        if device is None:
            device = x1.device

        x = x1.clone()
        s = 0.0
        h = 1.0 / max(int(num_steps), 1)

        for _ in range(max(int(num_steps), 1)):
            step = min(h, 1.0 - s)
            x = self._step_heun(model, x, s, step, label, device)
            s += step

        return x

    def sample_ode_dopri5(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        num_steps: int = 50,
        label: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> torch.Tensor:
        """
        Sample using adaptive Dormand-Prince 5(4) solver.

        num_steps sets the initial step size and soft budget for the solver.
        """
        if device is None:
            device = x1.device

        x = x1.clone()
        s = 0.0
        remaining = 1.0 - s
        num_steps = max(int(num_steps), 1)
        h = 1.0 / num_steps
        min_step = 1e-6
        safety = 0.9
        min_factor = 0.2
        max_factor = 5.0
        max_attempts = max(1000, num_steps * 50)
        attempts = 0

        def rhs(x_state: torch.Tensor, s_state: float) -> torch.Tensor:
            return self._rhs(model, x_state, s_state, label, device)

        while s < 1.0 - 1e-10 and attempts < max_attempts:
            attempts += 1
            remaining = 1.0 - s
            step = min(h, remaining)

            k1 = rhs(x, s)
            k2 = rhs(x + step * (1.0 / 5.0) * k1, s + step * (1.0 / 5.0))
            k3 = rhs(x + step * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2), s + step * (3.0 / 10.0))
            k4 = rhs(
                x + step * ((44.0 / 45.0) * k1 + (-56.0 / 15.0) * k2 + (32.0 / 9.0) * k3),
                s + step * (4.0 / 5.0),
            )
            k5 = rhs(
                x + step * ((19372.0 / 6561.0) * k1 + (-25360.0 / 2187.0) * k2 + (64448.0 / 6561.0) * k3 + (-212.0 / 729.0) * k4),
                s + step * (8.0 / 9.0),
            )
            k6 = rhs(
                x + step * ((9017.0 / 3168.0) * k1 + (-355.0 / 33.0) * k2 + (46732.0 / 5247.0) * k3 + (49.0 / 176.0) * k4 + (-5103.0 / 18656.0) * k5),
                s + step,
            )
            x5 = x + step * (
                (35.0 / 384.0) * k1
                + (500.0 / 1113.0) * k3
                + (125.0 / 192.0) * k4
                + (-2187.0 / 6784.0) * k5
                + (11.0 / 84.0) * k6
            )
            k7 = rhs(x5, s + step)
            x4 = x + step * (
                (5179.0 / 57600.0) * k1
                + (7571.0 / 16695.0) * k3
                + (393.0 / 640.0) * k4
                + (-92097.0 / 339200.0) * k5
                + (187.0 / 2100.0) * k6
                + (1.0 / 40.0) * k7
            )

            err = torch.max(torch.abs(x5 - x4))
            scale = torch.max(torch.maximum(x.abs(), x5.abs()))
            tol = atol + rtol * scale
            err_ratio = (err / tol).item() if tol.item() > 0 else 0.0

            if err_ratio <= 1.0 or step <= min_step:
                x = x5
                s += step
                if err_ratio == 0.0:
                    factor = max_factor
                else:
                    factor = safety * (err_ratio ** (-0.2))
                    factor = min(max_factor, max(min_factor, factor))
                h = max(min_step, min(1.0 - s, step * factor))
            else:
                factor = safety * (err_ratio ** (-0.2))
                factor = min(1.0, max(min_factor, factor))
                h = max(min_step, step * factor)

        if attempts >= max_attempts and s < 1.0 - 1e-8:
            raise RuntimeError("sample_ode_dopri5 exceeded the maximum number of solver attempts")

        return x
