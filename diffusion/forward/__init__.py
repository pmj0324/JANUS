"""
Forward Diffusion Process
=========================

Functions for applying forward diffusion (adding noise) to data.
"""

from .forward_diffusion import (
    q_sample_batch,
    apply_forward_diffusion,
)

__all__ = [
    "q_sample_batch",
    "apply_forward_diffusion",
]
