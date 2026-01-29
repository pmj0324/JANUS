"""
Device utilities for CUDA / MPS (Apple Silicon) / CPU.
"""
import torch


def get_default_device() -> torch.device:
    """Return best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
