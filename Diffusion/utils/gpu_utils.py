"""
Compatibility wrapper for legacy imports.

Older code imports GPU utilities from `utils.gpu_utils`, while the actual
implementation lives under `gpu_tools.utils.gpu_utils`. This module simply
re-exports the key functions to preserve backward compatibility.
"""

from gpu_tools.utils.gpu_utils import (
    print_gpu_info,
    print_memory_analysis,
    get_gpu_info,
    monitor_gpu_memory,
    estimate_model_memory,
    estimate_batch_memory,
    recommend_batch_size,
    auto_select_batch_size,
)

__all__ = [
    "print_gpu_info",
    "print_memory_analysis",
    "get_gpu_info",
    "monitor_gpu_memory",
    "estimate_model_memory",
    "estimate_batch_memory",
    "recommend_batch_size",
    "auto_select_batch_size",
]


