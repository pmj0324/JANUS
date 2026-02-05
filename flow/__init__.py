"""
Flow Matching implementations for GENESIS.
Supports Rectified Flow, Conditional Flow Matching, and Optimal Transport Flow.
"""

from .base import BaseFlowMatching
from .rectified_flow import RectifiedFlow
from .conditional_flow import ConditionalFlowMatching
from .optimal_transport import OptimalTransportFlow

__all__ = [
    "BaseFlowMatching",
    "RectifiedFlow",
    "ConditionalFlowMatching",
    "OptimalTransportFlow",
]
