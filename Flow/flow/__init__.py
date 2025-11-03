#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow Matching module for GENESIS.

This module provides Conditional Flow Matching for PMT signal generation.
"""

from .conditional_flow_matching import ConditionalFlowMatching, FlowConfig

__all__ = [
    "ConditionalFlowMatching",
    "FlowConfig",
]

