#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models package for Flow Matching.

The models are shared with Diffusion (both predict velocity/noise fields).
"""

from .factory import ModelFactory

__all__ = ["ModelFactory"]

