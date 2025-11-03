#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training package for GENESIS IceCube diffusion model.

This package provides comprehensive training functionality including:
- Trainer class with advanced features
- Scheduler support (cosine, plateau, step, linear)
- Logging and monitoring
- Checkpointing and resuming
- Mixed precision training
- Multi-GPU support
"""

from .trainer import Trainer, TrainingConfig, create_trainer
from .schedulers import create_scheduler, get_scheduler_info
from .logging import setup_logging, LoggingConfig
from .checkpointing import CheckpointManager
from .utils import (
    setup_training_environment,
    get_training_summary,
    validate_training_config,
    create_training_script,
    EarlyStopping
)
from .evaluation import compare_generated_vs_real

__all__ = [
    "Trainer",
    "TrainingConfig",
    "create_trainer",
    "create_scheduler",
    "get_scheduler_info",
    "setup_logging",
    "LoggingConfig",
    "CheckpointManager",
    "setup_training_environment",
    "get_training_summary",
    "validate_training_config",
    "create_training_script",
    "EarlyStopping",
    "compare_generated_vs_real",
]

__version__ = "1.0.0"
__author__ = "Minje Park"
__email__ = "pmj032400@naver.com"
