#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging utilities for GENESIS training.

This module provides comprehensive logging functionality including:
- TensorBoard integration
- Weights & Biases integration
- Console logging
- File logging
- Metric tracking and visualization
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    # Logging directories
    log_dir: str = "./logs"
    experiment_name: str = "experiment"
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_log_dir: Optional[str] = None
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "icecube-diffusion"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = None
    
    # Console logging
    use_console: bool = True
    log_level: str = "INFO"
    
    # File logging
    use_file: bool = True
    log_file: Optional[str] = None
    
    # Logging intervals
    log_interval: int = 50
    save_interval: int = 1000
    
    def __post_init__(self):
        if self.tensorboard_log_dir is None:
            self.tensorboard_log_dir = os.path.join(self.log_dir, self.experiment_name)
        
        if self.log_file is None:
            self.log_file = os.path.join(self.log_dir, f"{self.experiment_name}.log")
        
        if self.wandb_tags is None:
            self.wandb_tags = []


class Logger:
    """Comprehensive logger for training."""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.start_time = time.time()
        
        # Setup logging infrastructure
        self._setup_directories()
        self._setup_console_logging()
        self._setup_file_logging()
        self._setup_tensorboard()
        self._setup_wandb()
        
        # Metric tracking
        self.metrics_history = []
        self.current_epoch = 0
        self.current_step = 0
    
    def _setup_directories(self):
        """Create necessary directories."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        if self.config.use_tensorboard:
            os.makedirs(self.config.tensorboard_log_dir, exist_ok=True)
    
    def _setup_console_logging(self):
        """Setup console logging."""
        if not self.config.use_console:
            return
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.console_logger = logging.getLogger('genesis_training')
    
    def _setup_file_logging(self):
        """Setup file logging."""
        if not self.config.use_file:
            return
        
        # Create file handler
        file_handler = logging.FileHandler(self.config.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        if hasattr(self, 'console_logger'):
            self.console_logger.addHandler(file_handler)
        else:
            self.file_logger = logging.getLogger('genesis_file')
            self.file_logger.addHandler(file_handler)
            self.file_logger.setLevel(logging.INFO)
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        if not self.config.use_tensorboard:
            return
        
        self.tensorboard_writer = SummaryWriter(
            log_dir=self.config.tensorboard_log_dir
        )
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        if not self.config.use_wandb or not WANDB_AVAILABLE:
            return
        
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.experiment_name,
            tags=self.config.wandb_tags,
            dir=self.config.log_dir
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to all configured backends.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number (uses current step if None)
        """
        if step is None:
            step = self.current_step
        
        # Update current step
        self.current_step = step
        
        # Store metrics in history
        self.metrics_history.append({
            'step': step,
            'epoch': self.current_epoch,
            'metrics': metrics.copy(),
            'timestamp': time.time()
        })
        
        # Log to TensorBoard
        if hasattr(self, 'tensorboard_writer'):
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, step)
        
        # Log to Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)
        
        # Log to console
        if hasattr(self, 'console_logger'):
            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            self.console_logger.info(f"Step {step}: {metrics_str}")
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Log epoch-level metrics.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of epoch metrics
        """
        self.current_epoch = epoch
        
        # Add epoch prefix to metrics
        epoch_metrics = {f"epoch/{k}": v for k, v in metrics.items()}
        epoch_metrics['epoch'] = epoch
        
        self.log_metrics(epoch_metrics)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration to all backends."""
        # Log to TensorBoard
        if hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.add_text('config', json.dumps(config, indent=2))
        
        # Log to Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.config.update(config)
        
        # Log to console
        if hasattr(self, 'console_logger'):
            self.console_logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    def log_model_info(self, model: torch.nn.Module):
        """Log model information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024**2
        }
        
        # Log to TensorBoard
        if hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.add_text('model_info', json.dumps(model_info, indent=2))
        
        # Log to Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.config.update(model_info)
        
        # Log to console
        if hasattr(self, 'console_logger'):
            self.console_logger.info(f"Model info: {json.dumps(model_info, indent=2)}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters."""
        # Log to TensorBoard
        if hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.add_hparams(hyperparams, {})
        
        # Log to Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.config.update(hyperparams)
        
        # Log to console
        if hasattr(self, 'console_logger'):
            self.console_logger.info(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")
    
    def log_image(self, tag: str, image: np.ndarray, step: Optional[int] = None):
        """Log an image."""
        if step is None:
            step = self.current_step
        
        # Log to TensorBoard
        if hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.add_image(tag, image, step)
        
        # Log to Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({tag: wandb.Image(image)}, step=step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):
        """Log a histogram."""
        if step is None:
            step = self.current_step
        
        # Log to TensorBoard
        if hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.add_histogram(tag, values, step)
        
        # Log to Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({tag: wandb.Histogram(values)}, step=step)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text."""
        if step is None:
            step = self.current_step
        
        # Log to TensorBoard
        if hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.add_text(tag, text, step)
        
        # Log to Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({tag: text}, step=step)
    
    def save_metrics_history(self, filepath: Optional[str] = None):
        """Save metrics history to JSON file."""
        if filepath is None:
            filepath = os.path.join(self.config.log_dir, f"{self.config.experiment_name}_metrics.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        
        if hasattr(self, 'console_logger'):
            self.console_logger.info(f"Metrics history saved to {filepath}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        elapsed_time = time.time() - self.start_time
        
        return {
            'experiment_name': self.config.experiment_name,
            'elapsed_time': elapsed_time,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'total_metrics_logged': len(self.metrics_history),
            'log_dir': self.config.log_dir,
            'tensorboard_log_dir': self.config.tensorboard_log_dir if self.config.use_tensorboard else None,
            'wandb_project': self.config.wandb_project if self.config.use_wandb else None
        }
    
    def close(self):
        """Close all logging backends."""
        # Close TensorBoard
        if hasattr(self, 'tensorboard_writer'):
            self.tensorboard_writer.close()
        
        # Close Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        # Save metrics history
        self.save_metrics_history()
        
        # Log final summary
        if hasattr(self, 'console_logger'):
            summary = self.get_training_summary()
            self.console_logger.info(f"Training completed: {json.dumps(summary, indent=2)}")


def setup_logging(config: LoggingConfig) -> Logger:
    """
    Setup logging infrastructure.
    
    Args:
        config: Logging configuration
        
    Returns:
        Logger instance
    """
    return Logger(config)


def create_logger(
    experiment_name: str,
    log_dir: str = "./logs",
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    **kwargs
) -> Logger:
    """
    Create a logger with default settings.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for logs
        use_tensorboard: Whether to use TensorBoard
        use_wandb: Whether to use Weights & Biases
        **kwargs: Additional configuration options
        
    Returns:
        Logger instance
    """
    config = LoggingConfig(
        experiment_name=experiment_name,
        log_dir=log_dir,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        **kwargs
    )
    
    return setup_logging(config)


if __name__ == "__main__":
    # Test logging
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create logger
        logger = create_logger(
            experiment_name="test_logging",
            log_dir=temp_dir,
            use_tensorboard=True,
            use_wandb=False
        )
        
        # Test logging
        logger.log_config({"test": True, "value": 42})
        logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=1)
        logger.log_epoch(1, {"epoch_loss": 0.4, "epoch_time": 120.5})
        
        # Test summary
        summary = logger.get_training_summary()
        print(f"Training summary: {summary}")
        
        # Close logger
        logger.close()
        
        print("Logging test completed successfully!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
