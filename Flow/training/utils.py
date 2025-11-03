#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training utilities for GENESIS.

This module provides utility functions for training including:
- Environment setup
- Configuration validation
- Training script generation
- Performance monitoring
"""

from __future__ import annotations
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import time
import psutil
import torch
import numpy as np

from config import ExperimentConfig, TrainingConfig, ModelConfig, DataConfig, DiffusionConfig


def setup_training_environment(
    config: ExperimentConfig,
    force_cpu: bool = False,
    set_deterministic: bool = True
) -> Dict[str, Any]:
    """
    Setup training environment with proper device selection and settings.
    
    Args:
        config: Experiment configuration
        force_cpu: Force CPU usage even if GPU is available
        set_deterministic: Set deterministic behavior for reproducibility
        
    Returns:
        Environment information dictionary
    """
    env_info = {}
    
    # Device selection
    if force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        env_info['device'] = 'cpu'
        env_info['gpu_available'] = False
    else:
        device = torch.device("cuda")
        env_info['device'] = 'cuda'
        env_info['gpu_available'] = True
        env_info['gpu_count'] = torch.cuda.device_count()
        env_info['gpu_name'] = torch.cuda.get_device_name(0)
        env_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # Set deterministic behavior
    if set_deterministic:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        env_info['deterministic'] = True
    else:
        env_info['deterministic'] = False
    
    # System information
    env_info['python_version'] = sys.version
    env_info['pytorch_version'] = torch.__version__
    env_info['cuda_version'] = torch.version.cuda if torch.cuda.is_available() else None
    env_info['cpu_count'] = psutil.cpu_count()
    env_info['memory_gb'] = psutil.virtual_memory().total / 1024**3
    
    # Update config device
    config.device = str(device)
    
    return env_info


def validate_training_config(config: ExperimentConfig) -> List[str]:
    """
    Validate training configuration and return list of issues.
    
    Args:
        config: Experiment configuration to validate
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    # Check data configuration
    if not os.path.exists(config.data.h5_path):
        issues.append(f"Data file not found: {config.data.h5_path}")
    
    if config.data.batch_size <= 0:
        issues.append(f"Invalid batch size: {config.data.batch_size}")
    
    if config.data.num_workers < 0:
        issues.append(f"Invalid num_workers: {config.data.num_workers}")
    
    # Check training configuration
    if config.training.num_epochs <= 0:
        issues.append(f"Invalid num_epochs: {config.training.num_epochs}")
    
    if config.training.learning_rate <= 0:
        issues.append(f"Invalid learning_rate: {config.training.learning_rate}")
    
    if config.training.weight_decay < 0:
        issues.append(f"Invalid weight_decay: {config.training.weight_decay}")
    
    # Check model configuration
    if config.model.seq_len <= 0:
        issues.append(f"Invalid seq_len: {config.model.seq_len}")
    
    if config.model.hidden <= 0:
        issues.append(f"Invalid hidden: {config.model.hidden}")
    
    if config.model.depth <= 0:
        issues.append(f"Invalid depth: {config.model.depth}")
    
    if config.model.architecture not in ["dit", "cnn", "mlp", "hybrid", "resnet"]:
        issues.append(f"Unknown architecture: {config.model.architecture}")
    
    # Check diffusion configuration
    if config.diffusion.timesteps <= 0:
        issues.append(f"Invalid timesteps: {config.diffusion.timesteps}")
    
    if config.diffusion.beta_start <= 0 or config.diffusion.beta_end <= 0:
        issues.append(f"Invalid beta values: {config.diffusion.beta_start}, {config.diffusion.beta_end}")
    
    if config.diffusion.beta_start >= config.diffusion.beta_end:
        issues.append("beta_start must be less than beta_end")
    
    # Check scheduler configuration
    if config.training.scheduler is not None:
        valid_schedulers = ["cosine", "linear", "step", "plateau", "exponential", "polynomial"]
        if config.training.scheduler not in valid_schedulers:
            issues.append(f"Unknown scheduler: {config.training.scheduler}")
    
    # Check output directories
    for dir_path in [config.training.output_dir, config.training.checkpoint_dir, config.training.log_dir]:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create directory {dir_path}: {e}")
    
    return issues


def get_training_summary(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Get a summary of training configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Training summary dictionary
    """
    # Calculate estimated model size
    if config.model.architecture == "dit":
        # Rough estimate for DiT
        model_params = (
            config.model.hidden * config.model.hidden * config.model.depth * 4 +  # Attention
            config.model.hidden * config.model.hidden * config.model.depth * config.model.mlp_ratio * 2 +  # MLP
            config.model.hidden * config.model.seq_len * 2  # Embeddings
        )
    elif config.model.architecture == "cnn":
        # Rough estimate for CNN
        model_params = config.model.hidden * config.model.depth * 1000  # Rough estimate
    else:
        # Default estimate
        model_params = config.model.hidden * config.model.depth * 1000
    
    model_size_mb = model_params * 4 / 1024**2  # 4 bytes per float32
    
    # Calculate training steps
    # Note: This is a rough estimate, actual steps depend on dataset size
    estimated_steps_per_epoch = 1000  # Placeholder
    total_steps = config.training.num_epochs * estimated_steps_per_epoch
    
    # Calculate estimated training time (rough estimate)
    estimated_time_per_step = 0.1  # seconds (very rough estimate)
    estimated_total_time = total_steps * estimated_time_per_step / 3600  # hours
    
    return {
        'experiment_name': config.experiment_name,
        'architecture': config.model.architecture,
        'model_parameters': int(model_params),
        'model_size_mb': model_size_mb,
        'training_epochs': config.training.num_epochs,
        'estimated_total_steps': total_steps,
        'estimated_training_time_hours': estimated_total_time,
        'batch_size': config.data.batch_size,
        'learning_rate': config.training.learning_rate,
        'scheduler': config.training.scheduler or 'None',
        'optimizer': config.training.optimizer,
        'mixed_precision': config.training.use_amp,
        'device': config.device,
        'data_path': config.data.h5_path,
        'output_dir': config.training.output_dir,
        'checkpoint_dir': config.training.checkpoint_dir,
        'log_dir': config.training.log_dir
    }


def create_training_script(
    config: ExperimentConfig,
    output_path: str,
    script_type: str = "bash"
) -> str:
    """
    Create a training script from configuration.
    
    Args:
        config: Experiment configuration
        output_path: Path to save the script
        script_type: Type of script ("bash" or "python")
        
    Returns:
        Path to created script
    """
    if script_type == "bash":
        return _create_bash_script(config, output_path)
    elif script_type == "python":
        return _create_python_script(config, output_path)
    else:
        raise ValueError(f"Unknown script type: {script_type}")


def _create_bash_script(config: ExperimentConfig, output_path: str) -> str:
    """Create a bash training script."""
    script_content = f"""#!/bin/bash
# Training script for {config.experiment_name}
# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}

set -e  # Exit on error

# Configuration
EXPERIMENT_NAME="{config.experiment_name}"
CONFIG_PATH="configs/{config.experiment_name}.yaml"
DATA_PATH="{config.data.h5_path}"
OUTPUT_DIR="{config.training.output_dir}"
CHECKPOINT_DIR="{config.training.checkpoint_dir}"
LOG_DIR="{config.training.log_dir}"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# Training command
python train.py \\
    --config "$CONFIG_PATH" \\
    --data-path "$DATA_PATH" \\
    --experiment-name "$EXPERIMENT_NAME" \\
    --output-dir "$OUTPUT_DIR" \\
    --checkpoint-dir "$CHECKPOINT_DIR" \\
    --log-dir "$LOG_DIR" \\
    --epochs {config.training.num_epochs} \\
    --batch-size {config.data.batch_size} \\
    --lr {config.training.learning_rate} \\
    --scheduler {config.training.scheduler or 'none'} \\
    --optimizer {config.training.optimizer}

echo "Training completed for $EXPERIMENT_NAME"
"""
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(output_path, 0o755)
    
    return output_path


def _create_python_script(config: ExperimentConfig, output_path: str) -> str:
    """Create a Python training script."""
    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for {config.experiment_name}
Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import load_config_from_file
from training import create_trainer

def main():
    """Main training function."""
    # Load configuration
    config = load_config_from_file("configs/{config.experiment_name}.yaml")
    
    # Override with command line arguments if provided
    if len(sys.argv) > 1:
        config.data.h5_path = sys.argv[1]
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(output_path, 0o755)
    
    return output_path


def monitor_training_performance(
    process_id: Optional[int] = None,
    interval: float = 1.0,
    duration: float = 60.0
) -> Dict[str, List[float]]:
    """
    Monitor training performance metrics.
    
    Args:
        process_id: Process ID to monitor (None for current process)
        interval: Monitoring interval in seconds
        duration: Total monitoring duration in seconds
        
    Returns:
        Dictionary with performance metrics over time
    """
    if process_id is None:
        process = psutil.Process()
    else:
        process = psutil.Process(process_id)
    
    metrics = {
        'cpu_percent': [],
        'memory_percent': [],
        'memory_mb': [],
        'gpu_memory_mb': [],
        'timestamp': []
    }
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            # CPU and memory
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            metrics['cpu_percent'].append(cpu_percent)
            metrics['memory_percent'].append(memory_percent)
            metrics['memory_mb'].append(memory_info.rss / 1024**2)
            metrics['timestamp'].append(time.time())
            
            # GPU memory (if available)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
                metrics['gpu_memory_mb'].append(gpu_memory)
            else:
                metrics['gpu_memory_mb'].append(0.0)
            
            time.sleep(interval)
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
    
    return metrics


def estimate_training_time(
    config: ExperimentConfig,
    dataset_size: Optional[int] = None
) -> Dict[str, float]:
    """
    Estimate training time based on configuration.
    
    Args:
        config: Experiment configuration
        dataset_size: Size of dataset (if known)
        
    Returns:
        Dictionary with time estimates
    """
    # Rough estimates based on model architecture
    time_per_step = {
        'dit': 0.2,      # seconds per step
        'cnn': 0.05,     # seconds per step
        'mlp': 0.02,     # seconds per step
        'hybrid': 0.1,   # seconds per step
        'resnet': 0.08   # seconds per step
    }
    
    # Get base time per step
    base_time = time_per_step.get(config.model.architecture, 0.1)
    
    # Adjust for batch size
    batch_size_factor = 8 / config.data.batch_size  # Normalize to batch size 8
    adjusted_time = base_time * batch_size_factor
    
    # Estimate steps per epoch
    if dataset_size is None:
        # Rough estimate
        steps_per_epoch = 1000
    else:
        steps_per_epoch = dataset_size // config.data.batch_size
    
    # Calculate total time
    total_steps = config.training.num_epochs * steps_per_epoch
    total_time_seconds = total_steps * adjusted_time
    
    return {
        'time_per_step_seconds': adjusted_time,
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'total_time_seconds': total_time_seconds,
        'total_time_minutes': total_time_seconds / 60,
        'total_time_hours': total_time_seconds / 3600,
        'estimated_epoch_time_minutes': (steps_per_epoch * adjusted_time) / 60
    }


class EarlyStopping:
    """
    Early stopping to stop training when a monitored metric stops improving.
    
    Args:
        patience: Number of epochs with no improvement to wait before stopping
        min_delta: Minimum change in monitored value to qualify as improvement
        mode: One of 'min' or 'max'. In 'min' mode, training stops when metric stops decreasing;
              in 'max' mode it stops when metric stops increasing
        baseline: Baseline value for the monitored metric. Training will stop if metric doesn't
                  improve beyond baseline after patience epochs
        restore_best_weights: Whether to restore model weights from the epoch with best value
        verbose: If True, prints a message for each validation improvement
        
    Attributes:
        counter: Number of epochs without improvement
        best_score: Best monitored metric value observed
        early_stop: Whether early stopping condition has been met
        best_epoch: Epoch number with best metric value
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        baseline: Optional[float] = None,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        # Convert all numeric parameters to proper types
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = str(mode)
        self.baseline = baseline
        self.restore_best_weights = bool(restore_best_weights)
        self.verbose = bool(verbose)
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.best_weights = None
        
        # Set comparison operators based on mode
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        
        # Initialize best score with baseline if provided
        # Handle string "null", "None", or actual None
        if baseline is not None and baseline not in ["null", "None", ""]:
            try:
                self.best_score = float(baseline)
            except (ValueError, TypeError):
                self.best_score = None
        else:
            self.best_score = None
    
    def __call__(
        self, 
        current_score: float, 
        model: Optional[torch.nn.Module] = None,
        epoch: int = 0
    ) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            current_score: Current value of monitored metric
            model: Model to save weights from (if restore_best_weights=True)
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        # Ensure current_score is a float
        try:
            current_score = float(current_score)
        except (ValueError, TypeError) as e:
            raise TypeError(f"current_score must be a number, got {type(current_score)}: {current_score}") from e
        
        # Initialize best score on first call
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"EarlyStopping: Initial best score: {current_score:.6f}")
            return False
        
        # Check if current score is better than best
        if self.monitor_op(current_score - self.min_delta, self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"EarlyStopping: Metric improved to {current_score:.6f}")
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training at epoch {epoch}")
                    print(f"EarlyStopping: Best score {self.best_score:.6f} at epoch {self.best_epoch}")
                return True
        
        return False
    
    def restore_weights(self, model: torch.nn.Module) -> None:
        """
        Restore best weights to model.
        
        Args:
            model: Model to restore weights to
        """
        if self.best_weights is not None:
            model.load_state_dict({k: v.to(model.device if hasattr(model, 'device') else 'cpu') 
                                   for k, v in self.best_weights.items()})
            if self.verbose:
                print(f"EarlyStopping: Restored best weights from epoch {self.best_epoch}")
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.best_weights = None
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing."""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'best_epoch': self.best_epoch,
            'best_weights': self.best_weights,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
            'baseline': self.baseline,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary from checkpoint."""
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']
        self.best_epoch = state_dict['best_epoch']
        self.best_weights = state_dict['best_weights']
        self.patience = state_dict.get('patience', self.patience)
        self.min_delta = state_dict.get('min_delta', self.min_delta)
        self.mode = state_dict.get('mode', self.mode)
        self.baseline = state_dict.get('baseline', self.baseline)


if __name__ == "__main__":
    # Test utilities
    from config import get_default_config
    
    config = get_default_config()
    
    # Test environment setup
    env_info = setup_training_environment(config)
    print("Environment info:", json.dumps(env_info, indent=2))
    
    # Test validation
    issues = validate_training_config(config)
    if issues:
        print("Validation issues:", issues)
    else:
        print("Configuration is valid!")
    
    # Test summary
    summary = get_training_summary(config)
    print("Training summary:", json.dumps(summary, indent=2))
    
    # Test time estimation
    time_estimate = estimate_training_time(config)
    print("Time estimate:", json.dumps(time_estimate, indent=2))
    
    # Test early stopping
    print("\nTesting EarlyStopping:")
    early_stopping = EarlyStopping(patience=3, mode='min', verbose=True)
    
    # Simulate training with improving loss
    test_losses = [1.0, 0.9, 0.8, 0.85, 0.84, 0.83, 0.82]
    for i, loss in enumerate(test_losses):
        should_stop = early_stopping(loss, epoch=i)
        if should_stop:
            print(f"Would stop at epoch {i}")
            break
    
    print("\nTraining utilities test completed!")
