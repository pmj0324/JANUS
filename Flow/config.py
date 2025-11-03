#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration system for GENESIS Flow Matching model.

Provides centralized configuration management for Flow Matching.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import os


@dataclass
class ModelConfig:
    """Configuration for Flow Matching model architecture."""
    
    # Architecture selection
    architecture: str = "dit"  # "dit", "cnn", "mlp", "hybrid"
    
    # Model architecture
    seq_len: int = 5160
    hidden: int = 512
    depth: int = 8
    heads: int = 8
    dropout: float = 0.1
    
    # Fusion strategy (for dit)
    fusion: str = "FiLM"  # "SUM" or "FiLM"
    
    # Conditioning
    label_dim: int = 6
    t_embed_dim: int = 128
    mlp_ratio: float = 4.0
    
    # CNN configuration
    kernel_size: int = 3
    kernel_sizes: Tuple[int, ...] = (3, 5, 7, 9)
    
    # Normalization metadata
    affine_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    affine_scales: Tuple[float, ...] = (100.0, 10.0, 600.0, 550.0, 550.0)
    label_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    label_scales: Tuple[float, ...] = (5e7, 1.0, 1.0, 600.0, 550.0, 550.0)
    time_transform: Optional[str] = "ln"
    
    def __post_init__(self):
        """Convert all parameters to proper types."""
        self.architecture = str(self.architecture)
        self.seq_len = int(self.seq_len)
        self.hidden = int(self.hidden)
        self.depth = int(self.depth)
        self.heads = int(self.heads)
        self.dropout = float(self.dropout)
        self.fusion = str(self.fusion)
        self.label_dim = int(self.label_dim)
        self.t_embed_dim = int(self.t_embed_dim)
        self.mlp_ratio = float(self.mlp_ratio)
        self.kernel_size = int(self.kernel_size)
        
        if self.time_transform and self.time_transform not in ["null", "None", ""]:
            self.time_transform = str(self.time_transform)
        else:
            self.time_transform = "ln"


@dataclass
class FlowConfig:
    """Configuration for Flow Matching process."""
    
    # Flow Matching specific
    sigma_min: float = 1e-4  # Minimum noise level
    use_ode_solver: str = "euler"  # "euler", "midpoint", "rk4"
    num_steps: int = 50  # Number of sampling steps (can be much less than diffusion)
    
    # Conditional Flow Matching (Lipman et al. 2023)
    use_cfm: bool = True  # Use Conditional Flow Matching
    
    # Classifier-free guidance
    use_cfg: bool = True
    cfg_scale: float = 2.0
    cfg_dropout: float = 0.1
    
    def __post_init__(self):
        """Convert all parameters to proper types."""
        self.sigma_min = float(self.sigma_min)
        self.use_ode_solver = str(self.use_ode_solver)
        self.num_steps = int(self.num_steps)
        self.use_cfm = bool(self.use_cfm) if not isinstance(self.use_cfm, bool) else self.use_cfm
        self.use_cfg = bool(self.use_cfg) if not isinstance(self.use_cfg, bool) else self.use_cfg
        self.cfg_scale = float(self.cfg_scale)
        self.cfg_dropout = float(self.cfg_dropout)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    h5_path: str = "GENESIS-data/22644_0921_time_shift.h5"
    replace_time_inf_with: Optional[float] = 0.0
    channel_first: bool = True
    
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Normalization parameters
    time_transform: str = "ln"
    affine_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    affine_scales: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    label_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    label_scales: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    
    def __post_init__(self):
        """Convert all parameters to proper types."""
        import os
        import re
        import subprocess
        
        h5_path = str(self.h5_path)
        
        def get_git_root():
            try:
                result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], 
                                      capture_output=True, text=True, check=True)
                return result.stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                return os.getcwd()
        
        if '${' in h5_path:
            def replace_env_var(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) else ""
                
                if var_name == "GENESIS_ROOT":
                    return os.environ.get(var_name, get_git_root() + "/")
                else:
                    return os.environ.get(var_name, default_value)
            
            h5_path = re.sub(r'\$\{([^:}]+)(?::-(.*?))?\}', replace_env_var, h5_path)
        
        self.h5_path = os.path.expanduser(h5_path)
        if self.replace_time_inf_with is not None:
            self.replace_time_inf_with = float(self.replace_time_inf_with)
        self.channel_first = bool(self.channel_first) if not isinstance(self.channel_first, bool) else self.channel_first
        self.batch_size = int(self.batch_size)
        self.num_workers = int(self.num_workers)
        self.pin_memory = bool(self.pin_memory) if not isinstance(self.pin_memory, bool) else self.pin_memory
        self.shuffle = bool(self.shuffle) if not isinstance(self.shuffle, bool) else self.shuffle
        self.train_ratio = float(self.train_ratio)
        self.val_ratio = float(self.val_ratio)
        self.test_ratio = float(self.test_ratio)
        
        self.time_transform = str(self.time_transform)
        if isinstance(self.affine_offsets, list):
            self.affine_offsets = tuple(float(x) for x in self.affine_offsets)
        if isinstance(self.affine_scales, list):
            self.affine_scales = tuple(float(x) for x in self.affine_scales)
        if isinstance(self.label_offsets, list):
            self.label_offsets = tuple(float(x) for x in self.label_offsets)
        if isinstance(self.label_scales, list):
            self.label_scales = tuple(float(x) for x in self.label_scales)


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    num_epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    
    optimizer: str = "AdamW"
    scheduler: Optional[str] = None
    warmup_steps: int = 1000
    warmup_ratio: float = 0.04
    
    # Scheduler parameters
    cosine_t_max: Optional[int] = None
    plateau_patience: int = 10
    plateau_factor: float = 0.5
    plateau_mode: str = "min"
    plateau_threshold: float = 1e-4
    plateau_cooldown: int = 0
    step_size: int = 30
    step_gamma: float = 0.1
    linear_start_factor: float = 1.0
    linear_end_factor: float = 0.0
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 1e-4
    early_stopping_mode: str = "min"
    early_stopping_baseline: Optional[float] = None
    early_stopping_restore_best: bool = True
    early_stopping_verbose: bool = True
    
    log_interval: int = 50
    save_interval: int = 1000
    eval_interval: int = 500
    save_best_only: bool = False
    
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    resume_from_checkpoint: Optional[str] = None
    use_amp: bool = True
    
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    debug_mode: bool = False
    detect_anomaly: bool = False
    
    def __post_init__(self):
        """Convert all parameters to proper types."""
        self.num_epochs = int(self.num_epochs)
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.grad_clip_norm = float(self.grad_clip_norm)
        self.optimizer = str(self.optimizer) if self.optimizer else "AdamW"
        self.scheduler = str(self.scheduler) if self.scheduler and self.scheduler not in ["null", "None", ""] else None
        self.warmup_steps = int(self.warmup_steps)
        self.warmup_ratio = float(self.warmup_ratio)
        
        if self.cosine_t_max is not None:
            self.cosine_t_max = int(self.cosine_t_max)
        
        self.plateau_patience = int(self.plateau_patience)
        self.plateau_factor = float(self.plateau_factor)
        self.plateau_mode = str(self.plateau_mode)
        self.plateau_threshold = float(self.plateau_threshold)
        self.plateau_cooldown = int(self.plateau_cooldown)
        
        self.step_size = int(self.step_size)
        self.step_gamma = float(self.step_gamma)
        self.linear_start_factor = float(self.linear_start_factor)
        self.linear_end_factor = float(self.linear_end_factor)
        
        self.early_stopping = bool(self.early_stopping) if not isinstance(self.early_stopping, bool) else self.early_stopping
        self.early_stopping_patience = int(self.early_stopping_patience)
        self.early_stopping_min_delta = float(self.early_stopping_min_delta)
        self.early_stopping_mode = str(self.early_stopping_mode)
        
        if self.early_stopping_baseline is not None and self.early_stopping_baseline not in ["null", "None", ""]:
            try:
                self.early_stopping_baseline = float(self.early_stopping_baseline)
            except (ValueError, TypeError):
                self.early_stopping_baseline = None
        else:
            self.early_stopping_baseline = None
        
        self.early_stopping_restore_best = bool(self.early_stopping_restore_best) if not isinstance(self.early_stopping_restore_best, bool) else self.early_stopping_restore_best
        self.early_stopping_verbose = bool(self.early_stopping_verbose) if not isinstance(self.early_stopping_verbose, bool) else self.early_stopping_verbose
        
        self.log_interval = int(self.log_interval)
        self.save_interval = int(self.save_interval)
        self.eval_interval = int(self.eval_interval)
        self.save_best_only = bool(self.save_best_only) if not isinstance(self.save_best_only, bool) else self.save_best_only
        
        self.output_dir = str(self.output_dir)
        self.checkpoint_dir = str(self.checkpoint_dir)
        self.log_dir = str(self.log_dir)
        
        if self.resume_from_checkpoint and self.resume_from_checkpoint not in ["null", "None", ""]:
            self.resume_from_checkpoint = str(self.resume_from_checkpoint)
        else:
            self.resume_from_checkpoint = None
        
        self.use_amp = bool(self.use_amp) if not isinstance(self.use_amp, bool) else self.use_amp
        self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
        self.max_grad_norm = float(self.max_grad_norm)
        self.debug_mode = bool(self.debug_mode) if not isinstance(self.debug_mode, bool) else self.debug_mode
        self.detect_anomaly = bool(self.detect_anomaly) if not isinstance(self.detect_anomaly, bool) else self.detect_anomaly


@dataclass
class ExperimentConfig:
    """Complete experiment configuration for Flow Matching."""
    
    experiment_name: str = "icecube_flow"
    description: str = "IceCube Flow Matching model"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    device: str = "auto"
    seed: int = 42
    
    use_wandb: bool = False
    wandb_project: str = "icecube-flow"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        os.makedirs(self.training.output_dir, exist_ok=True)
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.training.log_dir, exist_ok=True)


def get_default_config() -> ExperimentConfig:
    """Get default configuration."""
    return ExperimentConfig()


def load_config_from_file(config_path: str) -> ExperimentConfig:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config_dict.pop('benchmark', None)
    
    if 'model' in config_dict:
        config_dict['model'] = ModelConfig(**config_dict['model'])
    if 'flow' in config_dict:
        config_dict['flow'] = FlowConfig(**config_dict['flow'])
    if 'data' in config_dict:
        config_dict['data'] = DataConfig(**config_dict['data'])
    if 'training' in config_dict:
        config_dict['training'] = TrainingConfig(**config_dict['training'])
    
    return ExperimentConfig(**config_dict)


def save_config_to_file(config: ExperimentConfig, config_path: str):
    """Save configuration to YAML file."""
    import yaml
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

