#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration system for GENESIS IceCube diffusion model.

Provides centralized configuration management for model hyperparameters,
training settings, and data processing options.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import os


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Architecture selection
    architecture: str = "dit"  # "dit", "cnn", "mlp", "hybrid", "resnet"
    
    # Model architecture
    seq_len: int = 5160  # Number of PMTs
    hidden: int = 512    # Hidden dimension
    depth: int = 8       # Number of blocks/layers
    heads: int = 8       # Number of attention heads (for dit/hybrid)
    dropout: float = 0.1 # Dropout rate
    
    # Fusion strategy (for dit)
    fusion: str = "FiLM"  # "SUM" or "FiLM"
    
    # Conditioning
    label_dim: int = 6      # Event condition dimension [E, zenith, azimuth, X, Y, Z]
    t_embed_dim: int = 128  # Timestep embedding dimension
    
    # MLP configuration
    mlp_ratio: float = 4.0  # MLP expansion ratio
    
    # CNN/Conv configuration
    kernel_size: int = 3    # Base kernel size
    kernel_sizes: Tuple[int, ...] = (3, 5, 7, 9)  # Multi-scale kernels (for cnn)
    
    # C-Dit specific parameters (for architecture="c-dit")
    classifier_size: int = 128    # Classifier token dimension
    factor: int = 2               # MLP expansion factor for C-Dit
    combine: str = "add"          # Signal+geometry combination: "add" or "concat"
    update_with_classifier: bool = False  # PMT tokens use classifier info
    n_cls_tokens: int = 1         # Number of classifier tokens
    
    # Affine normalization (per-channel)
    # Formula: (x - offset) / scale → charge/time to reasonable range, pos[-1,1]
    # Inverse: x_original = (x_normalized * scale) + offset
    affine_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)  # [npe, time, xpmt, ypmt, zpmt]
    affine_scales: Tuple[float, ...] = (100.0, 10.0, 600.0, 550.0, 550.0)  # Scale factors
    
    # Label affine normalization (per-label)
    # Formula: (x - offset) / scale (same as signal/geometry)
    label_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # [Energy, Zenith, Azimuth, X, Y, Z]
    label_scales: Tuple[float, ...] = (5e7, 1.0, 1.0, 600.0, 550.0, 550.0)  # [Energy, Zenith, Azimuth, X, Y, Z]
    
    # Time transformation: Always use log(1+x) to handle zeros naturally
    # - "ln": ln(1+x) - natural log, ln(1+0)=0
    # - "log10": log10(1+x) - base-10 log, log10(1+0)=0
    time_transform: Optional[str] = "ln"  # "log10" or "ln"
    exclude_zero_time: bool = True  # Exclude zero time values (recommended for log transforms)
    
    def __post_init__(self):
        """Convert all parameters to proper types."""
        # Architecture
        self.architecture = str(self.architecture)
        self.seq_len = int(self.seq_len)
        self.hidden = int(self.hidden)
        self.depth = int(self.depth)
        self.heads = int(self.heads)
        self.dropout = float(self.dropout)
        self.fusion = str(self.fusion)
        
        # Conditioning
        self.label_dim = int(self.label_dim)
        self.t_embed_dim = int(self.t_embed_dim)
        self.mlp_ratio = float(self.mlp_ratio)
        
        # CNN/Conv
        self.kernel_size = int(self.kernel_size)
        
        # C-Dit specific
        self.classifier_size = int(self.classifier_size)
        self.factor = int(self.factor)
        self.combine = str(self.combine)
        self.update_with_classifier = bool(self.update_with_classifier)
        self.n_cls_tokens = int(self.n_cls_tokens)
        
        # Time transformation
        if self.time_transform and self.time_transform not in ["null", "None", ""]:
            self.time_transform = str(self.time_transform)
        else:
            self.time_transform = "ln"  # Default to ln


@dataclass
class DiffusionConfig:
    """Configuration for diffusion process."""
    
    timesteps: int = 1000      # Number of diffusion timesteps
    beta_start: float = 1e-4   # Starting noise schedule
    beta_end: float = 2e-2     # Ending noise schedule
    objective: str = "eps"     # Training objective: "eps" or "x0"
    schedule: str = "linear"   # Noise schedule: "linear", "cosine", "quadratic", "sigmoid"
    
    # Schedule-specific parameters
    cosine_s: float = 0.008    # Small offset for cosine schedule to prevent β_t from being too small
    
    # Classifier-free guidance
    use_cfg: bool = True       # Use classifier-free guidance
    cfg_scale: float = 2.0     # Guidance scale (1.0 = no guidance, higher = stronger)
    cfg_dropout: float = 0.1   # Probability of dropping condition during training
    
    def __post_init__(self):
        """Convert all parameters to proper types."""
        self.timesteps = int(self.timesteps)
        self.beta_start = float(self.beta_start)
        self.beta_end = float(self.beta_end)
        self.objective = str(self.objective)
        self.schedule = str(self.schedule)
        self.cosine_s = float(self.cosine_s)
        self.use_cfg = bool(self.use_cfg) if not isinstance(self.use_cfg, bool) else self.use_cfg
        self.cfg_scale = float(self.cfg_scale)
        self.cfg_dropout = float(self.cfg_dropout)


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    
    NORMALIZATION POLICY:
    ---------------------
    Normalization is applied in the Dataloader, not in the model.
    
    Pipeline:
    1. Load raw data from HDF5
    2. Apply time transform: ln(1+x) or log10(1+x)
    3. Apply affine normalization: (x - offset) / scale
    4. Return normalized data to model
    
    The model stores these parameters as metadata for denormalization.
    """
    
    # Data paths
    h5_path: str = "GENESIS-data/22644_0921_time_shift.h5"
    
    # Data preprocessing
    replace_time_inf_with: Optional[float] = 0.0  # Replace inf time values
    channel_first: bool = True  # Return data in (C, L) format
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    
    # Data splitting
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # =========================================================================
    # NORMALIZATION PARAMETERS - Applied in Dataloader
    # =========================================================================
    time_transform: str = "ln"  # "ln" or "log10" - always log(1+x)
    
    # Affine: [charge, time, x, y, z] - Formula: (x - offset) / scale
    affine_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    affine_scales: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    
    # Labels: [Energy, Zenith, Azimuth, X, Y, Z]
    label_offsets: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    label_scales: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    # =========================================================================
    
    def __post_init__(self):
        """Convert all parameters to proper types."""
        import os
        import re
        import subprocess
        
        # 환경변수 처리: ${VAR:-default} 형태 지원
        h5_path = str(self.h5_path)
        
        # Git repository 루트 자동 감지 함수
        def get_git_root():
            try:
                # 현재 파일의 디렉토리에서 Git 루트 찾기
                result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], 
                                      capture_output=True, text=True, check=True)
                return result.stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Git이 없거나 Git repository가 아닌 경우, 현재 디렉토리 반환
                return os.getcwd()
        
        if '${' in h5_path:
            # 환경변수 치환: ${GENESIS_ROOT:-../} 형태 처리
            def replace_env_var(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) else ""
                
                if var_name == "GENESIS_ROOT":
                    # GENESIS_ROOT는 Git repository 루트로 설정
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
        
        # Normalization parameters
        self.time_transform = str(self.time_transform)
        # Convert tuples to lists if needed (YAML compatibility)
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
    
    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    
    # Optimization
    optimizer: str = "AdamW"  # "AdamW", "Adam", "SGD"
    scheduler: Optional[str] = None  # "cosine", "linear", "step", "plateau"
    warmup_steps: int = 1000  # Absolute steps (deprecated, use warmup_ratio)
    warmup_ratio: float = 0.04  # Warmup as % of total steps (default: 4%)
    
    # Scheduler-specific parameters
    # Cosine annealing
    cosine_t_max: Optional[int] = None  # If None, uses num_epochs
    
    # Plateau scheduler
    plateau_patience: int = 10
    plateau_factor: float = 0.5
    plateau_mode: str = "min"  # "min" or "max"
    plateau_threshold: float = 1e-4
    plateau_cooldown: int = 0
    plateau_verbose: bool = False
    
    # Step scheduler
    step_size: int = 30
    step_gamma: float = 0.1
    
    # Linear scheduler
    linear_start_factor: float = 1.0
    linear_end_factor: float = 0.0
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 1e-4
    early_stopping_mode: str = "min"  # "min" or "max"
    early_stopping_baseline: Optional[float] = None
    early_stopping_restore_best: bool = True
    early_stopping_verbose: bool = True
    
    # Logging and checkpointing
    log_interval: int = 50
    save_interval: int = 1000
    eval_interval: int = 500
    save_best_only: bool = False
    
    # Output directories
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Advanced features
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Debugging
    debug_mode: bool = False
    detect_anomaly: bool = False
    
    def __post_init__(self):
        """Convert all parameters to proper types (YAML sometimes parses as strings)."""
        # Training parameters
        self.num_epochs = int(self.num_epochs)
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.grad_clip_norm = float(self.grad_clip_norm)
        
        # Optimizer and scheduler
        self.optimizer = str(self.optimizer) if self.optimizer else "AdamW"
        self.scheduler = str(self.scheduler) if self.scheduler and self.scheduler not in ["null", "None", ""] else None
        self.warmup_steps = int(self.warmup_steps)
        self.warmup_ratio = float(self.warmup_ratio)
        
        # Cosine scheduler
        if self.cosine_t_max is not None:
            self.cosine_t_max = int(self.cosine_t_max)
        
        # Plateau scheduler
        self.plateau_patience = int(self.plateau_patience)
        self.plateau_factor = float(self.plateau_factor)
        self.plateau_mode = str(self.plateau_mode)
        self.plateau_threshold = float(self.plateau_threshold)
        self.plateau_cooldown = int(self.plateau_cooldown)
        self.plateau_verbose = bool(self.plateau_verbose) if not isinstance(self.plateau_verbose, bool) else self.plateau_verbose
        
        # Step scheduler
        self.step_size = int(self.step_size)
        self.step_gamma = float(self.step_gamma)
        
        # Linear scheduler
        self.linear_start_factor = float(self.linear_start_factor)
        self.linear_end_factor = float(self.linear_end_factor)
        
        # Early stopping
        self.early_stopping = bool(self.early_stopping) if not isinstance(self.early_stopping, bool) else self.early_stopping
        self.early_stopping_patience = int(self.early_stopping_patience)
        self.early_stopping_min_delta = float(self.early_stopping_min_delta)
        self.early_stopping_mode = str(self.early_stopping_mode)
        # Handle baseline
        if self.early_stopping_baseline is not None and self.early_stopping_baseline not in ["null", "None", ""]:
            try:
                self.early_stopping_baseline = float(self.early_stopping_baseline)
            except (ValueError, TypeError):
                self.early_stopping_baseline = None
        else:
            self.early_stopping_baseline = None
        self.early_stopping_restore_best = bool(self.early_stopping_restore_best) if not isinstance(self.early_stopping_restore_best, bool) else self.early_stopping_restore_best
        self.early_stopping_verbose = bool(self.early_stopping_verbose) if not isinstance(self.early_stopping_verbose, bool) else self.early_stopping_verbose
        
        # Logging and checkpointing
        self.log_interval = int(self.log_interval)
        self.save_interval = int(self.save_interval)
        self.eval_interval = int(self.eval_interval)
        self.save_best_only = bool(self.save_best_only) if not isinstance(self.save_best_only, bool) else self.save_best_only
        
        # Output directories
        self.output_dir = str(self.output_dir)
        self.checkpoint_dir = str(self.checkpoint_dir)
        self.log_dir = str(self.log_dir)
        
        # Resume training
        if self.resume_from_checkpoint and self.resume_from_checkpoint not in ["null", "None", ""]:
            self.resume_from_checkpoint = str(self.resume_from_checkpoint)
        else:
            self.resume_from_checkpoint = None
        
        # Mixed precision
        self.use_amp = bool(self.use_amp) if not isinstance(self.use_amp, bool) else self.use_amp
        
        # Advanced features
        self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
        self.max_grad_norm = float(self.max_grad_norm)
        
        # Debugging
        self.debug_mode = bool(self.debug_mode) if not isinstance(self.debug_mode, bool) else self.debug_mode
        self.detect_anomaly = bool(self.detect_anomaly) if not isinstance(self.detect_anomaly, bool) else self.detect_anomaly


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Experiment metadata
    experiment_name: str = "icecube_diffusion"
    description: str = "IceCube muon neutrino event diffusion model"
    
    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # System settings
    device: str = "auto"  # "auto", "cuda", "cpu"
    seed: int = 42
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "icecube-diffusion"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Auto-detect device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directories
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
    
    # Remove benchmark-specific config (not part of ExperimentConfig)
    config_dict.pop('benchmark', None)
    
    # Convert nested dictionaries to config objects
    if 'model' in config_dict:
        config_dict['model'] = ModelConfig(**config_dict['model'])
    if 'diffusion' in config_dict:
        config_dict['diffusion'] = DiffusionConfig(**config_dict['diffusion'])
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


# Predefined configurations for common use cases
def get_small_model_config() -> ExperimentConfig:
    """Configuration for small model (faster training/testing)."""
    config = get_default_config()
    config.model.hidden = 256
    config.model.depth = 4
    config.model.heads = 4
    config.diffusion.timesteps = 100
    config.training.num_epochs = 10
    config.data.batch_size = 16
    return config


def get_large_model_config() -> ExperimentConfig:
    """Configuration for large model (better quality)."""
    config = get_default_config()
    config.model.hidden = 768
    config.model.depth = 12
    config.model.heads = 12
    config.diffusion.timesteps = 2000
    config.training.num_epochs = 200
    config.data.batch_size = 4
    return config


def get_debug_config() -> ExperimentConfig:
    """Configuration for debugging."""
    config = get_default_config()
    config.model.hidden = 128
    config.model.depth = 2
    config.model.heads = 2
    config.diffusion.timesteps = 10
    config.training.num_epochs = 2
    config.data.batch_size = 2
    config.training.debug_mode = True
    config.training.detect_anomaly = True
    return config


def get_cnn_config() -> ExperimentConfig:
    """Configuration for CNN architecture."""
    config = get_default_config()
    config.model.architecture = "cnn"
    config.model.hidden = 256
    config.model.depth = 6
    config.model.kernel_sizes = (3, 5, 7, 9)
    config.training.num_epochs = 50
    config.data.batch_size = 16
    return config


def get_mlp_config() -> ExperimentConfig:
    """Configuration for MLP architecture."""
    config = get_default_config()
    config.model.architecture = "mlp"
    config.model.hidden = 512
    config.model.depth = 6
    config.training.num_epochs = 50
    config.data.batch_size = 16
    return config


def get_hybrid_config() -> ExperimentConfig:
    """Configuration for Hybrid architecture."""
    config = get_default_config()
    config.model.architecture = "hybrid"
    config.model.hidden = 384
    config.model.depth = 6
    config.model.heads = 6
    config.training.num_epochs = 50
    config.data.batch_size = 12
    return config


def get_resnet_config() -> ExperimentConfig:
    """Configuration for ResNet architecture."""
    config = get_default_config()
    config.model.architecture = "resnet"
    config.model.hidden = 256
    config.model.depth = 8
    config.training.num_epochs = 50
    config.data.batch_size = 16
    return config


if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    print("Default configuration:")
    print(f"Model: {config.model.hidden}D, {config.model.depth} layers")
    print(f"Diffusion: {config.diffusion.timesteps} timesteps")
    print(f"Training: {config.training.num_epochs} epochs, lr={config.training.learning_rate}")
    
    # Save example config
    save_config_to_file(config, "example_config.yaml")
    print("Saved example configuration to example_config.yaml")
