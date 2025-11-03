#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced trainer for GENESIS IceCube diffusion model.

This module provides a comprehensive training class with advanced features
including multiple schedulers, logging, checkpointing, and monitoring.
"""

from __future__ import annotations
import os
import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Optional imports for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Project imports
from dataloader.pmt_dataloader import make_dataloader, check_dataset_health
from models.factory import ModelFactory
from diffusion import GaussianDiffusion, DiffusionConfig
from config import ExperimentConfig
from utils.gpu_utils import print_memory_analysis, print_gpu_info
from .schedulers import create_scheduler
from .logging import setup_logging, LoggingConfig
from .checkpointing import CheckpointManager
from .utils import EarlyStopping

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.cuda.amp import GradScaler
    from torch.amp import autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Enhanced training configuration."""
    
    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    
    # Optimization
    optimizer: str = "AdamW"  # "AdamW", "Adam", "SGD"
    scheduler: Optional[str] = None  # "cosine", "linear", "step", "plateau"
    warmup_steps: int = 1000
    
    # Scheduler-specific parameters
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
    
    # Logging and checkpointing
    log_interval: int = 50
    save_interval: int = 1000
    eval_interval: int = 500
    
    # Output directories (relative to current working directory)
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Mixed precision
    use_amp: bool = True
    
    # Debugging
    debug_mode: bool = False
    detect_anomaly: bool = False
    
    # Advanced features
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 50
    save_best_only: bool = True


class Trainer:
    """Enhanced trainer for IceCube diffusion model."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
        
        # Initialize components
        self._setup_logging()
        self._setup_model()
        self._setup_optimizer()
        self._setup_data()
        self._setup_checkpointing()
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Early stopping
        if config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                min_delta=config.training.early_stopping_min_delta,
                mode=config.training.early_stopping_mode,
                baseline=config.training.early_stopping_baseline,
                restore_best_weights=config.training.early_stopping_restore_best,
                verbose=config.training.early_stopping_verbose
            )
        else:
            self.early_stopping = None
        
        # Mixed precision
        self.scaler = GradScaler() if AMP_AVAILABLE and config.training.use_amp else None
        
        # Resume from checkpoint if specified
        if config.training.resume_from_checkpoint:
            self._load_checkpoint(config.training.resume_from_checkpoint)
    
    def _setup_logging(self):
        """Setup logging infrastructure."""
        # Create output directories
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.training.log_dir, exist_ok=True)
        
        # Setup text log file
        self.log_file_path = os.path.join(self.config.training.log_dir, f"{self.config.experiment_name}_training.txt")
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
        
        # Store original print function
        self.original_print = print
        
        # Override print to also write to log file
        def log_print(*args, **kwargs):
            # Print to console
            self.original_print(*args, **kwargs)
            # Write to log file
            message = ' '.join(str(arg) for arg in args)
            self.log_file.write(message + '\n')
            self.log_file.flush()
        
        # Replace global print function
        import builtins
        builtins.print = log_print
        
        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.config.training.log_dir, self.config.experiment_name)
        )
        
        # Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.experiment_name,
                config=self.config.__dict__,
                dir=self.config.training.log_dir
            )
    
    def _setup_model(self):
        """Initialize model and diffusion wrapper."""
        # Create model using factory
        self.model = ModelFactory.create_model_from_config(self.config.model).to(self.device)
        
        # Create diffusion wrapper using factory
        self.diffusion = ModelFactory.create_diffusion_wrapper(self.model, self.config.diffusion).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized:")
        print(f"  Architecture: {self.config.model.architecture}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024**2:.1f} MB")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Optimizer
        if self.config.training.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        # Scheduler
        self.scheduler = create_scheduler(self.optimizer, self.config.training)
        
        print(f"Optimizer: {self.config.training.optimizer}")
        print(f"Scheduler: {self.config.training.scheduler or 'None'}")
    
    def _setup_data(self):
        """Setup data loaders."""
        import h5py
        import numpy as np
        
        # Get total number of samples
        with h5py.File(self.config.data.h5_path, 'r') as f:
            total_samples = len(f['input'])
        
        # Create indices for train/val/test split
        indices = np.arange(total_samples)
        if self.config.data.shuffle:
            np.random.seed(self.config.seed)
            np.random.shuffle(indices)
        
        # Calculate split sizes
        train_ratio = self.config.data.train_ratio
        val_ratio = self.config.data.val_ratio
        test_ratio = self.config.data.test_ratio
        
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = int(total_samples * test_ratio)
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:train_size + val_size + test_size]
        
        # =================================================================
        # NORMALIZATION PARAMETERS
        # =================================================================
        # Normalization happens in Dataloader. Get params from data config.
        # =================================================================
        # Prefer data config, fall back to model config
        time_transform = getattr(self.config.data, 'time_transform', self.config.model.time_transform)
        affine_offsets = getattr(self.config.data, 'affine_offsets', self.config.model.affine_offsets)
        affine_scales = getattr(self.config.data, 'affine_scales', self.config.model.affine_scales)
        label_offsets = getattr(self.config.data, 'label_offsets', self.config.model.label_offsets)
        label_scales = getattr(self.config.data, 'label_scales', self.config.model.label_scales)
        
        # Create data loaders
        self.train_loader = make_dataloader(
            h5_path=self.config.data.h5_path,
            batch_size=self.config.data.batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            replace_time_inf_with=self.config.data.replace_time_inf_with,
            channel_first=self.config.data.channel_first,
            indices=train_indices,
            time_transform=time_transform,
            affine_offsets=affine_offsets,
            affine_scales=affine_scales,
            label_offsets=label_offsets,
            label_scales=label_scales,
        )
        
        self.val_loader = make_dataloader(
            h5_path=self.config.data.h5_path,
            batch_size=self.config.data.batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            replace_time_inf_with=self.config.data.replace_time_inf_with,
            channel_first=self.config.data.channel_first,
            indices=val_indices,
            time_transform=time_transform,
            affine_offsets=affine_offsets,
            affine_scales=affine_scales,
            label_offsets=label_offsets,
            label_scales=label_scales,
        )
        
        self.test_loader = make_dataloader(
            h5_path=self.config.data.h5_path,
            batch_size=self.config.data.batch_size,
            shuffle=False,  # Don't shuffle test data
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            replace_time_inf_with=self.config.data.replace_time_inf_with,
            channel_first=self.config.data.channel_first,
            indices=test_indices,
            time_transform=time_transform,
            affine_offsets=affine_offsets,
            affine_scales=affine_scales,
            label_offsets=label_offsets,
            label_scales=label_scales,
        )
        
        # Calculate statistics
        train_samples = len(self.train_loader.dataset)
        val_samples = len(self.val_loader.dataset)
        test_samples = len(self.test_loader.dataset)
        used_samples = train_samples + val_samples + test_samples
        unused_samples = total_samples - used_samples
        
        print(f"\n{'='*70}")
        print(f"📊 Dataset Information")
        print(f"{'='*70}")
        print(f"  Total samples in file: {total_samples:,}")
        print(f"  Samples for training:  {used_samples:,} ({used_samples/total_samples*100:.1f}%)")
        print(f"  Unused samples:        {unused_samples:,} ({unused_samples/total_samples*100:.1f}%)")
        print(f"\n  Split breakdown:")
        print(f"    Train: {train_samples:,} ({train_samples/total_samples*100:.1f}%) - {len(self.train_loader):,} batches")
        print(f"    Val:   {val_samples:,} ({val_samples/total_samples*100:.1f}%) - {len(self.val_loader):,} batches")
        print(f"    Test:  {test_samples:,} ({test_samples/total_samples*100:.1f}%) - {len(self.test_loader):,} batches")
        print(f"{'='*70}\n")
    
    def _setup_checkpointing(self):
        """Setup checkpoint manager."""
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.training.checkpoint_dir,
            experiment_name=self.config.experiment_name,
            save_best_only=self.config.training.save_best_only
        )
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore early stopping state if available
        if self.early_stopping is not None and 'early_stopping_state_dict' in checkpoint:
            self.early_stopping.load_state_dict(checkpoint['early_stopping_state_dict'])
            print(f"Restored early stopping state (counter: {self.early_stopping.counter}/{self.early_stopping.patience})")
        
        print(f"Resumed from epoch {self.start_epoch}, step {self.global_step}")
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False, suffix: str = ''):
        """Save model checkpoint using CheckpointManager."""
        checkpoint_data = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.__dict__,
            'metrics': metrics,
        }
        
        if self.scheduler:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.early_stopping is not None:
            checkpoint_data['early_stopping_state_dict'] = self.early_stopping.state_dict()
        
        metric_value = metrics.get('eval/loss', metrics.get('train/epoch_loss', float('inf')))
        self.checkpoint_manager.save_checkpoint(
            checkpoint_data, epoch, is_best, metric_value=metric_value, suffix=suffix
        )
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to TensorBoard and Wandb."""
        # TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        
        # Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        epoch_start_time = time.time()
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}",
            leave=False
        )
        
        for step, (x_sig, geom, label, idx) in enumerate(progress_bar):
            # Move to device
            x_sig = x_sig.to(self.device)
            geom = geom.to(self.device)
            label = label.to(self.device)
            
            # Handle geometry shape
            if geom.ndim == 2:  # (3, L)
                geom = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)
            
            # Print data statistics for first batch of first epoch
            if epoch == 0 and step == 0:
                print(f"\n{'='*70}")
                print("📊 First Batch - Data Flow & Normalization")
                print(f"{'='*70}")
                
                # Sample first 128 samples for statistics
                batch_size = x_sig.size(0)
                sample_size = min(128, batch_size)
                print(f"  ℹ️  Showing statistics for first {sample_size} samples (batch_size={batch_size})")
                
                # Sample data
                x_sig_sample = x_sig[:sample_size]
                geom_sample = geom[:sample_size]
                label_sample = label[:sample_size]
                
                # These are ALREADY NORMALIZED by dataloader!
                print(f"\n  📥 Data from Dataloader (ALREADY NORMALIZED):")
                print(f"     Charge: [{x_sig_sample[:, 0, :].min():.6f}, {x_sig_sample[:, 0, :].max():.6f}] "
                      f"mean={x_sig_sample[:, 0, :].mean():.6f}")
                print(f"     Time:   [{x_sig_sample[:, 1, :].min():.6f}, {x_sig_sample[:, 1, :].max():.6f}] "
                      f"mean={x_sig_sample[:, 1, :].mean():.6f}")
                print(f"     X PMT:  [{geom_sample[:, 0, :].min():.6f}, {geom_sample[:, 0, :].max():.6f}] "
                      f"mean={geom_sample[:, 0, :].mean():.6f}")
                print(f"     Y PMT:  [{geom_sample[:, 1, :].min():.6f}, {geom_sample[:, 1, :].max():.6f}] "
                      f"mean={geom_sample[:, 1, :].mean():.6f}")
                print(f"     Z PMT:  [{geom_sample[:, 2, :].min():.6f}, {geom_sample[:, 2, :].max():.6f}] "
                      f"mean={geom_sample[:, 2, :].mean():.6f}")
                
                print(f"\n  🏷️  Labels from Dataloader (ALREADY NORMALIZED):")
                label_names = ['Energy', 'Zenith', 'Azimuth', 'X', 'Y', 'Z']
                for ch_idx, name in enumerate(label_names):
                    print(f"     {name:8s}: [{label_sample[:, ch_idx].min():.6f}, {label_sample[:, ch_idx].max():.6f}] "
                          f"mean={label_sample[:, ch_idx].mean():.6f}")
                
                print(f"\n  📋 Normalization Parameters (metadata in model):")
                print(f"     Signal+Geometry offsets: {self.model.affine_offset.squeeze().cpu().numpy()}")
                print(f"     Signal+Geometry scales:  {self.model.affine_scale.squeeze().cpu().numpy()}")
                print(f"     Label offsets: {self.model.label_offset.cpu().numpy()}")
                print(f"     Label scales:  {self.model.label_scale.cpu().numpy()}")
                
                print(f"\n  ⚠️  IMPORTANT:")
                print(f"     1. Dataloader applies: time_transform (ln) → affine normalization")
                print(f"     2. Model receives ALREADY NORMALIZED data")
                print(f"     3. Model does NOT normalize in forward()")
                print(f"     4. Normalization params are metadata for denormalization")
                print(f"{'='*70}\n")
            
            # Forward pass
            if self.scaler:
                with autocast('cuda'):
                    loss = self.diffusion.loss(x_sig, geom, label)
                    loss = loss / self.config.training.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss = self.diffusion.loss(x_sig, geom, label)
                loss = loss / self.config.training.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            epoch_losses.append(loss.item() * self.config.training.gradient_accumulation_steps)
            self.global_step += 1
            
            # Logging (step-based, epoch summary handled at epoch end)
            if self.global_step % self.config.training.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                metrics = {
                    'train/loss': loss.item() * self.config.training.gradient_accumulation_steps,
                    'train/learning_rate': current_lr,
                    'train/epoch': epoch,
                }
                self._log_metrics(metrics, self.global_step)
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * self.config.training.gradient_accumulation_steps:.6f}",
                    'lr': f"{current_lr:.2e}"
                })
            
            # Note: Checkpoint saving moved to epoch end (every epoch + best model)
        
        # Calculate epoch metrics
        avg_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start_time
        
        return {
            'train/epoch_loss': avg_loss,
            'train/epoch_time': epoch_time,
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for x_sig, geom, label, idx in self.val_loader:
                # Move to device
                x_sig = x_sig.to(self.device)
                geom = geom.to(self.device)
                label = label.to(self.device)
                
                # Handle geometry shape
                if geom.ndim == 2:  # (3, L)
                    geom = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)
                
                # Forward pass
                if self.scaler:
                    with autocast('cuda'):
                        loss = self.diffusion.loss(x_sig, geom, label)
                else:
                    loss = self.diffusion.loss(x_sig, geom, label)
                
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        return {
            'eval/loss': avg_val_loss,
            'eval/num_batches': len(val_losses)
        }
    
    def train(self):
        """Main training loop."""
        # Print configuration at start
        if self.start_epoch == 0:
            print(f"\n{'='*70}")
            print("⚙️  Configuration Loaded")
            print(f"{'='*70}")
            print(f"\nModel Config:")
            print(f"  Architecture: {self.config.model.architecture}")
            print(f"  Hidden: {self.config.model.hidden}, Depth: {self.config.model.depth}")
            print(f"  Fusion: {self.config.model.fusion}")
            print(f"  Signal+Geometry scales: {self.config.model.affine_scales}")
            print(f"  Label scales: {self.config.model.label_scales}")
            print(f"  Time transform: {self.config.model.time_transform} (always log(1+x))")
            
            print(f"\nDiffusion Config:")
            print(f"  Timesteps: {self.config.diffusion.timesteps}")
            print(f"  Beta: [{self.config.diffusion.beta_start}, {self.config.diffusion.beta_end}]")
            print(f"  Objective: {self.config.diffusion.objective}")
            print(f"  Schedule: {getattr(self.config.diffusion, 'schedule', 'linear')}")
            print(f"  Use CFG: {getattr(self.config.diffusion, 'use_cfg', False)}")
            print(f"  CFG Scale: {getattr(self.config.diffusion, 'cfg_scale', 1.0)}")
            print(f"  CFG Dropout: {getattr(self.config.diffusion, 'cfg_dropout', 0.0)}")
            
            # Data Config removed - already shown in Dataset Information section above
            
            print(f"\nTraining Config:")
            print(f"  Epochs: {self.config.training.num_epochs}")
            print(f"  Batch size: {self.config.data.batch_size}")
            print(f"  Learning rate: {self.config.training.learning_rate}")
            print(f"  Scheduler: {self.config.training.scheduler}")
            print(f"  Early stopping: {self.config.training.early_stopping} (patience={self.config.training.early_stopping_patience})")
            print(f"{'='*70}\n")
        
        # Calculate training steps
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.config.training.num_epochs
        
        # Calculate warmup steps (use ratio if set, otherwise use absolute)
        if hasattr(self.config.training, 'warmup_ratio') and self.config.training.warmup_ratio > 0:
            warmup_steps = int(total_steps * self.config.training.warmup_ratio)
        else:
            warmup_steps = self.config.training.warmup_steps
        
        print(f"\n{'='*70}")
        print(f"📊 Training Overview")
        print(f"{'='*70}")
        print(f"  Total Epochs:     {self.config.training.num_epochs}")
        print(f"  Steps per Epoch:  {steps_per_epoch:,}")
        print(f"  Total Steps:      {total_steps:,}")
        print(f"  Warmup Steps:     {warmup_steps:,} ({warmup_steps/total_steps*100:.1f}% of total)")
        print(f"  Device:           {self.device} {'✅ CUDA' if torch.cuda.is_available() else '⚠️  CPU'}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"  GPU(s):           {gpu_count}x {gpu_name}")
            print(f"  Using GPU:        cuda:0")
        
        print(f"  Mixed Precision:  {'✅ Enabled (float16)' if self.scaler is not None else '❌ Disabled (float32)'}")
        print(f"  Gradient Accum:   {self.config.training.gradient_accumulation_steps}")
        print(f"  Effective Batch:  {self.config.data.batch_size * self.config.training.gradient_accumulation_steps}")
        print(f"{'='*70}\n")
        
        # GPU memory analysis on first epoch
        if self.start_epoch == 0 and torch.cuda.is_available():
            print_memory_analysis(
                self.model,
                batch_size=self.config.data.batch_size,
                device_id=0 if self.device.type == 'cuda' else 0,
                mixed_precision=self.config.training.use_amp,
                model_depth=self.config.model.depth
            )
        
        # Data health check on first epoch
        if self.start_epoch == 0:
            print("\n" + "="*70)
            print("🔍 Running data health check before training...")
            print("="*70)
            check_dataset_health(self.train_loader, num_batches=5, verbose=True)
        
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            # Train epoch
            epoch_metrics = self.train_epoch(epoch)
            
            # Log epoch metrics
            self._log_metrics(epoch_metrics, epoch)
            
            # Evaluate (for both early stopping and plateau scheduler)
            eval_metrics = {}
            if (epoch + 1) % self.config.training.eval_interval == 0:
                eval_metrics = self.evaluate()
                self._log_metrics(eval_metrics, epoch)
                current_loss = eval_metrics.get('eval/loss', epoch_metrics['train/epoch_loss'])
            else:
                current_loss = epoch_metrics['train/epoch_loss']
            
            # Update scheduler (after evaluation)
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # Plateau scheduler needs validation loss
                    self.scheduler.step(current_loss)
                else:
                    # Other schedulers step every epoch
                    self.scheduler.step()
            
            # Save checkpoint every epoch + best model separately
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss
                # Use same format as epoch summary for consistency
                val_loss_display = current_loss if (epoch + 1) % self.config.training.eval_interval == 0 else 0.0
                print(f"  💾 New best model! Val loss: {val_loss_display:.6f}")
                self._save_checkpoint(epoch, epoch_metrics, is_best=True)
            
            # Save regular checkpoint every epoch
            self._save_checkpoint(epoch, epoch_metrics, is_best=False)
            
            # Early stopping check (using validation loss)
            if self.early_stopping is not None:
                should_stop = self.early_stopping(
                    current_score=current_loss,
                    model=self.model,
                    epoch=epoch
                )
                
                if should_stop:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                    print(f"Best val loss: {self.early_stopping.best_score:.6f} at epoch {self.early_stopping.best_epoch+1}")
                    
                    # Restore best weights if configured
                    if self.config.training.early_stopping_restore_best:
                        self.early_stopping.restore_weights(self.model)
                        print("✅ Restored best model weights")
                    
                    # Save final checkpoint
                    self._save_checkpoint(epoch, epoch_metrics, is_best=False, suffix='early_stop')
                    break
            
            # Print epoch summary
            epoch_time = epoch_metrics['train/epoch_time']
            train_loss = epoch_metrics['train/epoch_loss']
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Get validation loss if available
            if (epoch + 1) % self.config.training.eval_interval == 0:
                val_loss = eval_metrics.get('eval/loss', 0.0)
                val_loss_str = f"{val_loss:.6f}"
            else:
                val_loss_str = "N/A"
            
            # Early stopping patience info
            if self.early_stopping is not None:
                patience_used = self.early_stopping.counter
                patience_total = self.early_stopping.patience
                patience_str = f"patience={patience_used}/{patience_total}"
            else:
                patience_str = "patience=N/A"
            
            print(f"\n{'='*70}")
            print(f"📊 Epoch {epoch+1}/{self.config.training.num_epochs} Summary")
            print(f"{'='*70}")
            print(f"  Time:       {epoch_time:.2f}s")
            print(f"  LR:         {current_lr:.2e}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss_str}")
            print(f"  Early Stop: {patience_str}")
            print(f"{'='*70}\n")
        
        # Save final checkpoint if training completed normally
        if epoch + 1 == self.config.training.num_epochs:
            self._save_checkpoint(epoch, epoch_metrics, is_best, suffix='final')
        
        # Post-training evaluation: Compare generated vs real samples
        print(f"\n{'='*70}")
        print("🎨 Post-Training Evaluation")
        print(f"{'='*70}")
        
        try:
            from .evaluation import compare_generated_vs_real
            from .diffusion_process_viz import visualize_diffusion_steps
            from .npz_visualization import visualize_sample_as_npz
            
            # Get a batch of real data for comparison (use test set for final evaluation)
            real_batch = next(iter(self.test_loader))
            real_x_sig, real_geom, real_label, _ = real_batch
            
            print("📊 Using test set for final evaluation...")
            
            # 1. Compare generated vs real (4 samples from test set)
            compare_generated_vs_real(
                self.diffusion,
                real_x_sig,
                real_geom,
                real_label,
                num_samples=4,
                save_dir=Path(self.config.training.output_dir) / "final_evaluation"
            )
            
            # 2. Visualize diffusion process step-by-step (1 sample from test set)
            print("\n")
            visualize_diffusion_steps(
                self.diffusion,
                real_x_sig[:1],  # Take only first sample
                real_geom[:1],
                real_label[:1],
                num_steps=10,
                save_path=Path(self.config.training.output_dir) / "final_evaluation" / "diffusion_process_steps.png"
            )
            
            # 3. NPZ-style 3D visualization (1 sample from test set)
            print("\n")
            visualize_sample_as_npz(
                self.diffusion,
                real_x_sig[:1],
                real_geom[:1],
                real_label[:1],
                save_path=Path(self.config.training.output_dir) / "final_evaluation" / "generated_sample_3d.png",
                detector_csv="csv/detector_geometry.csv"
            )
            
            print("✅ Post-training evaluation complete!")
        except Exception as e:
            print(f"⚠️  Post-training evaluation failed: {e}")
        
        # Generate training curves
        self._generate_training_curves()
        
        # Run forward diffusion test
        self._run_forward_diffusion_test()
        
        # Close logging
        self.writer.close()
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        # Close log file and restore original print
        if hasattr(self, 'log_file'):
            self.log_file.close()
        if hasattr(self, 'original_print'):
            import builtins
            builtins.print = self.original_print
        
        print(f"\n{'='*70}")
        print("🎉 Training completed!")
        print(f"📝 Training log saved to: {self.log_file_path}")
        print(f"📊 Training curves saved to: {Path(self.config.training.output_dir) / 'plots' / 'training_curves.png'}")
        print(f"📊 Training curves (PDF) saved to: {Path(self.config.training.output_dir) / 'plots' / 'training_curves.pdf'}")
        print(f"📁 All outputs saved in: {self.config.training.output_dir}")
        print(f"{'='*70}")

    def _generate_training_curves(self):
        """Generate and save training curves."""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Matplotlib not available - skipping training curves generation")
            return
            
        try:
            print("\n📊 Generating training curves...")
            
            # Create plots directory
            plots_dir = Path(self.config.training.output_dir) / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Parse training log to extract losses
            train_losses = []
            val_losses = []
            epochs = []
            
            if hasattr(self, 'log_file_path') and os.path.exists(self.log_file_path):
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    # Parse epoch summary lines
                    if "Epoch Summary" in line and "Train Loss:" in line and "Val Loss:" in line:
                        try:
                            # Extract epoch number
                            epoch_match = line.split("Epoch ")[1].split("/")[0]
                            epoch = int(epoch_match)
                            
                            # Extract train loss
                            train_match = line.split("Train Loss: ")[1].split()[0]
                            train_loss = float(train_match)
                            
                            # Extract val loss
                            val_match = line.split("Val Loss: ")[1].split()[0]
                            if val_match != "N/A":
                                val_loss = float(val_match)
                                
                                epochs.append(epoch)
                                train_losses.append(train_loss)
                                val_losses.append(val_loss)
                        except (ValueError, IndexError) as e:
                            continue
            
            if len(epochs) == 0:
                print("⚠️  No epoch data found for plotting")
                return
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Plot train and val losses
            plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
            plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, alpha=0.8)
            
            # Add best val loss point
            if len(val_losses) > 0:
                best_epoch_idx = val_losses.index(min(val_losses))
                best_epoch = epochs[best_epoch_idx]
                best_val_loss = val_losses[best_epoch_idx]
                plt.scatter([best_epoch], [best_val_loss], color='red', s=100, 
                           label=f'Best Val Loss: {best_val_loss:.6f}', zorder=5)
            
            # Customize plot
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title(f'Training Curves - {self.config.experiment_name}', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            # Add statistics text
            if len(train_losses) > 0 and len(val_losses) > 0:
                final_train = train_losses[-1]
                final_val = val_losses[-1]
                min_val = min(val_losses)
                
                stats_text = f'Final Train Loss: {final_train:.6f}\n'
                stats_text += f'Final Val Loss: {final_val:.6f}\n'
                stats_text += f'Best Val Loss: {min_val:.6f}\n'
                stats_text += f'Total Epochs: {len(epochs)}'
                
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Save plot
            plot_path = plots_dir / "training_curves.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Training curves saved to: {plot_path}")
            
            # Also save as PDF for better quality
            plot_pdf_path = plots_dir / "training_curves.pdf"
            plt.figure(figsize=(12, 8))
            plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
            plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, alpha=0.8)
            
            if len(val_losses) > 0:
                best_epoch_idx = val_losses.index(min(val_losses))
                best_epoch = epochs[best_epoch_idx]
                best_val_loss = val_losses[best_epoch_idx]
                plt.scatter([best_epoch], [best_val_loss], color='red', s=100, 
                           label=f'Best Val Loss: {best_val_loss:.6f}', zorder=5)
            
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title(f'Training Curves - {self.config.experiment_name}', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            if len(train_losses) > 0 and len(val_losses) > 0:
                final_train = train_losses[-1]
                final_val = val_losses[-1]
                min_val = min(val_losses)
                
                stats_text = f'Final Train Loss: {final_train:.6f}\n'
                stats_text += f'Final Val Loss: {final_val:.6f}\n'
                stats_text += f'Best Val Loss: {min_val:.6f}\n'
                stats_text += f'Total Epochs: {len(epochs)}'
                
                plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(plot_pdf_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Training curves (PDF) saved to: {plot_pdf_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to generate training curves: {e}")

    def _run_forward_diffusion_test(self):
        """Run forward diffusion test to verify Gaussian convergence."""
        try:
            print("\n🔬 Running forward diffusion test...")
            
            # Get a small batch of validation data
            val_batch = next(iter(self.val_loader))
            x_sig, geom, labels = val_batch['x_sig'], val_batch['x_geo'], val_batch['labels']
            
            # Use only first few samples for efficiency
            batch_size = min(8, x_sig.size(0))
            x_sig = x_sig[:batch_size].to(self.device)
            geom = geom[:batch_size].to(self.device)
            labels = labels[:batch_size].to(self.device)
            
            # Test key timesteps
            T = self.diffusion.cfg.timesteps
            test_timesteps = [0, T//4, T//2, (3*T)//4, T-1]
            
            print(f"\n📊 Forward Diffusion Statistics (batch_size={batch_size})")
            print(f"{'Timestep':<10} {'Charge Range':<30} {'Time Range':<30} {'SNR':<10}")
            print(f"{'-'*80}")
            
            for t_val in test_timesteps:
                t = torch.full((batch_size,), t_val, device=self.device, dtype=torch.long)
                x_t = self.diffusion.q_sample(x_sig, t)
                
                charge_range = f"[{x_t[:,0].min():.3f}, {x_t[:,0].max():.3f}]"
                time_range = f"[{x_t[:,1].min():.3f}, {x_t[:,1].max():.3f}]"
                
                sqrt_alpha_bar = self.diffusion.sqrt_alphas_cumprod[t_val].item()
                sqrt_one_minus = self.diffusion.sqrt_one_minus_alphas_cumprod[t_val].item()
                snr = sqrt_alpha_bar / sqrt_one_minus if sqrt_one_minus > 0 else float('inf')
                
                print(f"{t_val:<10} {charge_range:<30} {time_range:<30} {snr:<10.4f}")
            
            # Per-sample statistics for final timestep
            print(f"\n📈 Per-Sample Statistics (t={T-1})")
            print(f"{'Sample':<8} {'Original':<20} {'Final':<20} {'Change':<15}")
            print(f"{'Index':<8} {'Charge Mean':<10} {'Charge Mean':<10} {'Charge':<7} {'Time':<7}")
            print(f"{'-'*80}")
            
            # Original statistics
            orig_charge_means = x_sig[:, 0].mean(dim=1)
            orig_time_means = x_sig[:, 1].mean(dim=1)
            
            # Final timestep
            t_final = torch.full((batch_size,), T-1, device=self.device, dtype=torch.long)
            x_final = self.diffusion.q_sample(x_sig, t_final)
            final_charge_means = x_final[:, 0].mean(dim=1)
            final_time_means = x_final[:, 1].mean(dim=1)
            
            # Calculate changes
            charge_changes = final_charge_means - orig_charge_means
            time_changes = final_time_means - orig_time_means
            
            # Print per-sample stats
            for i in range(batch_size):
                print(f"{i:<8} {orig_charge_means[i]:<10.4f} {final_charge_means[i]:<10.4f} {charge_changes[i]:<7.4f} {time_changes[i]:<7.4f}")
            
            # Overall statistics
            print(f"\n📊 Overall Statistics:")
            print(f"Charge: orig={orig_charge_means.mean():.4f}±{orig_charge_means.std():.4f}, "
                  f"final={final_charge_means.mean():.4f}±{final_charge_means.std():.4f}")
            print(f"Time:   orig={orig_time_means.mean():.4f}±{orig_time_means.std():.4f}, "
                  f"final={final_time_means.mean():.4f}±{final_time_means.std():.4f}")
            print(f"Changes: charge={charge_changes.mean():.4f}±{charge_changes.std():.4f}, "
                  f"time={time_changes.mean():.4f}±{time_changes.std():.4f}")
            
            print("✅ Forward diffusion test complete!")
            
        except Exception as e:
            print(f"⚠️  Forward diffusion test failed: {e}")
            import traceback
            traceback.print_exc()


def create_trainer(config: ExperimentConfig) -> Trainer:
    """Create a trainer instance from configuration."""
    return Trainer(config)


if __name__ == "__main__":
    # Test trainer
    from config import get_default_config
    
    config = get_default_config()
    trainer = create_trainer(config)
    print("Trainer created successfully!")
