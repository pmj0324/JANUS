#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer for GENESIS Flow Matching model.

Simplified trainer for Flow Matching (similar structure to Diffusion trainer).
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from dataloader.pmt_dataloader import make_dataloader
from models.factory import ModelFactory
from flow import ConditionalFlowMatching
from config import ExperimentConfig
from .schedulers import create_scheduler
from .checkpointing import CheckpointManager
from .utils import EarlyStopping

try:
    from torch.cuda.amp import GradScaler
    from torch.amp import autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


class Trainer:
    """Trainer for Flow Matching model."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # Initialize
        self._setup_logging()
        self._setup_model()
        self._setup_optimizer()
        self._setup_data()
        self._setup_checkpointing()
        
        # State
        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Early stopping
        if config.training.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                min_delta=config.training.early_stopping_min_delta,
                mode=config.training.early_stopping_mode,
                restore_best_weights=config.training.early_stopping_restore_best,
                verbose=config.training.early_stopping_verbose
            )
        else:
            self.early_stopping = None
        
        # Mixed precision
        self.scaler = GradScaler() if AMP_AVAILABLE and config.training.use_amp else None
        
        # Resume
        if config.training.resume_from_checkpoint:
            self._load_checkpoint(config.training.resume_from_checkpoint)
    
    def _setup_logging(self):
        """Setup logging."""
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.training.log_dir, exist_ok=True)
        
        self.log_file = open(
            os.path.join(self.config.training.log_dir, f"{self.config.experiment_name}_training.txt"),
            'w', encoding='utf-8'
        )
        
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.config.training.log_dir, self.config.experiment_name)
        )
    
    def _setup_model(self):
        """Initialize model and flow wrapper."""
        self.model = ModelFactory.create_model_from_config(self.config.model).to(self.device)
        
        # Create Flow Matching wrapper
        from flow import FlowConfig as FlowCfg
        flow_cfg = FlowCfg(
            sigma_min=self.config.flow.sigma_min,
            use_ode_solver=self.config.flow.use_ode_solver,
            num_steps=self.config.flow.num_steps,
            use_cfm=self.config.flow.use_cfm,
            use_cfg=self.config.flow.use_cfg,
            cfg_scale=self.config.flow.cfg_scale,
            cfg_dropout=self.config.flow.cfg_dropout
        )
        self.flow = ConditionalFlowMatching(self.model, flow_cfg).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model initialized: {self.config.model.architecture}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Model size: {total_params * 4 / 1024**2:.1f} MB")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
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
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        self.scheduler = create_scheduler(self.optimizer, self.config.training)
        print(f"Optimizer: {self.config.training.optimizer}")
        print(f"Scheduler: {self.config.training.scheduler or 'None'}")
    
    def _setup_data(self):
        """Setup dataloaders."""
        import h5py
        
        with h5py.File(self.config.data.h5_path, 'r') as f:
            total_samples = len(f['input'])
        
        indices = np.arange(total_samples)
        if self.config.data.shuffle:
            np.random.seed(self.config.seed)
            np.random.shuffle(indices)
        
        train_size = int(total_samples * self.config.data.train_ratio)
        val_size = int(total_samples * self.config.data.val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        
        time_transform = getattr(self.config.data, 'time_transform', self.config.model.time_transform)
        affine_offsets = getattr(self.config.data, 'affine_offsets', self.config.model.affine_offsets)
        affine_scales = getattr(self.config.data, 'affine_scales', self.config.model.affine_scales)
        label_offsets = getattr(self.config.data, 'label_offsets', self.config.model.label_offsets)
        label_scales = getattr(self.config.data, 'label_scales', self.config.model.label_scales)
        
        self.train_loader = make_dataloader(
            h5_path=self.config.data.h5_path,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            replace_time_inf_with=self.config.data.replace_time_inf_with,
            channel_first=self.config.data.channel_first,
            indices=train_indices,
            time_transform=time_transform,
            affine_offsets=affine_offsets,
            affine_scales=affine_scales,
            label_offsets=label_offsets,
            label_scales=label_scales
        )
        
        self.val_loader = make_dataloader(
            h5_path=self.config.data.h5_path,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            replace_time_inf_with=self.config.data.replace_time_inf_with,
            channel_first=self.config.data.channel_first,
            indices=val_indices,
            time_transform=time_transform,
            affine_offsets=affine_offsets,
            affine_scales=affine_scales,
            label_offsets=label_offsets,
            label_scales=label_scales
        )
        
        print(f"Data loaded: {len(train_indices)} train, {len(val_indices)} val")
    
    def _setup_checkpointing(self):
        """Setup checkpoint manager."""
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.training.checkpoint_dir,
            max_checkpoints=5
        )
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"Resumed from epoch {self.start_epoch}, step {self.global_step}")
    
    def train_epoch(self, epoch: int):
        """Train one epoch."""
        self.model.train()
        self.flow.train()
        
        epoch_losses = []
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for step, (x_sig, geom, label, _) in enumerate(pbar):
            x_sig = x_sig.to(self.device)
            geom = geom.to(self.device)
            label = label.to(self.device)
            
            # Forward pass
            if self.scaler:
                with autocast('cuda'):
                    loss = self.flow.loss(x_sig, geom, label)
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
                loss = self.flow.loss(x_sig, geom, label)
                loss = loss / self.config.training.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            epoch_losses.append(loss.item() * self.config.training.gradient_accumulation_steps)
            self.global_step += 1
            
            # Logging
            if step % self.config.training.log_interval == 0:
                avg_loss = np.mean(epoch_losses[-self.config.training.log_interval:])
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                self.writer.add_scalar('train/loss', avg_loss, self.global_step)
        
        return {'loss': np.mean(epoch_losses)}
    
    @torch.no_grad()
    def validate_epoch(self, epoch: int):
        """Validate one epoch."""
        self.model.eval()
        self.flow.eval()
        
        val_losses = []
        
        for x_sig, geom, label, _ in self.val_loader:
            x_sig = x_sig.to(self.device)
            geom = geom.to(self.device)
            label = label.to(self.device)
            
            loss = self.flow.loss(x_sig, geom, label)
            val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        self.writer.add_scalar('val/loss', val_loss, epoch)
        
        return {'loss': val_loss}
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*70}")
        print(f"Training Flow Matching Model")
        print(f"{'='*70}\n")
        
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}")
            
            # Scheduler step
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
            
            # Save checkpoint
            val_loss = val_metrics['loss']
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            checkpoint = {
                'epoch': epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_loss': self.best_loss,
                'config': self.config
            }
            
            self.checkpoint_manager.save_checkpoint(
                checkpoint, epoch, val_loss, is_best=is_best
            )
            
            # Early stopping
            if self.early_stopping:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        self.log_file.close()
        self.writer.close()
        print("\nTraining completed!")


def create_trainer(config: ExperimentConfig) -> Trainer:
    """Create trainer instance."""
    return Trainer(config)

