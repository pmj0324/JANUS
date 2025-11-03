#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checkpointing utilities for GENESIS training.

This module provides comprehensive checkpointing functionality including:
- Automatic checkpoint saving
- Best model tracking
- Checkpoint loading and resuming
- Checkpoint cleanup and management
"""

from __future__ import annotations
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch


class CheckpointManager:
    """Manages model checkpoints during training."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        experiment_name: str,
        save_best_only: bool = True,
        max_checkpoints: int = 5,
        cleanup_old_checkpoints: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name of the experiment
            save_best_only: Whether to save only the best checkpoint
            max_checkpoints: Maximum number of checkpoints to keep
            cleanup_old_checkpoints: Whether to clean up old checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.save_best_only = save_best_only
        self.max_checkpoints = max_checkpoints
        self.cleanup_old_checkpoints = cleanup_old_checkpoints
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best checkpoint
        self.best_checkpoint_path = None
        self.best_metric_value = float('inf')
        
        # Track checkpoint history
        self.checkpoint_history = []
    
    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        epoch: int,
        is_best: bool = False,
        metric_value: Optional[float] = None,
        suffix: str = ''
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            checkpoint_data: Dictionary containing checkpoint data
            epoch: Current epoch number
            is_best: Whether this is the best checkpoint so far
            metric_value: Metric value for this checkpoint
            suffix: Optional suffix for checkpoint filename (e.g., 'early_stop', 'final')
            
        Returns:
            Path to saved checkpoint
        """
        # Determine checkpoint filename
        if is_best:
            checkpoint_filename = f"{self.experiment_name}_best.pt"
        elif suffix:
            checkpoint_filename = f"{self.experiment_name}_epoch_{epoch:04d}_{suffix}.pt"
        else:
            checkpoint_filename = f"{self.experiment_name}_epoch_{epoch:04d}.pt"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # Add metadata to checkpoint
        checkpoint_data['checkpoint_metadata'] = {
            'experiment_name': self.experiment_name,
            'epoch': epoch,
            'is_best': is_best,
            'metric_value': metric_value,
            'timestamp': torch.utils.data.get_worker_info()
        }
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update best checkpoint if necessary
        if is_best and metric_value is not None:
            if metric_value < self.best_metric_value:
                self.best_metric_value = metric_value
                self.best_checkpoint_path = checkpoint_path
        
        # Track checkpoint history
        self.checkpoint_history.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'is_best': is_best,
            'metric_value': metric_value
        })
        
        # Cleanup old checkpoints if enabled
        if self.cleanup_old_checkpoints and not is_best:
            self._cleanup_old_checkpoints()
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data dictionary
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Validate checkpoint
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
        for key in required_keys:
            if key not in checkpoint_data:
                raise ValueError(f"Invalid checkpoint: missing key '{key}'")
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint_data
    
    def load_best_checkpoint(self) -> Dict[str, Any]:
        """
        Load the best checkpoint.
        
        Returns:
            Best checkpoint data dictionary
        """
        if self.best_checkpoint_path is None:
            raise ValueError("No best checkpoint available")
        
        return self.load_checkpoint(self.best_checkpoint_path)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None
        """
        if not self.checkpoint_history:
            return None
        
        # Return the most recent checkpoint
        latest = max(self.checkpoint_history, key=lambda x: x['epoch'])
        return latest['path']
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob(f"{self.experiment_name}_*.pt"):
            try:
                # Load checkpoint metadata without loading the full model
                checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
                metadata = checkpoint_data.get('checkpoint_metadata', {})
                
                checkpoints.append({
                    'path': str(checkpoint_file),
                    'epoch': metadata.get('epoch', 0),
                    'is_best': metadata.get('is_best', False),
                    'metric_value': metadata.get('metric_value', None),
                    'size_mb': checkpoint_file.stat().st_size / 1024**2
                })
            except Exception as e:
                print(f"Warning: Could not load metadata for {checkpoint_file}: {e}")
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x['epoch'])
        return checkpoints
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints to save disk space."""
        if not self.cleanup_old_checkpoints:
            return
        
        # Get all non-best checkpoints
        non_best_checkpoints = [
            cp for cp in self.checkpoint_history 
            if not cp['is_best']
        ]
        
        # Keep only the most recent ones
        if len(non_best_checkpoints) > self.max_checkpoints:
            # Sort by epoch and keep only the most recent
            non_best_checkpoints.sort(key=lambda x: x['epoch'], reverse=True)
            checkpoints_to_remove = non_best_checkpoints[self.max_checkpoints:]
            
            for checkpoint_info in checkpoints_to_remove:
                checkpoint_path = Path(checkpoint_info['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    print(f"Removed old checkpoint: {checkpoint_path}")
            
            # Update history
            self.checkpoint_history = [
                cp for cp in self.checkpoint_history 
                if cp not in checkpoints_to_remove
            ]
    
    def save_training_summary(self, summary_data: Dict[str, Any]) -> str:
        """
        Save training summary to JSON file.
        
        Args:
            summary_data: Training summary data
            
        Returns:
            Path to summary file
        """
        summary_path = self.checkpoint_dir / f"{self.experiment_name}_summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"Training summary saved: {summary_path}")
        return str(summary_path)
    
    def cleanup_all_checkpoints(self):
        """Remove all checkpoints for this experiment."""
        for checkpoint_file in self.checkpoint_dir.glob(f"{self.experiment_name}_*.pt"):
            checkpoint_file.unlink()
            print(f"Removed checkpoint: {checkpoint_file}")
        
        # Clear history
        self.checkpoint_history = []
        self.best_checkpoint_path = None
        self.best_metric_value = float('inf')
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get information about a checkpoint without loading the full model.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint information dictionary
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load only metadata
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        metadata = checkpoint_data.get('checkpoint_metadata', {})
        
        # Get file info
        file_stat = Path(checkpoint_path).stat()
        
        return {
            'path': checkpoint_path,
            'epoch': metadata.get('epoch', 0),
            'is_best': metadata.get('is_best', False),
            'metric_value': metadata.get('metric_value', None),
            'size_mb': file_stat.st_size / 1024**2,
            'created': file_stat.st_ctime,
            'modified': file_stat.st_mtime,
            'config': checkpoint_data.get('config', {}),
            'has_model': 'model_state_dict' in checkpoint_data,
            'has_optimizer': 'optimizer_state_dict' in checkpoint_data,
            'has_scheduler': 'scheduler_state_dict' in checkpoint_data,
        }


def create_checkpoint_manager(
    checkpoint_dir: str,
    experiment_name: str,
    **kwargs
) -> CheckpointManager:
    """
    Create a checkpoint manager with default settings.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        experiment_name: Name of the experiment
        **kwargs: Additional arguments for CheckpointManager
        
    Returns:
        CheckpointManager instance
    """
    return CheckpointManager(checkpoint_dir, experiment_name, **kwargs)


def load_checkpoint_safely(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Safely load a checkpoint, returning None if it fails.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Checkpoint data or None if loading fails
    """
    try:
        return torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
        return None


def validate_checkpoint(checkpoint_data: Dict[str, Any]) -> bool:
    """
    Validate that a checkpoint contains required data.
    
    Args:
        checkpoint_data: Checkpoint data dictionary
        
    Returns:
        True if checkpoint is valid, False otherwise
    """
    required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
    
    for key in required_keys:
        if key not in checkpoint_data:
            print(f"Invalid checkpoint: missing key '{key}'")
            return False
    
    return True


if __name__ == "__main__":
    # Test checkpoint manager
    import tempfile
    import shutil
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create checkpoint manager
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            experiment_name="test_experiment",
            save_best_only=False,
            max_checkpoints=3
        )
        
        # Create dummy checkpoint data
        dummy_data = {
            'model_state_dict': {'dummy': torch.tensor([1.0])},
            'optimizer_state_dict': {'dummy': torch.tensor([1.0])},
            'epoch': 0,
            'config': {'test': True}
        }
        
        # Save some checkpoints
        for epoch in range(5):
            is_best = epoch == 2  # Make epoch 2 the best
            metric_value = 1.0 - epoch * 0.1  # Decreasing metric
            
            manager.save_checkpoint(
                dummy_data.copy(),
                epoch,
                is_best=is_best,
                metric_value=metric_value
            )
        
        # List checkpoints
        checkpoints = manager.list_checkpoints()
        print(f"Found {len(checkpoints)} checkpoints:")
        for cp in checkpoints:
            print(f"  Epoch {cp['epoch']}: {cp['path']} (best: {cp['is_best']})")
        
        # Test loading
        best_checkpoint = manager.load_best_checkpoint()
        print(f"Loaded best checkpoint from epoch {best_checkpoint['epoch']}")
        
        print("Checkpoint manager test completed successfully!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
