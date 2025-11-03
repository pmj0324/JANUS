#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENESIS Unified Training Script
================================

Train either Diffusion or Flow Matching models.

Usage:
    # Train Diffusion model
    python train.py --config examples/diffusion_default.yaml --model diffusion
    
    # Train Flow Matching model
    python train.py --config examples/flow_default.yaml --model flow
"""

import sys
import os
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch


def main():
    parser = argparse.ArgumentParser(description="Train GENESIS IceCube generative model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration YAML file")
    parser.add_argument("--model", type=str, choices=["diffusion", "flow"], required=True,
                       help="Model type: diffusion or flow")
    parser.add_argument("--data-path", type=str, default=None,
                       help="Override data path from config")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: auto, cuda, cpu")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    
    # Training overrides
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Override number of dataloader workers")
    
    args = parser.parse_args()
    
    # Load appropriate config and trainer based on model type
    if args.model == "diffusion":
        print("\n" + "="*70)
        print("Training Diffusion Model")
        print("="*70 + "\n")
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Diffusion"))
        from Diffusion.config import load_config_from_file
        from Diffusion.training import create_trainer
        
    elif args.model == "flow":
        print("\n" + "="*70)
        print("Training Flow Matching Model")
        print("="*70 + "\n")
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Flow"))
        from Flow.config import load_config_from_file
        from Flow.training import create_trainer
    
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    # Load configuration
    config = load_config_from_file(args.config)
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    config.device = device
    
    # Apply overrides
    if args.data_path is not None:
        config.data.h5_path = args.data_path
        print(f"📝 Override data_path: {args.data_path}")
    
    if args.resume:
        config.training.resume_from_checkpoint = args.resume
        print(f"📝 Resume from: {args.resume}")
    
    if args.batch_size is not None:
        print(f"📝 Override batch_size: {config.data.batch_size} → {args.batch_size}")
        config.data.batch_size = args.batch_size
    
    if args.lr is not None:
        print(f"📝 Override learning_rate: {config.training.learning_rate} → {args.lr}")
        config.training.learning_rate = args.lr
    
    if args.epochs is not None:
        print(f"📝 Override num_epochs: {config.training.num_epochs} → {args.epochs}")
        config.training.num_epochs = args.epochs
    
    if args.num_workers is not None:
        print(f"📝 Override num_workers: {config.data.num_workers} → {args.num_workers}")
        config.data.num_workers = args.num_workers
    
    # Print config summary
    print(f"\n{'='*70}")
    print(f"Configuration Summary")
    print(f"{'='*70}")
    print(f"  Model Type: {args.model.upper()}")
    print(f"  Architecture: {config.model.architecture}")
    print(f"  Device: {config.device}")
    print(f"  Batch Size: {config.data.batch_size}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Data: {config.data.h5_path}")
    print(f"{'='*70}\n")
    
    # Create and run trainer
    print(f"📊 Loading data from {config.data.h5_path}")
    print(f"🏗️  Creating model: {config.model.architecture}")
    print(f"🚀 Initializing trainer\n")
    
    trainer = create_trainer(config)
    
    print(f"🎯 Starting training\n")
    trainer.train()
    
    print(f"\n{'='*70}")
    print(f"✅ Training completed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

