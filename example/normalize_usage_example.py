#!/usr/bin/env python3
"""
Normalize Decorator Usage Examples
===================================

This file demonstrates how to use the normalize decorator in:
1. Forward diffusion process
2. Visualization functions
3. Training loops

Examples show channel-wise normalization for sig (B, 2, L) where:
- Channel 0: npe (number of photoelectrons)
- Channel 1: firstTime (first hit time)
"""

import torch as th
import argparse
import yaml
import dataloader
import diffusion
from diffusion.schedules import (
    linear_beta_schedule,
    cosine_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
)
from utils.normalize import normalize


# ============================================================================
# Example 1: Forward Diffusion with Normalization
# ============================================================================

@normalize(
    channel_methods=['minmax', 'log_minmax'],
    arg_index=0  # Normalize first argument (x0)
)
def apply_forward_diffusion_normalized(
    x0: th.Tensor,
    betas: th.Tensor,
    timesteps: th.Tensor,
    noise=None
) -> th.Tensor:
    """
    Apply forward diffusion with automatic normalization.
    
    Before calling this function, x0[:, 0, :] (npe) is normalized with minmax,
    and x0[:, 1, :] (firstTime) is normalized with log_minmax.
    
    Args:
        x0: Clean samples (B, 2, L) - automatically normalized
        betas: Beta schedule (T,)
        timesteps: Timesteps to apply (B,)
        noise: Optional pre-generated noise
    
    Returns:
        Noised samples x_t (B, 2, L)
    """
    # x0 is already normalized by the decorator
    return diffusion.apply_forward_diffusion(
        x0=x0,
        betas=betas,
        timesteps=timesteps,
        noise=noise
    )


# Alternative: Using dict format for clarity
@normalize(
    channel_methods={'npe': 'zscore', 'firstTime': 'log_minmax'},
    feature_ranges={'npe': (0, 1), 'firstTime': (-1, 1)}
)
def apply_forward_diffusion_normalized_dict(
    x0: th.Tensor,
    betas: th.Tensor,
    timesteps: th.Tensor,
    noise=None
) -> th.Tensor:
    """
    Apply forward diffusion with channel-specific normalization.
    
    - npe channel: zscore normalization
    - firstTime channel: log_minmax normalization to [-1, 1]
    """
    return diffusion.apply_forward_diffusion(
        x0=x0,
        betas=betas,
        timesteps=timesteps,
        noise=noise
    )


# ============================================================================
# Example 2: Visualization with Normalization
# ============================================================================

@normalize(
    channel_methods=['minmax', 'log_minmax'],
    arg_index=0  # Normalize x0_sig
)
def visualize_with_normalization(
    x0_sig: th.Tensor,
    geom: th.Tensor,
    label: th.Tensor,
    schedules: list,
    timesteps: list,
    output_dir: str = "./forward_visualization"
):
    """
    Visualize forward diffusion with automatic normalization.
    
    x0_sig is normalized before visualization:
    - npe: minmax to [0, 1]
    - firstTime: log_minmax to [0, 1]
    """
    from utils.vis.visualize_forward_diffusion import visualize_forward_diffusion
    
    # x0_sig is already normalized by the decorator
    visualize_forward_diffusion(
        x0_sig=x0_sig,
        geom=geom,
        label=label,
        schedules=schedules,
        timesteps=timesteps,
        output_dir=output_dir,
        save_3d=True,
        save_histograms=True,
    )


# ============================================================================
# Example 3: Training Loop with Normalization
# ============================================================================

@normalize(
    channel_methods=['zscore', 'log_minmax'],
    arg_index=0  # Normalize sig
)
def training_step_normalized(
    sig: th.Tensor,
    geo: th.Tensor,
    label: th.Tensor,
    betas: th.Tensor,
    timesteps: th.Tensor
):
    """
    Training step with automatic normalization.
    
    sig is normalized before processing:
    - npe: zscore (mean=0, std=1)
    - firstTime: log_minmax to [0, 1]
    """
    # sig is already normalized
    device = sig.device
    
    # Apply forward diffusion
    sig_noised = diffusion.apply_forward_diffusion(
        x0=sig,
        betas=betas,
        timesteps=timesteps,
    )
    
    # Your model training code here
    # loss = model(sig_noised, timesteps, sig)
    # ...
    
    return sig_noised


# ============================================================================
# Example 4: Complete Training Script
# ============================================================================

def main_with_normalization():
    """
    Complete training script showing normalize decorator usage.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    
    # Create dataset
    loader_type = config["data"]["loader"]
    if loader_type in ["h5", "hdf5"]:
        dataset = dataloader.H5Dataset(
            h5_path=config["data"]["h5_path"],
            angle_conversion=config["data"].get("angle_conversion", False),
            num_workers=config["data"].get("num_workers"),
            shuffle=config["data"].get("shuffle"),
        )
    else:
        raise ValueError(f"Unsupported loader: {loader_type}")
    
    # Create DataLoader
    data_loader = th.utils.data.DataLoader(
        dataset,
        batch_size=config["data"]["bsz"],
        shuffle=dataset.shuffle if dataset.shuffle is not None else config["data"].get("shuffle", False),
        num_workers=dataset.num_workers if dataset.num_workers is not None else config["data"].get("num_workers", 0),
        pin_memory=config["data"].get("pin_memory", False),
    )
    
    # Get diffusion configuration
    diffusion_config = config.get("diffusion", {})
    schedule_config = diffusion_config.get("schedule", {})
    
    schedule_type = schedule_config["type"]
    timesteps = schedule_config["timesteps"]
    
    # Create beta schedule
    if schedule_type == "linear":
        betas = linear_beta_schedule(
            timesteps, 
            schedule_config["beta_start"], 
            schedule_config["beta_end"]
        )
    elif schedule_type == "cosine":
        betas = cosine_beta_schedule(timesteps, schedule_config["s"])
    elif schedule_type == "quadratic":
        betas = quadratic_beta_schedule(
            timesteps,
            schedule_config["beta_start"],
            schedule_config["beta_end"]
        )
    elif schedule_type == "sigmoid":
        betas = sigmoid_beta_schedule(
            timesteps,
            schedule_config["beta_start"],
            schedule_config["beta_end"]
        )
    
    # Setup device
    _default = "cuda" if th.cuda.is_available() else ("mps" if getattr(th.backends, "mps", None) and th.backends.mps.is_available() else "cpu")
    device = th.device(config.get("device", _default))
    betas = betas.to(device)
    
    print("\nApplying forward diffusion with normalization...")
    print("Normalization:")
    print("  - npe channel: minmax to [0, 1]")
    print("  - firstTime channel: log_minmax to [0, 1]")
    
    for batch_idx, batch in enumerate(data_loader):
        sig, geo, label = batch
        sig = sig.to(device)  # (B, 2, L)
        
        # Random timesteps for each sample in batch
        B = sig.shape[0]
        timesteps_tensor = th.randint(0, timesteps, (B,), device=device)
        
        # Apply forward diffusion with normalization
        # The decorator automatically normalizes sig before processing
        sig_noised = apply_forward_diffusion_normalized(
            x0=sig,
            betas=betas,
            timesteps=timesteps_tensor,
        )
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Original signal shape: {sig.shape}")
        print(f"  Original signal range: [{sig.min():.4f}, {sig.max():.4f}]")
        print(f"  Normalized signal range: [{sig_noised.min():.4f}, {sig_noised.max():.4f}]")
        print(f"  Timesteps: {timesteps_tensor.tolist()}")
        
        if batch_idx == 0:
            break
    
    print("\nDone")


# ============================================================================
# Example 5: Visualization Script
# ============================================================================

def visualize_example():
    """
    Example of using normalization in visualization.
    """
    # Load your data
    # sig, geo, label = load_your_data()
    
    # Define normalization methods
    # Option 1: List format
    @normalize(
        channel_methods=['minmax', 'log_minmax'],
        arg_index=0
    )
    def visualize_normalized(sig, geo, label, schedules, timesteps):
        from utils.vis.visualize_forward_diffusion import visualize_forward_diffusion
        
        visualize_forward_diffusion(
            x0_sig=sig,
            geom=geo,
            label=label,
            schedules=schedules,
            timesteps=timesteps,
            output_dir="./visualization_output",
        )
    
    # Option 2: Dict format with custom ranges
    @normalize(
        channel_methods={'npe': 'zscore', 'firstTime': 'log_minmax'},
        feature_ranges={'npe': (0, 1), 'firstTime': (-1, 1)},
        arg_index=0
    )
    def visualize_normalized_dict(sig, geo, label, schedules, timesteps):
        from utils.vis.visualize_forward_diffusion import visualize_forward_diffusion
        
        visualize_forward_diffusion(
            x0_sig=sig,
            geom=geo,
            label=label,
            schedules=schedules,
            timesteps=timesteps,
            output_dir="./visualization_output",
        )
    
    # Usage:
    # schedules = [("linear", {}), ("cosine", {"s": 0.008})]
    # timesteps = [0, 100, 500, 999]
    # visualize_normalized(sig, geo, label, schedules, timesteps)


# ============================================================================
# Example 6: Different Normalization for Different Use Cases
# ============================================================================

# For training: zscore for npe (better for gradient flow)
@normalize(channel_methods=['zscore', 'log_minmax'])
def training_function(sig, geo, label):
    """Training with zscore for npe."""
    pass

# For visualization: minmax for npe (easier to interpret)
@normalize(channel_methods=['minmax', 'log_minmax'])
def visualization_function(sig, geo, label):
    """Visualization with minmax for npe."""
    pass

# For inference: same as training
@normalize(channel_methods=['zscore', 'log_minmax'])
def inference_function(sig, geo, label):
    """Inference with same normalization as training."""
    pass


if __name__ == '__main__':
    # Run the example
    main_with_normalization()
