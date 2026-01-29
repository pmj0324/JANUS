#!/usr/bin/env python3
"""
Normalize Decorator Integration Examples
========================================

  normalize     .

1. Forward Diffusion  
2. Visualization  
3. train.py   
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
#  1: Wrapper   ( )
# ============================================================================

@normalize(
    channel_methods=['minmax', 'log_minmax'],
    arg_index=0  #    (sig) 
)
def apply_forward_diffusion_with_normalization(
    x0: th.Tensor,
    betas: th.Tensor,
    timesteps: th.Tensor,
    noise=None
) -> th.Tensor:
    """
    Forward diffusion ,  sig  .
    
    - sig[:, 0, :] (npe): minmax  [0, 1]
    - sig[:, 1, :] (firstTime): log_minmax  [0, 1]
    """
    return diffusion.apply_forward_diffusion(
        x0=x0,  #  
        betas=betas,
        timesteps=timesteps,
        noise=noise
    )


# ============================================================================
#  2: train.py  
# ============================================================================

def train_with_normalization_example():
    """
    train.py normalize   
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()
    
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    
    #  
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
    
    data_loader = th.utils.data.DataLoader(
        dataset,
        batch_size=config["data"]["bsz"],
        shuffle=dataset.shuffle if dataset.shuffle is not None else config["data"].get("shuffle", False),
        num_workers=dataset.num_workers if dataset.num_workers is not None else config["data"].get("num_workers", 0),
        pin_memory=config["data"].get("pin_memory", False),
    )
    
    # Diffusion 
    diffusion_config = config.get("diffusion", {})
    schedule_config = diffusion_config.get("schedule", {})
    schedule_type = schedule_config["type"]
    timesteps = schedule_config["timesteps"]
    
    # Beta schedule 
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
    
    _default = "cuda" if th.cuda.is_available() else ("mps" if getattr(th.backends, "mps", None) and th.backends.mps.is_available() else "cpu")
    device = th.device(config.get("device", _default))
    betas = betas.to(device)
    
    print("\n" + "="*80)
    print("Forward Diffusion with Channel-wise Normalization")
    print("="*80)
    print("Normalization methods:")
    print("  - npe channel (sig[:, 0, :]): minmax → [0, 1]")
    print("  - firstTime channel (sig[:, 1, :]): log_minmax → [0, 1]")
    print("="*80)
    
    for batch_idx, batch in enumerate(data_loader):
        sig, geo, label = batch
        sig = sig.to(device)  # (B, 2, L)
        
        #    
        print(f"\nBatch {batch_idx + 1} - Before normalization:")
        print(f"  npe range: [{sig[:, 0, :].min():.4f}, {sig[:, 0, :].max():.4f}]")
        print(f"  firstTime range: [{sig[:, 1, :].min():.4f}, {sig[:, 1, :].max():.4f}]")
        
        # Random timesteps
        B = sig.shape[0]
        timesteps_tensor = th.randint(0, timesteps, (B,), device=device)
        
        #  forward diffusion 
        #   sig 
        sig_noised = apply_forward_diffusion_with_normalization(
            x0=sig,  #   
            betas=betas,
            timesteps=timesteps_tensor,
        )
        
        print(f"\n  After normalization and forward diffusion:")
        print(f"  npe range: [{sig_noised[:, 0, :].min():.4f}, {sig_noised[:, 0, :].max():.4f}]")
        print(f"  firstTime range: [{sig_noised[:, 1, :].min():.4f}, {sig_noised[:, 1, :].max():.4f}]")
        print(f"  Timesteps: {timesteps_tensor.tolist()}")
        
        if batch_idx == 0:
            break
    
    print("\n Done")


# ============================================================================
#  3: Visualization  
# ============================================================================

@normalize(
    channel_methods={'npe': 'minmax', 'firstTime': 'log_minmax'},
    feature_ranges={'npe': (0, 1), 'firstTime': (0, 1)},
    arg_index=0  # x0_sig 
)
def visualize_forward_diffusion_normalized(
    x0_sig: th.Tensor,
    geom: th.Tensor,
    label: th.Tensor,
    schedules: list,
    timesteps: list,
    output_dir: str = "./forward_visualization"
):
    """
    Forward diffusion  ,  sig  .
    
    Args:
        x0_sig: Clean signals (B, 2, L) -  
        geom: Geometry (B, 3, L)
        label: Labels (B, 6)
        schedules: List of (schedule_name, schedule_kwargs) tuples
        timesteps: List of timesteps to visualize
        output_dir: Output directory
    """
    from utils.vis.visualize_forward_diffusion import visualize_forward_diffusion
    
    # x0_sig  
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


def visualize_example():
    """
    Visualization  
    """
    #   ()
    # sig, geo, label = load_data()
    
    #  
    schedules = [
        ("linear", {"beta_start": 1e-4, "beta_end": 2e-2}),
        ("cosine", {"s": 0.008})
    ]
    timesteps = [0, 100, 500, 999]
    
    #   
    # visualize_forward_diffusion_normalized(
    #     x0_sig=sig,
    #     geom=geo,
    #     label=label,
    #     schedules=schedules,
    #     timesteps=timesteps,
    #     output_dir="./visualization_output"
    # )


# ============================================================================
#  4:    
# ============================================================================

#  1: npe zscore, firstTime log_minmax ( )
@normalize(channel_methods=['zscore', 'log_minmax'])
def training_function_1(sig, geo, label):
    """npe: zscore, firstTime: log_minmax"""
    pass

#  2:   minmax ( )
@normalize(channel_methods=['minmax', 'minmax'])
def visualization_function_1(sig, geo, label):
    """npe: minmax, firstTime: minmax"""
    pass

#  3: npe minmax, firstTime log_minmax ()
@normalize(channel_methods=['minmax', 'log_minmax'])
def general_function(sig, geo, label):
    """npe: minmax, firstTime: log_minmax"""
    pass

#  4:   log_minmax (skewed  )
@normalize(channel_methods=['log_minmax', 'log_minmax'])
def skewed_data_function(sig, geo, label):
    """npe: log_minmax, firstTime: log_minmax"""
    pass

#  5: Dict   
@normalize(
    channel_methods={'npe': 'zscore', 'firstTime': 'log_minmax'},
    feature_ranges={'npe': (0, 1), 'firstTime': (-1, 1)}
)
def explicit_function(sig, geo, label):
    """     """
    pass


# ============================================================================
#   : train.py  
# ============================================================================

if __name__ == '__main__':
    #  1: train.py 
    print("Example 1: Training with normalization")
    print("-" * 80)
    # train_with_normalization_example()
    
    #  2: Visualization 
    print("\nExample 2: Visualization with normalization")
    print("-" * 80)
    # visualize_example()
    
    print("\n All examples ready to use!")
    print("\nUsage in your code:")
    print("  1. Import: from utils.normalize import normalize")
    print("  2. Decorate your function:")
    print("     @normalize(channel_methods=['minmax', 'log_minmax'], arg_index=0)")
    print("     def your_function(sig, ...):")
    print("         # sig is automatically normalized")
    print("         return result")
