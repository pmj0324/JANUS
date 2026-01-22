#!/usr/bin/env python3
"""
Quick Start: Normalize Decorator 
=======================================

   .
"""

import torch as th
from utils.normalize import normalize
import diffusion


# ============================================================================
#  1: Forward Diffusion  ( )
# ============================================================================

#  A:   
@normalize(
    channel_methods=['minmax', 'log_minmax'],  # npe: minmax, firstTime: log_minmax
    arg_index=0  #    (sig) 
)
def my_forward_diffusion(sig, betas, timesteps):
    """
    sig   :
    - sig[:, 0, :] (npe) → minmax [0, 1]
    - sig[:, 1, :] (firstTime) → log_minmax [0, 1]
    """
    return diffusion.apply_forward_diffusion(
        x0=sig,  #  
        betas=betas,
        timesteps=timesteps
    )


#  :
# sig = ...  # (B, 2, L)   
# betas = ...
# timesteps = ...
# sig_noised = my_forward_diffusion(sig, betas, timesteps)  #  !


# ============================================================================
#  2: Visualization 
# ============================================================================

@normalize(
    channel_methods={'npe': 'minmax', 'firstTime': 'log_minmax'},
    arg_index=0
)
def my_visualize(sig, geo, label, schedules, timesteps):
    """  sig  """
    from utils.vis.visualize_forward_diffusion import visualize_forward_diffusion
    
    visualize_forward_diffusion(
        x0_sig=sig,  #  
        geom=geo,
        label=label,
        schedules=schedules,
        timesteps=timesteps,
        output_dir="./output"
    )


# ============================================================================
#  3: train.py  
# ============================================================================

def train_example():
    """train.py  """
    
    # ...    ...
    # sig, geo, label = batch
    
    #  forward diffusion  
    @normalize(channel_methods=['minmax', 'log_minmax'], arg_index=0)
    def forward_with_norm(sig, betas, timesteps):
        return diffusion.apply_forward_diffusion(sig, betas, timesteps)
    
    # 
    # sig_noised = forward_with_norm(sig, betas, timesteps)


# ============================================================================
#  4:    
# ============================================================================

# npe: zscore, firstTime: log_minmax
@normalize(channel_methods=['zscore', 'log_minmax'])
def method1(sig):
    return sig

# npe: minmax, firstTime: log_minmax ( )
@normalize(channel_methods=['minmax', 'log_minmax'])
def method2(sig):
    return sig

# npe: log_minmax, firstTime: log_minmax
@normalize(channel_methods=['log_minmax', 'log_minmax'])
def method3(sig):
    return sig

# Dict  ( )
@normalize(
    channel_methods={'npe': 'minmax', 'firstTime': 'log_minmax'},
    feature_ranges={'npe': (0, 1), 'firstTime': (-1, 1)}
)
def method4(sig):
    return sig


if __name__ == '__main__':
    print("="*80)
    print("Normalize Decorator Quick Start")
    print("="*80)
    print("\n1. Forward Diffusion :")
    print("   @normalize(channel_methods=['minmax', 'log_minmax'], arg_index=0)")
    print("   def my_forward_diffusion(sig, betas, timesteps):")
    print("       return diffusion.apply_forward_diffusion(sig, betas, timesteps)")
    print("\n2. Visualization :")
    print("   @normalize(channel_methods=['minmax', 'log_minmax'], arg_index=0)")
    print("   def my_visualize(sig, geo, label, ...):")
    print("       visualize_forward_diffusion(x0_sig=sig, ...)")
    print("\n3. :")
    print("   sig_noised = my_forward_diffusion(sig, betas, timesteps)")
    print("   # sig   !")
    print("\n" + "="*80)
