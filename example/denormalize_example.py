#!/usr/bin/env python3
"""
Denormalize  
=====================

normalize  denormalize   .
    .
"""

import torch as th
from utils.normalize import normalize
import diffusion


# ============================================================================
#  1: Forward Diffusion with Auto Denormalize
# ============================================================================

@normalize(
    channel_methods=['minmax', 'log_minmax'],
    arg_index=0,  #    (sig) 
    denormalize=True  #   
)
def apply_forward_diffusion_with_denormalize(sig, betas, timesteps):
    """
    sig , forward diffusion   .
    
    -  sig:  (npe: minmax, firstTime: log_minmax)
    - :   
    """
    return diffusion.apply_forward_diffusion(
        x0=sig,  #  
        betas=betas,
        timesteps=timesteps
    )


#  :
# sig = ...  # (B, 2, L)  
# sig_noised = apply_forward_diffusion_with_denormalize(sig, betas, timesteps)
# # sig_noised    !


# ============================================================================
#  2: Visualization with Denormalize
# ============================================================================

@normalize(
    channel_methods={'npe': 'zscore', 'firstTime': 'log_minmax'},
    feature_ranges={'npe': (0, 1), 'firstTime': (0, 1)},
    arg_index=0,
    denormalize=True  #     
)
def visualize_with_denormalize(sig, geo, label, schedules, timesteps, output_dir):
    """
      ,    .
    """
    from utils.vis.visualize_forward_diffusion import visualize_forward_diffusion
    
    # sig  forward diffusion 
    #    
    visualize_forward_diffusion(
        x0_sig=sig,
        geom=geo,
        label=label,
        schedules=schedules,
        timesteps=timesteps,
        output_dir=output_dir
    )


# ============================================================================
#  3:    + 
# ============================================================================

@normalize(
    method='minmax',
    feature_range=(0, 1),
    denormalize=True
)
def process_single_channel(data):
    """
         .
    """
    # data  
    #     
    return data * 2  #  


# ============================================================================
#  4: denormalize=False ()
# ============================================================================

@normalize(
    channel_methods=['minmax', 'log_minmax'],
    denormalize=False  # ,    
)
def process_without_denormalize(sig):
    """
        .
    """
    return sig


# ============================================================================
#  5: train.py 
# ============================================================================

def train_with_denormalize_example():
    """
    train.py denormalize   
    """
    # ...   ...
    # sig, geo, label = batch
    
    @normalize(
        channel_methods=['minmax', 'log_minmax'],
        arg_index=0,
        denormalize=True  #   
    )
    def forward_with_denorm(sig, betas, timesteps):
        return diffusion.apply_forward_diffusion(sig, betas, timesteps)
    
    # 
    # sig_noised = forward_with_denorm(sig, betas, timesteps)
    # # sig_noised   (   )


# ============================================================================
#  6:  - denormalize=True vs False
# ============================================================================

def compare_denormalize():
    """
    denormalize    
    """
    sig = th.randn(2, 2, 100)  # (B, 2, L)
    
    # denormalize=False:   
    @normalize(channel_methods=['minmax', 'log_minmax'], denormalize=False)
    def process_false(sig):
        return sig
    
    result_false = process_false(sig)
    print(f"denormalize=False: range [{result_false.min():.4f}, {result_false.max():.4f}]")
    # :   (: [0, 1])
    
    # denormalize=True:   
    @normalize(channel_methods=['minmax', 'log_minmax'], denormalize=True)
    def process_true(sig):
        return sig
    
    result_true = process_true(sig)
    print(f"denormalize=True: range [{result_true.min():.4f}, {result_true.max():.4f}]")
    # :   (  )


if __name__ == '__main__':
    print("="*80)
    print("Denormalize   ")
    print("="*80)
    print("\n1.  :")
    print("   @normalize(channel_methods=['minmax', 'log_minmax'], denormalize=True)")
    print("   def my_function(sig, ...):")
    print("       # sig  ")
    print("       #     ")
    print("       return result")
    print("\n2. denormalize=False ():")
    print("   #    ")
    print("\n3. denormalize=True:")
    print("   #     ")
    print("\n" + "="*80)
