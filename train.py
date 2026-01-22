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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    # Bring the config file
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

    # Create dataset based on loader type
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
    # Use dataset's num_workers and shuffle if provided, otherwise use config
    data_loader = th.utils.data.DataLoader(
        dataset,
        batch_size=config["data"]["bsz"],
        shuffle=dataset.shuffle if dataset.shuffle is not None else config["data"].get("shuffle", False),
        num_workers=dataset.num_workers if dataset.num_workers is not None else config["data"].get("num_workers", 0),
        pin_memory=config["data"].get("pin_memory", False),
    )


    # for batch in data_loader:
    #     sig, geo, label = batch
    #     print(f"Signal shape: {sig.shape}")     # (B, 2, L)
    #     print(f"Signal : sig[0] = {sig[0]}")
    #     print(f"Geometry shape: {geo.shape}")    # (B, 3, L)
    #     print(f"Geometry : geo[0] = {geo[0]}")
    #     print(f"Label shape: {label.shape}")     # (B, 6)
    #     print(f"Label : label[0] = {label[0]}")
    #     break

    # print("Done")

    
    # Get diffusion configuration
    diffusion_config = config.get("diffusion", {})
    schedule_config = diffusion_config.get("schedule", {})
    
    # Step 1: Check if schedule type is declared
    if "type" not in schedule_config:
        raise ValueError(
            "Schedule type is not specified in config. "
            "Please add 'type' field in diffusion.schedule (e.g., 'linear', 'cosine', 'quadratic', 'sigmoid')"
        )
    
    schedule_type = schedule_config["type"]
    
    # Step 2: Validate schedule type
    valid_types = ["linear", "cosine", "quadratic", "sigmoid"]
    if schedule_type not in valid_types:
        raise ValueError(
            f"Unknown schedule type: {schedule_type}. "
            f"Choose from: {valid_types}"
        )
    
    # Step 3: Check required parameters for each schedule type
    if "timesteps" not in schedule_config:
        raise ValueError(
            f"Required parameter 'timesteps' is missing for schedule type '{schedule_type}'. "
            f"Please add 'timesteps' in diffusion.schedule"
        )
    timesteps = schedule_config["timesteps"]
    
    # Step 4: Validate type-specific parameters
    if schedule_type == "linear":
        required_params = ["beta_start", "beta_end"]
        missing_params = [p for p in required_params if p not in schedule_config]
        if missing_params:
            raise ValueError(
                f"Required parameters for 'linear' schedule are missing: {missing_params}. "
                f"Please add {missing_params} in diffusion.schedule"
            )
        beta_start = schedule_config["beta_start"]
        beta_end = schedule_config["beta_end"]
        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        
    elif schedule_type == "cosine":
        required_params = ["s"]
        missing_params = [p for p in required_params if p not in schedule_config]
        if missing_params:
            raise ValueError(
                f"Required parameters for 'cosine' schedule are missing: {missing_params}. "
                f"Please add {missing_params} in diffusion.schedule"
            )
        s = schedule_config["s"]
        betas = cosine_beta_schedule(timesteps, s)
        
    elif schedule_type == "quadratic":
        required_params = ["beta_start", "beta_end"]
        missing_params = [p for p in required_params if p not in schedule_config]
        if missing_params:
            raise ValueError(
                f"Required parameters for 'quadratic' schedule are missing: {missing_params}. "
                f"Please add {missing_params} in diffusion.schedule"
            )
        beta_start = schedule_config["beta_start"]
        beta_end = schedule_config["beta_end"]
        betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        
    elif schedule_type == "sigmoid":
        required_params = ["beta_start", "beta_end"]
        missing_params = [p for p in required_params if p not in schedule_config]
        if missing_params:
            raise ValueError(
                f"Required parameters for 'sigmoid' schedule are missing: {missing_params}. "
                f"Please add {missing_params} in diffusion.schedule"
            )
        beta_start = schedule_config["beta_start"]
        beta_end = schedule_config["beta_end"]
        betas = sigmoid_beta_schedule(timesteps, beta_start, beta_end)
    
    print(f"Creating noise schedule: {schedule_type}")
    print(f"  Timesteps: {timesteps}")
    if schedule_type == "linear":
        print(f"  beta_start: {schedule_config['beta_start']}, beta_end: {schedule_config['beta_end']}")
    elif schedule_type == "cosine":
        print(f"  Parameter s: {schedule_config['s']}")
    elif schedule_type == "quadratic":
        print(f"  beta_start: {schedule_config['beta_start']}, beta_end: {schedule_config['beta_end']}")
    elif schedule_type == "sigmoid":
        print(f"  beta_start: {schedule_config['beta_start']}, beta_end: {schedule_config['beta_end']}")
    print(f"  Beta schedule shape: {betas.shape}, range: [{betas.min():.6f}, {betas.max():.6f}]")
    
    # Setup device
    device = th.device(config.get("device", "cuda" if th.cuda.is_available() else "cpu"))
    betas = betas.to(device)
    print(f"Device: {device}")
    
    print("\nApplying forward diffusion...")
    for batch_idx, batch in enumerate(data_loader):
        sig, geo, label = batch
        sig = sig.to(device)  # (B, 2, L)
        
        # Random timesteps for each sample in batch
        B = sig.shape[0]
        timesteps_tensor = th.randint(0, timesteps, (B,), device=device)
        
        # Apply forward diffusion (add noise)
        sig_noised = diffusion.apply_forward_diffusion(
            x0=sig,
            betas=betas,
            timesteps=timesteps_tensor,
        )
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Original signal shape: {sig.shape}")
        print(f"  Original signal range: [{sig.min():.4f}, {sig.max():.4f}]")
        print(f"  Original signal mean: {sig.mean():.4f}, std: {sig.std():.4f}")
        print(f"  Timesteps: {timesteps_tensor.tolist()}")
        print(f"  Noised signal shape: {sig_noised.shape}")
        print(f"  Noised signal range: [{sig_noised.min():.4f}, {sig_noised.max():.4f}]")
        print(f"  Noised signal mean: {sig_noised.mean():.4f}, std: {sig_noised.std():.4f}")
        
        if batch_idx == 0:
            break
    
    print("\nDone")


    