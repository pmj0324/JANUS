import torch as th
import argparse
import yaml
import dataloader

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, required=True)
args = parser.parse_args()

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

for batch in data_loader:
    sig, geo, label = batch
    print(f"Signal shape: {sig.shape}")      # (B, 2, L)
    print(f"Geometry shape: {geo.shape}")    # (B, 3, L)
    print(f"Label shape: {label.shape}")     # (B, 6)
    break

print("Done")