# GENESIS

**Generative Engine for Neutrino Event Simulation and Infernce System**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

GENESIS is a **generative model** for synthesizing IceCube PMT signals from neutrino events. It uses a **Diffusion Transformer (DiT)** architecture to learn the complex patterns of photomultiplier tube (PMT) responses conditioned on neutrino event properties.

### Key Features

- 🚀 **State-of-the-art DiT architecture** for high-quality signal generation
- 📊 **Efficient normalization** in dataloader (not model) for faster training
- 🎨 **3D visualization** of generated events
- ⚡ **GPU-optimized** training with automatic batch size selection
- 🔧 **Flexible configuration** via YAML files
- 📦 **Task management** system for organized experiments
- 🎯 **Classifier-free guidance** for better conditioning

---

## 🚀 Quick Start

### Installation

```bash
# Create environment
micromamba create -n genesis python=3.10 pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge
micromamba activate genesis

# Install dependencies
pip install h5py numpy matplotlib scipy tqdm tensorboard pyyaml
```

### Train Your First Model

```bash
# Train with default configuration
python scripts/train.py --config configs/default.yaml
```

### Generate Samples

```bash
# Sample from trained model
python scripts/sample.py \
    --config outputs/config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --n-samples 100
```

**→ See [Quick Start Guide](docs/setup/QUICK_START.md) for detailed instructions**

---

## 📚 Documentation

### 🚀 Getting Started
- **[Quick Start](docs/setup/QUICK_START.md)** - Get up and running in 5 minutes
- **[Getting Started](docs/setup/GETTING_STARTED.md)** - Complete setup tutorial
- **[Environment Setup](docs/setup/MICROMAMBA_SETUP.md)** - Conda/Micromamba installation

### 🏗️ Architecture
- **[Model Architecture](docs/architecture/MODEL_ARCHITECTURE.md)** - DiT model & normalization system
- **[Normalization](docs/architecture/NORMALIZATION.md)** - Complete normalization guide
- **[Diffusion Module](docs/architecture/DIFFUSION_MODULE.md)** - Diffusion process details

### 📖 Guides
- **[Training Guide](docs/guides/TRAINING.md)** - Train models
- **[Sampling Guide](docs/guides/SAMPLING.md)** - Generate samples
- **[GPU Optimization](docs/guides/GPU_OPTIMIZATION.md)** - Maximize performance

### 📋 Reference
- **[API Reference](docs/reference/API.md)** - Complete API documentation
- **[Full Documentation Index](docs/README.md)** - Browse all documentation

---

## 🏗️ Architecture

### Model Structure

```
Input: [charge, time, x, y, z] + Event Labels
   ↓
Dataloader Normalization (ln + affine)
   ↓
DiT Model (Transformer-based)
   ↓
Noise Prediction (2 channels: charge, time)
   ↓
Denormalization
   ↓
Output: Generated PMT Signals
```

### Normalization System

GENESIS uses a **two-stage normalization** applied in the **dataloader**:

1. **Time transformation**: `ln(1 + time)` handles zeros naturally
2. **Affine normalization**: `(x - offset) / scale` brings data to normalized range

**Key benefit**: Normalization happens **once per sample** (at loading), not every forward pass → faster training!

**→ See [Model Architecture](docs/architecture/MODEL_ARCHITECTURE.md) for details**

---

## 📁 Project Structure

```
GENESIS/
├── dataloader/          # Data loading & normalization
├── models/              # Model architectures
│   ├── pmt_dit.py      # DiT model (main)
│   └── architectures.py # Alternative architectures
├── diffusion/           # Diffusion process
├── training/            # Training infrastructure
├── scripts/             # Main scripts
│   ├── train.py        # Training
│   └── sample.py       # Sampling
├── utils/               # Utilities
├── configs/             # Configuration files
│   ├── default.yaml    # Default config
│   ├── testing.yaml    # Fast testing
│   ├── models/         # Model configs
│   ├── data/           # Data configs
│   └── training/       # Training configs
├── tasks/               # Experiment management
│   └── create_task.sh  # Create new experiment
└── docs/                # Documentation
    ├── setup/          # Getting started
    ├── architecture/   # System design
    ├── guides/         # How-to guides
    └── reference/      # API reference
```

---

## 🎓 Example Usage

### Training

```python
from config import load_config_from_file
from training import Trainer

# Load configuration
config = load_config_from_file("configs/default.yaml")

# Create and run trainer
trainer = Trainer(config)
trainer.train()
```

### Sampling

```python
from models.factory import ModelFactory
from utils.denormalization import denormalize_signal

# Load model
model, diffusion = ModelFactory.create_model_and_diffusion(
    model_config, diffusion_config
)

# Generate samples
samples_normalized = diffusion.sample(
    label=labels,  # (N, 6): [Energy, Zenith, Azimuth, X, Y, Z]
    geom=geom,     # (N, 3, 5160): [x, y, z]
    shape=(N, 2, 5160)  # N samples, 2 channels, 5160 PMTs
)

# Denormalize to real scale
norm_params = model.get_normalization_params()
samples_raw = denormalize_signal(
    samples_normalized,
    norm_params['affine_offsets'],
    norm_params['affine_scales'],
    norm_params['time_transform']
)
```

**→ See [Sampling Guide](docs/guides/SAMPLING.md) for more examples**

---

## ⚙️ Configuration

### Model Configuration

```yaml
model:
  seq_len: 5160        # Number of PMTs
  hidden: 512          # Hidden dimension
  depth: 8             # Transformer layers
  heads: 8             # Attention heads
  dropout: 0.1
  fusion: "SUM"        # Token fusion strategy
  
  # Normalization metadata (for denormalization)
  affine_offsets: [0.0, 0.0, 0.0, 0.0, 0.0]
  affine_scales: [100.0, 10.0, 600.0, 550.0, 550.0]
  label_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  label_scales: [50000000.0, 1.0, 1.0, 600.0, 550.0, 550.0]
  time_transform: "ln"
```

### Data Configuration

```yaml
data:
  h5_path: "path/to/data.h5"
  batch_size: 512
  num_workers: 40
  
  # Normalization (applied in Dataloader)
  time_transform: "ln"
  affine_offsets: [0.0, 0.0, 0.0, 0.0, 0.0]
  affine_scales: [100.0, 10.0, 600.0, 550.0, 550.0]
  label_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  label_scales: [50000000.0, 1.0, 1.0, 600.0, 550.0, 550.0]
```

**→ See [Configuration Reference](docs/reference/CONFIG.md) for all options**

---

## 🔬 Experiments & Tasks

Organize experiments using the task management system:

```bash
# Create a new experiment
./tasks/create_task.sh my_experiment

# This creates:
# tasks/YYYYMMDD_my_experiment/
#   ├── config.yaml      # Configuration
#   ├── run.sh           # Training script
#   ├── logs/            # Training logs
#   └── outputs/         # Checkpoints & samples

# Run the experiment
cd tasks/YYYYMMDD_my_experiment
bash run.sh
```

---

## 🎨 Visualization

GENESIS automatically generates 3D visualizations of sampled events:

```python
from utils.event_visualization.event_show import show_event_from_npz

show_event_from_npz(
    npz_path="sample_0000.npz",
    output_path="sample_0000_3d.png"
)
```

Features:
- ✅ PMT positions in 3D
- ✅ Charge shown as marker size
- ✅ Time shown as color gradient
- ✅ Compatible with `npz-show-event.py` format

---

## 🔧 GPU Optimization

GENESIS includes tools for automatic GPU optimization:

```bash
# Quick optimization (recommended)
python gpu_tools/benchmark_gpu.py --quick

# Full optimization (thorough)
python gpu_tools/benchmark_gpu.py --full

# Generates optimized config automatically
```

This automatically finds:
- ✅ Optimal batch size
- ✅ Best number of workers
- ✅ Memory-efficient settings

**→ See [GPU Optimization Guide](docs/guides/GPU_OPTIMIZATION.md)**

---

## 📊 Model Zoo

| Model | Hidden | Depth | Params | Performance |
|-------|--------|-------|--------|-------------|
| **Small** | 64 | 2 | ~0.5M | Fast, baseline |
| **Medium** | 256 | 4 | ~8M | Balanced |
| **Large** | 512 | 8 | ~50M | High quality |
| **XLarge** | 1024 | 12 | ~200M | Best quality |

Configurations available in `configs/models/`

---

## 🧪 Testing & Validation

### Test Diffusion Process

```bash
# Verify forward diffusion converges to Gaussian
python diffusion/test_diffusion_process.py \
    --config configs/default.yaml \
    --analyze-only
```

### Quick Testing

```bash
# Fast testing with 10% of data
python scripts/train.py --config configs/testing.yaml
```

---

## 📈 Performance

Typical performance on NVIDIA A100:

| Batch Size | Throughput | Memory | Training Time |
|------------|------------|--------|---------------|
| 256 | ~400 samples/s | 12 GB | ~6 hours |
| 512 | ~700 samples/s | 20 GB | ~3 hours |
| 1024 | ~1200 samples/s | 35 GB | ~1.5 hours |

*Based on default model (hidden=512, depth=8)*

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- 🔬 New architectures
- 📊 Better evaluation metrics
- 🎨 Visualization tools
- 📚 Documentation
- 🐛 Bug fixes

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **IceCube Collaboration** for neutrino data
- **DiT paper** for transformer-based diffusion
- **DDPM paper** for diffusion fundamentals

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: [docs/README.md](docs/README.md)
- **Questions**: Open an issue with the `question` tag

---

## 🎓 Citation

If you use GENESIS in your research, please cite:

```bibtex
@software{genesis2025,
  title={GENESIS: Generative IceCube Neutrino Event Synthesis},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/GENESIS}
}
```

---

**Happy modeling! 🚀**

For detailed documentation, see **[docs/README.md](docs/README.md)**
