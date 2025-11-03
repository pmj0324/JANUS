# GENESIS

**Generative Engine for Neutrino Event Simulation and Inference System**

---

## 🎯 Overview

GENESIS provides two state-of-the-art generative models for IceCube PMT signal synthesis:

1. **Diffusion Model** - DDPM-based approach with 1000 timesteps
2. **Flow Matching** - Fast ODE-based generation with 50 steps

Both models use DiT (Diffusion Transformer) architecture and support classifier-free guidance.

---

## 📁 Structure

```
GENESIS/
├── Diffusion/          # Diffusion model
│   ├── models/         # DiT architecture
│   ├── dataloader/     # Data loading & normalization
│   ├── training/       # Training infrastructure
│   ├── diffusion/      # Diffusion process
│   ├── utils/          # Utilities
│   └── config.py       # Configuration
│
├── Flow/               # Flow Matching model
│   ├── models/         # DiT architecture (shared)
│   ├── dataloader/     # Data loading (shared)
│   ├── training/       # Training infrastructure
│   ├── flow/           # Flow Matching process
│   ├── utils/          # Utilities
│   └── config.py       # Configuration
│
├── examples/           # Configuration examples
│   ├── diffusion_default.yaml
│   └── flow_default.yaml
│
├── docs/               # Documentation
│
└── train.py            # Unified training script
```

---

## 🚀 Quick Start

### Installation

```bash
# Create environment
micromamba create -n genesis python=3.10 -c conda-forge
micromamba activate genesis

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install h5py numpy matplotlib scipy tqdm tensorboard pyyaml
```

### Train Diffusion Model

```bash
python train.py \
    --config examples/diffusion_default.yaml \
    --model diffusion
```

### Train Flow Matching Model

```bash
python train.py \
    --config examples/flow_default.yaml \
    --model flow
```

---

## 🔬 Model Comparison

| Feature | Diffusion | Flow Matching |
|---------|-----------|---------------|
| **Training** | Noise prediction | Velocity field |
| **Sampling Steps** | 1000 | 50 |
| **Sampling Speed** | Slower | **20x Faster** |
| **Quality** | High | High |
| **Stability** | Very stable | Stable |
| **Best for** | High quality | Fast generation |

---

## 📊 Key Features

### Diffusion Model
- DDPM with 1000 timesteps
- Multiple noise schedules (linear, cosine)
- DDPM/DDIM sampling
- Classifier-free guidance

### Flow Matching
- Conditional Flow Matching (Lipman et al. 2023)
- ODE-based sampling (Euler, Midpoint)
- 50 steps only (vs 1000 for diffusion)
- Faster training convergence

### Shared Features
- DiT architecture with SUM/FiLM fusion
- Normalization in dataloader (not model)
- Mixed precision training (AMP)
- Multiple schedulers (Plateau, Cosine, etc.)
- Early stopping
- TensorBoard logging

---

## 🎓 Usage Examples

### Basic Training

```bash
# Diffusion with default settings
python train.py --config examples/diffusion_default.yaml --model diffusion

# Flow Matching with custom batch size
python train.py --config examples/flow_default.yaml --model flow --batch-size 512
```

### Advanced Options

```bash
python train.py \
    --config examples/flow_default.yaml \
    --model flow \
    --batch-size 256 \
    --lr 0.0002 \
    --epochs 50 \
    --num-workers 20 \
    --device cuda
```

### Resume Training

```bash
python train.py \
    --config examples/diffusion_default.yaml \
    --model diffusion \
    --resume checkpoints/diffusion/best_model.pt
```

---

## ⚙️ Configuration

### Model Config

```yaml
model:
  architecture: "dit"
  hidden: 512
  depth: 8
  heads: 8
  fusion: "SUM"  # or "FiLM"
```

### Diffusion Config

```yaml
diffusion:
  timesteps: 1000
  schedule: "linear"  # or "cosine"
  use_cfg: true
  cfg_scale: 2.0
```

### Flow Config

```yaml
flow:
  num_steps: 50
  use_ode_solver: "euler"  # or "midpoint"
  use_cfg: true
  cfg_scale: 2.0
```

---

## 📈 Performance

On NVIDIA A100 (batch_size=256):

| Model | Training Speed | Sampling Speed | Memory |
|-------|---------------|----------------|--------|
| Diffusion | ~700 samples/s | ~50 samples/s | 20 GB |
| Flow | ~700 samples/s | **~1000 samples/s** | 20 GB |

---

## 🔍 Normalization

**Important**: Normalization happens in the **dataloader**, not the model!

Pipeline:
1. Load HDF5 data
2. Time transformation: `ln(1+time)`
3. Affine normalization: `(x - offset) / scale`
4. Model processes normalized data
5. Denormalize for visualization

---

## 📚 Documentation

- **Full Documentation**: `docs/README.md`
- **Diffusion Details**: `docs/architecture/DIFFUSION_MODULE.md`
- **Flow Matching Details**: `docs/architecture/FLOW_MATCHING.md` (if exists)
- **Training Guide**: `docs/guides/TRAINING.md`

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- New architectures
- Better evaluation metrics
- Visualization tools
- Documentation

---

## 📄 License

MIT License

---

## 📞 Contact

- **Issues**: GitHub Issues
- **Email**: pmj032400@naver.com

---

## 🙏 Acknowledgments

- IceCube Collaboration
- Diffusion Models: Ho et al. (DDPM)
- Flow Matching: Lipman et al. (2023)
- DiT: Peebles & Xie (2023)

---

**Happy Modeling! 🚀**

