# GENESIS Project Structure

## 📁 Directory Layout

```
GENESIS/
├── Diffusion/              # Diffusion Model Implementation
│   ├── __init__.py
│   ├── config.py          # Diffusion configuration
│   ├── models/            # DiT architecture
│   │   ├── pmt_dit.py
│   │   ├── architectures.py
│   │   └── factory.py
│   ├── dataloader/        # Data loading & normalization
│   │   └── pmt_dataloader.py
│   ├── training/          # Training infrastructure
│   │   ├── trainer.py
│   │   ├── schedulers.py
│   │   ├── checkpointing.py
│   │   └── utils.py
│   ├── diffusion/         # Diffusion process
│   │   ├── gaussian_diffusion.py
│   │   └── noise_schedules.py
│   └── utils/             # Utilities
│       ├── denormalization.py
│       └── gpu_utils.py
│
├── Flow/                   # Flow Matching Implementation
│   ├── __init__.py
│   ├── config.py          # Flow configuration
│   ├── models/            # DiT architecture (shared with Diffusion)
│   │   ├── pmt_dit.py
│   │   ├── architectures.py
│   │   └── factory.py
│   ├── dataloader/        # Data loading (shared)
│   │   └── pmt_dataloader.py
│   ├── training/          # Training infrastructure
│   │   ├── trainer.py
│   │   ├── schedulers.py
│   │   ├── checkpointing.py
│   │   └── utils.py
│   ├── flow/              # Flow Matching process
│   │   └── conditional_flow_matching.py
│   └── utils/             # Utilities
│       ├── denormalization.py
│       └── gpu_utils.py
│
├── examples/               # Configuration Examples
│   ├── diffusion_default.yaml
│   └── flow_default.yaml
│
├── docs/                   # Documentation
│   ├── README.md
│   ├── architecture/      # Architecture docs
│   ├── guides/            # How-to guides
│   ├── reference/         # API reference
│   └── setup/             # Setup guides
│
├── GENESIS-data/           # Data directory
│   └── 22644_0921_time_shift.h5
│
├── train.py                # Unified training script
├── README.md               # Main README
├── QUICK_START.md          # Quick start guide
└── STRUCTURE.md            # This file
```

---

## 🎯 Design Philosophy

### 1. Separation of Concerns

- **Diffusion/** contains all DDPM-related code
- **Flow/** contains all Flow Matching-related code
- **Shared components** (models, dataloader) are copied for independence

### 2. Unified Interface

- Single `train.py` for both models
- Consistent configuration structure
- Same CLI arguments

### 3. Minimal Root

Root directory contains only:
- `train.py` - main script
- `examples/` - configuration files
- `docs/` - documentation
- Model directories (Diffusion, Flow)

---

## 🔄 Data Flow

### Diffusion Model

```
HDF5 Data → Dataloader (normalize) → Model (DiT) → Diffusion → Loss
                                                   ↓
                                            Noise Prediction
```

### Flow Matching Model

```
HDF5 Data → Dataloader (normalize) → Model (DiT) → Flow → Loss
                                                   ↓
                                            Velocity Field
```

---

## 📝 Configuration Files

### Diffusion Config (`examples/diffusion_default.yaml`)

```yaml
model:
  architecture: "dit"
  hidden: 512
  depth: 8

diffusion:
  timesteps: 1000
  schedule: "linear"

training:
  num_epochs: 100
  learning_rate: 0.0001
```

### Flow Config (`examples/flow_default.yaml`)

```yaml
model:
  architecture: "dit"
  hidden: 512
  depth: 8

flow:
  num_steps: 50
  use_ode_solver: "euler"

training:
  num_epochs: 100
  learning_rate: 0.0002
```

---

## 🚀 Usage

### Train Diffusion

```bash
python train.py --config examples/diffusion_default.yaml --model diffusion
```

### Train Flow

```bash
python train.py --config examples/flow_default.yaml --model flow
```

---

## 📊 Outputs

### Checkpoints

```
checkpoints/
├── diffusion/
│   ├── best_model.pt
│   └── latest_model.pt
└── flow/
    ├── best_model.pt
    └── latest_model.pt
```

### Logs

```
logs/
├── diffusion/
│   ├── events.out.tfevents.*  # TensorBoard
│   └── icecube_diffusion_default_training.txt
└── flow/
    ├── events.out.tfevents.*
    └── icecube_flow_default_training.txt
```

---

## 🔧 Key Components

### Models (`models/`)

- **pmt_dit.py**: DiT architecture
- **architectures.py**: Alternative architectures
- **factory.py**: Model creation factory

### Dataloader (`dataloader/`)

- **pmt_dataloader.py**: HDF5 loading + normalization
  - Time transformation: `ln(1+x)`
  - Affine normalization: `(x - offset) / scale`

### Training (`training/`)

- **trainer.py**: Main training loop
- **schedulers.py**: LR schedulers (Plateau, Cosine, etc.)
- **checkpointing.py**: Checkpoint management
- **utils.py**: Training utilities (early stopping, etc.)

### Diffusion (`Diffusion/diffusion/`)

- **gaussian_diffusion.py**: DDPM implementation
- **noise_schedules.py**: Noise schedule functions

### Flow (`Flow/flow/`)

- **conditional_flow_matching.py**: Flow Matching implementation
  - Optimal transport path
  - ODE solvers (Euler, Midpoint)

---

## 🎓 Adding New Models

To add a new generative model:

1. Create new directory: `GENESIS/NewModel/`
2. Copy structure from `Diffusion/` or `Flow/`
3. Implement your generative process in `NewModel/new_process/`
4. Update `train.py` to support `--model newmodel`
5. Add config: `examples/newmodel_default.yaml`

---

## 📚 Documentation

- **Main README**: `README.md`
- **Quick Start**: `QUICK_START.md`
- **Full Docs**: `docs/README.md`
- **Architecture**: `docs/architecture/`
- **Training Guides**: `docs/guides/`

---

## ✅ Advantages

1. **Clear Separation**: Diffusion and Flow are completely independent
2. **Easy Comparison**: Same interface, different methods
3. **Scalable**: Easy to add new generative models
4. **Clean Root**: Minimal files in root directory
5. **Unified Training**: Single script for all models

---

## 🔄 Migration from Old Structure

Old structure:
```
GENESIS/
├── models/
├── diffusion/
├── training/
└── scripts/train.py
```

New structure:
```
GENESIS/
├── Diffusion/
│   ├── models/
│   ├── diffusion/
│   └── training/
├── Flow/
│   ├── models/
│   ├── flow/
│   └── training/
└── train.py
```

Benefits:
- Clearer organization
- Multiple generative models supported
- Easier to maintain and extend

