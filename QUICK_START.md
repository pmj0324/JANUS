# GENESIS Quick Start Guide

## 📦 Installation

```bash
# Create environment
micromamba create -n genesis python=3.10 -c conda-forge
micromamba activate genesis

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install h5py numpy matplotlib scipy tqdm tensorboard pyyaml
```

## 🚀 Training

### Diffusion Model

```bash
# Train with default configuration
python train.py --config examples/diffusion_default.yaml --model diffusion

# With custom settings
python train.py \
    --config examples/diffusion_default.yaml \
    --model diffusion \
    --batch-size 256 \
    --lr 0.0001 \
    --epochs 100
```

### Flow Matching Model

```bash
# Train with default configuration
python train.py --config examples/flow_default.yaml --model flow

# With custom settings
python train.py \
    --config examples/flow_default.yaml \
    --model flow \
    --batch-size 256 \
    --lr 0.0002 \
    --epochs 100
```

## 📊 Monitoring

### TensorBoard

```bash
# For Diffusion
tensorboard --logdir logs/diffusion

# For Flow
tensorboard --logdir logs/flow
```

### Check Training Logs

```bash
# Diffusion logs
tail -f logs/diffusion/icecube_diffusion_default_training.txt

# Flow logs
tail -f logs/flow/icecube_flow_default_training.txt
```

## 💾 Checkpoints

Models are saved to:
- Diffusion: `checkpoints/diffusion/`
- Flow: `checkpoints/flow/`

Best model: `best_model.pt`
Latest model: `latest_model.pt`

## 🔧 Configuration

### Model Architecture

Edit `model` section in YAML:
```yaml
model:
  architecture: "dit"
  hidden: 512
  depth: 8
  heads: 8
  fusion: "SUM"  # or "FiLM"
```

### Training Settings

Edit `training` section:
```yaml
training:
  num_epochs: 100
  learning_rate: 0.0001
  batch_size: 256
  optimizer: "AdamW"
  scheduler: "plateau"
```

### Data Path

Edit `data` section:
```yaml
data:
  h5_path: "${GENESIS_ROOT}GENESIS-data/22644_0921_time_shift.h5"
  batch_size: 256
  num_workers: 16
```

## 🎯 Tips

### For Fast Testing

Use smaller model:
```yaml
model:
  hidden: 128
  depth: 4
  heads: 4
```

```bash
python train.py --config examples/flow_default.yaml --model flow --epochs 5
```

### For Best Quality

Use larger model and more epochs:
```yaml
model:
  hidden: 1024
  depth: 12
  heads: 12
```

### GPU Memory Issues

Reduce batch size:
```bash
python train.py --config examples/flow_default.yaml --model flow --batch-size 128
```

## 📈 Performance Comparison

| Model | Sampling Steps | Speed | Quality |
|-------|----------------|-------|---------|
| Diffusion | 1000 | Slow | High |
| Flow | 50 | **20x Faster** | High |

## 🆘 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size`
- Reduce `hidden` size
- Set `use_amp: false`

### NaN Loss
- Check normalization parameters
- Reduce learning rate
- Check data for inf/nan values

### Slow Training
- Increase `num_workers`
- Use `use_amp: true`
- Check GPU utilization

## 📚 More Information

- Full Documentation: `docs/README.md`
- Training Guide: `docs/guides/TRAINING.md`
- Architecture Details: `docs/architecture/MODEL_ARCHITECTURE.md`

