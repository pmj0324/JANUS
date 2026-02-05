#!/usr/bin/env python3
"""
Training script for GENESIS diffusion model with CFG, validation, and early stopping.
Supports YAML configuration files.
"""

import math
import os
import sys
import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
# Add GENESIS to path
sys.path.insert(0, os.path.join(os.getcwd(), "GENESIS"))
from dataloader.h5 import H5Dataset
from diffusion.schedules import sigmoid_beta_schedule, compute_alpha_schedule
from diffusion.forward import apply_forward_diffusion
from utils.normalize import normalize, denormalize_log_minmax, denormalize_minmax, apply_minmax_geo
from utils.vis.event_show import show_event_dual_plot
from utils.device import get_default_device

# ============================================================================
# Configuration Parameters (can be overridden by YAML)
# ============================================================================

# ============================================================================
# Save Paths (모델과 플롯 저장 경로 지정) - 맨 앞에 위치
# ============================================================================
# 기본 출력 디렉토리 지정 (이 경로 아래에 models/와 plots/ 폴더가 자동 생성됨)
output_dir = Path("./output")
# 플롯 저장 활성화 여부 (False면 plots 폴더 생성 안 함)
save_plots = True

# Diffusion schedule
T = 1000
beta_start, beta_end = 1e-4, 2e-2  # sigmoid schedule params

# Training
batch_size = 256
num_workers = 32
lr = 3e-4
num_epochs = 20
val_ratio = 0.1  # 10% for validation
val_every = 1  # validate every N epochs (항상 1로 설정 - ReduceLROnPlateau와 호환)

# Learning rate scheduler (ReduceLROnPlateau)
# 주의: step()은 매 epoch마다 호출하며, validation한 epoch에는 val_loss, 아니면 train_loss 사용.
# val_every=1 권장. val_every>1이면 patience는 "validation 횟수" 기준이 됨 (patience*val_every epoch).
lr_scheduler_patience = 5  # 개선 없을 때 step() 호출 횟수만큼 기다린 뒤 LR 감소
lr_scheduler_factor = 0.5  # factor to reduce LR by
lr_scheduler_min = 1e-6  # minimum learning rate

# Early stopping
early_stopping_patience = 10  # epochs to wait before stopping
early_stopping_min_delta = 1e-6  # minimum change to qualify as improvement

# Classifier-Free Guidance (CFG)
use_cfg = True  # Enable CFG training (기본값: True)
cfg_dropout = 0.1  # Probability to drop label (use null label) during training
cfg_scale = 2.0  # CFG scale for inference (not used during training)

# Data normalization
npe_clip = 1000.0
ftime_clip = 21000.0
log_min = 0.0
ftime_log_max = float(np.log1p(ftime_clip))
_feature_range = (-1, 1)

# Label normalization: [Energy, ux, uy, X, Y, Z]
_label_methods = ["log_minmax", "identity", "identity", "minmax", "minmax", "minmax"]
_label_feature_ranges = [_feature_range] * 6
_ENERGY_PEV_MINMAX = {"min": 1.0, "max": 100.0}
energy_log_min = float(np.log1p(_ENERGY_PEV_MINMAX["min"]))
energy_log_max = float(np.log1p(_ENERGY_PEV_MINMAX["max"]))
_LABEL_XYZ_MINMAX = [
    {"min": -570.9000244140625, "max": 576.3699951171875},   # X
    {"min": -521.0800170898438, "max": 509.5},               # Y
    {"min": -509.8599853515625, "max": 506.0566711425781},   # Z
]
_GEO_XYZ_MINMAX = [
    {"min": -570.9000244140625, "max": 576.3699951171875},   # x
    {"min": -521.0800170898438, "max": 509.5},               # y
    {"min": -509.8599853515625, "max": 506.0566711425781},   # z
]
geo_min = np.array([_GEO_XYZ_MINMAX[j]["min"] for j in range(3)], dtype=np.float32)
geo_max = np.array([_GEO_XYZ_MINMAX[j]["max"] for j in range(3)], dtype=np.float32)
_label_stats = [
    {"log_min": energy_log_min, "log_max": energy_log_max},
    {},
    {},
    _LABEL_XYZ_MINMAX[0],
    _LABEL_XYZ_MINMAX[1],
    _LABEL_XYZ_MINMAX[2],
]

LABEL_NAMES = ["Energy (PeV)", "ux", "uy", "X", "Y", "Z"]

# Model parameters
model_d_model = 256
model_nhead = 8
model_depth = 6
model_mlp_ratio = 4.0
model_dropout = 0.0
model_label_dim = 6

# Data paths
h5_path = "./GENESIS-data/22644_0921_time_shift.h5"

# Other
seed = 42
compile_model = False
print_every = 50

# ============================================================================
# Helper Functions
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def apply_config(config: dict):
    """Apply configuration to global variables."""
    global T, beta_start, beta_end
    global batch_size, num_workers, lr, num_epochs, val_ratio, val_every
    global lr_scheduler_patience, lr_scheduler_factor, lr_scheduler_min
    global early_stopping_patience, early_stopping_min_delta
    global use_cfg, cfg_dropout, cfg_scale
    global model_d_model, model_nhead, model_depth, model_mlp_ratio, model_dropout, model_label_dim
    global h5_path, output_dir, save_plots, seed, compile_model, print_every
    
    # Diffusion
    if 'diffusion' in config:
        diff = config['diffusion']
        if 'schedule' in diff:
            sched = diff['schedule']
            T = sched.get('timesteps', T)
            beta_start = sched.get('beta_start', beta_start)
            beta_end = sched.get('beta_end', beta_end)
    
    # Training
    if 'training' in config:
        train = config['training']
        num_epochs = train.get('nepochs', num_epochs)
        lr = train.get('lr', lr)
        batch_size = train.get('bsz', batch_size)
    
    # Validation
    if 'val_ratio' in config:
        val_ratio = config['val_ratio']
    if 'val_every' in config:
        val_every_config = config['val_every']
        if val_every_config != 1:
            print(f"Warning: val_every={val_every_config} is set in config, but forcing to 1 for ReduceLROnPlateau compatibility")
        val_every = 1  # 항상 1로 강제 설정
    
    # LR Scheduler
    if 'lr_scheduler' in config:
        lr_sched = config['lr_scheduler']
        lr_scheduler_patience = lr_sched.get('patience', lr_scheduler_patience)
        lr_scheduler_factor = lr_sched.get('factor', lr_scheduler_factor)
        lr_scheduler_min = lr_sched.get('min_lr', lr_scheduler_min)
    
    # Early stopping
    if 'early_stopping' in config:
        es = config['early_stopping']
        early_stopping_patience = es.get('patience', early_stopping_patience)
        early_stopping_min_delta = es.get('min_delta', early_stopping_min_delta)
    
    # CFG
    if 'cfg' in config:
        cfg = config['cfg']
        use_cfg = cfg.get('use_cfg', use_cfg)
        cfg_dropout = cfg.get('cfg_dropout', cfg_dropout)
        cfg_scale = cfg.get('cfg_scale', cfg_scale)
    
    # Model
    if 'model' in config:
        model = config['model']
        if 'options' in model:
            opts = model['options']
            model_d_model = opts.get('d_model', model_d_model)
            model_nhead = opts.get('nhead', model_nhead)
            model_depth = opts.get('depth', model_depth)
            model_mlp_ratio = opts.get('mlp_ratio', model_mlp_ratio)
            model_dropout = opts.get('dropout', model_dropout)
            model_label_dim = opts.get('label_dim', model_label_dim)
    
    # Data
    if 'data' in config:
        data = config['data']
        h5_path = data.get('h5_path', h5_path)
        num_workers = data.get('num_workers', num_workers)
    
    # Paths - 저장 경로 설정
    if 'path' in config:
        output_dir = Path(config['path'])
    
    if 'save_plots' in config:
        save_plots = config['save_plots']
    if 'seed' in config:
        seed = config['seed']
    if 'compile_model' in config:
        compile_model = config['compile_model']
    if 'print_every' in config:
        print_every = config['print_every']


def _print_label_normalize_config():
    """Label별 정규화 설정 출력."""
    print("label normalize (per column):")
    for j, name in enumerate(LABEL_NAMES):
        m = _label_methods[j]
        fr = _label_feature_ranges[j]
        st = _label_stats[j] if _label_stats and j < len(_label_stats) else {}
        if m == "identity":
            detail = "identity (no transform)"
        elif m == "log_minmax":
            detail = f"log_minmax -> {fr}  stats={st}"
        elif m == "minmax":
            if st and "min" in st and "max" in st:
                detail = f"minmax -> {fr}  stats={st} (dataset min/max)"
            else:
                detail = f"minmax -> {fr}  stats={st} (empty => batch min/max)"
        else:
            detail = f"{m} -> {fr}  stats={st}"
        print(f"  [{j}] {name}: {detail}")


def _clamp_sig(sig: torch.Tensor) -> torch.Tensor:
    """Clamp npe/ftime before normalize. Returns clamped copy."""
    s = sig.clone()
    s[:, 0] = torch.clamp(s[:, 0], min=0.0, max=npe_clip)
    s[:, 1] = torch.clamp(s[:, 1], min=0.0, max=ftime_clip)
    return s


def sample_timesteps(batch: int, T: int, device: torch.device) -> torch.Tensor:
    """t in [1, T] (t=0은 노이즈가 없는 원본이므로 학습에서 제외)"""
    return torch.randint(low=1, high=T + 1, size=(batch,), device=device, dtype=torch.long)


def get_null_label(batch_size: int, label_dim: int, device: torch.device) -> torch.Tensor:
    """Create null label for CFG (unconditional generation)."""
    return torch.zeros(batch_size, label_dim, device=device)


# ============================================================================
# Model Definition
# ============================================================================

def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    t: (B,) int/float
    return: (B, dim)
    """
    if t.dim() != 1:
        t = t.view(-1)
    t = t.float()

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )  # (half,)
    args = t[:, None] * freqs[None, :]  # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device, dtype=emb.dtype)], dim=-1)
    return emb


class DiTBlock(nn.Module):
    """
    DiT-style Transformer block with AdaLN modulation from conditioning vector c.
    x: (B, L, d)
    c: (B, d)
    """
    def __init__(self, d: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

        # cond -> (shift1, scale1, gate1, shift2, scale2, gate2)
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, 6 * d),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        params = self.ada(c).view(B, 6, d)
        shift1, scale1, gate1, shift2, scale2, gate2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]

        # Attention
        x1 = self.norm1(x)
        x1 = x1 * (1.0 + scale1[:, None, :]) + shift1[:, None, :]
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + gate1[:, None, :] * attn_out

        # MLP
        x2 = self.norm2(x)
        x2 = x2 * (1.0 + scale2[:, None, :]) + shift2[:, None, :]
        mlp_out = self.mlp(x2)
        x = x + gate2[:, None, :] * mlp_out

        return x


class DiffusionDiTTransformer(nn.Module):
    def __init__(
        self,
        geo: torch.Tensor,          # (3, 5160) or (1, 3, 5160)
        d_model: int = 256,
        nhead: int = 8,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        label_dim: int = 6,
    ):
        super().__init__()
        self.d_model = d_model

        # ---- geo buffer (고정) ----
        # geo: (3, L) -> (1, L, 3)
        if geo.dim() == 2:
            geo_tok = geo.transpose(0, 1).unsqueeze(0)  # (1, L, 3)
        elif geo.dim() == 3:
            # (1, 3, L) -> (1, L, 3)
            geo_tok = geo.permute(0, 2, 1)
        else:
            raise ValueError(f"geo must be (3,L) or (1,3,L). got {geo.shape}")

        self.register_buffer("geo_tokens", geo_tok.contiguous(), persistent=True)
        L = self.geo_tokens.shape[1]
        self.L = L

        # ---- input embedding: (B, L, 2) -> (B, L, d) ----
        self.in_proj = nn.Linear(2, d_model)

        # ---- geo positional embedding: (1, L, 3) -> (1, L, d) ----
        self.geo_mlp = nn.Sequential(
            nn.Linear(3, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.use_index_pos = False
        if self.use_index_pos:
            self.index_pos = nn.Parameter(torch.zeros(1, L, d_model))

        # ---- conditioning: time + label -> (B, d) ----
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.label_mlp = nn.Sequential(
            nn.Linear(label_dim, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        # ---- DiT blocks ----
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        # ---- final layer (DiT 스타일로 마지막에도 cond로 LN modulation) ----
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),  # shift, scale
        )
        self.out_proj = nn.Linear(d_model, 2)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, 2, L)
        t: (B,)
        label: (B, 6)
        return: eps_hat (B, 2, L)
        """
        B, C, L = x_t.shape
        assert L == self.L, f"Expected L={self.L}, got L={L}"

        # (B, 2, L) -> (B, L, 2)
        tokens = x_t.permute(0, 2, 1)
        h = self.in_proj(tokens)  # (B, L, d)

        # geo positional embedding
        pos_geo = self.geo_mlp(self.geo_tokens)  # (1, L, d)
        h = h + pos_geo

        if self.use_index_pos:
            h = h + self.index_pos

        # conditioning
        t_emb = sinusoidal_timestep_embedding(t, self.d_model)  # (B, d)
        c = self.time_mlp(t_emb) + self.label_mlp(label)        # (B, d)

        # DiT blocks
        for blk in self.blocks:
            h = blk(h, c)

        # final AdaLN + output
        shift, scale = self.final_ada(c).chunk(2, dim=-1)       # (B,d), (B,d)
        h = self.final_norm(h)
        h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]
        out = self.out_proj(h)                                   # (B, L, 2)
        return out.permute(0, 2, 1)                             # (B, 2, L)


# ============================================================================
# Normalization Functions
# ============================================================================

_channel_stats = [
    {"min": 0.0, "max": npe_clip},   # nPE: log 없이 minmax
    {"log_min": log_min, "log_max": ftime_log_max},
]

@normalize(
    channel_methods=["minmax", "log_minmax"],
    feature_ranges=[_feature_range, _feature_range],
    channel_stats=_channel_stats,
    arg_index=0,
    label_arg_index=1,
    label_methods=_label_methods,
    label_feature_ranges=_label_feature_ranges,
    label_stats=_label_stats,
)
def prepare_batch(
    sig: torch.Tensor, label: torch.Tensor, *, verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """입력: (sig, label). 데코레이터가 sig는 [-1,1] log_minmax, label은 Energy log_minmax / ux,uy identity / X,Y,Z minmax.
    출력: (sig_norm, label_norm). verbose=True면 print."""
    if verbose:
        print("prepare_batch: label", label)
    return (sig, label)


def denormalize_sig(sig: torch.Tensor) -> torch.Tensor:
    """정규화된 sig를 원 스케일로 역정규화. nPE는 minmax 역변환, FirstTime은 log_minmax 역변환."""
    out = sig.clone()
    if sig.dim() == 3:
        out[:, 0, :] = denormalize_minmax(sig[:, 0, :], 0.0, npe_clip, _feature_range)
        out[:, 1, :] = denormalize_log_minmax(sig[:, 1, :], log_min, ftime_log_max, _feature_range)
    else:
        out[0, :] = denormalize_minmax(sig[0, :], 0.0, npe_clip, _feature_range)
        out[1, :] = denormalize_log_minmax(sig[1, :], log_min, ftime_log_max, _feature_range)
    return out


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train GENESIS diffusion model with CFG and validation")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        apply_config(config)
        print(f"Loaded configuration from: {args.config}")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Device setup
    device = get_default_device()
    print("device:", device)
    
    # Create output directories (기본 경로 아래에 models/와 plots/ 자동 생성)
    output_dir.mkdir(exist_ok=True, parents=True)
    model_save_dir = output_dir / "models"
    model_save_dir.mkdir(exist_ok=True, parents=True)
    
    if save_plots:
        plot_save_dir = output_dir / "plots"
        plot_save_dir.mkdir(exist_ok=True, parents=True)
    else:
        plot_save_dir = None
    
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Model save directory: {model_save_dir.absolute()}")
    if save_plots:
        print(f"Plot save directory: {plot_save_dir.absolute()}")
    else:
        print("Plot saving: disabled")
    
    # Load dataset
    print(f"Loading dataset from: {h5_path}")
    dataset = H5Dataset(h5_path=h5_path)
    print(f"Dataset length: {len(dataset)}")
    
    # Train/Val split
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # DataLoaders
    _print_label_normalize_config()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    # Get sample data for shape info
    sig0, geo0, label0 = dataset[0]
    print("sig:", sig0.shape, sig0.dtype)
    print("geo:", geo0.shape, geo0.dtype)
    print("label:", label0.shape, label0.dtype)
    
    # Setup geo (all samples use same geo)
    _geo_raw = dataset[0][1]  # (3, L)
    _geo = apply_minmax_geo(_geo_raw, geo_min, geo_max, feature_range=(0, 1))
    
    # Create model
    model = DiffusionDiTTransformer(
        geo=_geo,
        d_model=model_d_model,
        nhead=model_nhead,
        depth=model_depth,
        mlp_ratio=model_mlp_ratio,
        dropout=model_dropout,
        label_dim=model_label_dim
    ).to(device)
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"[GPU] after model: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    
    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Learning rate scheduler (ReduceLROnPlateau)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode='min',
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=lr_scheduler_min,
        verbose=True
    )
    
    # AMP
    try:
        scaler = GradScaler(device.type) if device.type in ("cuda", "mps") else None
    except (TypeError, ValueError):
        scaler = GradScaler() if device.type == "cuda" else None
    print("AMP enabled:", scaler is not None)
    
    # Compile model
    if compile_model:
        try:
            if hasattr(torch, 'compile'):
                print("Compiling model with torch.compile()...")
                model = torch.compile(model, mode="reduce-overhead")
                print("Model compilation successful!")
            else:
                print("torch.compile() not available (requires PyTorch 2.0+)")
        except Exception as e:
            print(f"Model compilation failed (continuing without compile): {e}")
    
    print("params:", sum(p.numel() for p in model.parameters())/1e6, "M")
    print(f"CFG enabled: {use_cfg} (dropout={cfg_dropout}, scale={cfg_scale})")
    
    # Noise schedule
    betas = sigmoid_beta_schedule(timesteps=T, beta_start=beta_start, beta_end=beta_end).to(device)
    alpha_schedule = compute_alpha_schedule(betas)
    alphas = alpha_schedule["alphas"]
    alphas_cumprod = alpha_schedule["alphas_cumprod"]
    print("betas:", betas.shape, betas.min().item(), betas.max().item())
    
    # Training loop
    model.train()
    
    train_loss_hist = []
    val_loss_hist = []
    best_val_loss = float('inf')
    best_checkpoint_path = None
    epochs_without_improvement = 0
    
    steps_per_epoch = len(train_loader)
    val_steps_per_epoch = len(val_loader)
    total_steps = num_epochs * steps_per_epoch
    
    # Print training configuration summary
    print("\n" + "="*60)
    print("Training Configuration Summary")
    print("="*60)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Initial LR: {lr:.2e}")
    print(f"LR Scheduler: ReduceLROnPlateau (patience={lr_scheduler_patience}, factor={lr_scheduler_factor})")
    print(f"Early Stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    print(f"Train batches per epoch: {steps_per_epoch}")
    print(f"Val batches per epoch: {val_steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    print("="*60)
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*60 + "\n")
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        epoch_train_losses = []
        
        pbar = tqdm(enumerate(train_loader, 1), total=steps_per_epoch, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        
        for batch_idx, (sig, geo, label) in pbar:
            sig = sig.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            # CFG: randomly replace label with null label
            if use_cfg:
                mask = torch.rand(label.shape[0], device=device) < cfg_dropout
                label_cfg = label.clone()
                label_cfg[mask] = get_null_label(mask.sum().item(), model_label_dim, device)
            else:
                label_cfg = label
            
            sig_clamp = _clamp_sig(sig)
            x0, label_norm = prepare_batch(sig_clamp, label_cfg)
            
            B = x0.shape[0]
            t = sample_timesteps(B, T, device)
            
            noise = torch.randn_like(x0)
            x_t = apply_forward_diffusion(x0=x0, betas=betas, timesteps=t, noise=noise)
            
            # Forward pass
            with autocast(device.type, enabled=(scaler is not None)):
                eps_hat = model(x_t, t, label_norm)
                loss = F.mse_loss(eps_hat, noise)
            
            # Backward pass
            optim.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            
            loss_val = float(loss.item())
            epoch_train_losses.append(loss_val)
            train_loss_hist.append(loss_val)
            
            current_step = (epoch - 1) * steps_per_epoch + batch_idx
            avg_loss_so_far = np.mean(epoch_train_losses)
            pbar.set_postfix({"loss": f"{avg_loss_so_far:.6f}", "lr": f"{optim.param_groups[0]['lr']:.2e}"})
        
        epoch_train_loss = np.mean(epoch_train_losses)
        
        # Validation phase
        if epoch % val_every == 0:
            model.eval()
            epoch_val_losses = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]")
                for sig, geo, label in val_pbar:
                    sig = sig.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)
                    
                    sig_clamp = _clamp_sig(sig)
                    x0, label_norm = prepare_batch(sig_clamp, label)
                    
                    B = x0.shape[0]
                    t = sample_timesteps(B, T, device)
                    
                    noise = torch.randn_like(x0)
                    x_t = apply_forward_diffusion(x0=x0, betas=betas, timesteps=t, noise=noise)
                    
                    with autocast(device.type, enabled=(scaler is not None)):
                        eps_hat = model(x_t, t, label_norm)
                        loss = F.mse_loss(eps_hat, noise)
                    
                    loss_val = float(loss.item())
                    epoch_val_losses.append(loss_val)
                    # Update progress bar with current batch loss
                    val_pbar.set_postfix({"loss": f"{loss_val:.6f}"})
            
            epoch_val_loss = np.mean(epoch_val_losses)
            val_loss_hist.append(epoch_val_loss)
            
            # Update LR scheduler (validation epoch: use val_loss)
            old_lr = optim.param_groups[0]['lr']
            lr_scheduler.step(epoch_val_loss)
            new_lr = optim.param_groups[0]['lr']
            lr_reduced = (old_lr != new_lr)
            
            # Check for improvement
            improvement = best_val_loss - epoch_val_loss
            if improvement > early_stopping_min_delta:
                best_val_loss = epoch_val_loss
                epochs_without_improvement = 0
                
                # Save best model
                if best_checkpoint_path is not None and best_checkpoint_path.exists():
                    best_checkpoint_path.unlink()
                
                best_checkpoint_path = model_save_dir / f"best_checkpoint_epoch_{epoch:03d}_val_loss_{best_val_loss:.6f}.pt"
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "best_val_loss": best_val_loss,
                    "betas": betas.cpu(),
                    "alphas": alphas.cpu(),
                    "alphas_cumprod": alphas_cumprod.cpu(),
                    "T": T,
                    "beta_start": beta_start,
                    "beta_end": beta_end,
                    "d_model": model_d_model,
                    "nhead": model_nhead,
                    "depth": model_depth,
                    "mlp_ratio": model_mlp_ratio,
                    "dropout": model_dropout,
                    "label_dim": model_label_dim,
                    "use_cfg": use_cfg,
                    "cfg_dropout": cfg_dropout,
                    "cfg_scale": cfg_scale,
                }
                torch.save(checkpoint, best_checkpoint_path)
                print(f"\n✓ New best model saved: {best_checkpoint_path.name}")
            else:
                epochs_without_improvement += 1
            
            print(f"\nEpoch {epoch:3d}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | Best Val: {best_val_loss:.6f}")
            lr_msg = f"LR: {optim.param_groups[0]['lr']:.2e}"
            if lr_reduced:
                lr_msg += f" (reduced from {old_lr:.2e})"
            print(f"{lr_msg} | Epochs without improvement: {epochs_without_improvement}/{early_stopping_patience}")
            print("-"*60)
            
            # Early stopping check
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {early_stopping_patience} epochs)")
                break
        else:
            # Validation 없음: 스케줄러는 호출하지 않음 (val_loss만 사용하므로).
            # val_every=1 권장. val_every>1이면 patience는 "validation 횟수" 기준 (실제 대기 = patience * val_every epoch).
            print(f"\nEpoch {epoch:3d}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | LR: {optim.param_groups[0]['lr']:.2e}")
            print("-"*60)
    
    print("\nTraining done!")
    
    # Save final model
    final_checkpoint_path = model_save_dir / "model_checkpoint_final.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "epoch": epoch,
        "final_train_loss": train_loss_hist[-1] if train_loss_hist else None,
        "final_val_loss": val_loss_hist[-1] if val_loss_hist else None,
        "best_val_loss": best_val_loss,
        "betas": betas.cpu(),
        "alphas": alphas.cpu(),
        "alphas_cumprod": alphas_cumprod.cpu(),
        "T": T,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "d_model": model_d_model,
        "nhead": model_nhead,
        "depth": model_depth,
        "mlp_ratio": model_mlp_ratio,
        "dropout": model_dropout,
        "label_dim": model_label_dim,
        "use_cfg": use_cfg,
        "cfg_dropout": cfg_dropout,
        "cfg_scale": cfg_scale,
    }
    torch.save(checkpoint, final_checkpoint_path)
    print(f"Final model saved to: {final_checkpoint_path}")
    if best_checkpoint_path:
        print(f"Best model (val_loss={best_val_loss:.6f}) saved to: {best_checkpoint_path.name}")
    
    # Sampling with corrected indexing
    print("\n" + "="*60)
    print("Sampling from trained model...")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        ref_idx = 0
        sig_ref_raw, geo_ref_raw, label_ref_raw = dataset[ref_idx]
        
        sig_ref_clamp = _clamp_sig(sig_ref_raw.unsqueeze(0).to(device))
        label_ref = label_ref_raw.unsqueeze(0).to(device)
        _, label_ref_norm = prepare_batch(sig_ref_clamp, label_ref, verbose=False)
        
        num_samples = 1
        B = num_samples
        
        # x_T ~ N(0, I)로 시작 (정규화된 공간)
        x = torch.randn(B, 2, model.L, device=device)
        
        # DDPM 역확산: T -> 1 (인덱싱 수정)
        print("Running reverse diffusion (T -> 1)...")
        sampling_pbar = tqdm(reversed(range(1, T + 1)), total=T, desc="Sampling progress")
        for t_val in sampling_pbar:
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            
            # CFG for sampling (if enabled)
            if use_cfg:
                # Unconditional prediction
                with autocast(device.type, enabled=(scaler is not None)):
                    eps_uncond = model(x, t_batch, get_null_label(B, model_label_dim, device))
                # Conditional prediction
                with autocast(device.type, enabled=(scaler is not None)):
                    eps_cond = model(x, t_batch, label_ref_norm)
                # CFG combination
                eps_hat = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                with autocast(device.type, enabled=(scaler is not None)):
                    eps_hat = model(x, t_batch, label_ref_norm)
            
            # DDPM 업데이트 (인덱싱 수정: t_val는 1~T, 인덱스는 0~T-1)
            idx = t_val - 1  # t_val=1000 -> idx=999, t_val=1 -> idx=0
            alpha_t = alphas[idx]
            alpha_bar_t = alphas_cumprod[idx]
            
            # 평균 계산
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (betas[idx] / torch.sqrt(1.0 - alpha_bar_t)) * eps_hat
            )
            
            # 분산 계산
            if t_val > 1:
                alpha_bar_prev = alphas_cumprod[idx - 1]  # idx > 0이므로 안전
                posterior_variance = betas[idx] * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                var = torch.sqrt(posterior_variance)
                noise = torch.randn_like(x)
                x = mean + var * noise
            else:
                # t=1: 최종 단계
                x = mean
            
            # Update progress bar
            if t_val % 100 == 0 or t_val <= 10:
                sampling_pbar.set_postfix({
                    "t": t_val,
                    "mean": f"{x.mean().item():.4f}",
                    "std": f"{x.std().item():.4f}"
                })
        
        # 샘플 역정규화
        samples_denorm = denormalize_sig(x)
        sample_np = samples_denorm[0].detach().cpu().numpy()
        
        print("\nSampling completed!")
        print(f"Sample shape: {sample_np.shape}")
        print(f"Sample nPE range: [{sample_np[0].min():.2f}, {sample_np[0].max():.2f}]")
        print(f"Sample FirstTime range: [{sample_np[1].min():.2f}, {sample_np[1].max():.2f}]")
        
        # Save sample plot if plot_save_dir is set
        if plot_save_dir is not None:
            geo_ref_np = geo_ref_raw.detach().cpu().numpy()
            label_ref_np = label_ref_raw.detach().cpu().numpy()
            
            sampled_output_path = plot_save_dir / f"sampled_event_{ref_idx}.png"
            fig_sampled, _ = show_event_dual_plot(
                sig=sample_np,
                geo=geo_ref_np,
                label=label_ref_np,
                output_path=str(sampled_output_path),
                figure_size=(18, 8),
                marker_size=8.0,
                show_detector_hull=True,
                show=False,
                title_prefix=f"train_exp_new_cfg.py | Sampled data | using label from event {ref_idx}",
                firsttime_title="FirstTime (sampled)",
                npe_title="nPE (sampled)",
            )
            print(f"Sampled event plot saved to: {sampled_output_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()
