#!/usr/bin/env python3
"""
Training script for GENESIS diffusion model.
"""

import math
import os
import sys
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
from torch.utils.data import DataLoader
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

device = get_default_device()
print("device:", device)

# ---- 출력 폴더 생성 ----
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)
print(f"Output directory: {output_dir.absolute()}")

# ---- config (원하는대로 바꿔도 됨) ----
T = 1000
beta_start, beta_end = 1e-4, 2e-2  # sigmoid schedule params

batch_size = 256
num_workers = 32  # 데이터 로딩 병렬화 (CPU 코어 수에 맞게 조정 가능, 0은 병렬화 없음)
lr = 3e-4
num_epochs = 20

# 데이터/정규화: nPE는 clamp 후 minmax만, FirstTime은 clamp 후 log_minmax
npe_clip = 1000.0
ftime_clip = 21000.0
log_min = 0.0
npe_log_max = float(np.log1p(npe_clip))  # nPE는 minmax만 쓰지만 denormalize 호환용 유지
ftime_log_max = float(np.log1p(ftime_clip))
_feature_range = (-1, 1)

# label 정규화: [Energy, ux, uy, X, Y, Z] -> Energy log_minmax, ux/uy identity, X/Y/Z minmax (dataset min/max)
_label_methods = ["log_minmax", "identity", "identity", "minmax", "minmax", "minmax"]
_label_feature_ranges = [_feature_range] * 6
# Energy (PeV) min/max: 하드코딩 → log_minmax용 log_min, log_max 계산
_ENERGY_PEV_MINMAX = {"min": 1.0, "max": 100.0}
energy_log_min = float(np.log1p(_ENERGY_PEV_MINMAX["min"]))
energy_log_max = float(np.log1p(_ENERGY_PEV_MINMAX["max"]))
# X,Y,Z min/max: 22644_0921_time_shift.h5 전체 데이터셋 기준 (하드코딩)
_LABEL_XYZ_MINMAX = [
    {"min": -570.9000244140625, "max": 576.3699951171875},   # X
    {"min": -521.0800170898438, "max": 509.5},               # Y
    {"min": -509.8599853515625, "max": 506.0566711425781},   # Z
]
# Geo (xpmt, ypmt, zpmt) min/max: 동일 H5 기준 하드코딩
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


# h5 경로 및 데이터셋
h5_path = "./GENESIS-data/22644_0921_time_shift.h5"
dataset = H5Dataset(h5_path=h5_path)

# ---- dataloader ----
_print_label_normalize_config()

loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers, 
    drop_last=True,
    pin_memory=(device.type == "cuda"),  # CUDA only; MPS/CPU use False
    persistent_workers=True if num_workers > 0 else False,  # 워커 재사용으로 오버헤드 감소
    prefetch_factor=2 if num_workers > 0 else None,  # 미리 로드할 배치 수
)

print("dataset length:", len(dataset))
sig0, geo0, label0 = dataset[0]
print("sig:", sig0.shape, sig0.dtype)
print("geo:", geo0.shape, geo0.dtype)
print("label:", label0.shape, label0.dtype, label0)
print("geo_min (x,y,z):", geo_min)
print("geo_max (x,y,z):", geo_max)

# ---- normalization: decorator-wrapped prepare_batch ----
# nPE: clamp [0, npe_clip] 후 minmax to [-1, 1]. FirstTime: clamp 후 log_minmax to [-1, 1]
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


def _clamp_sig(sig: torch.Tensor) -> torch.Tensor:
    """Clamp npe/ftime before normalize. Returns clamped copy."""
    s = sig.clone()
    s[:, 0] = torch.clamp(s[:, 0], min=0.0, max=npe_clip)
    s[:, 1] = torch.clamp(s[:, 1], min=0.0, max=ftime_clip)
    return s


def sample_timesteps(batch: int, T: int, device: torch.device) -> torch.Tensor:
    """t in [1, T] (t=0은 노이즈가 없는 원본이므로 학습에서 제외)"""
    return torch.randint(low=1, high=T + 1, size=(batch,), device=device, dtype=torch.long)

# ---- sigmoid noise schedule ----
betas = sigmoid_beta_schedule(timesteps=T, beta_start=beta_start, beta_end=beta_end).to(device)
alpha_schedule = compute_alpha_schedule(betas)
alphas = alpha_schedule["alphas"]
alphas_cumprod = alpha_schedule["alphas_cumprod"]
print("betas:", betas.shape, betas.min().item(), betas.max().item())

# ---- Transformer (DiT-style): token=DOM, conditional on t + label ----

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


# Fixed geo from first sample, minmax-normalized with H5 min/max (model uses single geometry for all samples)
_geo_raw = dataset[0][1]  # (3, L)
_geo = apply_minmax_geo(_geo_raw, geo_min, geo_max, feature_range=(0, 1))
model = DiffusionDiTTransformer(geo=_geo, d_model=256, nhead=8, depth=6, mlp_ratio=4.0, dropout=0.0, label_dim=6).to(device)
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
    print(f"[GPU] after model: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
optim = torch.optim.AdamW(model.parameters(), lr=lr)

# AMP (Automatic Mixed Precision): CUDA/MPS 지원
try:
    scaler = GradScaler(device.type) if device.type in ("cuda", "mps") else None
except (TypeError, ValueError):
    scaler = GradScaler() if device.type == "cuda" else None
print("AMP enabled:", scaler is not None)

# torch.compile() 최적화 (PyTorch 2.0+)
# 모델을 컴파일하여 실행 속도 향상 (약 20-30% 빠름)
# Triton이 없으면 에러가 발생하므로 안전하게 처리
try:
    if hasattr(torch, 'compile'):
        # Triton 없이도 작동하도록 에러 suppress 설정
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")  # 또는 "max-autotune" (더 느린 컴파일, 더 빠른 실행)
        print("Model compilation successful!")
    else:
        print("torch.compile() not available (requires PyTorch 2.0+)")
except Exception as e:
    print(f"Model compilation failed (continuing without compile): {e}")
    print("Note: If Triton is not installed, torch.compile() will fall back to eager mode.")

print("params:", sum(p.numel() for p in model.parameters())/1e6, "M")

# ---- training loop (objective: eps) ----
model.train()

loss_hist = []
steps_per_epoch = len(loader)
total_steps = num_epochs * steps_per_epoch

# 최고 성능 모델 추적 (loss가 낮을수록 좋음)
best_loss = float('inf')
best_checkpoint_path = None

print(f"Training for {num_epochs} epochs ({steps_per_epoch} batches per epoch, {total_steps} total steps)")
print("="*60)

for epoch in range(1, num_epochs + 1):
    epoch_losses = []
    
    # tqdm으로 진행률 표시
    pbar = tqdm(enumerate(loader, 1), total=steps_per_epoch, desc=f"Epoch {epoch}/{num_epochs}")
    
    for batch_idx, (sig, geo, label) in pbar:
        # pin_memory=True일 때 non_blocking으로 전송 속도 향상
        sig = sig.to(device, non_blocking=True)         # (B, 2, L)
        label = label.to(device, non_blocking=True)     # (B, 6)

        sig_clamp = _clamp_sig(sig)
        if epoch == 1 and batch_idx == 1:
            _label_raw_before = label.clone()
        x0, label = prepare_batch(sig_clamp, label, verbose=(epoch == 1 and batch_idx == 1))  # (B, 2, L) in [-1, 1]
        if epoch == 1 and batch_idx == 1:
            print("prepare_batch: label (raw, same batch)", _label_raw_before)
            for j in [1, 2]:
                ok = torch.allclose(_label_raw_before[:, j], label[:, j])
                print(f"  identity col {j} ({LABEL_NAMES[j]}): {'OK' if ok else 'MISMATCH'} raw={_label_raw_before[0, j].item():.6f} norm={label[0, j].item():.6f}")
            if device.type == "cuda":
                print(f"[GPU] after first batch (before step): {torch.cuda.memory_allocated() / 1e9:.3f} GB")

        B = x0.shape[0]
        t = sample_timesteps(B, T, device)

        noise = torch.randn_like(x0)
        x_t = apply_forward_diffusion(x0=x0, betas=betas, timesteps=t, noise=noise)

        # AMP: forward pass를 autocast로 감싸기
        with autocast(device.type, enabled=(scaler is not None)):
            eps_hat = model(x_t, t, label)
            loss = F.mse_loss(eps_hat, noise)

        optim.zero_grad(set_to_none=True)
        # AMP: loss를 scaler로 감싸서 backward
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
        if device.type == "cuda" and epoch == 1 and batch_idx == 1:
            print(f"[GPU] after first step: {torch.cuda.memory_allocated() / 1e9:.3f} GB (peak: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB)")
        loss_hist.append(loss_val)
        epoch_losses.append(loss_val)

        current_step = (epoch - 1) * steps_per_epoch + batch_idx
        
        # 성능이 개선되면 (loss가 낮아지면) 모델 저장
        if loss_val < best_loss:
            best_loss = loss_val
            
            # 이전 최고 모델 삭제 (선택사항)
            if best_checkpoint_path is not None and best_checkpoint_path.exists():
                best_checkpoint_path.unlink()
            
            # 새로운 최고 모델 저장
            best_checkpoint_path = output_dir / f"best_checkpoint_epoch_{epoch:03d}_batch_{batch_idx:05d}_step_{current_step:05d}_loss_{best_loss:.6f}.pt"
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "epoch": epoch,
                "batch_idx": batch_idx,
                "step": current_step,
                "loss": loss_val,
                "best_loss": best_loss,
                "betas": betas.cpu(),
                "alphas": alphas.cpu(),
                "alphas_cumprod": alphas_cumprod.cpu(),
                "T": T,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "d_model": model.d_model,
                "nhead": 8,
                "depth": 6,
                "mlp_ratio": 4.0,
                "label_dim": 6,
            }
            torch.save(checkpoint, best_checkpoint_path)
        
        # tqdm 업데이트
        avg_loss_so_far = np.mean(epoch_losses)
        pbar.set_postfix({
            "loss": f"{avg_loss_so_far:.6f}", 
            "step": current_step,
            "best": f"{best_loss:.6f}"
        })
    
    # Epoch summary
    epoch_avg_loss = np.mean(epoch_losses)
    print(f"\nepoch {epoch:3d}/{num_epochs} completed | avg loss: {epoch_avg_loss:.6f} | best loss: {best_loss:.6f}")
    if best_checkpoint_path:
        print(f"Best model saved: {best_checkpoint_path.name}")
    print("-"*60)

print("Training done!")

# ---- quick visualization: one event, different t ----
model.eval()

event_idx = 0
sig_raw, geo_raw, label_raw = dataset[event_idx]

geo_np = geo_raw.detach().cpu().numpy()
label_np = label_raw.detach().cpu().numpy()

sig = sig_raw.unsqueeze(0).to(device)  # (1,2,L)
label = label_raw.unsqueeze(0).to(device)  # (1,6)

sig_clamp = _clamp_sig(sig)
x0, label = prepare_batch(sig_clamp, label, verbose=True)

for t_val in [0, 250, 500, 750, 1000]:
    t = torch.tensor([t_val], device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = apply_forward_diffusion(x0=x0, betas=betas, timesteps=t, noise=noise)
    # 출력 역정규화 후 시각화 (prepare_batch 출력용 denormalize_sig)
    x_t_denorm = denormalize_sig(x_t)[0].detach().cpu().numpy()

    output_path = output_dir / f"event_{event_idx}_t_{t_val}.png"
    fig, _ = show_event_dual_plot(
        sig=x_t_denorm,
        geo=geo_np,
        label=label_np,
        output_path=str(output_path),
        figure_size=(18, 8),
        marker_size=8.0,
        show_detector_hull=True,
        show=False,
        title_prefix=f"train_exp.py | Sigmoid schedule | event {event_idx} | t={t_val}",
        firsttime_title="FirstTime (x_t, denorm)",
        npe_title="nPE (x_t, denorm)",
    )

# ---- Sampling: Generate new samples using label from first index ----
print("\n" + "="*60)
print("Sampling from trained model using label from first index...")
print("="*60)

model.eval()
with torch.no_grad():
    # 첫 번째 인덱스의 실제 데이터 가져오기
    ref_idx = 0
    sig_ref_raw, geo_ref_raw, label_ref_raw = dataset[ref_idx]
    
    # 실제 데이터 준비 (clamp만 적용한 상태로 그림 그리기)
    sig_ref_clamp = _clamp_sig(sig_ref_raw.unsqueeze(0).to(device))
    sig_ref_denorm = sig_ref_clamp[0].detach().cpu().numpy()
    geo_ref_np = geo_ref_raw.detach().cpu().numpy()
    label_ref_np = label_ref_raw.detach().cpu().numpy()
    
    # 실제 데이터 그림 그리기
    print(f"Plotting actual data from index {ref_idx}...")
    actual_output_path = output_dir / f"actual_event_{ref_idx}.png"
    fig_actual, _ = show_event_dual_plot(
        sig=sig_ref_denorm,
        geo=geo_ref_np,
        label=label_ref_np,
        output_path=str(actual_output_path),
        figure_size=(18, 8),
        marker_size=8.0,
        show_detector_hull=True,
        show=False,
        title_prefix=f"train_exp.py | Actual data | event {ref_idx}",
        firsttime_title="FirstTime (actual)",
        npe_title="nPE (actual)",
    )
    print(f"Actual data saved to: {actual_output_path}")
    
    # 샘플링을 위한 label 정규화 (prepare_batch 사용)
    label_ref = label_ref_raw.unsqueeze(0).to(device)  # (1, 6) - raw label
    _, label_ref_norm = prepare_batch(sig_ref_clamp, label_ref, verbose=False)  # label만 정규화
    
    # 샘플링 파라미터
    num_samples = 1
    B = num_samples
    
    # x_T ~ N(0, I)로 시작 (정규화된 공간)
    x = torch.randn(B, 2, model.L, device=device)
    
    # DDPM 역확산: T -> 1
    print("Running reverse diffusion (T -> 1)...")
    for t_val in reversed(range(1, T + 1)):
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
        
        # 모델로 노이즈 예측 (정규화된 label 사용) - AMP 적용
        with autocast(device.type, enabled=(scaler is not None)):
            eps_hat = model(x, t_batch, label_ref_norm)  # (B, 2, L)
        
        # DDPM 업데이트
        idx = t_val - 1  # t > 0이므로 t-1을 인덱스로 사용
        alpha_t = alphas[idx]
        alpha_bar_t = alphas_cumprod[idx]
        
        # 평균 계산: μ_θ(x_t, t) = (1/sqrt(α_t)) * (x_t - (β_t / sqrt(1-ᾱ_t)) * ε_θ)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (betas[idx] / torch.sqrt(1.0 - alpha_bar_t)) * eps_hat
        )
        
        # 분산 계산 (posterior variance)
        if t_val > 1:
            alpha_bar_prev = alphas_cumprod[idx - 1] if idx > 0 else torch.tensor(1.0, device=device)
            posterior_variance = betas[idx] * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            var = torch.sqrt(posterior_variance)
            noise = torch.randn_like(x)
            x = mean + var * noise
        else:
            # t=1: 최종 단계, 평균 반환
            x = mean
        
        if t_val % 100 == 0 or t_val <= 10:
            print(f"  t={t_val:4d} | x_norm: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
    
    # 샘플 역정규화
    samples_denorm = denormalize_sig(x)  # (B, 2, L)
    sample_np = samples_denorm[0].detach().cpu().numpy()
    
    print("Sampling completed!")
    print(f"Sample shape: {sample_np.shape}")
    print(f"Sample nPE range: [{sample_np[0].min():.2f}, {sample_np[0].max():.2f}]")
    print(f"Sample FirstTime range: [{sample_np[1].min():.2f}, {sample_np[1].max():.2f}]")
    
    # 샘플 시각화 및 저장
    print(f"Plotting sampled data using label from index {ref_idx}...")
    sampled_output_path = output_dir / f"sampled_event_{ref_idx}.png"
    fig_sampled, _ = show_event_dual_plot(
        sig=sample_np,
        geo=geo_ref_np,
        label=label_ref_np,
        output_path=str(sampled_output_path),
        figure_size=(18, 8),
        marker_size=8.0,
        show_detector_hull=True,
        show=False,
        title_prefix=f"train_exp.py | Sampled data | using label from event {ref_idx}",
        firsttime_title="FirstTime (sampled)",
        npe_title="nPE (sampled)",
    )
    print(f"Sampled event saved to: {sampled_output_path}")

# ---- Save final model ----
print("\n" + "="*60)
print("Saving final model...")
print("="*60)

model_save_path = output_dir / "model_checkpoint_final.pt"
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optim.state_dict(),
    "epoch": num_epochs,
    "final_loss": loss_hist[-1] if loss_hist else None,
    "best_loss": best_loss,
    "betas": betas.cpu(),
    "alphas": alphas.cpu(),
    "alphas_cumprod": alphas_cumprod.cpu(),
    "T": T,
    "beta_start": beta_start,
    "beta_end": beta_end,
    "d_model": model.d_model,
    "nhead": 8,
    "depth": 6,
    "mlp_ratio": 4.0,
    "label_dim": 6,
}

torch.save(checkpoint, model_save_path)
print(f"Final model saved to: {model_save_path}")
if best_checkpoint_path:
    print(f"Best model (loss={best_loss:.6f}) saved to: {best_checkpoint_path.name}")
print("Done!")
