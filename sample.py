#!/usr/bin/env python3
"""
Sampling script for GENESIS diffusion model.
Loads a trained model and generates samples.
"""

import math
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

# Add GENESIS to path
sys.path.insert(0, os.path.join(os.getcwd(), "GENESIS"))
from dataloader.h5 import H5Dataset
from diffusion.schedules import sigmoid_beta_schedule, compute_alpha_schedule
from utils.normalize import normalize, denormalize_log_minmax, apply_minmax_geo
from utils.vis.event_show import show_event_dual_plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ---- 정규화 설정 (train_exp.py와 동일) ----
npe_clip = 1000.0
ftime_clip = 8.0
log_min = 0.0
npe_log_max = float(np.log1p(npe_clip))
ftime_log_max = float(np.log1p(ftime_clip))
_feature_range = (-1, 1)

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

_channel_stats = [
    {"log_min": log_min, "log_max": npe_log_max},
    {"log_min": log_min, "log_max": ftime_log_max},
]

@normalize(
    channel_methods=["log_minmax", "log_minmax"],
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
    """입력: (sig, label). 데코레이터가 sig는 [-1,1] log_minmax, label은 Energy log_minmax / ux,uy identity / X,Y,Z minmax."""
    if verbose:
        print("prepare_batch: label", label)
    return (sig, label)


def _clamp_sig(sig: torch.Tensor) -> torch.Tensor:
    """Clamp npe/ftime before normalize."""
    s = sig.clone()
    s[:, 0] = torch.clamp(s[:, 0], min=0.0, max=npe_clip)
    s[:, 1] = torch.clamp(s[:, 1], min=0.0, max=ftime_clip)
    return s


def denormalize_sig(sig: torch.Tensor) -> torch.Tensor:
    """정규화된 sig ([-1, 1] log_minmax)를 원 스케일로 역정규화."""
    out = sig.clone()
    if sig.dim() == 3:
        out[:, 0, :] = denormalize_log_minmax(sig[:, 0, :], log_min, npe_log_max, _feature_range)
        out[:, 1, :] = denormalize_log_minmax(sig[:, 1, :], log_min, ftime_log_max, _feature_range)
    else:
        out[0, :] = denormalize_log_minmax(sig[0, :], log_min, npe_log_max, _feature_range)
        out[1, :] = denormalize_log_minmax(sig[1, :], log_min, ftime_log_max, _feature_range)
    return out


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """t: (B,) int/float, return: (B, dim)"""
    if t.dim() != 1:
        t = t.view(-1)
    t = t.float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device, dtype=emb.dtype)], dim=-1)
    return emb


class DiTBlock(nn.Module):
    """DiT-style Transformer block with AdaLN modulation."""
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
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, 6 * d),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        params = self.ada(c).view(B, 6, d)
        shift1, scale1, gate1, shift2, scale2, gate2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]
        x1 = self.norm1(x)
        x1 = x1 * (1.0 + scale1[:, None, :]) + shift1[:, None, :]
        attn_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + gate1[:, None, :] * attn_out
        x2 = self.norm2(x)
        x2 = x2 * (1.0 + scale2[:, None, :]) + shift2[:, None, :]
        mlp_out = self.mlp(x2)
        x = x + gate2[:, None, :] * mlp_out
        return x


class DiffusionDiTTransformer(nn.Module):
    def __init__(
        self,
        geo: torch.Tensor,
        d_model: int = 256,
        nhead: int = 8,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        label_dim: int = 6,
    ):
        super().__init__()
        self.d_model = d_model
        if geo.dim() == 2:
            geo_tok = geo.transpose(0, 1).unsqueeze(0)
        elif geo.dim() == 3:
            geo_tok = geo.permute(0, 2, 1)
        else:
            raise ValueError(f"geo must be (3,L) or (1,3,L). got {geo.shape}")
        self.register_buffer("geo_tokens", geo_tok.contiguous(), persistent=True)
        L = self.geo_tokens.shape[1]
        self.L = L
        self.in_proj = nn.Linear(2, d_model)
        self.geo_mlp = nn.Sequential(
            nn.Linear(3, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.use_index_pos = False
        if self.use_index_pos:
            self.index_pos = nn.Parameter(torch.zeros(1, L, d_model))
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
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.out_proj = nn.Linear(d_model, 2)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        B, C, L = x_t.shape
        assert L == self.L, f"Expected L={self.L}, got L={L}"
        tokens = x_t.permute(0, 2, 1)
        h = self.in_proj(tokens)
        pos_geo = self.geo_mlp(self.geo_tokens)
        h = h + pos_geo
        if self.use_index_pos:
            h = h + self.index_pos
        t_emb = sinusoidal_timestep_embedding(t, self.d_model)
        c = self.time_mlp(t_emb) + self.label_mlp(label)
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        h = self.final_norm(h)
        h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]
        out = self.out_proj(h)
        return out.permute(0, 2, 1)


def remove_orig_mod_prefix(state_dict: dict) -> dict:
    """torch.compile()으로 컴파일된 모델의 state_dict에서 _orig_mod. 접두사 제거."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod."):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_model(checkpoint_path: str, device: torch.device):
    """모델 체크포인트를 로드하고 모델을 반환."""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 체크포인트에서 설정 읽기
    T = checkpoint.get("T", 1000)
    beta_start = checkpoint.get("beta_start", 1e-4)
    beta_end = checkpoint.get("beta_end", 2e-2)
    d_model = checkpoint.get("d_model", 256)
    nhead = checkpoint.get("nhead", 8)
    depth = checkpoint.get("depth", 6)
    mlp_ratio = checkpoint.get("mlp_ratio", 4.0)
    label_dim = checkpoint.get("label_dim", 6)
    
    print(f"Model config: T={T}, d_model={d_model}, nhead={nhead}, depth={depth}")
    
    # 데이터셋에서 geo 가져오기
    h5_path = "./GENESIS-data/22644_0921_time_shift.h5"
    dataset = H5Dataset(h5_path=h5_path)
    geo_raw = dataset[0][1]  # (3, L)
    geo = apply_minmax_geo(geo_raw, geo_min, geo_max, feature_range=(0, 1))
    
    # 모델 생성
    model = DiffusionDiTTransformer(
        geo=geo,
        d_model=d_model,
        nhead=nhead,
        depth=depth,
        mlp_ratio=mlp_ratio,
        dropout=0.0,
        label_dim=label_dim,
    ).to(device)
    
    # 가중치 로드 (torch.compile() 접두사 제거)
    state_dict = checkpoint["model_state_dict"]
    state_dict = remove_orig_mod_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # Noise schedule 재계산
    betas = sigmoid_beta_schedule(timesteps=T, beta_start=beta_start, beta_end=beta_end).to(device)
    alpha_schedule = compute_alpha_schedule(betas)
    alphas = alpha_schedule["alphas"]
    alphas_cumprod = alpha_schedule["alphas_cumprod"]
    
    return model, betas, alphas, alphas_cumprod, T, dataset


def sample(
    model: nn.Module,
    label: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    T: int,
    num_samples: int = 1,
    device: torch.device = None,
    scaler: GradScaler = None,
    output_dir: Path = None,
    ref_idx: int = 0,
    geo_np: np.ndarray = None,
    label_np: np.ndarray = None,
):
    """DDPM 역확산을 통한 샘플링."""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    B = num_samples
    
    # Label 정규화 (dummy sig 사용)
    dummy_sig = torch.zeros(1, 2, model.L, device=device)
    _, label_norm = prepare_batch(dummy_sig, label.unsqueeze(0) if label.dim() == 1 else label, verbose=False)
    if label_norm.dim() == 1:
        label_norm = label_norm.unsqueeze(0)
    
    # x_T ~ N(0, I)로 시작
    x = torch.randn(B, 2, model.L, device=device)
    
    # DDPM 역확산: T -> 1
    print("Running reverse diffusion (T -> 1)...")
    pbar = tqdm(reversed(range(1, T + 1)), total=T, desc="Sampling")
    
    for t_val in pbar:
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
        
        with autocast('cuda', enabled=(scaler is not None)):
            eps_hat = model(x, t_batch, label_norm)
        
        idx = t_val - 1
        alpha_t = alphas[idx]
        alpha_bar_t = alphas_cumprod[idx]
        
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (betas[idx] / torch.sqrt(1.0 - alpha_bar_t)) * eps_hat
        )
        
        if t_val > 1:
            alpha_bar_prev = alphas_cumprod[idx - 1] if idx > 0 else torch.tensor(1.0, device=device)
            posterior_variance = betas[idx] * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            var = torch.sqrt(posterior_variance)
            noise = torch.randn_like(x)
            x = mean + var * noise
        else:
            x = mean
        
        pbar.set_postfix({"t": t_val, "mean": f"{x.mean().item():.4f}", "std": f"{x.std().item():.4f}"})
    
    # 샘플 역정규화
    samples_denorm = denormalize_sig(x)
    sample_np = samples_denorm[0].detach().cpu().numpy()
    
    print("Sampling completed!")
    print(f"Sample shape: {sample_np.shape}")
    print(f"Sample nPE range: [{sample_np[0].min():.2f}, {sample_np[0].max():.2f}]")
    print(f"Sample FirstTime range: [{sample_np[1].min():.2f}, {sample_np[1].max():.2f}]")
    
    # 시각화 및 저장
    if output_dir:
        # numpy 배열 저장
        np_output_path = output_dir / f"sampled_event_{ref_idx}.npy"
        np.save(np_output_path, sample_np)
        print(f"Sample numpy array saved to: {np_output_path}")
        
        # 이미지 저장 (geo와 label이 있을 때만)
        if geo_np is not None and label_np is not None:
            img_output_path = output_dir / f"sampled_event_{ref_idx}.png"
            print(f"Plotting sampled data...")
            fig_sampled, _ = show_event_dual_plot(
                sig=sample_np,
                geo=geo_np,
                label=label_np,
                output_path=str(img_output_path),
                figure_size=(18, 8),
                marker_size=8.0,
                show_detector_hull=True,
                show=False,
                title_prefix=f"sample.py | Sampled data | using label from event {ref_idx}",
                firsttime_title="FirstTime (sampled)",
                npe_title="nPE (sampled)",
            )
            print(f"Sample image saved to: {img_output_path}")
        else:
            print("Note: geo_np or label_np not provided, skipping image generation")
    
    return sample_np


def main():
    parser = argparse.ArgumentParser(description="Sample from trained GENESIS diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for samples")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--ref_idx", type=int, default=0, help="Dataset index to use label from")
    parser.add_argument("--label", type=str, default=None, help="Custom label as comma-separated values: Energy,ux,uy,X,Y,Z")
    
    args = parser.parse_args()
    
    # 출력 폴더 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # 모델 로드
    model, betas, alphas, alphas_cumprod, T, dataset = load_model(args.checkpoint, device)
    
    # AMP 설정
    scaler = GradScaler() if device.type == "cuda" else None
    
    # Label 준비
    geo_np = None
    if args.label:
        # 사용자 지정 label
        label_values = [float(x.strip()) for x in args.label.split(",")]
        if len(label_values) != 6:
            raise ValueError("Label must have 6 values: Energy,ux,uy,X,Y,Z")
        label = torch.tensor(label_values, device=device, dtype=torch.float32)
        label_np = label.detach().cpu().numpy()
        ref_idx = 0  # 사용자 지정 label일 때는 ref_idx를 0으로 설정
        # geo는 데이터셋에서 가져오기
        sig_ref_raw, geo_ref_raw, _ = dataset[0]
        geo_np = geo_ref_raw.detach().cpu().numpy()
    else:
        # 데이터셋에서 label 가져오기
        ref_idx = args.ref_idx
        sig_ref_raw, geo_ref_raw, label_ref_raw = dataset[ref_idx]
        label = label_ref_raw.to(device)
        label_np = label_ref_raw.detach().cpu().numpy()
        geo_np = geo_ref_raw.detach().cpu().numpy()
    
    print(f"Using label: {label_np}")
    print(f"Label names: {LABEL_NAMES}")
    
    # 샘플링
    sample_np = sample(
        model=model,
        label=label,
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        T=T,
        num_samples=args.num_samples,
        device=device,
        scaler=scaler,
        output_dir=output_dir,
        ref_idx=ref_idx,
        geo_np=geo_np,
        label_np=label_np,
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
