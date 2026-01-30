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
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add GENESIS to path
sys.path.insert(0, os.path.join(os.getcwd(), "GENESIS"))
from dataloader.h5 import H5Dataset
from diffusion.schedules import sigmoid_beta_schedule, compute_alpha_schedule
from utils.normalize import normalize, denormalize_log_minmax, denormalize_minmax, apply_minmax_geo
from utils.vis.event_show import show_event_dual_plot
from utils.device import get_default_device


def get_device(gpu_id: int = None) -> torch.device:
    """GPU 디바이스 반환. CUDA 우선, 없으면 MPS(맥), 없으면 CPU. gpu_id는 CUDA 다중 GPU용."""
    if torch.cuda.is_available():
        if gpu_id is not None:
            if gpu_id >= torch.cuda.device_count():
                raise ValueError(f"GPU {gpu_id} not available. Only {torch.cuda.device_count()} GPU(s) available.")
            return torch.device(f"cuda:{gpu_id}")
        for i in range(torch.cuda.device_count()):
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            if mem_reserved < 1.0:
                return torch.device(f"cuda:{i}")
        return torch.device("cuda:0")
    return get_default_device()

device = get_device()
print("device:", device)

# ---- 정규화 설정 (train_exp.py와 동일) ----
npe_clip = 1000.0
ftime_clip = 21000.0
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
    """정규화된 sig를 원 스케일로 역정규화. nPE는 minmax 역변환, FirstTime은 log_minmax 역변환."""
    out = sig.clone()
    if sig.dim() == 3:
        out[:, 0, :] = denormalize_minmax(sig[:, 0, :], 0.0, npe_clip, _feature_range)
        out[:, 1, :] = denormalize_log_minmax(sig[:, 1, :], log_min, ftime_log_max, _feature_range)
    else:
        out[0, :] = denormalize_minmax(sig[0, :], 0.0, npe_clip, _feature_range)
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


def plot_histogram(
    sig: np.ndarray,
    output_path: Path,
    title_suffix: str = "",
    cut_npe: float = 0.0,
    cut_firsttime: float = 0.0,
):
    """
    nPE와 FirstTime의 히스토그램을 그려서 저장.
    
    Args:
        sig: (2, L) 형태의 샘플 데이터
        output_path: 저장 경로
        title_suffix: 제목에 추가할 접미사
        cut_npe: 이 값 이하의 nPE는 히스토그램에 포함하지 않음 (0이면 전부 포함)
        cut_firsttime: 이 값 이하의 FirstTime은 히스토그램에 포함하지 않음 (0이면 전부 포함)
    """
    npe = sig[0]  # nPE 채널
    ftime = sig[1]  # FirstTime 채널
    
    # cut 이하 제외. cut=0(디폴트)이면 전부 포함, cut>0이면 해당 값 초과만
    npe_for_hist = npe if cut_npe == 0 else npe[npe > cut_npe]
    ftime_for_hist = ftime if cut_firsttime == 0 else ftime[ftime > cut_firsttime]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # nPE 히스토그램
    ax1 = axes[0]
    if len(npe_for_hist) > 0:
        ax1.hist(npe_for_hist, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('nPE')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'nPE Distribution{title_suffix}' + (f' (cut>{cut_npe})' if cut_npe > 0 else ''))
        ax1.grid(True, alpha=0.3)
        ax1.axvline(npe_for_hist.mean(), color='red', linestyle='--', label=f'Mean: {npe_for_hist.mean():.2f}')
        ax1.axvline(np.median(npe_for_hist), color='green', linestyle='--', label=f'Median: {np.median(npe_for_hist):.2f}')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No nPE values above cut', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'nPE Distribution{title_suffix} (empty)')
    
    # FirstTime 히스토그램
    ax2 = axes[1]
    if len(ftime_for_hist) > 0:
        ax2.hist(ftime_for_hist, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('FirstTime')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'FirstTime Distribution{title_suffix}' + (f' (cut>{cut_firsttime})' if cut_firsttime > 0 else ''))
        ax2.grid(True, alpha=0.3)
        ax2.axvline(ftime_for_hist.mean(), color='red', linestyle='--', label=f'Mean: {ftime_for_hist.mean():.2f}')
        ax2.axvline(np.median(ftime_for_hist), color='green', linestyle='--', label=f'Median: {np.median(ftime_for_hist):.2f}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No FirstTime values above cut', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'FirstTime Distribution{title_suffix} (empty)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved histogram: {output_path}")


def _apply_cuts(sig: np.ndarray, cut_npe: float, cut_firsttime: float) -> np.ndarray:
    """cut_npe/cut_firsttime 이하 값을 0으로 만들어 시각화에서 보이지 않게 함. 0이면 변경 없음."""
    out = sig.copy()
    if cut_npe > 0:
        out[0] = np.where(out[0] <= cut_npe, 0.0, out[0])
    if cut_firsttime > 0:
        out[1] = np.where(out[1] <= cut_firsttime, 0.0, out[1])
    return out


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
    save_histogram: bool = False,
    cut_npe: float = 0.0,
    cut_firsttime: float = 0.0,
):
    """DDPM 역확산을 통한 샘플링."""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    B = num_samples
    
    # Label 정규화 (dummy sig 사용)
    with torch.no_grad():
        dummy_sig = torch.zeros(1, 2, model.L, device=device)
        _, label_norm = prepare_batch(dummy_sig, label.unsqueeze(0) if label.dim() == 1 else label, verbose=False)
        if label_norm.dim() == 1:
            label_norm = label_norm.unsqueeze(0)
        del dummy_sig
    
    # x_T ~ N(0, I)로 시작
    x = torch.randn(B, 2, model.L, device=device)
    
    # DDPM 역확산: T -> 1
    print("Running reverse diffusion (T -> 1)...")
    pbar = tqdm(reversed(range(1, T + 1)), total=T, desc="Sampling")
    
    with torch.no_grad():  # Gradient 계산 불필요 (inference)
        for t_val in pbar:
            t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
            
            with autocast(device.type, enabled=(scaler is not None)):
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
                del noise, var, posterior_variance, alpha_bar_prev
            else:
                x = mean
            
            # 중간 변수 정리
            del eps_hat, mean, alpha_t, alpha_bar_t, t_batch
            
            pbar.set_postfix({"t": t_val, "mean": f"{x.mean().item():.4f}", "std": f"{x.std().item():.4f}"})
    
    # 샘플 역정규화
    samples_denorm = denormalize_sig(x)  # (B, 2, L)
    samples_np = samples_denorm.detach().cpu().numpy()  # (B, 2, L)
    
    # GPU 메모리 정리
    del x, samples_denorm, label_norm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and getattr(torch.mps, "empty_cache", None):
        torch.mps.empty_cache()
    
    print("Sampling completed!")
    print(f"Generated {num_samples} sample(s)")
    
    # 각 샘플 저장
    saved_samples = []
    for i in range(num_samples):
        sample_np = samples_np[i]  # (2, L)
        
        print(f"\nSample {i+1}/{num_samples}:")
        print(f"  Shape: {sample_np.shape}")
        print(f"  nPE range: [{sample_np[0].min():.2f}, {sample_np[0].max():.2f}]")
        print(f"  FirstTime range: [{sample_np[1].min():.2f}, {sample_np[1].max():.2f}]")
        
        # 시각화 및 저장
        if output_dir:
            # numpy 배열 저장
            np_output_path = output_dir / f"sampled_event_{ref_idx}_sample_{i+1:03d}.npy"
            np.save(np_output_path, sample_np)
            print(f"  Saved numpy: {np_output_path}")
            
            # 이미지 저장 (geo와 label이 있을 때만) — cut 적용 시 해당 값 이하는 미표시
            if geo_np is not None and label_np is not None:
                sig_vis = _apply_cuts(sample_np, cut_npe, cut_firsttime)
                img_output_path = output_dir / f"sampled_event_{ref_idx}_sample_{i+1:03d}.png"
                fig_sampled, _ = show_event_dual_plot(
                    sig=sig_vis,
                    geo=geo_np,
                    label=label_np,
                    output_path=str(img_output_path),
                    figure_size=(18, 8),
                    marker_size=8.0,
                    show_detector_hull=True,
                    show=False,
                    title_prefix=f"sample.py | Sampled data #{i+1} | using label from event {ref_idx}",
                    firsttime_title="FirstTime (sampled)",
                    npe_title="nPE (sampled)",
                )
                print(f"  Saved image: {img_output_path}")
            
            # 히스토그램 저장 (cut 적용 시 해당 값 이하는 히스토그램에서 제외)
            if save_histogram:
                hist_output_path = output_dir / f"sampled_event_{ref_idx}_sample_{i+1:03d}_histogram.png"
                plot_histogram(
                    sample_np, hist_output_path, title_suffix=f" (Sample #{i+1})",
                    cut_npe=cut_npe, cut_firsttime=cut_firsttime,
                )
        
        saved_samples.append(sample_np)
    
    # 첫 번째 샘플 반환 (하위 호환성)
    return saved_samples[0] if len(saved_samples) == 1 else saved_samples


def main():
    parser = argparse.ArgumentParser(description="Sample from trained GENESIS diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for samples")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--ref_idx", type=int, default=0, help="Dataset index to use label from")
    parser.add_argument("--label", type=str, default=None, help="Custom label as comma-separated values: Energy,ux,uy,X,Y,Z")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (default: auto-select free GPU)")
    parser.add_argument("--histogram", action="store_true", help="Save histogram plots for each sample")
    parser.add_argument("--cut_npe", type=float, default=0.0, help="nPE cut: values <= this are hidden in plots/histogram (default: 0 = show all)")
    parser.add_argument("--cut_firsttime", type=float, default=0.0, help="FirstTime cut: values <= this are hidden in plots/histogram (default: 0 = show all)")
    
    args = parser.parse_args()
    
    # GPU 선택
    global device
    device = get_device(args.gpu)
    print(f"Using device: {device}")
    
    # 출력 폴더 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # 모델 로드
    model, betas, alphas, alphas_cumprod, T, dataset = load_model(args.checkpoint, device)
    
    # AMP 설정 (샘플링에는 필요 없지만 호환성을 위해)
    scaler = None  # 샘플링은 inference이므로 GradScaler 불필요
    
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
    
    # 원본 이미지 저장 (데이터셋에서 가져온 경우만)
    if not args.label:  # 사용자 지정 label이 아닐 때만
        print(f"\nPlotting actual data from index {ref_idx}...")
        sig_ref_clamp = _clamp_sig(sig_ref_raw.unsqueeze(0).to(device))
        sig_ref_denorm = sig_ref_clamp[0].detach().cpu().numpy()  # clamp만 적용된 상태
        
        sig_actual_vis = _apply_cuts(sig_ref_denorm, args.cut_npe, args.cut_firsttime)
        actual_output_path = output_dir / f"actual_event_{ref_idx}.png"
        fig_actual, _ = show_event_dual_plot(
            sig=sig_actual_vis,
            geo=geo_np,
            label=label_np,
            output_path=str(actual_output_path),
            figure_size=(18, 8),
            marker_size=8.0,
            show_detector_hull=True,
            show=False,
            title_prefix=f"sample.py | Actual data | event {ref_idx}",
            firsttime_title="FirstTime (actual)",
            npe_title="nPE (actual)",
        )
        print(f"Actual data saved to: {actual_output_path}")
        
        # 원본 히스토그램도 저장 (옵션이 켜져있을 때, cut 적용)
        if args.histogram:
            actual_hist_path = output_dir / f"actual_event_{ref_idx}_histogram.png"
            plot_histogram(
                sig_ref_denorm, actual_hist_path, title_suffix=" (Actual)",
                cut_npe=args.cut_npe, cut_firsttime=args.cut_firsttime,
            )
        
        del sig_ref_clamp, sig_ref_denorm
    
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
        save_histogram=args.histogram,
        cut_npe=args.cut_npe,
        cut_firsttime=args.cut_firsttime,
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
