#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmt_dataloader.py

HDF5 구조
- info  : (N, 9)         float32
- input : (N, 2, 5160)   float32   # [npe, time]
- label : (N, 6)         float32   # [Energy, Zenith, Azimuth, X, Y, Z]
- xpmt  : (5160,)        float32
- ypmt  : (5160,)        float32
- zpmt  : (5160,)        float32

반환 (한 이벤트):
- x_sig : (2, L)    # [npe, time]
- geom  : (3, L)    # [x, y, z]
- y     : (6,)
"""

from __future__ import annotations
import os
from typing import Tuple, Optional

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


ArrayLike = np.ndarray


class PMTSignalsH5(Dataset):
    def __init__(
        self,
        h5_path: str,
        replace_time_inf_with: Optional[float] = None,  # 예: 0.0 (기본: None → 원본 그대로)
        channel_first: bool = True,  # (2, L) 형태로 반환. False면 (L, 2)
        dtype: np.dtype = np.float32,
        indices: Optional[np.ndarray] = None,  # 서브셋 학습 시 사용
        time_transform: str = "ln",  # "log10" or "ln" - always use log(1+x)
        # Normalization parameters (applied AFTER time transformation)
        affine_offsets: Optional[tuple] = None,  # [charge, time, x, y, z]
        affine_scales: Optional[tuple] = None,   # [charge, time, x, y, z]
        label_offsets: Optional[tuple] = None,   # [Energy, Zenith, Azimuth, X, Y, Z]
        label_scales: Optional[tuple] = None,    # [Energy, Zenith, Azimuth, X, Y, Z]
    ):
        super().__init__()
        self.h5_path = os.path.expanduser(h5_path)
        self.replace_time_inf_with = replace_time_inf_with
        self.channel_first = channel_first
        self.dtype = dtype
        self.time_transform = time_transform
        
        # Store normalization parameters
        # Default: no normalization (offset=0, scale=1)
        self.affine_offsets = np.array(affine_offsets if affine_offsets is not None 
                                       else [0.0, 0.0, 0.0, 0.0, 0.0], dtype=self.dtype)
        self.affine_scales = np.array(affine_scales if affine_scales is not None 
                                      else [1.0, 1.0, 1.0, 1.0, 1.0], dtype=self.dtype)
        self.label_offsets = np.array(label_offsets if label_offsets is not None 
                                      else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=self.dtype)
        self.label_scales = np.array(label_scales if label_scales is not None 
                                     else [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=self.dtype)
        
        # Reshape for broadcasting: (5,) -> (5, 1) for signals+geom
        self.affine_offsets = self.affine_offsets.reshape(5, 1)
        self.affine_scales = self.affine_scales.reshape(5, 1)
        # Labels: (6,) stays (6,) for per-label normalization

        # 파일 메타만 먼저 확인 (빠르게)
        with h5py.File(self.h5_path, "r", swmr=True, libver="latest") as f:
            assert "input" in f and "label" in f, "HDF5 must contain 'input' and 'label'"
            N, C, L = f["input"].shape
            assert C == 2, f"input should have 2 channels (npe,time), got {C}"
            assert "xpmt" in f and "ypmt" in f and "zpmt" in f, "HDF5 must have xpmt, ypmt, zpmt"
            self.N = N
            self.L = L
            # 지오메트리는 파일에서 한 번만 읽어 캐시 (모든 이벤트에서 동일)
            xpmt = np.asarray(f["xpmt"], dtype=self.dtype)
            ypmt = np.asarray(f["ypmt"], dtype=self.dtype)
            zpmt = np.asarray(f["zpmt"], dtype=self.dtype)
            assert xpmt.shape == (L,) and ypmt.shape == (L,) and zpmt.shape == (L,)
            self.geom_np = np.stack([xpmt, ypmt, zpmt], axis=0)  # (3, L)

        # 인덱스 서브셋
        if indices is None:
            self.indices = np.arange(self.N, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return self.indices.shape[0]

    def _read_event(self, f: h5py.File, i: int) -> Tuple[ArrayLike, ArrayLike]:
        """input(2,L), label(6,) numpy 반환 (원본 스케일)"""
        x_sig = np.asarray(f["input"][i, :, :], dtype=self.dtype)   # (2, L)
        y     = np.asarray(f["label"][i, :], dtype=self.dtype)      # (6,)
        
        # Handle inf/nan in both npe and time
        # NPE (channel 0)
        npe = x_sig[0, :]
        mask_invalid_npe = ~np.isfinite(npe)
        if mask_invalid_npe.any():
            npe[mask_invalid_npe] = 0.0  # Replace inf/nan in npe with 0
            x_sig[0, :] = npe
        
        # Time (channel 1)
        t = x_sig[1, :]
        mask_invalid_time = ~np.isfinite(t)
        if mask_invalid_time.any():
            replacement_val = self.replace_time_inf_with if self.replace_time_inf_with is not None else 0.0
            t[mask_invalid_time] = replacement_val
            x_sig[1, :] = t
        
        # Time transformation: Always use log(1+x)
        # - log(1 + 0) = 0 (zeros handled naturally)
        # - No -inf issue
        # - Inverse: exp(y) - 1 or 10^y - 1
        if self.time_transform == "log10":
            t = np.log10(1.0 + t)  # log10(1+0)=0, log10(1+10000)≈4
        elif self.time_transform == "ln":
            t = np.log(1.0 + t)    # ln(1+0)=0, ln(1+10000)≈9.2
        else:
            raise ValueError(f"time_transform must be 'ln' or 'log10', got {self.time_transform}")
        
        x_sig[1, :] = t
        
        # Handle inf/nan in labels
        mask_invalid_label = ~np.isfinite(y)
        if mask_invalid_label.any():
            y[mask_invalid_label] = 0.0
        
        return x_sig, y

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, int]:
        real_i = int(self.indices[idx])
        # 매 호출마다 안전하게 open (DataLoader num_workers>0 호환)
        with h5py.File(self.h5_path, "r", swmr=True, libver="latest") as f:
            x_sig_np, y_np = self._read_event(f, real_i)

        # ================================================================
        # NORMALIZATION APPLIED HERE (once per sample, not per batch)
        # ================================================================
        #  At this point:
        #   x_sig_np: (2, L) - [charge_raw, time_transformed]
        #              - charge is raw NPE
        #              - time is ln(1+time) or log10(1+time)
        #   geom_np:  (3, L) - [x, y, z] in raw scale
        #   y_np:     (6,)   - [Energy, Zenith, Azimuth, X, Y, Z] in raw scale
        
        # Concatenate signal + geom -> (5, L)
        geom_np = self.geom_np  # (3, L)
        x5_np = np.concatenate([x_sig_np, geom_np], axis=0)  # (5, L)
        
        # Apply affine normalization: (x - offset) / scale
        # Broadcast: (5, 1) with (5, L) -> (5, L)
        x5_np = (x5_np - self.affine_offsets) / self.affine_scales
        
        # Split back
        x_sig_np = x5_np[0:2, :]  # (2, L)
        geom_np = x5_np[2:5, :]   # (3, L)
        
        # Normalize labels: (6,) - (6,) / (6,) -> (6,)
        y_np = (y_np - self.label_offsets) / self.label_scales
        # ================================================================

        if not self.channel_first:
            # (L, 2) 로 바꿔달라는 경우
            x_sig_np = np.transpose(x_sig_np, (1, 0))

        x_sig = torch.from_numpy(x_sig_np)     # (2, L) or (L,2)
        geom  = torch.from_numpy(geom_np)      # (3, L)
        y     = torch.from_numpy(y_np)         # (6,)

        return x_sig, geom, y, real_i


def validate_data_batch(x_sig: Tensor, geom: Tensor, label: Tensor, batch_idx: int = 0) -> dict:
    """
    Validate a batch of data and return statistics.
    
    Args:
        x_sig: (B, 2, L) signal tensor
        geom: (B, 3, L) or (3, L) geometry tensor
        label: (B, 6) label tensor
        batch_idx: Batch index for logging
        
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'batch_idx': batch_idx,
        'batch_size': x_sig.shape[0],
        'has_nan': False,
        'has_inf': False,
        'warnings': []
    }
    
    # Check x_sig (signals)
    if torch.isnan(x_sig).any():
        stats['has_nan'] = True
        nan_count = torch.isnan(x_sig).sum().item()
        stats['warnings'].append(f"x_sig contains {nan_count} NaN values")
    
    if torch.isinf(x_sig).any():
        stats['has_inf'] = True
        inf_count = torch.isinf(x_sig).sum().item()
        stats['warnings'].append(f"x_sig contains {inf_count} inf values")
    
    # Check label
    if torch.isnan(label).any():
        stats['has_nan'] = True
        nan_count = torch.isnan(label).sum().item()
        stats['warnings'].append(f"label contains {nan_count} NaN values")
    
    if torch.isinf(label).any():
        stats['has_inf'] = True
        inf_count = torch.isinf(label).sum().item()
        stats['warnings'].append(f"label contains {inf_count} inf values")
    
    # Statistics
    stats['x_sig_min'] = x_sig.min().item()
    stats['x_sig_max'] = x_sig.max().item()
    stats['x_sig_mean'] = x_sig.mean().item()
    stats['x_sig_std'] = x_sig.std().item()
    
    stats['label_min'] = label.min().item()
    stats['label_max'] = label.max().item()
    stats['label_mean'] = label.mean().item()
    stats['label_std'] = label.std().item()
    
    return stats


def check_dataset_health(dataloader: DataLoader, num_batches: int = 10, verbose: bool = True) -> dict:
    """
    Check the health of the dataset by sampling a few batches.
    
    Args:
        dataloader: DataLoader to check
        num_batches: Number of batches to sample
        verbose: Whether to print detailed statistics
        
    Returns:
        Dictionary with overall statistics
    """
    print(f"\n{'='*70}")
    print("📊 Dataset Health Check")
    print(f"{'='*70}")
    
    total_stats = {
        'total_batches_checked': 0,
        'batches_with_nan': 0,
        'batches_with_inf': 0,
        'all_batch_stats': []
    }
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        x_sig, geom, label, idx = batch
        
        # Expand geom if needed
        if geom.ndim == 2:
            geom = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)
        
        stats = validate_data_batch(x_sig, geom, label, batch_idx=i)
        total_stats['all_batch_stats'].append(stats)
        total_stats['total_batches_checked'] += 1
        
        if stats['has_nan']:
            total_stats['batches_with_nan'] += 1
        if stats['has_inf']:
            total_stats['batches_with_inf'] += 1
        
        if verbose and (stats['has_nan'] or stats['has_inf'] or i == 0):
            print(f"\nBatch {i}:")
            print(f"  Shape: x_sig={x_sig.shape}, geom={geom.shape}, label={label.shape}")
            
            # Detailed channel-wise statistics
            print(f"\n  📊 Signal Channels (NORMALIZED by dataloader):")
            print(f"    Charge (ch 0): [{x_sig[:, 0, :].min():.6f}, {x_sig[:, 0, :].max():.6f}] "
                  f"mean={x_sig[:, 0, :].mean():.6f} std={x_sig[:, 0, :].std():.6f}")
            print(f"    Time   (ch 1): [{x_sig[:, 1, :].min():.6f}, {x_sig[:, 1, :].max():.6f}] "
                  f"mean={x_sig[:, 1, :].mean():.6f} std={x_sig[:, 1, :].std():.6f}")
            
            print(f"\n  📍 Geometry Channels (NORMALIZED):")
            print(f"    X PMT  (ch 0): [{geom[:, 0, :].min():.6f}, {geom[:, 0, :].max():.6f}] "
                  f"mean={geom[:, 0, :].mean():.6f} std={geom[:, 0, :].std():.6f}")
            print(f"    Y PMT  (ch 1): [{geom[:, 1, :].min():.6f}, {geom[:, 1, :].max():.6f}] "
                  f"mean={geom[:, 1, :].mean():.6f} std={geom[:, 1, :].std():.6f}")
            print(f"    Z PMT  (ch 2): [{geom[:, 2, :].min():.6f}, {geom[:, 2, :].max():.6f}] "
                  f"mean={geom[:, 2, :].mean():.6f} std={geom[:, 2, :].std():.6f}")
            
            print(f"\n  🏷️  Label Channels (NORMALIZED):")
            label_names = ['Energy', 'Zenith', 'Azimuth', 'X', 'Y', 'Z']
            for ch_idx, name in enumerate(label_names):
                print(f"    {name:8s} (ch {ch_idx}): [{label[:, ch_idx].min():.6f}, {label[:, ch_idx].max():.6f}] "
                      f"mean={label[:, ch_idx].mean():.6f} std={label[:, ch_idx].std():.6f}")
            
            if stats['warnings']:
                print(f"\n  ⚠️  Warnings:")
                for warning in stats['warnings']:
                    print(f"    - {warning}")
    
    print(f"\n{'='*70}")
    print("📋 Summary:")
    print(f"  Total batches checked: {total_stats['total_batches_checked']}")
    print(f"  Batches with NaN: {total_stats['batches_with_nan']}")
    print(f"  Batches with inf: {total_stats['batches_with_inf']}")
    
    if total_stats['batches_with_nan'] == 0 and total_stats['batches_with_inf'] == 0:
        print(f"  ✅ All checked batches are healthy!")
    else:
        print(f"  ⚠️  Some batches contain invalid values (NaN/inf)")
    print(f"{'='*70}\n")
    
    return total_stats


def make_dataloader(
    h5_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    replace_time_inf_with: Optional[float] = None,
    channel_first: bool = True,
    indices: Optional[np.ndarray] = None,
    time_transform: str = "ln",  # "log10" or "ln" - always use log(1+x)
    affine_offsets: Optional[tuple] = None,
    affine_scales: Optional[tuple] = None,
    label_offsets: Optional[tuple] = None,
    label_scales: Optional[tuple] = None,
) -> DataLoader:
    ds = PMTSignalsH5(
        h5_path=h5_path,
        replace_time_inf_with=replace_time_inf_with,
        channel_first=channel_first,
        indices=indices,
        time_transform=time_transform,
        affine_offsets=affine_offsets,
        affine_scales=affine_scales,
        label_offsets=label_offsets,
        label_scales=label_scales,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


# ---------------------------
# 간단 사용 예 (스모크 테스트)
# ---------------------------
if __name__ == "__main__":
    path = "/home/work/GENESIS/GENESIS-data/22644_0921.h5"
    loader = make_dataloader(path, batch_size=2, num_workers=0, replace_time_inf_with=None)

    x_sig, geom, y, idx = next(iter(loader))
    # geom 차원에 따라 안전하게 배치 차원 맞추기
    if geom.ndim == 2:
        # (3,L) -> (B,3,L)
        geom_batched = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)
    elif geom.ndim == 3:
        # 이미 (B,3,L)
        geom_batched = geom
    else:
        raise ValueError(f"Unexpected geom shape: {geom.shape}")

    print("x_sig:", x_sig.shape)                 # (B,2,L)
    print("geom (per-batch):", geom_batched.shape)  # (B,3,L)
    print("label:", y.shape)                     # (B,6)
    print("indices:", idx)