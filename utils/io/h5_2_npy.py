#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h5_to_npy.py

HDF5 파일에서 하나의 이벤트를 로드하여 numpy 배열로 반환

HDF5 구조:
- info  : (N, 9)         float32
- input : (N, 2, 5160)   float32   # [npe, time]
- label : (N, 6)         float32   # [Energy, Zenith, Azimuth, X, Y, Z]
- xpmt  : (5160,)        float32
- ypmt  : (5160,)        float32
- zpmt  : (5160,)        float32

반환 형식:
- sig : (2, L)    # [npe, firstTime]
- geo : (3, L)    # [x, y, z]
- label : (6,)    # [Energy, Zenith, Azimuth, X, Y, Z]
"""

import os
from typing import Tuple

import h5py
import numpy as np


def load_event_from_h5(
    h5_path: str,
    event_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HDF5 파일에서 하나의 이벤트를 로드하여 numpy 배열로 반환
    
    Args:
        h5_path: HDF5 파일 경로
        event_index: 이벤트 인덱스
        
    Returns:
        (sig, geo, label) 튜플
        - sig: (2, L) numpy array - [npe, firstTime]
        - geo: (3, L) numpy array - [x, y, z]
        - label: (6,) numpy array - [Energy, Zenith, Azimuth, X, Y, Z]
    """
    h5_path = os.path.expanduser(h5_path)
    
    with h5py.File(h5_path, "r", swmr=True, libver="latest") as f:
        # Signal 데이터 로드
        sig = np.asarray(f["input"][event_index, :, :], dtype=np.float32)  # (2, L)
        label = np.asarray(f["label"][event_index, :], dtype=np.float32)    # (6,)
        
        # Geometry 데이터 로드
        xpmt = np.asarray(f["xpmt"], dtype=np.float32)
        ypmt = np.asarray(f["ypmt"], dtype=np.float32)
        zpmt = np.asarray(f["zpmt"], dtype=np.float32)
        geo = np.stack([xpmt, ypmt, zpmt], axis=0)  # (3, L)
    
    return sig, geo, label


# ---------------------------
# 사용 예시
# ---------------------------
if __name__ == "__main__":
    h5_path = "/home/work/GENESIS/0121/data/22644_0921_time_shift.h5"
    event_idx = 0
    
    sig, geo, label = load_event_from_h5(h5_path, event_idx)
    
    print(f"sig shape: {sig.shape}")      # (2, 5160)
    print(f"geo shape: {geo.shape}")      # (3, 5160)
    print(f"label shape: {label.shape}")  # (6,)
    print(f"label: Energy={label[0]:.4f}, Zenith={label[1]:.4f}, Azimuth={label[2]:.4f}")