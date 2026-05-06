#!/usr/bin/env python3
"""
DOM 이벤트 형 HDF5 (예: input shape (N, 2, N_dom), xpmt/ypmt/zpmt)를
multifield_map 평가용 (N, 3, H, W) 격자 텐서로 변환해 새 HDF5를 만든다.

채널 정의 (기본):
  ch0: 격자 셀별 NPE(또는 charge) 합
  ch1: 격자 셀별 NPE 가중 평균 first time  (가중치 0인 셀은 0)
  ch2: 격자 셀별 활성 DOM 개수 (NPE>threshold 인 hit 수)

좌표 투영: xpmt, ypmt, zpmt 중 두 축을 골라 2D 히스토그램 빈에 누적한다.

사용 예:
  python dom_event_h5_to_multifield_grid_h5.py ^
    --input_h5 "C:/Users/user/Downloads/22644_0921_time_shift (1).h5" ^
    --output_h5 "C:/Users/user/Downloads/22644_0921_time_shift_grid3c.h5" ^
    --grid_h 64 --grid_w 64 --plane xy --auto_bounds --npe_threshold 0

그 다음 make_sample_ver2.py 에서 --h5_path 를 출력 파일로 바꾸고
--eval_mode multifield_map --channel_names Mcdm,Mgas,T 등으로 실행한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np


PLANE_COORDS = {
    "xy": (0, 1),  # xpmt, ypmt
    "xz": (0, 2),
    "yz": (1, 2),
}


def pick_coords(xyz: Tuple[np.ndarray, np.ndarray, np.ndarray], plane: str) -> Tuple[np.ndarray, np.ndarray]:
    i, j = PLANE_COORDS[plane.lower()]
    pts = (xyz[0], xyz[1], xyz[2])
    return np.asarray(pts[i], dtype=np.float64), np.asarray(pts[j], dtype=np.float64)


def auto_bounds(u: np.ndarray, v: np.ndarray, lo_q: float, hi_q: float, pad_frac: float) -> Tuple[float, float, float, float]:
    u = u[np.isfinite(u)]
    v = v[np.isfinite(v)]
    u_lo, u_hi = float(np.quantile(u, lo_q)), float(np.quantile(u, hi_q))
    v_lo, v_hi = float(np.quantile(v, lo_q)), float(np.quantile(v, hi_q))
    if u_hi <= u_lo:
        u_hi = u_lo + 1.0
    if v_hi <= v_lo:
        v_hi = v_lo + 1.0
    pu = pad_frac * (u_hi - u_lo)
    pv = pad_frac * (v_hi - v_lo)
    return u_lo - pu, u_hi + pu, v_lo - pv, v_hi + pv


def digitize_2d(u: np.ndarray, v: np.ndarray, u0: float, u1: float, v0: float, v1: float, gw: int, gh: int) -> Tuple[np.ndarray, np.ndarray]:
    """각 DOM 인덱스에 대해 [0,gw), [0,gh) 빈 인덱스."""
    tu = (u - u0) / max(u1 - u0, 1e-12) * gw
    tv = (v - v0) / max(v1 - v0, 1e-12) * gh
    ix = np.floor(tu).astype(np.int64)
    iy = np.floor(tv).astype(np.int64)
    ix = np.clip(ix, 0, gw - 1)
    iy = np.clip(iy, 0, gh - 1)
    return ix, iy


def project_event(
    npe: np.ndarray,
    tme: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    gh: int,
    gw: int,
    npe_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """단일 이벤트 (N_dom,) -> (3, gh, gw).  np.add.at 으로 벡터화."""
    ch0 = np.zeros((gh, gw), dtype=np.float64)
    wnum = np.zeros((gh, gw), dtype=np.float64)
    wden = np.zeros((gh, gw), dtype=np.float64)
    ch2 = np.zeros((gh, gw), dtype=np.float64)
    active = np.isfinite(npe) & np.isfinite(tme) & (npe > npe_threshold)
    iy_a = iy[active]
    ix_a = ix[active]
    w = npe[active].astype(np.float64)
    t = tme[active].astype(np.float64)
    np.add.at(ch0, (iy_a, ix_a), w)
    np.add.at(wnum, (iy_a, ix_a), w * t)
    np.add.at(wden, (iy_a, ix_a), w)
    np.add.at(ch2, (iy_a, ix_a), 1.0)
    ch1 = np.zeros_like(ch0)
    m = wden > 1e-12
    ch1[m] = wnum[m] / wden[m]
    return ch0.astype(np.float32), ch1.astype(np.float32), ch2.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_h5", type=Path, required=True)
    p.add_argument("--output_h5", type=Path, required=True)
    p.add_argument("--grid_h", type=int, default=64)
    p.add_argument("--grid_w", type=int, default=64)
    p.add_argument("--plane", choices=["xy", "xz", "yz"], default="xy")
    p.add_argument("--auto_bounds", action="store_true", help="DOM 좌표 분위수로 투영 범위 자동 설정")
    p.add_argument("--lo_q", type=float, default=0.001)
    p.add_argument("--hi_q", type=float, default=0.999)
    p.add_argument("--pad_frac", type=float, default=0.02)
    p.add_argument("--u_min", type=float, default=None)
    p.add_argument("--u_max", type=float, default=None)
    p.add_argument("--v_min", type=float, default=None)
    p.add_argument("--v_max", type=float, default=None)
    p.add_argument("--npe_threshold", type=float, default=0.0)
    p.add_argument("--start", type=int, default=0, help="처리 시작 이벤트 인덱스")
    p.add_argument("--end", type=int, default=None, help="처리 끝(배타); 기본 전체")
    p.add_argument("--chunk", type=int, default=512, help="한 번에 쓰는 이벤트 청크 크기")
    args = p.parse_args()

    gh, gw = int(args.grid_h), int(args.grid_w)
    if gh < 2 or gw < 2:
        raise ValueError("grid_h, grid_w 는 2 이상이어야 합니다.")

    inp_path = args.input_h5.expanduser().resolve()
    out_path = args.output_h5.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(inp_path, "r") as fin:
        if "input" not in fin:
            raise KeyError(f"{inp_path} 에 'input' 데이터셋이 없습니다.")
        ds_in = fin["input"]
        if ds_in.ndim != 3 or ds_in.shape[1] != 2:
            raise ValueError(f"input shape 기대 (N,2,N_dom), 실제={ds_in.shape}")
        n_total = int(ds_in.shape[0])
        n_dom = int(ds_in.shape[2])

        for k in ("xpmt", "ypmt", "zpmt"):
            if k not in fin:
                raise KeyError(f"{inp_path} 에 '{k}' 가 없습니다.")
            if int(fin[k].shape[0]) != n_dom:
                raise ValueError(f"{k} 길이 {fin[k].shape[0]} != N_dom {n_dom}")

        xp = np.asarray(fin["xpmt"][:], dtype=np.float64)
        yp = np.asarray(fin["ypmt"][:], dtype=np.float64)
        zp = np.asarray(fin["zpmt"][:], dtype=np.float64)
        u, v = pick_coords((xp, yp, zp), args.plane)

        if args.auto_bounds:
            u0, u1, v0, v1 = auto_bounds(u, v, args.lo_q, args.hi_q, args.pad_frac)
        else:
            if None in (args.u_min, args.u_max, args.v_min, args.v_max):
                raise ValueError("--auto_bounds 가 아니면 u_min,u_max,v_min,v_max 를 모두 지정하세요.")
            u0, u1 = float(args.u_min), float(args.u_max)
            v0, v1 = float(args.v_min), float(args.v_max)

        ix, iy = digitize_2d(u, v, u0, u1, v0, v1, gw, gh)

        start = max(0, int(args.start))
        end = n_total if args.end is None else min(int(args.end), n_total)
        if start >= end:
            raise ValueError("start < end 가 되도록 조정하세요.")

        # 출력 파일 (덮어쓰기)
        if out_path.exists():
            out_path.unlink()

        with h5py.File(out_path, "w") as fout:
            fout.attrs["source_h5"] = str(inp_path)
            fout.attrs["plane"] = args.plane
            fout.attrs["grid_h"] = gh
            fout.attrs["grid_w"] = gw
            fout.attrs["u_min"], fout.attrs["u_max"] = u0, u1
            fout.attrs["v_min"], fout.attrs["v_max"] = v0, v1
            fout.attrs["ch0"] = "sum_npe"
            fout.attrs["ch1"] = "npe_weighted_mean_time"
            fout.attrs["ch2"] = "active_dom_count"
            fout.attrs["npe_threshold"] = float(args.npe_threshold)

            n_out = end - start
            ds_out = fout.create_dataset(
                "input",
                shape=(n_out, 3, gh, gw),
                dtype="float32",
                chunks=(min(256, n_out), 3, min(gh, 64), min(gw, 64)),
            )

            # 부가 데이터셋 복사 (동일 인덱스 구간)
            def copy_slice(name: str) -> None:
                if name not in fin:
                    return
                src = fin[name]
                sl = slice(start, end)
                arr = np.asarray(src[sl])
                fout.create_dataset(name, data=arr, compression="gzip", compression_opts=4)

            copy_slice("label")
            copy_slice("info")
            fout.create_dataset("xpmt", data=np.asarray(fin["xpmt"][:], dtype=np.float32))
            fout.create_dataset("ypmt", data=np.asarray(fin["ypmt"][:], dtype=np.float32))
            fout.create_dataset("zpmt", data=np.asarray(fin["zpmt"][:], dtype=np.float32))

            chunk = max(1, int(args.chunk))
            for a in range(start, end, chunk):
                b = min(a + chunk, end)
                block = np.asarray(ds_in[a:b], dtype=np.float32)  # (b-a, 2, n_dom)
                for k in range(block.shape[0]):
                    npe = block[k, 0, :]
                    tme = block[k, 1, :]
                    c0, c1, c2 = project_event(npe, tme, ix, iy, gh, gw, args.npe_threshold)
                    out_i = a - start + k
                    ds_out[out_i, 0] = c0
                    ds_out[out_i, 1] = c1
                    ds_out[out_i, 2] = c2
                print(f"[INFO] wrote events [{a}, {b}) / [{start}, {end})")

    print(f"[INFO] done: {out_path}  input shape = ({n_out}, 3, {gh}, {gw})")


if __name__ == "__main__":
    main()
