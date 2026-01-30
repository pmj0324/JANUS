#!/usr/bin/env python3
"""
이미 생성된 샘플 .npy 파일을 시각화하는 스크립트.
히스토그램, nPE/FirstTime cut 등을 적용해 이벤트 플롯과 히스토그램을 저장할 수 있음.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataloader.h5 import H5Dataset
from utils.vis.event_show import show_event_dual_plot


def _apply_cuts(sig: np.ndarray, cut_npe: float, cut_firsttime: float) -> np.ndarray:
    """cut_npe/cut_firsttime 이하 값을 0으로 만들어 시각화에서 보이지 않게 함. 0이면 변경 없음."""
    out = sig.copy()
    if cut_npe > 0:
        out[0] = np.where(out[0] <= cut_npe, 0.0, out[0])
    if cut_firsttime > 0:
        out[1] = np.where(out[1] <= cut_firsttime, 0.0, out[1])
    return out


def plot_histogram(
    sig: np.ndarray,
    output_path: Path,
    title_suffix: str = "",
    cut_npe: float = 0.0,
    cut_firsttime: float = 0.0,
):
    """nPE와 FirstTime 히스토그램 저장."""
    npe = sig[0]
    ftime = sig[1]
    npe_plot = npe[npe > cut_npe]
    ftime_plot = ftime[ftime > cut_firsttime]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    if len(npe_plot) > 0:
        ax1.hist(npe_plot, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('nPE')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'nPE Distribution{title_suffix}')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(npe_plot.mean(), color='red', linestyle='--', label=f'Mean: {npe_plot.mean():.2f}')
        ax1.axvline(np.median(npe_plot), color='green', linestyle='--', label=f'Median: {np.median(npe_plot):.2f}')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No nPE above cut', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'nPE Distribution{title_suffix} (empty)')

    ax2 = axes[1]
    if len(ftime_plot) > 0:
        ax2.hist(ftime_plot, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('FirstTime')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'FirstTime Distribution{title_suffix}')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(ftime_plot.mean(), color='red', linestyle='--', label=f'Mean: {ftime_plot.mean():.2f}')
        ax2.axvline(np.median(ftime_plot), color='green', linestyle='--', label=f'Median: {np.median(ftime_plot):.2f}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No FirstTime above cut', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'FirstTime Distribution{title_suffix} (empty)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved histogram: {output_path}")


def collect_npy_paths_from_one(npy_path: Path) -> list[Path]:
    """단일 파일 또는 디렉터리에서 .npy 경로 목록 반환."""
    npy_path = npy_path.resolve()
    if npy_path.is_file():
        if npy_path.suffix.lower() != '.npy':
            raise ValueError(f"Not a .npy file: {npy_path}")
        return [npy_path]
    if npy_path.is_dir():
        paths = sorted(npy_path.glob("*.npy"))
        if not paths:
            raise ValueError(f"No .npy files in directory: {npy_path}")
        return paths
    raise FileNotFoundError(f"Not found: {npy_path}")


def collect_npy_paths(npy_inputs: list[str]) -> list[Path]:
    """여러 경로(파일/디렉터리)에서 .npy 목록 수집. 순서 유지, 중복 제거."""
    seen = set()
    out = []
    for s in npy_inputs:
        p = Path(s)
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")
        for q in collect_npy_paths_from_one(p):
            r = q.resolve()
            if r not in seen:
                seen.add(r)
                out.append(q)
    if not out:
        raise ValueError("No .npy files found from given paths.")
    return out


# 3D 플롯용 geo/label은 고정 경로에서만 로드 (사용자 인자 아님)
_DEFAULT_H5 = "./GENESIS-data/22644_0921_time_shift.h5"
_REF_IDX = 0


def main():
    parser = argparse.ArgumentParser(
        description="샘플 .npy 파일 시각화 (이벤트 플롯, 히스토그램, cut)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--npy",
        type=str,
        nargs="+",
        required=True,
        help=".npy 파일 또는 .npy들이 있는 디렉터리 (여러 개 가능)",
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None,
        help="출력 디렉터리",
    )
    parser.add_argument(
        "-c", "--cut_npe",
        type=float,
        default=0.0,
        help="nPE cut (이하 미표시)",
    )
    parser.add_argument(
        "-f", "--cut_firsttime",
        type=float,
        default=0.0,
        help="FirstTime cut (이하 미표시)",
    )
    parser.add_argument(
        "-H", "--histogram",
        action="store_true",
        help="히스토그램도 저장",
    )
    parser.add_argument(
        "-p", "--prefix",
        type=str,
        default="vis",
        help="출력 파일 접두사",
    )
    args = parser.parse_args()

    npy_list = collect_npy_paths(args.npy)

    if args.output_dir is None:
        output_dir = npy_list[0].parent if len(npy_list) == 1 else Path(".")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cut_suffix = ""
    if args.cut_npe > 0 or args.cut_firsttime > 0:
        parts = [f"cutNpe{args.cut_npe}"] * (args.cut_npe > 0) + [f"cutFt{args.cut_firsttime}"] * (args.cut_firsttime > 0)
        cut_suffix = "_" + "_".join(parts)

    # 3D 플롯용 geo/label (고정 경로, 실패 시 플롯만 스킵)
    geo_np, label_np = None, None
    try:
        dataset = H5Dataset(h5_path=_DEFAULT_H5)
        _, geo_ref, label_ref = dataset[_REF_IDX]
        geo_np = geo_ref.numpy()
        label_np = label_ref.numpy()
    except Exception as e:
        print(f"Warning: geo/label 로드 실패 ({_DEFAULT_H5}), 3D 플롯 스킵. 히스토그램만 가능.")

    for i, p in enumerate(npy_list):
        name = p.stem
        sig = np.load(p)
        if sig.shape[0] != 2:
            print(f"  Skip {p.name}: shape {sig.shape}")
            continue
        print(f"[{i+1}/{len(npy_list)}] {p.name}")

        if geo_np is not None and label_np is not None:
            sig_vis = _apply_cuts(sig, args.cut_npe, args.cut_firsttime)
            img_path = output_dir / f"{args.prefix}{cut_suffix}_{name}.png"
            show_event_dual_plot(
                sig=sig_vis,
                geo=geo_np,
                label=label_np,
                output_path=str(img_path),
                figure_size=(18, 8),
                marker_size=8.0,
                show_detector_hull=True,
                show=False,
                title_prefix=f"sample_npy | {name} | cut_npe<={args.cut_npe} cut_ftime<={args.cut_firsttime}",
                firsttime_title="FirstTime",
                npe_title="nPE",
            )
            print(f"  Saved {img_path}")

        if args.histogram:
            hist_path = output_dir / f"{args.prefix}{cut_suffix}_{name}_histogram.png"
            plot_histogram(
                sig, hist_path,
                title_suffix=f" ({name})",
                cut_npe=args.cut_npe,
                cut_firsttime=args.cut_firsttime,
            )

    print("Done.")


if __name__ == "__main__":
    main()
