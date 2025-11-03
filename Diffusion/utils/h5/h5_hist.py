#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
h5_hist.py - Beautiful Histogram Plotter for IceCube HDF5 Data
==============================================================

HDF5 파일의 input 데이터셋에서 charge와 time의 아름다운 히스토그램을 생성합니다.

Features:
- 아름다운 시각화 디자인
- 정확한 통계선과 범위 표시
- 로그 변환 지원 (log10, ln)
- 다양한 필터링 옵션
- 반응형 레이아웃
- 명확한 범례와 통계 정보

사용 예시:
================================================================

1. 기본 사용:
   python utils/h5_hist.py -p /path/to/data.h5

2. 로그 변환과 필터링:
   python utils/h5_hist.py -p /path/to/data.h5 --log-time log10 --exclude-zero --min-time 1000

3. 아름다운 스타일링:
   python utils/h5_hist.py -p /path/to/data.h5 --style modern --logy --bins 300

4. 완전 커스텀:
   python utils/h5_hist.py -p /path/to/data.h5 \
     --log-time ln --exclude-zero --min-time 5000 \
     --style elegant --figsize 16 10 --bins 250

CLI Options:
================================================================
-p, --path          : HDF5 파일 경로 (필수)
--bins              : 히스토그램 빈 개수 (기본: 200)
--chunk             : 스트리밍 배치 크기 (기본: 1024)
--range-charge      : Charge 수동 범위 (예: 0 50)
--range-time        : Time 수동 범위 (예: 0 20000)
--out               : 출력 파일 접두사 (기본: hist_input)
--logy              : Y축 로그 스케일
--logx              : X축 로그 스케일
--log-time          : Time 값 로그 변환 (log10, ln) - 기본값: ln
--exclude-zero      : 0 값을 제외한 히스토그램만 생성
--plot-both         : 모든 값과 0 제외 값 모두 플롯
--min-time          : Time 최소 임계값 (ns)
--style             : 스타일 선택 (modern, elegant, classic) - 기본값: modern
--figsize           : 그림 크기 (예: 12 8)
--pclip             : 자동 범위 퍼센타일 (기본: 0.5 99.5)

출력:
================================================================
- {out_prefix}_charge.png: Charge 히스토그램
- {out_prefix}_time[_log10/ln][_nonzero][_min{threshold}].png: Time 히스토그램
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 스타일 설정
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5
})

def apply_style(style: str = "modern"):
    """적용할 스타일 설정"""
    if style == "modern":
        plt.rcParams.update({
            'axes.facecolor': '#f8f9fa',
            'figure.facecolor': 'white',
            'grid.color': '#e0e0e0',
            'axes.edgecolor': '#333333',
            'xtick.color': '#666666',
            'ytick.color': '#666666',
        })
    elif style == "elegant":
        plt.rcParams.update({
            'axes.facecolor': '#fafafa',
            'figure.facecolor': '#ffffff',
            'grid.color': '#e8e8e8',
            'axes.edgecolor': '#2c2c2c',
            'xtick.color': '#555555',
            'ytick.color': '#555555',
        })
    elif style == "classic":
        plt.rcParams.update({
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'grid.color': '#cccccc',
            'axes.edgecolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
        })

def get_colors(style: str = "modern"):
    """스타일에 따른 색상 팔레트"""
    if style == "modern":
        return {
            'charge': '#2E86AB',
            'charge_edge': '#1A5F7A',
            'time': '#A23B72',
            'time_edge': '#7D2A5A',
            'mean': '#E63946',
            'median': '#F77F00',
            'std_fill': '#FFE5E5',
            'p10_p90': '#457B9D',
            'p25_p75': '#7209B7',
            'text_bg': '#FFFFFF',
            'text_border': '#E0E0E0'
        }
    elif style == "elegant":
        return {
            'charge': '#6C5CE7',
            'charge_edge': '#5F3DC4',
            'time': '#00B894',
            'time_edge': '#00A085',
            'mean': '#E17055',
            'median': '#FDCB6E',
            'std_fill': '#FFF5F5',
            'p10_p90': '#74B9FF',
            'p25_p75': '#A29BFE',
            'text_bg': '#FFFFFF',
            'text_border': '#DDD6FE'
        }
    else:  # classic
        return {
            'charge': '#1f77b4',
            'charge_edge': '#0d5aa7',
            'time': '#ff7f0e',
            'time_edge': '#cc5c00',
            'mean': '#d62728',
            'median': '#ff7f0e',
            'std_fill': '#ffe5e5',
            'p10_p90': '#2ca02c',
            'p25_p75': '#9467bd',
            'text_bg': '#FFFFFF',
            'text_border': '#CCCCCC'
        }

def calculate_percentile_range(dset, channel: int, chunk: int = 1024, 
                              sample_chunks: int = 8, p_low: float = 0.5, 
                              p_high: float = 99.5) -> Tuple[float, float]:
    """퍼센타일 기반 자동 범위 계산"""
    total_chunks = dset.shape[0] // chunk
    sample_size = min(sample_chunks, total_chunks)
    
    # 랜덤 샘플링
    chunk_indices = np.random.choice(total_chunks, size=sample_size, replace=False)
    
    values = []
    for i in chunk_indices:
        start_idx = i * chunk
        end_idx = min(start_idx + chunk, dset.shape[0])
        chunk_data = dset[start_idx:end_idx, channel, :].flatten()
        values.extend(chunk_data[~np.isnan(chunk_data)])
    
    if not values:
        return 0.0, 1.0
    
    values = np.array(values)
    lo, hi = np.percentile(values, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    
    return float(lo), float(hi)

def process_data_stream(dset, channel: int, bins: int, v_range: Tuple[float, float], 
                       chunk: int, exclude_zero: bool = False, 
                       min_threshold: Optional[float] = None,
                       log_transform: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """스트리밍 데이터 처리 및 히스토그램 생성"""
    
    # 히스토그램 엣지 생성
    edges = np.linspace(v_range[0], v_range[1], bins + 1)
    counts = np.zeros(bins, dtype=np.int64)
    
    # 통계 수집
    all_values = []
    zero_count = 0
    total_count = 0
    
    # 데이터 스트리밍 처리
    total_chunks = dset.shape[0] // chunk
    for i in range(total_chunks):
        start_idx = i * chunk
        end_idx = min(start_idx + chunk, dset.shape[0])
        
        # 데이터 로드
        x = dset[start_idx:end_idx, channel, :].flatten()
        x = x[~np.isnan(x)]  # NaN 제거
        
        total_count += len(x)
        
        # 0 제외 옵션
        if exclude_zero:
            zero_mask = (x == 0)
            zero_count += np.sum(zero_mask)
            x = x[~zero_mask]
        
        # 최소 임계값 적용
        if min_threshold is not None:
            x = x[x >= min_threshold]
        
        # 로그 변환 적용
        if log_transform is not None and len(x) > 0:
            if log_transform == "log10":
                x = np.log10(x + 1e-10)
            elif log_transform == "ln":
                x = np.log(x + 1e-10)
        
        # 통계 수집 (샘플링으로 메모리 절약)
        if len(all_values) < 100000:
            all_values.extend(x.tolist())
        
        # 히스토그램 계산
        if len(x) > 0 and edges[0] < edges[-1]:
            x_clipped = np.clip(x, edges[0], edges[-1])
            c, _ = np.histogram(x_clipped, bins=edges)
        counts += c
    
    # 빈 중심점 계산
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    # 통계 계산
    if all_values:
        all_values = np.array(all_values)
        stats_dict = {
            'count': len(all_values),
            'zero_count': zero_count,
            'total_count': total_count,
            'zero_fraction': zero_count / total_count if total_count > 0 else 0,
            'min_threshold': min_threshold,
            'transform': log_transform,
            'mean': np.mean(all_values),
            'std': np.std(all_values),
            'median': np.median(all_values),
            'min': np.min(all_values),
            'max': np.max(all_values),
            'p10': np.percentile(all_values, 10),
            'p25': np.percentile(all_values, 25),
            'p75': np.percentile(all_values, 75),
            'p90': np.percentile(all_values, 90),
        }
    else:
        stats_dict = {
            'count': 0, 'zero_count': 0, 'total_count': 0, 'zero_fraction': 0,
            'min_threshold': min_threshold, 'transform': log_transform,
            'mean': 0, 'std': 0, 'median': 0, 'min': 0, 'max': 0,
            'p10': 0, 'p25': 0, 'p75': 0, 'p90': 0
        }
    
    return centers, counts, stats_dict

def create_beautiful_histogram(x, y, stats, title, xlabel, colors, 
                              logy=False, logx=False, figsize=(12, 8),
                              show_stats=True, show_percentiles=True):
    """아름다운 히스토그램 생성"""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 데이터가 있는지 확인
    if len(x) == 0 or np.sum(y) == 0:
        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                ha='center', va='center', fontsize=16, color='red',
                bbox=dict(boxstyle='round,pad=1', facecolor='white', edgecolor='red'))
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        return fig, ax
    
    # 히스토그램 플롯
    if len(x) > 0 and np.sum(y) > 0:
        # 막대 히스토그램
        bars = ax.bar(x, y, width=x[1]-x[0] if len(x) > 1 else 1, 
                     alpha=0.8, color=colors['charge'], 
                     edgecolor=colors['charge_edge'], linewidth=0.5)
        
        # 그라데이션 효과 (선택적)
        for i, bar in enumerate(bars):
            bar.set_alpha(0.8 - (i % 3) * 0.1)
    
    # 통계선 추가
    if show_stats and stats['count'] > 0:
        # 평균선
        ax.axvline(stats['mean'], color=colors['mean'], linestyle='--', 
                  linewidth=2.5, alpha=0.8, label=f'Mean: {stats["mean"]:.2f}')
        
        # 중앙값선
        ax.axvline(stats['median'], color=colors['median'], linestyle=':', 
                  linewidth=2.5, alpha=0.8, label=f'Median: {stats["median"]:.2f}')
        
        # 표준편차 영역
        std_low = stats['mean'] - stats['std']
        std_high = stats['mean'] + stats['std']
        ax.axvspan(std_low, std_high, alpha=0.2, color=colors['std_fill'],
                  label=f'±1σ: [{std_low:.2f}, {std_high:.2f}]')
    
    # 백분위수선 추가
    if show_percentiles and stats['count'] > 0:
        ax.axvline(stats['p10'], color=colors['p10_p90'], linestyle='--', 
                  linewidth=1.5, alpha=0.7, label=f'P10: {stats["p10"]:.2f}')
        ax.axvline(stats['p90'], color=colors['p10_p90'], linestyle='--', 
                  linewidth=1.5, alpha=0.7, label=f'P90: {stats["p90"]:.2f}')
        ax.axvline(stats['p25'], color=colors['p25_p75'], linestyle=':', 
                  linewidth=1.5, alpha=0.7, label=f'P25: {stats["p25"]:.2f}')
        ax.axvline(stats['p75'], color=colors['p25_p75'], linestyle=':', 
                  linewidth=1.5, alpha=0.7, label=f'P75: {stats["p75"]:.2f}')
    
    # 축 설정
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 로그 스케일 적용
    if logy:
        ax.set_yscale("log")
        ax.set_ylabel("Count (log scale)", fontsize=12, fontweight='bold')
    if logx:
        ax.set_xscale("log")
    
    # 범례
    if show_stats or show_percentiles:
        legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                          fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
    
    # 통계 정보 박스
    if stats['count'] > 0:
        info_lines = [
            f"Count: {stats['count']:,}",
            f"Range: [{stats['min']:.3f}, {stats['max']:.3f}]",
            f"Mean: {stats['mean']:.3f} ± {stats['std']:.3f}",
            f"Median: {stats['median']:.3f}"
        ]
        
        if stats['zero_fraction'] > 0:
            info_lines.append(f"Zeros: {stats['zero_count']:,} ({stats['zero_fraction']:.1%})")
        
        if stats['min_threshold'] is not None:
            info_lines.append(f"Min Threshold: {stats['min_threshold']:.1f}")
        
        if stats['transform'] is not None:
            info_lines.append(f"Transform: {stats['transform']}")
        
        info_text = "\n".join(info_lines)
        
        # 아름다운 텍스트 박스
        textbox = ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                         verticalalignment='top', fontsize=10, fontweight='normal',
                         bbox=dict(boxstyle='round,pad=0.8', 
                                 facecolor=colors['text_bg'], 
                                 edgecolor=colors['text_border'],
                                 alpha=0.95, linewidth=1.5))
    
    # 그리드 및 레이아웃
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # X축 범위 자동 조정
    if len(x) > 0 and np.sum(y) > 0:
        data_min = np.min(x[y > 0])
        data_max = np.max(x[y > 0])
        margin = (data_max - data_min) * 0.02  # 2% 마진
        # 로그 변환된 경우 음수 방지
        x_min = max(data_min - margin, data_min * 0.95) if data_min > 0 else data_min - margin
        ax.set_xlim(x_min, data_max + margin)
    
    plt.tight_layout()
    return fig, ax

def plot_hist_pair(h5_path: str, bins: int = 200, chunk: int = 1024,
    range_charge: Optional[Tuple[float, float]] = None,
    range_time: Optional[Tuple[float, float]] = None,
                   out_prefix: str = "hist_input", logy: bool = False,
                   logx: bool = False, pclip: Tuple[float, float] = (0.5, 99.5),
                   show_stats: bool = True, show_percentiles: bool = True,
                   figsize: Tuple[int, int] = (12, 8), exclude_zero: bool = False,
                   plot_both: bool = False, min_time_threshold: Optional[float] = None,
                   log_time_transform: Optional[str] = None, style: str = "modern"):
    """메인 히스토그램 플롯 함수"""
    
    # 스타일 적용
    apply_style(style)
    colors = get_colors(style)

    with h5py.File(h5_path, "r") as f:
        if "input" not in f:
            raise KeyError("Dataset 'input' not found")

        dset = f["input"]

        # 자동 범위 계산
        if range_charge is None:
            range_charge = calculate_percentile_range(dset, 0, chunk, 8, pclip[0], pclip[1])
        if range_time is None:
            range_time = calculate_percentile_range(dset, 1, chunk, 8, pclip[0], pclip[1])
            
        # 임계값이 있으면 범위 조정
        if min_time_threshold is not None:
            range_time = (min_time_threshold, max(range_time[1], min_time_threshold + 1))
        
        # 로그 변환이 적용될 경우 범위도 변환
        original_range_time = range_time
        if log_time_transform is not None:
            if log_time_transform == "log10":
                range_time = (np.log10(range_time[0] + 1e-10), np.log10(range_time[1] + 1e-10))
            elif log_time_transform == "ln":
                range_time = (np.log(range_time[0] + 1e-10), np.log(range_time[1] + 1e-10))
            print(f"🔄 Time range transformation:")
            print(f"   Original: [{original_range_time[0]:.1f}, {original_range_time[1]:.1f}]")
            print(f"   {log_time_transform} transformed: [{range_time[0]:.3f}, {range_time[1]:.3f}]")
        
        # Charge 히스토그램
        x_c, y_c, stats_c = process_data_stream(dset, 0, bins, range_charge, chunk, 
                                               exclude_zero, None, None)
        
        # Time 히스토그램
        x_t, y_t, stats_t = process_data_stream(dset, 1, bins, range_time, chunk,
                                               exclude_zero, min_time_threshold, log_time_transform)
    
    # Charge 히스토그램 생성
    charge_title = "Charge Distribution"
    if exclude_zero:
        charge_title += " (Non-zero only)"
    
    charge_xlabel = "Charge (NPE)"
    
    fig_c, ax_c = create_beautiful_histogram(
        x_c, y_c, stats_c, charge_title, charge_xlabel, colors,
        logy, logx, figsize, show_stats, show_percentiles
    )
    
    # 파일 저장
    suffix = "_nonzero" if exclude_zero else ""
    plt.savefig(f"{out_prefix}_charge{suffix}.png", dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig_c)
    
    # Time 히스토그램 생성
    time_title = "Time Distribution"
    if exclude_zero:
        time_title += " (Non-zero only)"
    if min_time_threshold is not None:
        time_title += f" (≥{min_time_threshold:.1f}ns)"
    if log_time_transform is not None:
        transform_name = "log₁₀" if log_time_transform == "log10" else "ln"
        time_title += f" [{transform_name} transformed]"
    
    time_xlabel = "Time (ns)"
    if log_time_transform == "log10":
        time_xlabel = "log₁₀(Time + ε) (ns)"
    elif log_time_transform == "ln":
        time_xlabel = "ln(Time + ε) (ns)"
    
    fig_t, ax_t = create_beautiful_histogram(
        x_t, y_t, stats_t, time_title, time_xlabel, colors,
        logy, logx, figsize, show_stats, show_percentiles
    )
    
    # 파일 저장
    suffix = "_nonzero" if exclude_zero else ""
    if min_time_threshold is not None:
        suffix += f"_min{min_time_threshold:.0f}"
    if log_time_transform is not None:
        suffix += f"_{log_time_transform}"
    
    plt.savefig(f"{out_prefix}_time{suffix}.png", dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig_t)
    
    # 통계 정보 출력
    print(f"\n🎨 Style: {style.title()}")
    print(f"📊 Charge Statistics{' (Non-zero only)' if exclude_zero else ''}:")
    if stats_c['count'] > 0:
        print(f"  Count: {stats_c['count']:,}")
        print(f"  Range: [{stats_c['min']:.3f}, {stats_c['max']:.3f}]")
        print(f"  Mean ± Std: {stats_c['mean']:.3f} ± {stats_c['std']:.3f}")
        print(f"  Median: {stats_c['median']:.3f}")
        print(f"  Percentiles: P10={stats_c['p10']:.3f}, P90={stats_c['p90']:.3f}")
        if stats_c['zero_fraction'] > 0:
            print(f"  Zeros: {stats_c['zero_count']:,} ({stats_c['zero_fraction']:.1%})")
    else:
        print("  No data available")
    
    print(f"\n⏱️  Time Statistics{' (Non-zero only)' if exclude_zero else ''}{' (≥' + str(min_time_threshold) + 'ns)' if min_time_threshold else ''}{' [' + log_time_transform + ' transformed]' if log_time_transform else ''}:")
    if stats_t['count'] > 0:
        print(f"  Count: {stats_t['count']:,}")
        print(f"  Range: [{stats_t['min']:.3f}, {stats_t['max']:.3f}]")
        print(f"  Mean ± Std: {stats_t['mean']:.3f} ± {stats_t['std']:.3f}")
        print(f"  Median: {stats_t['median']:.3f}")
        print(f"  Percentiles: P10={stats_t['p10']:.3f}, P90={stats_t['p90']:.3f}")
        if stats_t['zero_fraction'] > 0:
            print(f"  Zeros: {stats_t['zero_count']:,} ({stats_t['zero_fraction']:.1%})")
        if min_time_threshold is not None:
            print(f"  Min Threshold: {min_time_threshold:.1f}ns")
        if log_time_transform is not None:
            print(f"  Log Transform: {log_time_transform}")
    else:
        print("  No data available")

def main():
    parser = argparse.ArgumentParser(description="Create beautiful histograms for IceCube HDF5 data")
    parser.add_argument("-p", "--path", required=True, help="Path to HDF5 file")
    parser.add_argument("--bins", type=int, default=200, help="Number of histogram bins")
    parser.add_argument("--chunk", type=int, default=1024, help="Chunk size for streaming")
    parser.add_argument("--range-charge", type=float, nargs=2, metavar=("MIN", "MAX"),
                       help="Manual range for charge")
    parser.add_argument("--range-time", type=float, nargs=2, metavar=("MIN", "MAX"),
                       help="Manual range for time")
    parser.add_argument("--out", default="hist_input", help="Output file prefix")
    parser.add_argument("--logy", action="store_true", help="Use log-scale on y-axis")
    parser.add_argument("--logx", action="store_true", help="Use log-scale on x-axis")
    parser.add_argument("--log-time", choices=["log10", "ln"], default="ln",
                       help="Log transformation for time values")
    parser.add_argument("--no-stats", action="store_true", help="Hide statistical lines")
    parser.add_argument("--no-percentiles", action="store_true", help="Hide percentile lines")
    parser.add_argument("--figsize", type=int, nargs=2, default=(12, 8), metavar=("WIDTH", "HEIGHT"),
                       help="Figure size")
    parser.add_argument("--exclude-zero", action="store_true", help="Exclude zero values")
    parser.add_argument("--plot-both", action="store_true", help="Plot both all and non-zero values")
    parser.add_argument("--min-time", type=float, help="Minimum time threshold (ns)")
    parser.add_argument("--style", choices=["modern", "elegant", "classic"], default="modern",
                       help="Visual style")
    parser.add_argument("--pclip", type=float, nargs=2, default=(0.5, 99.5),
                       metavar=("LOW", "HIGH"), help="Percentiles for auto-range")
    
    args = parser.parse_args()
    
    # 기본 플롯 (모든 값 포함)
    if not args.exclude_zero or args.plot_both:
        plot_hist_pair(
            h5_path=args.path,
            bins=args.bins,
            chunk=args.chunk,
            range_charge=tuple(args.range_charge) if args.range_charge else None,
            range_time=tuple(args.range_time) if args.range_time else None,
            out_prefix=args.out,
            logy=args.logy,
            logx=args.logx,
            show_stats=not args.no_stats,
            show_percentiles=not args.no_percentiles,
            figsize=tuple(args.figsize),
            exclude_zero=False,
            plot_both=args.plot_both,
            min_time_threshold=args.min_time,
            log_time_transform=args.log_time,
            style=args.style,
            pclip=tuple(args.pclip),
        )
    
    # 0 제외 플롯
    if args.exclude_zero or args.plot_both:
        plot_hist_pair(
        h5_path=args.path,
        bins=args.bins,
        chunk=args.chunk,
        range_charge=tuple(args.range_charge) if args.range_charge else None,
        range_time=tuple(args.range_time) if args.range_time else None,
        out_prefix=args.out,
        logy=args.logy,
            logx=args.logx,
            show_stats=not args.no_stats,
            show_percentiles=not args.no_percentiles,
            figsize=tuple(args.figsize),
            exclude_zero=True,
            plot_both=args.plot_both,
            min_time_threshold=args.min_time,
            log_time_transform=args.log_time,
            style=args.style,
        pclip=tuple(args.pclip),
    )
    
    # 출력 파일 목록
    time_suffix = ""
    if args.min_time is not None:
        time_suffix += f"_min{args.min_time:.0f}"
    if args.log_time:
        time_suffix += f"_{args.log_time}"
    
    if args.plot_both:
        print(f"\n✅ Saved: {args.out}_charge.png, {args.out}_time{time_suffix}.png, "
              f"{args.out}_charge_nonzero.png, {args.out}_time_nonzero{time_suffix}.png")
    elif args.exclude_zero:
        print(f"\n✅ Saved: {args.out}_charge_nonzero.png, {args.out}_time_nonzero{time_suffix}.png")
    else:
        print(f"\n✅ Saved: {args.out}_charge.png, {args.out}_time{time_suffix}.png")

if __name__ == "__main__":
    main()