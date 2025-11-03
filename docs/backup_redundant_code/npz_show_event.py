#!/usr/bin/env python3
"""
npz-show-event.py

DEPRECATED: This module is maintained for backward compatibility only.
For new code, use utils.event_visualization.event_show instead:

    from utils.event_visualization.event_show import show_event_from_npz

NPZ의 한 이벤트(키: info, label, input)를 3D로 시각화.
- input[0,:] = NPE       → 구 크기
- input[1,:] = FirstTime → 색
- detector_geometry.csv 에서 x,y,z 좌표를 읽어 각 위치에 구를 그림.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def show_event(
    npz_path: str,
    detector_csv: str = "../configs/detector_geometry.csv",
    out_path: str | None = "./event.png",
    sphere_res: tuple[int, int] = (40, 20),
    base_radius: float = 5.0,
    radius_scale: float = 0.2,
    skip_nonfinite: bool = True,
    scatter_background: bool = True,
    figure_size: tuple[int, int] = (15, 10),
    separate_plots: bool = False,  # Create separate NPE and time plots
):
    """
    DEPRECATED: Use utils.event_visualization.event_show.show_event_from_npz instead.
    
    NPZ 파일(키: info, label, input)을 읽어 3D 시각화를 만듭니다.
    - detector_csv: x,y,z 컬럼이 있는 CSV 경로
    - out_path: 저장 경로 (None이면 저장하지 않음)
    Returns:
        (fig, ax): matplotlib Figure, Axes3D
    """
    import warnings
    warnings.warn(
        "utils.npz_show_event.show_event is deprecated. "
        "Use utils.event_visualization.event_show.show_event_from_npz instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # --- load geometry ---
    df_geo = pd.read_csv(detector_csv)
    x = np.asarray(df_geo["x"], dtype=np.float32)
    y = np.asarray(df_geo["y"], dtype=np.float32)
    z = np.asarray(df_geo["z"], dtype=np.float32)
    L = len(x)

    # --- load event npz ---
    with np.load(npz_path) as data:
        arr = data["input"]  # shape (2, L)
        label = data["label"]  # 필요시 사용
        print(label)
    assert arr.shape == (2, L), f"input shape must be (2,{L}), got {arr.shape}"

    energy, zenith, azimuth, x_pos, y_pos, z_pos = label

    npe   = arr[0, :].astype(np.float32)
    ftime = arr[1, :].astype(np.float32)

    # --- sanitize firstTime: ±inf → 0 ---
    ftime[np.isinf(ftime)] = 0.0

    # --- color scale: 0 제외하고 min/max 계산 ---
    nonzero_mask = (ftime != 0) & np.isfinite(ftime)
    if not nonzero_mask.any():
        vmin, vmax = 0.0, 1.0  # 모두 0이면 기본값
    else:
        vmin = float(np.min(ftime[nonzero_mask]))
        vmax = float(np.max(ftime[nonzero_mask]))
        if vmin == vmax:
            vmax = vmin + 1.0  # 모든 값이 같은 경우 대비

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps["jet"]

    # --- figure/axes ---
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection="3d")

    title_line1 = f"Energy = {energy:.3f}"
    title_line2 = f"Zenith = {zenith:.3f}, Azimuth = {azimuth:.3f}"
    title_line3 = f"X = {x_pos:.2f}, Y = {y_pos:.2f}, Z = {z_pos:.2f}"
    fig.suptitle(f"{title_line1}\n{title_line2}\n{title_line3}", fontsize=14, y=0.98)

    # --- detector hull (optional) ---
    edge_string_idx = [1, 6, 50, 74, 73, 78, 75, 31]
    top_xy, bottom_xy = [], []
    for i in edge_string_idx:
        top_xy.append([x[(i - 1) * 60],          y[(i - 1) * 60]])
        bottom_xy.append([x[(i - 1) * 60 + 59],  y[(i - 1) * 60 + 59]])
    top_xy.append(top_xy[0]); bottom_xy.append(bottom_xy[0])

    z_bottom, z_top = -500, 500
    for poly in (top_xy, bottom_xy):
        for i in range(len(poly) - 1):
            x0, y0 = poly[i]; x1, y1 = poly[i + 1]
            zc = z_top if poly is top_xy else z_bottom
            ax.plot([x0, x1], [y0, y1], [zc, zc], color="gray")
    for _x, _y in top_xy[:-1]:
        ax.plot([_x, _x], [_y, _y], [z_bottom, z_top], color="gray")

    # --- background dots ---
    if scatter_background:
        ax.scatter(x, y, z, s=1, c="gray", alpha=0.5)

    if separate_plots:
        # Create separate NPE and time plots
        return _create_separate_plots(
            x, y, z, npe, ftime, norm, cmap, 
            energy, zenith, azimuth, x_pos, y_pos, z_pos,
            out_path, figure_size, scatter_background
        )
    else:
        # Original combined plot (NPE as size, time as color)
        u_steps, v_steps = sphere_res
        u, v = np.mgrid[0:2 * np.pi:complex(u_steps), 0:np.pi:complex(v_steps)]

        # Count valid PMTs for progress
        valid_count = 0
        total_valid = np.sum((npe > 0) & np.isfinite(ftime) & (ftime != 0)) if skip_nonfinite else np.sum(npe > 0)
        
        for xi, yi, zi, ri, ci in zip(x, y, z, npe, ftime):
            # 0 값(치환된 inf 포함), 비유효값, 음수 NPE 등 스킵
            if skip_nonfinite and ((ri <= 0) or (not np.isfinite(ci)) or (ci == 0)):
                continue

            valid_count += 1

            radius = base_radius + radius_scale * (1.0 + ri)
            color = cmap(norm(ci))

            xs = radius * np.cos(u) * np.sin(v) + xi
            ys = radius * np.sin(u) * np.sin(v) + yi
            zs = radius * np.cos(v) + zi

            facecolor = np.ones(xs.shape + (4,), dtype=np.float32)
            facecolor[...] = color

            ax.plot_surface(
                xs, ys, zs,
                facecolors=facecolor,
                rstride=1, cstride=1,
                linewidth=0, antialiased=True,
                shade=True, alpha=0.9
            )
        
        print(f"📊 Rendered {valid_count}/{total_valid} PMTs")

    # --- colorbar (0 제외 범위로 생성) ---
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.373, 0.15, 0.3, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("firstTime (ns)", fontsize=16)

    # --- axes style ---
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_visible(False)
        axis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.dist = 3

    # --- save ---
    if out_path:
        fig.savefig(out_path, transparent=True, bbox_inches="tight")

    return fig, ax


def _create_separate_plots(
    x, y, z, npe, ftime, norm, cmap,
    energy, zenith, azimuth, x_pos, y_pos, z_pos,
    out_path, figure_size, scatter_background
):
    """Create separate NPE and time plots."""
    import time as time_module
    
    print("📊 Creating separate NPE and time plots...")
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(figure_size[0] * 2, figure_size[1]))
    
    # Common title
    title_line1 = f"Energy = {energy:.3f}"
    title_line2 = f"Zenith = {zenith:.3f}, Azimuth = {azimuth:.3f}"
    title_line3 = f"X = {x_pos:.2f}, Y = {y_pos:.2f}, Z = {z_pos:.2f}"
    fig.suptitle(f"{title_line1}\n{title_line2}\n{title_line3}", fontsize=14, y=0.98)
    
    # --- NPE Plot (Left) ---
    ax1 = fig.add_subplot(121, projection="3d")
    print("🎨 Creating NPE plot...")
    npe_start = time_module.perf_counter()
    
    # NPE color mapping
    npe_nonzero = npe[npe > 0]
    if len(npe_nonzero) > 0:
        npe_vmin = float(np.min(npe_nonzero))
        npe_vmax = float(np.max(npe_nonzero))
        if npe_vmin == npe_vmax:
            npe_vmax = npe_vmin + 1.0
    else:
        npe_vmin, npe_vmax = 0.0, 1.0
    
    npe_norm = Normalize(vmin=npe_vmin, vmax=npe_vmax)
    npe_cmap = colormaps["viridis"]
    
    # Detector hull for NPE plot
    _draw_detector_hull(ax1, x, y, z)
    
    # Background dots
    if scatter_background:
        ax1.scatter(x, y, z, s=1, c="gray", alpha=0.3)
    
    # NPE spheres (all PMTs, NPE as color)
    u_steps, v_steps = 20, 10  # Reduced resolution for speed
    u, v = np.mgrid[0:2 * np.pi:complex(u_steps), 0:np.pi:complex(v_steps)]
    
    valid_count = 0
    for xi, yi, zi, ri in zip(x, y, z, npe):
        if ri <= 0:
            continue
        
        valid_count += 1
        radius = 3.0 + 0.1 * (1.0 + ri)  # Fixed size, NPE as color
        color = npe_cmap(npe_norm(ri))
        
        xs = radius * np.cos(u) * np.sin(v) + xi
        ys = radius * np.sin(u) * np.sin(v) + yi
        zs = radius * np.cos(v) + zi
        
        facecolor = np.ones(xs.shape + (4,), dtype=np.float32)
        facecolor[...] = color
        
        ax1.plot_surface(
            xs, ys, zs,
            facecolors=facecolor,
            rstride=1, cstride=1,
            linewidth=0, antialiased=False,
            shade=True, alpha=0.8
        )
    
    # NPE colorbar
    sm1 = ScalarMappable(norm=npe_norm, cmap=npe_cmap)
    sm1.set_array([])
    cbar_ax1 = fig.add_axes([0.15, 0.15, 0.3, 0.02])
    cbar1 = fig.colorbar(sm1, cax=cbar_ax1, orientation="horizontal")
    cbar1.ax.tick_params(labelsize=12)
    cbar1.set_label("NPE", fontsize=14)
    
    ax1.set_title("NPE Distribution", fontsize=12, fontweight='bold')
    _style_axes(ax1)
    
    npe_time = time_module.perf_counter() - npe_start
    print(f"   ✅ NPE plot: {npe_time:.2f}s ({valid_count} PMTs)")
    
    # --- Time Plot (Right) ---
    ax2 = fig.add_subplot(122, projection="3d")
    print("🎨 Creating Time plot...")
    time_start = time_module.perf_counter()
    
    # Detector hull for Time plot
    _draw_detector_hull(ax2, x, y, z)
    
    # Background dots
    if scatter_background:
        ax2.scatter(x, y, z, s=1, c="gray", alpha=0.3)
    
    # Time spheres (all PMTs, time as color)
    valid_count = 0
    for xi, yi, zi, ci in zip(x, y, z, ftime):
        if not np.isfinite(ci) or ci == 0:
            continue
        
        valid_count += 1
        radius = 3.0 + 0.1  # Fixed size, time as color
        color = cmap(norm(ci))
        
        xs = radius * np.cos(u) * np.sin(v) + xi
        ys = radius * np.sin(u) * np.sin(v) + yi
        zs = radius * np.cos(v) + zi
        
        facecolor = np.ones(xs.shape + (4,), dtype=np.float32)
        facecolor[...] = color
        
        ax2.plot_surface(
            xs, ys, zs,
            facecolors=facecolor,
            rstride=1, cstride=1,
            linewidth=0, antialiased=False,
            shade=True, alpha=0.8
        )
    
    # Time colorbar
    sm2 = ScalarMappable(norm=norm, cmap=cmap)
    sm2.set_array([])
    cbar_ax2 = fig.add_axes([0.55, 0.15, 0.3, 0.02])
    cbar2 = fig.colorbar(sm2, cax=cbar_ax2, orientation="horizontal")
    cbar2.ax.tick_params(labelsize=12)
    cbar2.set_label("firstTime (ns)", fontsize=14)
    
    ax2.set_title("Time Distribution", fontsize=12, fontweight='bold')
    _style_axes(ax2)
    
    time_time = time_module.perf_counter() - time_start
    print(f"   ✅ Time plot: {time_time:.2f}s ({valid_count} PMTs)")
    
    # Save plots
    if out_path:
        base_path = out_path.rsplit('.', 1)[0] if '.' in out_path else out_path
        npe_path = f"{base_path}_npe.png"
        time_path = f"{base_path}_time.png"
        combined_path = f"{base_path}_combined.png"
        
        # Save individual plots
        fig1 = plt.figure(figsize=figure_size)
        ax1_copy = fig1.add_subplot(111, projection="3d")
        _copy_plot_content(ax1, ax1_copy, "NPE Distribution", npe_norm, npe_cmap, "NPE")
        fig1.savefig(npe_path, transparent=True, bbox_inches="tight", dpi=150)
        plt.close(fig1)
        
        fig2 = plt.figure(figsize=figure_size)
        ax2_copy = fig2.add_subplot(111, projection="3d")
        _copy_plot_content(ax2, ax2_copy, "Time Distribution", norm, cmap, "firstTime (ns)")
        fig2.savefig(time_path, transparent=True, bbox_inches="tight", dpi=150)
        plt.close(fig2)
        
        # Save combined plot
        fig.savefig(combined_path, transparent=True, bbox_inches="tight", dpi=150)
        
        print(f"📁 Saved: {npe_path}, {time_path}, {combined_path}")
    
    total_time = time_module.perf_counter() - npe_start
    print(f"✅ Separate plots completed in {total_time:.2f}s")
    
    return fig, (ax1, ax2)


def _draw_detector_hull(ax, x, y, z):
    """Draw detector hull outline."""
    edge_string_idx = [1, 6, 50, 74, 73, 78, 75, 31]
    top_xy, bottom_xy = [], []
    for i in edge_string_idx:
        top_xy.append([x[(i - 1) * 60], y[(i - 1) * 60]])
        bottom_xy.append([x[(i - 1) * 60 + 59], y[(i - 1) * 60 + 59]])
    top_xy.append(top_xy[0])
    bottom_xy.append(bottom_xy[0])

    z_bottom, z_top = -500, 500
    for poly in (top_xy, bottom_xy):
        for i in range(len(poly) - 1):
            x0, y0 = poly[i]
            x1, y1 = poly[i + 1]
            zc = z_top if poly is top_xy else z_bottom
            ax.plot([x0, x1], [y0, y1], [zc, zc], color="gray", alpha=0.7, linewidth=0.5)
    for _x, _y in top_xy[:-1]:
        ax.plot([_x, _x], [_y, _y], [z_bottom, z_top], color="gray", alpha=0.7, linewidth=0.5)


def _style_axes(ax):
    """Style 3D axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_visible(False)
        axis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.dist = 3


def _copy_plot_content(source_ax, target_ax, title, norm, cmap, label):
    """Copy plot content from source to target axis."""
    # This is a simplified version - in practice you'd need to copy all the plot elements
    target_ax.set_title(title, fontsize=12, fontweight='bold')
    _style_axes(target_ax)


# ---------------------------
# CLI entry
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Show one NPZ event in 3D")
    parser.add_argument("-i", "--input", required=True, help="Path to .npz (keys: info,label,input)")
    parser.add_argument("-g", "--geom", default="./configs/detector_geometry.csv", help="Detector geometry CSV (x,y,z)")
    parser.add_argument("-o", "--out",  default="./event.png", help="Output image path (png/pdf/svg...)")
    parser.add_argument("--separate", action="store_true", help="Create separate NPE and time plots")
    args = parser.parse_args()

    show_event(
        args.input, 
        detector_csv=args.geom, 
        out_path=args.out,
        separate_plots=args.separate
    )
    plt.tight_layout()
    plt.show()