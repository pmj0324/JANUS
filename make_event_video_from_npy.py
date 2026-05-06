#!/usr/bin/env python3
"""Create PPT-friendly event videos from denormalized real/generated NPY files.

Each NPY is expected to contain one event signal with shape (2, L):
  - sig[0]: nPE
  - sig[1]: firstTime

The animation shows PMTs appearing over firstTime. Marker size represents nPE,
and marker color can represent nPE (default) or firstTime.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

from utils.vis.event_show import _draw_detector_hull


DEFAULT_H5_PATH = "GENESIS-data/22644_0921_time_shift.h5"


def load_event_npy(path: str | Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[1] == 2:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[0] != 2 and arr.shape[1] == 2:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"Expected NPY shape (2, L), got {arr.shape} from {path}")
    return arr


def load_geometry(
    h5_path: str | Path,
    *,
    x_key: str = "xpmt",
    y_key: str = "ypmt",
    z_key: str = "zpmt",
) -> np.ndarray:
    with h5py.File(str(Path(h5_path).expanduser()), "r") as f:
        geo = np.stack(
            [
                np.asarray(f[x_key][...], dtype=np.float32),
                np.asarray(f[y_key][...], dtype=np.float32),
                np.asarray(f[z_key][...], dtype=np.float32),
            ],
            axis=0,
        )
    if geo.ndim != 2 or geo.shape[0] != 3:
        raise ValueError(f"Expected geometry shape (3, L), got {geo.shape}")
    return geo


def clean_event(sig: np.ndarray, *, cut_npe: float, cut_firsttime: float) -> dict:
    npe = np.asarray(sig[0], dtype=np.float32).copy()
    ftime = np.asarray(sig[1], dtype=np.float32).copy()
    npe[~np.isfinite(npe)] = 0.0
    ftime[~np.isfinite(ftime)] = 0.0

    hit_mask = (npe > float(cut_npe)) & (ftime > float(cut_firsttime))
    return {
        "npe": npe,
        "ftime": ftime,
        "hit_mask": hit_mask,
        "hit_count": int(np.count_nonzero(hit_mask)),
        "npe_sum": float(np.sum(npe[hit_mask], dtype=np.float64)),
    }


def compute_shared_ranges(events: Iterable[dict], args: argparse.Namespace) -> dict:
    time_vals = []
    npe_vals = []
    for event in events:
        mask = event["hit_mask"]
        if np.any(mask):
            time_vals.append(event["ftime"][mask])
            npe_vals.append(event["npe"][mask])

    pct_lo = float(getattr(args, "clip_pct_low", 1.0))
    pct_hi = float(getattr(args, "clip_pct_high", 99.0))

    if time_vals:
        all_time = np.concatenate(time_vals)
        time_min = float(np.percentile(all_time, pct_lo)) if args.time_min is None else float(args.time_min)
        time_max = float(np.percentile(all_time, pct_hi)) if args.time_max is None else float(args.time_max)
    else:
        time_min = 0.0 if args.time_min is None else float(args.time_min)
        time_max = 1.0 if args.time_max is None else float(args.time_max)

    if not time_max > time_min:
        time_max = time_min + 1.0

    if npe_vals:
        all_npe = np.concatenate(npe_vals)
        npe_max = float(np.percentile(all_npe, pct_hi)) if args.npe_max is None else float(args.npe_max)
    else:
        npe_max = 1.0 if args.npe_max is None else float(args.npe_max)
    npe_max = max(npe_max, 1e-6)

    return {
        "time_min": time_min,
        "time_max": time_max,
        "npe_max": npe_max,
    }


def set_axes_equal(ax, geo: np.ndarray, pad_fraction: float = 0.05) -> None:
    x, y, z = geo
    mins = np.array([np.nanmin(x), np.nanmin(y), np.nanmin(z)], dtype=np.float64)
    maxs = np.array([np.nanmax(x), np.nanmax(y), np.nanmax(z)], dtype=np.float64)
    centers = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    span = span * (1.0 + float(pad_fraction))
    half = 0.5 * span
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def style_3d_axes(ax) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_alpha(0.0)
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            axis.line.set_color((1.0, 1.0, 1.0, 0.0))
        except Exception:
            pass
    ax.set_axis_off()


def scale_marker_sizes(npe: np.ndarray, npe_max: float, marker_size: float) -> np.ndarray:
    scaled = np.sqrt(np.clip(npe / float(npe_max), 0.0, 1.0))
    return float(marker_size) * (0.25 + 4.75 * scaled)


def visible_mask_for_time(
    event: dict,
    frame_time: float,
    *,
    mode: str,
    window_width: float | None,
) -> np.ndarray:
    hit = event["hit_mask"]
    ftime = event["ftime"]
    if mode == "cumulative":
        return hit & (ftime <= float(frame_time))

    width = float(window_width) if window_width is not None else 0.0
    if width <= 0.0:
        valid_times = ftime[hit]
        width = max(float(np.ptp(valid_times)) / 30.0, 1.0) if valid_times.size else 1.0
    half = 0.5 * width
    return hit & (ftime >= frame_time - half) & (ftime <= frame_time + half)


def figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        return rgba[:, :, :3].copy()
    if hasattr(fig.canvas, "tostring_rgb"):
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape((h, w, 3)).copy()
    raise RuntimeError("Could not read RGB pixels from matplotlib canvas")


def find_content_box(frame: np.ndarray, *, padding: int, tolerance: int = 4) -> tuple[int, int, int, int]:
    rgb = np.asarray(frame, dtype=np.uint8)
    corners = np.array([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], dtype=np.float32)
    background = np.median(corners, axis=0)
    diff = np.max(np.abs(rgb.astype(np.float32) - background), axis=2)
    mask = diff > float(tolerance)
    if not np.any(mask):
        h, w, _ = rgb.shape
        return 0, h, 0, w

    rows, cols = np.where(mask)
    h, w, _ = rgb.shape
    pad = max(0, int(padding))
    y0 = max(0, int(rows.min()) - pad)
    y1 = min(h, int(rows.max()) + pad + 1)
    x0 = max(0, int(cols.min()) - pad)
    x1 = min(w, int(cols.max()) + pad + 1)
    return y0, y1, x0, x1


def crop_to_box(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = box
    return frame[y0:y1, x0:x1].copy()


def ensure_even_rgb(frame: np.ndarray) -> np.ndarray:
    h, w, c = frame.shape
    if c != 3:
        raise ValueError(f"Expected RGB frame, got shape {frame.shape}")
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


class MP4PipeWriter:
    def __init__(self, output_path: Path, fps: float, crf: int, preset: str) -> None:
        self.output_path = Path(output_path)
        self.fps = float(fps)
        self.crf = int(crf)
        self.preset = str(preset)
        self.proc: subprocess.Popen | None = None
        self.frame_shape: tuple[int, int, int] | None = None

    def _start(self, frame: np.ndarray) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg was not found. Install ffmpeg or use --format gif.")
        h, w, _ = frame.shape
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{w}x{h}",
            "-pix_fmt",
            "rgb24",
            "-r",
            f"{self.fps:g}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(self.crf),
            "-preset",
            self.preset,
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.frame_shape = frame.shape

    def append(self, frame: np.ndarray) -> None:
        frame = ensure_even_rgb(np.asarray(frame, dtype=np.uint8))
        if self.proc is None:
            self._start(frame)
        if frame.shape != self.frame_shape:
            raise ValueError(f"Frame shape changed from {self.frame_shape} to {frame.shape}")
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.proc is None:
            return
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        stderr = self.proc.stderr.read().decode("utf-8", errors="replace") if self.proc.stderr else ""
        ret = self.proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {ret}:\n{stderr[-4000:]}")


class GIFWriter:
    def __init__(self, output_path: Path, fps: float) -> None:
        try:
            import imageio
        except ImportError as exc:
            raise RuntimeError("GIF output requires imageio. Install with: pip install imageio") from exc
        self.imageio = imageio
        self.output_path = Path(output_path)
        self.writer = imageio.get_writer(str(self.output_path), mode="I", duration=1.0 / float(fps))

    def append(self, frame: np.ndarray) -> None:
        self.writer.append_data(np.asarray(frame, dtype=np.uint8))

    def close(self) -> None:
        self.writer.close()


def make_writers(output_base: Path, output_format: str, fps: float, crf: int, preset: str):
    writers = []
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_format.lower()
    if fmt in {"mp4", "both"}:
        writers.append((output_base.with_suffix(".mp4"), MP4PipeWriter(output_base.with_suffix(".mp4"), fps, crf, preset)))
    if fmt in {"gif", "both"}:
        writers.append((output_base.with_suffix(".gif"), GIFWriter(output_base.with_suffix(".gif"), fps)))
    return writers


def render_frame(
    *,
    event: dict,
    geo: np.ndarray,
    label: str,
    frame_time: float,
    frame_index: int,
    num_frames: int,
    ranges: dict,
    args: argparse.Namespace,
) -> np.ndarray:
    fig = plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi), dpi=args.dpi)
    fig.patch.set_facecolor(args.background)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(args.background)

    x, y, z = geo
    if args.show_detector_hull:
        _draw_detector_hull(ax, x, y, z, alpha=0.22, linewidth=1.2)
    if args.show_background:
        ax.scatter(
            x,
            y,
            z,
            s=args.background_size,
            c=args.background_color,
            alpha=args.background_alpha,
            depthshade=False,
            edgecolors="none",
        )

    visible = visible_mask_for_time(
        event,
        frame_time,
        mode=args.mode,
        window_width=args.window_width,
    )
    npe_visible = event["npe"][visible]
    ftime_visible = event["ftime"][visible]

    if args.color_by == "firsttime":
        color_values = ftime_visible
        norm = Normalize(vmin=ranges["time_min"], vmax=ranges["time_max"])
        cbar_label = "firstTime"
    else:
        color_values = npe_visible
        norm = Normalize(vmin=0.0, vmax=ranges["npe_max"])
        cbar_label = "nPE"

    if np.any(visible):
        ax.scatter(
            x[visible],
            y[visible],
            z[visible],
            c=color_values,
            s=scale_marker_sizes(npe_visible, ranges["npe_max"], args.marker_size),
            cmap=args.cmap,
            norm=norm,
            alpha=args.hit_alpha,
            depthshade=False,
            edgecolors="none",
        )

    if args.show_colorbar:
        sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(args.cmap))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.55, aspect=30, pad=-0.08)
        cbar.set_label(cbar_label, labelpad=6)

    set_axes_equal(ax, geo)
    ax.view_init(elev=args.elev, azim=args.azim)
    style_3d_axes(ax)

    if args.show_overlay:
        total_hits = max(event["hit_count"], 1)
        shown_hits = int(np.count_nonzero(visible))
        if args.mode == "cumulative":
            time_text = f"t <= {frame_time:.1f}"
        else:
            width = args.window_width if args.window_width is not None else "auto"
            time_text = f"t = {frame_time:.1f}, window={width}"

        ax.text2D(
            0.02,
            0.96,
            f"{label}\n{time_text}\nhits {shown_hits}/{total_hits}   nPE sum {event['npe_sum']:.1f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=args.title_size,
            color=args.text_color,
        )
    if args.show_frame_counter:
        ax.text2D(
            0.02,
            0.04,
            f"frame {frame_index + 1}/{num_frames}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=max(8, args.title_size - 2),
            color=args.text_color,
            alpha=0.78,
        )

    frame = figure_to_rgb(fig)
    plt.close(fig)
    return frame


def render_event_video(
    *,
    event: dict,
    geo: np.ndarray,
    label: str,
    output_base: Path,
    ranges: dict,
    args: argparse.Namespace,
) -> list[Path]:
    writers = make_writers(output_base, args.format, args.fps, args.crf, args.preset)
    saved_paths = [path for path, _writer in writers]
    hit_ftimes = event["ftime"][event["hit_mask"]]
    actual_min = float(np.min(hit_ftimes)) if hit_ftimes.size > 0 else ranges["time_min"]
    time_span = ranges["time_max"] - actual_min
    empty_start = actual_min - time_span * 0.05
    times = np.linspace(empty_start, ranges["time_max"], int(args.num_frames), dtype=np.float64)
    tail_frames = max(0, int(round(float(args.tail_seconds) * float(args.fps))))
    crop_box = None
    if args.tight_detector:
        # Probe frame always uses cumulative mode so all PMT positions are visible,
        # giving a stable crop box regardless of the playback mode.
        crop_args = argparse.Namespace(**vars(args))
        crop_args.mode = "cumulative"
        crop_args.window_width = None
        probe_frame = render_frame(
            event=event,
            geo=geo,
            label=label,
            frame_time=ranges["time_max"],
            frame_index=max(len(times) - 1, 0),
            num_frames=len(times),
            ranges=ranges,
            args=crop_args,
        )
        crop_box = find_content_box(probe_frame, padding=args.tight_padding)

    try:
        last_frame = None
        for frame_index, frame_time in enumerate(times):
            frame = render_frame(
                event=event,
                geo=geo,
                label=label,
                frame_time=float(frame_time),
                frame_index=frame_index,
                num_frames=len(times),
                ranges=ranges,
                args=args,
            )
            if args.tight_detector:
                assert crop_box is not None
                frame = crop_to_box(frame, crop_box)
            last_frame = frame
            for _path, writer in writers:
                writer.append(frame)
            if args.progress and (frame_index == 0 or (frame_index + 1) % args.progress_every == 0 or frame_index + 1 == len(times)):
                print(f"  {label}: frame {frame_index + 1}/{len(times)}")

        if last_frame is not None:
            for _ in range(tail_frames):
                for _path, writer in writers:
                    writer.append(last_frame)
    finally:
        for _path, writer in writers:
            writer.close()

    return saved_paths


def build_output_base(out_dir: Path, kind: str, npy_path: Path) -> Path:
    safe_stem = npy_path.stem.replace(" ", "_")
    return out_dir / f"{kind}_{safe_stem}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create firstTime/nPE event videos from real and generated NPY files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real_npy", type=str, default=None, help="Real event .npy path")
    parser.add_argument("--generated_npy", "--gen_npy", dest="generated_npy", type=str, default=None, help="Generated event .npy path")
    parser.add_argument("--h5_path", type=str, default=DEFAULT_H5_PATH, help="H5 file used only for PMT geometry")
    parser.add_argument("--out_dir", type=str, default="event_videos", help="Output directory")
    parser.add_argument("--format", choices=["mp4", "gif", "both"], default="mp4", help="Output format")
    parser.add_argument("--num_frames", type=int, default=120, help="Rendered animation frames before final hold")
    parser.add_argument("--fps", type=float, default=15.0, help="Frames per second")
    parser.add_argument("--tail_seconds", type=float, default=1.0, help="Hold the final frame this many seconds")
    parser.add_argument("--mode", choices=["cumulative", "window"], default="cumulative", help="Show all hits up to time, or only a moving time window")
    parser.add_argument("--window_width", type=float, default=None, help="firstTime width for --mode window")
    parser.add_argument("--cut_npe", type=float, default=0.0, help="Ignore PMTs with nPE <= this")
    parser.add_argument("--cut_firsttime", type=float, default=0.0, help="Ignore PMTs with firstTime <= this")
    parser.add_argument("--time_min", type=float, default=None, help="Override shared minimum firstTime")
    parser.add_argument("--time_max", type=float, default=None, help="Override shared maximum firstTime")
    parser.add_argument("--npe_max", type=float, default=None, help="Override shared maximum nPE for marker/color scale")
    parser.add_argument("--clip_pct_low", type=float, default=1.0, help="Lower percentile for color range clipping (0=no clip)")
    parser.add_argument("--clip_pct_high", type=float, default=99.0, help="Upper percentile for color range clipping (100=no clip)")
    parser.add_argument("--color_by", choices=["npe", "firsttime"], default="firsttime", help="Marker color quantity")
    parser.add_argument("--cmap", type=str, default="jet", help="Matplotlib colormap")
    parser.add_argument("--width", type=int, default=1440, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=1440, help="Video height in pixels")
    parser.add_argument("--dpi", type=int, default=150, help="Matplotlib render DPI")
    parser.add_argument("--marker_size", type=float, default=30.0, help="Maximum-ish marker size scale for hit PMTs")
    parser.add_argument("--hit_alpha", type=float, default=0.90, help="Visible hit marker alpha")
    parser.add_argument("--show_background", action=argparse.BooleanOptionalAction, default=True, help="Draw all PMTs as background dots")
    parser.add_argument("--background_size", type=float, default=2.0, help="Background PMT marker size")
    parser.add_argument("--background_alpha", type=float, default=0.20, help="Background PMT alpha")
    parser.add_argument("--background_color", type=str, default="#9ca3af", help="Background PMT color")
    parser.add_argument("--background", type=str, default="white", help="Figure background color")
    parser.add_argument("--text_color", type=str, default="#111827", help="Overlay text color")
    parser.add_argument("--title_size", type=int, default=13, help="Overlay title font size")
    parser.add_argument("--show_overlay", action=argparse.BooleanOptionalAction, default=False, help="Draw event label, time, hit count, and nPE sum")
    parser.add_argument("--show_frame_counter", action=argparse.BooleanOptionalAction, default=False, help="Draw frame counter at the bottom")
    parser.add_argument("--show_colorbar", action=argparse.BooleanOptionalAction, default=True, help="Draw color scale bar")
    parser.add_argument("--tight_detector", action=argparse.BooleanOptionalAction, default=True, help="Let the detector fill the video canvas")
    parser.add_argument("--tight_padding", type=int, default=18, help="Pixel padding around detector content when --tight_detector is enabled")
    parser.add_argument("--show_detector_hull", action=argparse.BooleanOptionalAction, default=True, help="Draw detector hull")
    parser.add_argument("--elev", type=float, default=18.0, help="3D camera elevation")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth")
    parser.add_argument("--crf", type=int, default=20, help="MP4 quality: lower is better/larger")
    parser.add_argument("--preset", type=str, default="medium", help="ffmpeg x264 preset")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Print frame progress")
    parser.add_argument("--progress_every", type=int, default=20, help="Progress print interval in frames")
    args = parser.parse_args()

    if args.real_npy is None and args.generated_npy is None:
        parser.error("Provide at least one of --real_npy or --generated_npy")
    if args.num_frames <= 0:
        parser.error("--num_frames must be > 0")
    if args.fps <= 0:
        parser.error("--fps must be > 0")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be > 0")
    if args.tight_padding < 0:
        parser.error("--tight_padding must be >= 0")
    try:
        plt.get_cmap(args.cmap)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    geo = load_geometry(args.h5_path)
    jobs: list[tuple[str, Path, dict]] = []

    if args.real_npy is not None:
        real_path = Path(args.real_npy).expanduser().resolve()
        real_sig = load_event_npy(real_path)
        jobs.append(("real", real_path, clean_event(real_sig, cut_npe=args.cut_npe, cut_firsttime=args.cut_firsttime)))
    if args.generated_npy is not None:
        gen_path = Path(args.generated_npy).expanduser().resolve()
        gen_sig = load_event_npy(gen_path)
        jobs.append(("generated", gen_path, clean_event(gen_sig, cut_npe=args.cut_npe, cut_firsttime=args.cut_firsttime)))

    for kind, npy_path, event in jobs:
        if event["npe"].shape[0] != geo.shape[1]:
            raise ValueError(
                f"{kind} NPY length {event['npe'].shape[0]} does not match geometry length {geo.shape[1]}: {npy_path}"
            )

    ranges = compute_shared_ranges([event for _kind, _path, event in jobs], args)
    print("Shared scales:")
    print(f"  firstTime: {ranges['time_min']:.6g} to {ranges['time_max']:.6g}")
    print(f"  nPE max  : {ranges['npe_max']:.6g}")
    print(f"  output   : {out_dir}")

    all_saved: list[Path] = []
    for kind, npy_path, event in jobs:
        print(f"\nRendering {kind}: {npy_path}")
        print(f"  hits after cuts: {event['hit_count']}, nPE sum: {event['npe_sum']:.6g}")
        output_base = build_output_base(out_dir, kind, npy_path)
        saved = render_event_video(
            event=event,
            geo=geo,
            label=f"{kind}: {npy_path.stem}",
            output_base=output_base,
            ranges=ranges,
            args=args,
        )
        all_saved.extend(saved)
        for path in saved:
            print(f"  saved: {path}")

    print("\nDone.")
    for path in all_saved:
        print(path)


if __name__ == "__main__":
    main()
