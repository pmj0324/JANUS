#!/usr/bin/env python3
"""
Diffusion Process Video / GIF
==============================

Render forward diffusion (noise added step by step) as MP4 or GIF.
Uses the same forward diffusion and event visualization as visualize_forward_diffusion.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root for imports when run as script
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from diffusion.schedules import get_noise_schedule, compute_alpha_schedule
from diffusion.forward import apply_forward_diffusion
from utils.vis.event_show import show_event_dual_plot


def _figure_to_rgb(fig: plt.Figure, dpi: int = 100) -> np.ndarray:
    """Render a matplotlib figure to RGB array (H, W, 3) uint8."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # Matplotlib API differs by version/backends:
    # - Some provide tostring_rgb()
    # - Newer versions may only provide buffer_rgba() / tostring_argb()
    if hasattr(fig.canvas, "tostring_rgb"):
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape((h, w, 3))

    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)  # (h, w, 4)
        return rgba[:, :, :3].copy()

    if hasattr(fig.canvas, "tostring_argb"):
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
        # ARGB -> RGB
        return argb[:, :, 1:4].copy()

    raise AttributeError("Unsupported matplotlib canvas: cannot extract RGB buffer")


def _write_video(
    frames: List[np.ndarray],
    output_path: Path,
    fps: float = 10,
    format: str = "gif",
) -> None:
    """Write list of RGB frames (H,W,3) to GIF or MP4 using imageio."""
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "Saving video requires 'imageio'. Install with: pip install imageio\n"
            "For MP4 support also install: pip install imageio-ffmpeg"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if format.lower() == "gif":
        duration = 1.0 / fps
        imageio.mimsave(str(output_path), frames, duration=duration, loop=0)
    else:
        # mp4 via imageio (uses ffmpeg if available)
        try:
            imageio.mimsave(str(output_path), frames, fps=fps, codec="libx264", quality=8)
        except TypeError:
            imageio.mimsave(str(output_path), frames, fps=fps)


def save_diffusion_video(
    x0_sig: torch.Tensor,
    geom: torch.Tensor,
    label: torch.Tensor,
    schedule_name: str = "linear",
    schedule_kwargs: Optional[dict] = None,
    num_timesteps: int = 1000,
    step: int = 20,
    output_path: Union[str, Path] = "./diffusion_video",
    output_format: str = "both",
    fps: float = 10,
    figure_size: Tuple[int, int] = (14, 7),
    marker_size: float = 20.0,
    show_detector_hull: bool = True,
    fixed_noise: bool = True,
    seed: Optional[int] = 42,
    clamp_for_visualization: bool = False,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Render forward diffusion (noise added step by step) and save as MP4 and/or GIF.

    For each timestep t in [0, step, 2*step, ..., T], applies forward diffusion
    to get x_t from x_0, then renders the event with show_event_dual_plot and
    appends the frame to the video.

    Expects x0_sig already normalized to [0, 1] (e.g. log_minmax) as in training.
    If you have raw signals, normalize first (see example script).

    Args:
        x0_sig: Clean signals (B, 2, L) - normalized [0, 1]. Only first sample used if B > 1.
        geom: Geometry (B, 3, L). Only first sample used if B > 1.
        label: Labels (B, 6). Only first sample used if B > 1.
        schedule_name: "linear", "cosine", "quadratic", or "sigmoid".
        schedule_kwargs: Optional dict for schedule (e.g. {"s": 0.008} for cosine).
        num_timesteps: Total diffusion steps T.
        step: Render every `step` timesteps (0, step, 2*step, ... up to T).
        output_path: Output file path (without extension) or directory.
            If path has no extension, files are named diffusion_video.mp4 / .gif in that dir.
        output_format: "mp4", "gif", or "both".
        fps: Frames per second for the video.
        figure_size: Figure size (width, height) in inches.
        marker_size: Marker size for 3D scatter.
        show_detector_hull: Whether to draw detector hull.
        fixed_noise: If True, use fixed seed so the same noise is used for all steps (reproducible).
        seed: Random seed when fixed_noise is True.
        clamp_for_visualization: If True, clamp x_t to [0, 1] before plotting.
            Default False to match utils/vis/visualize_forward_diffusion.py behavior (no clamp).

    Returns:
        (path_mp4, path_gif) - paths to saved files; the unused format is None.

    Requires:
        pip install imageio
        pip install imageio-ffmpeg  # for MP4
    """
    schedule_kwargs = schedule_kwargs or {}
    output_path = Path(output_path)
    if output_path.suffix.lower() in (".mp4", ".gif"):
        out_dir = output_path.parent
        out_stem = output_path.stem
    else:
        out_dir = output_path
        out_stem = "diffusion_video"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = x0_sig.device
    B, C, L = x0_sig.shape
    assert C == 2, "Signal must have 2 channels (nPE, firstTime)"
    if B > 1:
        x0_sig = x0_sig[0:1]
        geom = geom[0:1]
        label = label[0:1]

    geom_np = geom[0].cpu().numpy()
    label_np = label[0].cpu().numpy()

    # Timesteps to render: 0, step, 2*step, ..., num_timesteps
    timestep_list = list(range(0, num_timesteps + 1, step))
    if timestep_list[-1] != num_timesteps:
        timestep_list.append(num_timesteps)

    betas = get_noise_schedule(
        schedule_name=schedule_name,
        timesteps=num_timesteps + 1,
        **schedule_kwargs
    )
    betas = betas.to(device)
    alpha_schedule = compute_alpha_schedule(betas)

    if fixed_noise and seed is not None:
        torch.manual_seed(seed)

    frames = []
    for t_val in timestep_list:
        # Sample fresh noise for each timestep (noise=None → re-sampled inside apply_forward_diffusion)
        t_tensor = torch.full((1,), t_val, device=device, dtype=torch.long)
        x_t = apply_forward_diffusion(
            x0=x0_sig,
            betas=betas,
            timesteps=t_tensor,
            noise=None,
        )
        x_t_np = x_t[0].cpu().numpy()
        if clamp_for_visualization:
            # Optional clamp (disabled by default to match visualize_forward_diffusion.py)
            x_t_np = np.clip(x_t_np, 0.0, 1.0)
        ftime = np.nan_to_num(x_t_np[1], nan=0.0, posinf=0.0, neginf=0.0)
        sig_vis = np.stack([x_t_np[0], ftime], axis=0)

        fig, _ = show_event_dual_plot(
            sig=sig_vis,
            geo=geom_np,
            label=label_np,
            output_path=None,
            figure_size=figure_size,
            marker_size=marker_size,
            show_detector_hull=show_detector_hull,
            show=False,
            title_prefix=f"Forward diffusion t={t_val} / {num_timesteps} ({schedule_name})",
            firsttime_title="FirstTime (norm)",
            npe_title="NPE (norm)",
            firsttime_cbar_label="FirstTime (normalized)",
            npe_cbar_label="NPE (normalized)",
        )
        frames.append(_figure_to_rgb(fig))
        plt.close(fig)

    path_mp4 = None
    path_gif = None
    if output_format.lower() in ("mp4", "both"):
        path_mp4 = out_dir / f"{out_stem}.mp4"
        try:
            _write_video(frames, path_mp4, fps=fps, format="mp4")
        except Exception as e:
            if "ffmpeg" in str(e).lower() or "imageio-ffmpeg" in str(e).lower():
                path_mp4 = None
                print("MP4 skipped (install imageio-ffmpeg for MP4):", e)
            else:
                raise
    if output_format.lower() in ("gif", "both"):
        path_gif = out_dir / f"{out_stem}.gif"
        _write_video(frames, path_gif, fps=fps, format="gif")

    return path_mp4, path_gif
