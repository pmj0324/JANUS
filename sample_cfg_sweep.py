#!/usr/bin/env python3
"""
다양한 이벤트 × 다양한 CFG 스케일로 여러 번 샘플링.
샘플링 코드는 수정하지 않고, sample_cfg.py를 반복 호출하는 스크립트.
체크포인트 기본: task/cfg_20/ 내 .pt 파일 사용.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent


def _find_checkpoint_in_cfg20() -> Path:
    """task/cfg_20/ 안의 첫 .pt 파일 경로 반환. 없으면 예외."""
    d = _ROOT / "task" / "cfg_20"
    if not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {d}")
    pts = sorted(d.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt checkpoint in {d}")
    return pts[0]


def _parse_events(s: str) -> list[int]:
    """'0,100,39276' 또는 '0:100:10' 형태 파싱. 범위는 [start:end:step] (end 미포함)."""
    s = s.strip()
    if ":" in s:
        parts = [int(x.strip()) for x in s.split(":")]
        if len(parts) == 2:
            start, end = parts
            step = 1
        else:
            start, end, step = parts[0], parts[1], parts[2]
        return list(range(start, end, max(1, step)))
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_cfg_scales(s: str) -> list[float]:
    """'1.0,2.0,5.0,10.0' 형태 파싱."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def get_event_indices_by_energy_bins(h5_path, n_bins=5, n_per_bin=5, energy_in_mev=True):
    import h5py
    with h5py.File(h5_path, "r") as f:
        labels = f["label"][...]
    energy = np.asarray(labels[:, 0], dtype=np.float64)
    if energy_in_mev:
        energy = energy / 1e6
    if len(energy) == 0:
        return []
    edges = np.percentile(energy, np.linspace(0, 100, n_bins + 1))
    edges[-1] += 1e-9
    out = []
    for i in range(n_bins):
        mask = (energy >= edges[i]) & (energy < edges[i + 1])
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        n_take = min(n_per_bin, len(indices))
        step = max(1, len(indices) // n_take)
        out.extend(indices[::step][:n_take].tolist())
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep over events and CFG scales by calling sample_cfg.py repeatedly. No changes to sampling code.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint. If omitted, uses first .pt in task/cfg_20/",
    )
    parser.add_argument(
        "-o", "--out_dir",
        type=str,
        default="./task/sweep_cfg",
        help="Base output directory. Results go to out_dir/idx_<ref_idx>/scale_<cfg>/",
    )
    parser.add_argument(
        "-e", "--events",
        type=str,
        default="0,50,100,200,300,500,800,1000,2000,3000,5000,8000,10000,15000,20000,25000,30000,39276",
        help="Comma-separated event indices or range 'start:end' or 'start:end:step' (e.g. 0:100:10)",
    )
    parser.add_argument(
        "-s", "--cfg_scales",
        type=str,
        default="1.0,1.5,2.0,3.0,4.0,5.0",
        help="Comma-separated CFG scales (default: 1,1.5,2,3,4,5)",
    )
    parser.add_argument(
        "-n", "--num_samples",
        type=int,
        default=2,
        help="Number of samples per (event, cfg_scale)",
    )
    parser.add_argument("--by_energy", action="store_true", help="Pick events by energy bins from H5")
    parser.add_argument("--h5_path", type=str, default="./GENESIS-data/22644_0921_time_shift.h5", help="H5 for --by_energy")
    parser.add_argument("--n_bins", type=int, default=5, help="Energy bins for --by_energy")
    parser.add_argument("--n_per_bin", type=int, default=5, help="Events per bin for --by_energy")
    parser.add_argument("-g", "--gpu", type=int, default=None, help="GPU ID")
    parser.add_argument("-H", "--histogram", action="store_true", help="Save histograms")
    parser.add_argument("--cut_npe", type=float, default=0.0, help="nPE cut for plots/histograms")
    parser.add_argument("--cut_firsttime", type=float, default=0.0, help="FirstTime cut for plots/histograms")

    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = str(_find_checkpoint_in_cfg20())
        print(f"Using checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = str(Path(checkpoint_path).resolve())
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if getattr(args, "by_energy", False):
        h5 = str(Path(args.h5_path).resolve())
        if not Path(h5).exists():
            raise FileNotFoundError(f"H5 not found: {h5}")
        event_indices = get_event_indices_by_energy_bins(h5, n_bins=args.n_bins, n_per_bin=args.n_per_bin, energy_in_mev=True)
    else:
        event_indices = _parse_events(args.events)
    cfg_scales = _parse_cfg_scales(args.cfg_scales)
    print(f"Events: {event_indices}")
    print(f"CFG scales: {cfg_scales}")
    print(f"Samples per (event, cfg): {args.num_samples}")
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    sample_cfg_script = _ROOT / "sample_cfg.py"
    if not sample_cfg_script.is_file():
        raise FileNotFoundError(f"sample_cfg.py not found: {sample_cfg_script}")

    total = len(event_indices) * len(cfg_scales)
    done = 0
    for ref_idx in event_indices:
        for cfg_scale in cfg_scales:
            output_dir = out_root / f"idx_{ref_idx}" / f"scale_{cfg_scale}"
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(sample_cfg_script),
                "--checkpoint", checkpoint_path,
                "--output_dir", str(output_dir),
                "--ref_idx", str(ref_idx),
                "--num_samples", str(args.num_samples),
                "--cfg_scale", str(cfg_scale),
            ]
            if args.gpu is not None:
                cmd.extend(["--gpu", str(args.gpu)])
            if args.histogram:
                cmd.append("--histogram")
            if args.cut_npe != 0.0:
                cmd.extend(["--cut_npe", str(args.cut_npe)])
            if args.cut_firsttime != 0.0:
                cmd.extend(["--cut_firsttime", str(args.cut_firsttime)])

            done += 1
            print(f"\n[{done}/{total}] event {ref_idx} cfg_scale={cfg_scale} -> {output_dir}")
            ret = subprocess.run(cmd, cwd=str(_ROOT))
            if ret.returncode != 0:
                print(f"sample_cfg.py failed with exit code {ret.returncode}", file=sys.stderr)
                sys.exit(ret.returncode)

    print(f"\nDone. Total (event, cfg) combinations: {done}. Output base: {out_root.absolute()}")


if __name__ == "__main__":
    main()
