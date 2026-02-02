#!/usr/bin/env python3
"""에너지대별 sweep: sample_cfg_sweep.py --by_energy 를 호출. 기본: task/cfg_0201_model_final 모델, 출력 task/cfg_0201_model_final/cfg_energy."""

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_MODEL_DIR = _ROOT / "task" / "cfg_0201_model_final"
_DEFAULT_CKPT = _MODEL_DIR / "model_checkpoint_final.pt"
_OUT_DIR = _MODEL_DIR  # 모델 있는 폴더에 바로 저장 (idx_*/scale_*/)


if __name__ == "__main__":
    ckpt_path = str(_DEFAULT_CKPT.resolve()) if _DEFAULT_CKPT.exists() else None
    cmd = [
        sys.executable,
        str(_ROOT / "sample_cfg_sweep.py"),
        "--by_energy",
        "--h5_path", "./GENESIS-data/22644_0921_time_shift.h5",
        "--n_bins", "5",
        "--n_per_bin", "2",
        "-o", str(_OUT_DIR),
        "-s", "1.0,1.2,1.4,1.5,1.8,2.0,2.5,3.0,4.0,5.0",
        "--cut_npe", "0.9",
        "--cut_firsttime", "0.9",
        "--histogram",
    ]
    if ckpt_path:
        cmd.extend(["-c", ckpt_path])
    cmd += sys.argv[1:]
    sys.exit(subprocess.run(cmd, cwd=str(_ROOT)).returncode)