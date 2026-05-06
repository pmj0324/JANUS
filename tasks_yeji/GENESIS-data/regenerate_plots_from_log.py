#!/usr/bin/env python3
"""
실험 로그 CSV만으로 summary CSV와 그래프를 다시 생성한다.
샘플(.npy) 재생성이나 H5/체크포인트 접근 없이 동작한다.

사용 예:
  python regenerate_plots_from_log.py \\
    --log_csv /path/to/experiment_log.csv \\
    --summary_dir /path/to/summary

multifield 로그에 대해 response 플롯까지 쓰려면 (선택):
  python regenerate_plots_from_log.py --log_csv ... --summary_dir ... \\
    --fiducial_ref_idx 0 --response_dim 2

event 모드일 때 `summary/event/plots/` 아래에 1x3 요약 PNG
(`*_vs_cfg_npecut_firsttimecut.png`)가 갱신된다. 구현은 make_sample_ver2와 동일하다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Recompute summary CSVs and plots from an existing experiment_log.csv (no resampling)."
    )
    p.add_argument(
        "--log_csv",
        type=Path,
        required=True,
        help="Path to experiment_log.csv (or equivalent main log CSV).",
    )
    p.add_argument(
        "--summary_dir",
        type=Path,
        required=True,
        help="Output summary directory (e.g. .../output/summary). Writes event/ and multifield/ subdirs.",
    )
    p.add_argument(
        "--fiducial_ref_idx",
        type=int,
        default=None,
        help="Optional: multifield parameter-response plots vs this reference index.",
    )
    p.add_argument(
        "--response_dim",
        type=int,
        default=None,
        help="Optional: label dimension index for response plots (requires --fiducial_ref_idx).",
    )
    p.add_argument("--response_tol", type=float, default=1e-8)
    p.add_argument("--max_response_values", type=int, default=12)
    p.add_argument("--min_diversity_actual_refs", type=int, default=2)
    p.add_argument("--min_diversity_samples", type=int, default=4)
    p.add_argument("--max_diversity_plot_groups", type=int, default=6)

    args = p.parse_args()
    log_csv = args.log_csv.resolve()
    summary_dir = args.summary_dir.resolve()

    if not log_csv.is_file():
        print(f"[ERROR] Log CSV not found: {log_csv}", file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    import make_sample_ver2 as ms

    plot_args = ms.default_summary_plot_args(
        min_diversity_actual_refs=args.min_diversity_actual_refs,
        min_diversity_samples=args.min_diversity_samples,
        max_diversity_plot_groups=args.max_diversity_plot_groups,
        response_dim=args.response_dim,
        fiducial_ref_idx=args.fiducial_ref_idx,
        response_tol=args.response_tol,
        max_response_values=args.max_response_values,
    )

    print(f"[INFO] Log CSV:    {log_csv}")
    print(f"[INFO] Summary dir: {summary_dir}")
    ms.regenerate_summary_and_plots(log_csv, summary_dir, plot_args)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
