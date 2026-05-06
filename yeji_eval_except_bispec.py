#!/usr/bin/env python3
"""replot_total_score_no_bispec.py

이미 생성된 CSV 로그에서 bispectrum_score를 제외하고
total_score를 재계산하여 그래프를 그리는 독립 스크립트.

사용법:
    python replot_total_score_no_bispec.py \
        --csv path/to/eval_log.csv \
        --outdir path/to/output_plots

기존 가중치 (bispectrum 포함):
    pdf=0.25, auto=0.25, cross=0.20, coherence=0.20, bispectrum=0.10

재계산 가중치 (bispectrum 제외, 나머지 합=1로 정규화):
    pdf=0.2778, auto=0.2778, cross=0.2222, coherence=0.2222
    (= 원래 비율 유지, bispectrum 몫을 나머지에 비례 배분)
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ── 가중치 정의 ──────────────────────────────────────────────────────────────
WEIGHTS_WITH_BISPEC = {
    "pdf_score":       0.25,
    "auto_score":      0.25,
    "cross_score":     0.20,
    "coherence_score": 0.20,
    "bispectrum_score": 0.10,
}

# bispectrum 제외 후 나머지 비율 유지하며 합=1로 정규화
_base = {k: v for k, v in WEIGHTS_WITH_BISPEC.items() if k != "bispectrum_score"}
_total = sum(_base.values())
WEIGHTS_NO_BISPEC = {k: v / _total for k, v in _base.items()}
# → pdf=0.2778, auto=0.2778, cross=0.2222, coherence=0.2222


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def _to_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nanmean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size > 0 else np.nan


def recompute_total_score(row: Dict[str, str], weights: Dict[str, float]) -> float:
    """주어진 가중치로 total_score 재계산 (aggregate_weighted_score와 동일 로직)."""
    vals, ws = [], []
    for key, w in weights.items():
        v = _to_float(row.get(key, ""), np.nan)
        if np.isfinite(v):
            vals.append(v)
            ws.append(w)
    if not vals:
        return np.nan
    wsum = sum(ws)
    if wsum <= 0:
        return np.nan
    return float(np.dot(vals, ws) / wsum)


# ── 그룹핑 ───────────────────────────────────────────────────────────────────
def group_by_cfg(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[_to_float(r.get("cfg_scale", np.nan))].append(r)
    return dict(groups)


def group_by_cfg_cut(rows):
    groups = defaultdict(list)
    for r in rows:
        key = (
            _to_float(r.get("cfg_scale", np.nan)),
            _to_float(r.get("cut_npe", np.nan)),
            _to_float(r.get("cut_firsttime", np.nan)),
        )
        groups[key].append(r)
    return dict(groups)


# ── 플롯 ─────────────────────────────────────────────────────────────────────
def plot_comparison(
    xs,
    ys_orig,
    ys_new,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys_orig, marker="o", label="원래 total_score (bispectrum 포함)", color="tab:blue")
    ax.plot(xs, ys_new,  marker="s", linestyle="--", label="재계산 total_score (bispectrum 제외)", color="tab:orange")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    #ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {save_path}")


def plot_new_only(
    xs,
    ys_new,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys_new, marker="s", color="tab:orange")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {save_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="bispectrum 제외 total_score 재계산 및 플롯")
    parser.add_argument("--csv", required=True, help="eval_log.csv 경로")
    parser.add_argument("--outdir", default="./replot_output", help="출력 디렉터리")
    parser.add_argument(
        "--eval_mode", default="multifield_map",
        help="필터할 eval_mode (기본: multifield_map)"
    )
    parser.add_argument(
        "--comparison", action="store_true", default=True,
        help="원래 total_score와 비교 곡선 함께 그리기 (기본: True)"
    )
    parser.add_argument(
        "--no_comparison", dest="comparison", action="store_false",
        help="비교 곡선 없이 새 total_score만 그리기"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir   = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[로드] {csv_path}")
    all_rows = read_csv_rows(csv_path)

    # multifield_map 행만 필터
    rows = [r for r in all_rows if r.get("eval_mode", "").strip() == args.eval_mode]
    if not rows:
        print(f"[경고] eval_mode='{args.eval_mode}'인 행이 없습니다. 전체 행을 사용합니다.")
        rows = all_rows

    print(f"[INFO] 대상 행 수: {len(rows)}")
    print(f"\n재계산 가중치 (bispectrum 제외):")
    for k, v in WEIGHTS_NO_BISPEC.items():
        print(f"  {k}: {v:.4f}")

    # 각 행에 재계산 total_score 추가
    for r in rows:
        r["_total_score_no_bispec"] = recompute_total_score(r, WEIGHTS_NO_BISPEC)
        # 원래 total_score도 가중치로 재계산해서 검증
        r["_total_score_orig_recomputed"] = recompute_total_score(r, WEIGHTS_WITH_BISPEC)

    # ── 1. cfg_scale 축 플롯 ─────────────────────────────────────────────────
    cfg_groups = group_by_cfg(rows)
    cfg_sorted = sorted(cfg_groups.keys())

    xs_cfg = cfg_sorted
    ys_orig_cfg = [nanmean([_to_float(r.get("total_score", "")) for r in cfg_groups[c]]) for c in cfg_sorted]
    ys_new_cfg  = [nanmean([r["_total_score_no_bispec"] for r in cfg_groups[c]]) for c in cfg_sorted]

    if args.comparison:
        plot_comparison(
            xs_cfg, ys_orig_cfg, ys_new_cfg,
            xlabel="cfg_scale",
            ylabel="mean total_score (except bispectrum)",
            title="Total score (except bispectrum) vs cfg_scale",
            save_path=outdir / "total_score_no_bispec_vs_cfg.png",
        )
    else:
        plot_new_only(
            xs_cfg, ys_new_cfg,
            xlabel="cfg_scale",
            ylabel="mean total_score (except bispectrum)",
            title="Total score (except bispectrum) vs cfg_scale",
            save_path=outdir / "total_score_no_bispec_vs_cfg.png",
        )

    # ── 2. cut_npe 축 플롯 (cut_npe가 여러 값이면) ───────────────────────────
    cfg_cut_groups = group_by_cfg_cut(rows)
    line_keys = sorted({(r.get("cfg_scale"), r.get("cut_firsttime")) for r in rows})
    cut_npe_vals = sorted({_to_float(r.get("cut_npe", np.nan)) for r in rows if np.isfinite(_to_float(r.get("cut_npe", np.nan)))})

    if len(cut_npe_vals) > 1:
        fig, ax = plt.subplots(figsize=(9, 6))
        for cfg_scale_str, cut_ft_str in line_keys:
            cfg_scale = _to_float(cfg_scale_str)
            cut_ft    = _to_float(cut_ft_str)
            sub = {
                _to_float(r.get("cut_npe", np.nan)): r["_total_score_no_bispec"]
                for r in rows
                if _to_float(r.get("cfg_scale", np.nan)) == cfg_scale
                and _to_float(r.get("cut_firsttime", np.nan)) == cut_ft
            }
            if not sub:
                continue
            xs_npe = sorted(sub.keys())
            ys_npe = [nanmean([r["_total_score_no_bispec"]
                               for r in rows
                               if _to_float(r.get("cfg_scale", np.nan)) == cfg_scale
                               and _to_float(r.get("cut_firsttime", np.nan)) == cut_ft
                               and _to_float(r.get("cut_npe", np.nan)) == x])
                      for x in xs_npe]
            ax.plot(xs_npe, ys_npe, marker="o", label=f"cfg={cfg_scale_str}, t={cut_ft_str}")

        ax.set_xlabel("cut_npe")
        ax.set_ylabel("mean total_score (except bispectrum)")
        ax.set_title("Total score (except bispectrum) vs cut_npe")
        ax.grid(True, alpha=0.3)
        #ax.legend(fontsize=8)
        fig.tight_layout()
        save_path = outdir / "total_score_no_bispec_vs_cut_npe.png"
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[저장] {save_path}")

    # ── 3. 수치 요약 출력 ────────────────────────────────────────────────────
    print("\n── cfg_scale별 요약 ─────────────────────────────────────────────")
    print(f"{'cfg_scale':>12}  {'orig total_score':>18}  {'no-bispec total_score':>22}  {'diff':>8}")
    for c, yo, yn in zip(xs_cfg, ys_orig_cfg, ys_new_cfg):
        diff = yn - yo if np.isfinite(yn) and np.isfinite(yo) else np.nan
        print(f"{c:>12.4g}  {yo:>18.6f}  {yn:>22.6f}  {diff:>+8.6f}")

    print(f"\n[완료] 플롯 저장 위치: {outdir}")


if __name__ == "__main__":
    main()