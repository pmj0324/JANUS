#!/bin/bash
# sample_cfg.py 풀옵션 (모든 히스토그램 켬, cut_npe=0.99, cut_firsttime=0.1)
# 사용: ./run_sample_cfg_full.sh [체크포인트] [출력디렉] [ref_idx]
# 예: ./run_sample_cfg_full.sh tasks/nb10_h4/nb10_h4_ep30_0.005571.pt tasks/nb10_h4/samples_full 100

set -e
cd "$(dirname "$0")"

CHECKPOINT="${1:-tasks/nb10_h4/nb10_h4_ep30_0.005571.pt}"
OUT_DIR="${2:-tasks/nb10_h4/samples_full}"
REF_IDX="${3:-100}"

python sample_cfg.py \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUT_DIR" \
  --num_samples 2 \
  --ref_idx "$REF_IDX" \
  --gpu 0 \
  --histogram \
  --histogram_combined \
  --histogram_noclip \
  --histogram_log \
  --cut_npe 0.99 \
  --cut_firsttime 0.1

echo "Done. Output: $OUT_DIR"
