#!/usr/bin/env bash
# sample_cfg.py 풀옵션 실행 (변수만 바꿔서 사용)
set -e
cd "$(dirname "$0")"

CHECKPOINT="${CHECKPOINT:-tasks/nb10_h4/nb10_h4_ep30_0.005571.pt}"
OUT_DIR="${OUT_DIR:-tasks/nb10_h4/samples_full}"
REF_IDX="${REF_IDX:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"

python3 sample_cfg.py \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUT_DIR" \
  --ref_idx "$REF_IDX" \
  --num_samples "$NUM_SAMPLES" \
  --gpu 0 \
  --histogram \
  --histogram_combined \
  --histogram_noclip \
  --histogram_log \
  --cut_npe 0.99 \
  --cut_firsttime 0.1
