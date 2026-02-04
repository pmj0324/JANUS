#!/bin/bash
# sample_by_zenith.py 풀옵션 실행 예시
# 사용 전: 체크포인트 경로, H5 파일 경로를 실제 값으로 바꾸고, 필요시 pip install tqdm

cd "$(dirname "$0")"

python sample_by_zenith.py \
  --checkpoint ./checkpoints/model.pt \
  --data_dir ./GENESIS-data \
  --label_key label \
  --n_bins 5 \
  --m_per_bin 3 \
  --output_dir ./output_zenith_sampling \
  --num_samples 2 \
  --gpu 0 \
  --histogram \
  --cut_npe 10.0 \
  --cut_firsttime 100.0 \
  --cfg_scale 2.5 \
  --seed 123 \
  --skip_existing
