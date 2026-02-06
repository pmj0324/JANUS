#!/usr/bin/env bash
#
# sample_cfg.py를 여러 CFG 스케일(1.5, 1.75, 2.0, 2.25)과
# 1~100 PeV 구간을 3개로 나눠 구간당 2개씩 뽑은 이벤트(총 6개)에 대해 실행.
# num_samples=2 로 고정.
#
# 사용: ./run_sample_cfg_energy_cfg_sweep.sh [체크포인트] [출력루트] [H5경로]
# 예:   ./run_sample_cfg_energy_cfg_sweep.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHECKPOINT="${1:-tasks/nb10_h4/nb10_h4_ep30_0.005571.pt}"
OUT_ROOT="${2:-tasks/nb10_h4/sweep_energy_cfg2}"
H5_PATH="${3:-/home/pmj0324/icecube-genesis/0121/GENESIS/GENESIS-data/22644_0921_time_shift.h5}"

CFG_SCALES="1.1 1.25 1.5 1.75 2.0"
NUM_SAMPLES=2
N_BINS=4
N_PER_BIN=2
ENERGY_PEV_MIN=1.0
ENERGY_PEV_MAX=100.0
SEED=42

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT"
  exit 1
fi
if [[ ! -f "$H5_PATH" ]]; then
  echo "H5 file not found: $H5_PATH"
  exit 1
fi

# H5에서 1~100 PeV를 3구간으로 나누고 구간당 2개 인덱스 뽑기 (Python)
INDICES_STR=$(python3 - "$H5_PATH" "$N_BINS" "$N_PER_BIN" "$ENERGY_PEV_MIN" "$ENERGY_PEV_MAX" "$SEED" << 'PY'
import sys
import numpy as np

h5_path = sys.argv[1]
n_bins = int(sys.argv[2])
n_per_bin = int(sys.argv[3])
e_min_pev = float(sys.argv[4])
e_max_pev = float(sys.argv[5])
seed = int(sys.argv[6])

try:
    import h5py
except ImportError:
    sys.exit(1)

with h5py.File(h5_path, "r") as f:
    labels = np.asarray(f["label"], dtype=np.float32)
# raw H5 energy is MeV -> convert to PeV
energy_mev = labels[:, 0]
energy_pev = energy_mev / 1e6
mask = (energy_pev >= e_min_pev) & (energy_pev <= e_max_pev)
indices_all = np.where(mask)[0]
energy_sel = energy_pev[mask]

if len(indices_all) == 0:
    sys.exit(1)

edges = np.linspace(e_min_pev, e_max_pev, n_bins + 1)
edges[-1] += 1e-9  # include right edge in last bin
rng = np.random.default_rng(seed)
out_indices = []
for i in range(n_bins):
    lo, hi = edges[i], edges[i + 1]
    in_bin = (energy_sel >= lo) & (energy_sel < hi)
    bin_global_idx = indices_all[in_bin]
    n_take = min(n_per_bin, len(bin_global_idx))
    if n_take > 0:
        picked = rng.choice(bin_global_idx, size=n_take, replace=False)
        out_indices.extend(picked.tolist())

out_indices = sorted(out_indices)
print(" ".join(map(str, out_indices)))
PY
)

if [[ -z "$INDICES_STR" ]]; then
  echo "Failed to get event indices from H5 (or no events in 1–100 PeV)."
  exit 1
fi

read -ra INDICES <<< "$INDICES_STR"
mkdir -p "$OUT_ROOT"

echo "Checkpoint: $CHECKPOINT"
echo "Output root: $OUT_ROOT"
echo "H5: $H5_PATH"
echo "Event indices (1–100 PeV, 3 bins x 2): ${INDICES[*]}"
echo "CFG scales: $CFG_SCALES"
echo "Num samples per run: $NUM_SAMPLES"
echo ""

total=$(( ${#INDICES[@]} * 4 ))
done=0
for ref_idx in "${INDICES[@]}"; do
  for cfg in $CFG_SCALES; do
    out_dir="$OUT_ROOT/idx_${ref_idx}/scale_${cfg}"
    mkdir -p "$out_dir"
    (( done++ )) || true
    echo "[$done/$total] ref_idx=$ref_idx cfg_scale=$cfg -> $out_dir"
    python3 sample_cfg.py \
      --checkpoint "$CHECKPOINT" \
      --output_dir "$out_dir" \
      --ref_idx "$ref_idx" \
      --num_samples "$NUM_SAMPLES" \
      --cfg_scale "$cfg" \
      --histogram \
      --histogram_combined \
      --histogram_noclip \
      --histogram_log \
      --cut_npe 0.99 \
      --cut_firsttime 0.1
  done
done

echo ""
echo "Done. Total runs: $done. Output: $OUT_ROOT"
