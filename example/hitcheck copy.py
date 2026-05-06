#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def scan_xor_zero_pairs(
    h5_path: Path,
    max_events: int | None = None,
    chunk_size: int = 4096,
    max_examples: int = 20,
):
    only_npe_zero = 0
    only_ftime_zero = 0
    both_zero = 0
    both_nonzero = 0
    events_with_any_xor = 0

    # 추가: 0이 아닌 값들의 최솟값
    min_nonzero_npe = np.inf
    min_nonzero_ftime = np.inf

    examples = []

    with h5py.File(h5_path, "r") as f:
        ds = f["input"]
        total_events = ds.shape[0]
        stop = total_events if max_events is None or max_events <= 0 else min(total_events, max_events)

        for start in tqdm(range(0, stop, chunk_size), desc="Scanning"):
            end = min(start + chunk_size, stop)
            sig = np.asarray(ds[start:end], dtype=np.float32)

            npe = sig[:, 0, :]
            ftime = sig[:, 1, :]

            finite = np.isfinite(npe) & np.isfinite(ftime)

            npe_zero = npe == 0.0
            ftime_zero = ftime == 0.0

            mask_only_npe_zero = finite & npe_zero & (~ftime_zero)
            mask_only_ftime_zero = finite & (~npe_zero) & ftime_zero
            mask_both_zero = finite & npe_zero & ftime_zero
            mask_both_nonzero = finite & (~npe_zero) & (~ftime_zero)

            only_npe_zero += int(mask_only_npe_zero.sum())
            only_ftime_zero += int(mask_only_ftime_zero.sum())
            both_zero += int(mask_both_zero.sum())
            both_nonzero += int(mask_both_nonzero.sum())

            event_has_xor = np.any(mask_only_npe_zero | mask_only_ftime_zero, axis=1)
            events_with_any_xor += int(event_has_xor.sum())

            # 추가: 0이 아닌 값들의 최솟값 계산
            npe_nonzero = npe[np.isfinite(npe) & (npe != 0.0)]
            ftime_nonzero = ftime[np.isfinite(ftime) & (ftime != 0.0)]

            if npe_nonzero.size > 0:
                min_nonzero_npe = min(min_nonzero_npe, float(np.min(npe_nonzero)))
            if ftime_nonzero.size > 0:
                min_nonzero_ftime = min(min_nonzero_ftime, float(np.min(ftime_nonzero)))

            if len(examples) < max_examples:
                rel_idx = np.argwhere(mask_only_npe_zero | mask_only_ftime_zero)
                for ev_rel, det_rel in rel_idx:
                    if len(examples) >= max_examples:
                        break
                    ev = start + int(ev_rel)
                    det = int(det_rel)
                    examples.append(
                        (
                            ev,
                            det,
                            float(npe[ev_rel, det_rel]),
                            float(ftime[ev_rel, det_rel]),
                        )
                    )

    if not np.isfinite(min_nonzero_npe):
        min_nonzero_npe = np.nan
    if not np.isfinite(min_nonzero_ftime):
        min_nonzero_ftime = np.nan

    return {
        "total_events": total_events,
        "scanned_events": stop,
        "only_npe_zero": only_npe_zero,
        "only_ftime_zero": only_ftime_zero,
        "both_zero": both_zero,
        "both_nonzero": both_nonzero,
        "events_with_any_xor": events_with_any_xor,
        "min_nonzero_npe": min_nonzero_npe,
        "min_nonzero_ftime": min_nonzero_ftime,
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser(description="Find positions where only one of nPE / FirstTime is zero")
    parser.add_argument(
        "--h5-path",
        type=str,
        default="./GENESIS-data/22644_0921_time_shift.h5",
        help="Path to the HDF5 file",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional cap on number of events to scan",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of events per chunk",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Number of example indices to print",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    result = scan_xor_zero_pairs(
        h5_path=h5_path,
        max_events=args.max_events,
        chunk_size=max(1, args.chunk_size),
        max_examples=max(1, args.max_examples),
    )

    print("\nSummary")
    print(f"  total events in file        : {result['total_events']:,}")
    print(f"  scanned events              : {result['scanned_events']:,}")
    print(f"  only nPE == 0, FirstTime != 0 : {result['only_npe_zero']:,}")
    print(f"  only nPE != 0, FirstTime == 0 : {result['only_ftime_zero']:,}")
    print(f"  both zero                     : {result['both_zero']:,}")
    print(f"  both non-zero                 : {result['both_nonzero']:,}")
    print(f"  events with any xor-zero pair : {result['events_with_any_xor']:,}")

    print("\nMin non-zero values")
    print(f"  nPE       min(non-zero) = {result['min_nonzero_npe']:.6g}")
    print(f"  FirstTime min(non-zero) = {result['min_nonzero_ftime']:.6g}")

    if result["examples"]:
        print("\nExamples")
        for ev, det, npe_val, ftime_val in result["examples"]:
            print(f"  event={ev}, det={det}, nPE={npe_val:g}, FirstTime={ftime_val:g}")
    else:
        print("\nNo xor-zero examples found.")


if __name__ == "__main__":
    main()

