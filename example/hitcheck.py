#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def find_multi_xor_events(
    h5_path: Path,
    max_events: int | None = None,
    chunk_size: int = 4096,
    max_events_to_print: int = 50,
):
    multi_events = []

    with h5py.File(h5_path, "r") as f:
        ds = f["input"]
        total = ds.shape[0]
        stop = total if max_events is None or max_events <= 0 else min(total, max_events)

        for start in tqdm(range(0, stop, chunk_size), desc="Scanning"):
            end = min(start + chunk_size, stop)
            sig = np.asarray(ds[start:end], dtype=np.float32)

            npe = sig[:, 0, :]
            ftime = sig[:, 1, :]

            finite = np.isfinite(npe) & np.isfinite(ftime)
            xor_zero = finite & ((npe == 0.0) ^ (ftime == 0.0))
            per_event_counts = np.sum(xor_zero, axis=1).astype(np.int64)

            multi_local = np.where(per_event_counts >= 2)[0]
            for local_idx in multi_local:
                ev = start + int(local_idx)
                det_idx = np.where(xor_zero[local_idx])[0]

                entries = []
                for d in det_idx:
                    entries.append(
                        {
                            "detector": int(d),
                            "npe": float(npe[local_idx, d]),
                            "ftime": float(ftime[local_idx, d]),
                        }
                    )

                multi_events.append(
                    {
                        "event": ev,
                        "count": int(per_event_counts[local_idx]),
                        "pairs": entries,
                    }
                )

                if len(multi_events) >= max_events_to_print:
                    return multi_events

    return multi_events


def main():
    parser = argparse.ArgumentParser(description="Print details for events with 2+ xor-zero detector pairs")
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
        "--max-events-to-print",
        type=int,
        default=50,
        help="Maximum number of multi-xor events to print",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    multi_events = find_multi_xor_events(
        h5_path=h5_path,
        max_events=args.max_events,
        chunk_size=max(1, args.chunk_size),
        max_events_to_print=max(1, args.max_events_to_print),
    )

    if not multi_events:
        print("No events with 2+ xor-zero pairs found.")
        return

    print(f"\nFound {len(multi_events)} example multi-xor events:\n")
    for item in multi_events:
        print(f"event={item['event']}, xor_zero_pairs={item['count']}")
        for p in item["pairs"]:
            print(f"  detector={p['detector']}, nPE={p['npe']:g}, FirstTime={p['ftime']:g}")
        print()


if __name__ == "__main__":
    main()
