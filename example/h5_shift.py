#!/usr/bin/env python3
"""
Create a new HDF5 file where nPE and FirstTime can be transformed for active hits.

Rule:
  - if nPE > 0, then nPE *= npe_scale
  - if nPE > 0, then FirstTime += ftime_shift
  - if nPE == 0, both values stay unchanged

This keeps zero reserved for inactive detectors while optionally removing
zero from active nPE / FirstTime values.

Usage:
    python make_firsttime_shifted_dataset.py \
        --input-h5 ./GENESIS-data/22644_0921_time_shift.h5 \
        --output-h5 ./GENESIS-data/22644_0921_time_shift_shifted.h5 \
        --npe-scale 1.0 \
        --ftime-shift 0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def _copy_dataset_like(src_ds: h5py.Dataset, dst_group: h5py.Group, name: str) -> h5py.Dataset:
    """Create a dataset in dst_group with the same storage settings as src_ds."""
    create_kwargs = {
        "shape": src_ds.shape,
        "dtype": src_ds.dtype,
        "chunks": src_ds.chunks,
        "compression": src_ds.compression,
        "compression_opts": src_ds.compression_opts,
        "shuffle": src_ds.shuffle,
        "fletcher32": src_ds.fletcher32,
        "fillvalue": src_ds.fillvalue,
        "maxshape": src_ds.maxshape,
    }
    if src_ds.scaleoffset is not None:
        create_kwargs["scaleoffset"] = src_ds.scaleoffset

    dst_ds = dst_group.create_dataset(name, **create_kwargs)
    for key, val in src_ds.attrs.items():
        dst_ds.attrs[key] = val
    return dst_ds


def _copy_object(src_obj: h5py.Dataset | h5py.Group, dst_parent: h5py.Group | h5py.File, name: str):
    """Copy a non-input object recursively."""
    if isinstance(src_obj, h5py.Group):
        dst_group = dst_parent.create_group(name)
        for key, val in src_obj.attrs.items():
            dst_group.attrs[key] = val
        for child_name, child_obj in src_obj.items():
            _copy_object(child_obj, dst_group, child_name)
    else:
        dst_parent.copy(src_obj, name)


def build_shifted_dataset(
    input_h5: Path,
    output_h5: Path,
    npe_scale: float = 1.0,
    ftime_shift: float = 0.0,
    chunk_size: int = 4096,
    overwrite: bool = False,
):
    if not input_h5.exists():
        raise FileNotFoundError(f"Input HDF5 file not found: {input_h5}")

    if output_h5.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_h5}. Use --overwrite to replace it."
            )
        output_h5.unlink()

    summary = {
        "events": 0,
        "detectors": 0,
        "scaled_npe_hits": 0,
        "shifted_ftime_hits": 0,
        "unchanged_inactive_hits": 0,
        "nonfinite_hits": 0,
    }

    if npe_scale <= 0.0 or not np.isfinite(npe_scale):
        raise ValueError(f"npe_scale must be a finite positive number, got {npe_scale!r}")

    with h5py.File(input_h5, "r") as src, h5py.File(output_h5, "w") as dst:
        for key, val in src.attrs.items():
            dst.attrs[key] = val

        if "input" not in src:
            raise KeyError("Source file does not contain required dataset: input")

        src_input = src["input"]
        dst_input = _copy_dataset_like(src_input, dst, "input")

        # Copy everything else verbatim.
        for name, obj in src.items():
            if name == "input":
                continue
            _copy_object(obj, dst, name)

        total_events = src_input.shape[0]
        summary["events"] = int(total_events)
        summary["detectors"] = int(src_input.shape[2]) if src_input.ndim >= 3 else 0

        for start in tqdm(range(0, total_events, chunk_size), desc="Writing shifted dataset"):
            end = min(start + chunk_size, total_events)
            sig = np.asarray(src_input[start:end], dtype=src_input.dtype)
            out = np.array(sig, copy=True)

            npe = out[:, 0, :]
            ftime = out[:, 1, :]
            active_mask = np.isfinite(npe) & (npe > 0.0)
            finite_time_mask = np.isfinite(ftime)
            shift_mask = active_mask & finite_time_mask

            if npe_scale != 1.0:
                out[:, 0, :][active_mask] = npe[active_mask] * npe_scale
                summary["scaled_npe_hits"] += int(np.count_nonzero(active_mask))

            if ftime_shift != 0.0:
                out[:, 1, :][shift_mask] = ftime[shift_mask] + ftime_shift
                summary["shifted_ftime_hits"] += int(np.count_nonzero(shift_mask))
            summary["unchanged_inactive_hits"] += int(np.count_nonzero(np.isfinite(npe) & (npe <= 0.0)))
            summary["nonfinite_hits"] += int(np.count_nonzero(~np.isfinite(npe) | ~np.isfinite(ftime)))

            dst_input[start:end] = out

    return summary


def main():
    parser = argparse.ArgumentParser(description="Shift nPE / FirstTime by configurable amounts for active hits only")
    parser.add_argument(
        "--input-h5",
        type=str,
        required=True,
        help="Path to the source HDF5 file",
    )
    parser.add_argument(
        "--output-h5",
        type=str,
        required=True,
        help="Path to the new HDF5 file to create",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=0.0,
        help="Backward-compatible alias for --ftime-shift",
    )
    parser.add_argument(
        "--npe-scale",
        type=float,
        default=1.0,
        help="Factor to multiply nPE by when nPE > 0",
    )
    parser.add_argument(
        "--npe-shift",
        type=float,
        default=None,
        help="Backward-compatible alias for --npe-scale",
    )
    parser.add_argument(
        "--ftime-shift",
        type=float,
        default=None,
        help="Amount to add to FirstTime when nPE > 0",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of events to process per chunk",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists",
    )
    args = parser.parse_args()

    input_h5 = Path(args.input_h5)
    output_h5 = Path(args.output_h5)

    summary = build_shifted_dataset(
        input_h5=input_h5,
        output_h5=output_h5,
        npe_scale=float(args.npe_scale if args.npe_shift is None else args.npe_shift),
        ftime_shift=float(args.shift if args.ftime_shift is None else args.ftime_shift),
        chunk_size=max(1, args.chunk_size),
        overwrite=args.overwrite,
    )

    print("\nDone")
    print(f"  input  : {input_h5}")
    print(f"  output : {output_h5}")
    print(f"  npe scale   : {float(args.npe_scale if args.npe_shift is None else args.npe_shift):g}")
    print(f"  ftime shift : {float(args.shift if args.ftime_shift is None else args.ftime_shift):g}")
    print(f"  events : {summary['events']:,}")
    print(f"  detectors/event : {summary['detectors']:,}")
    print(f"  scaled nPE hits   : {summary['scaled_npe_hits']:,}")
    print(f"  shifted FirstTime hits : {summary['shifted_ftime_hits']:,}")
    print(f"  inactive hits    : {summary['unchanged_inactive_hits']:,}")
    print(f"  non-finite hits  : {summary['nonfinite_hits']:,}")


if __name__ == "__main__":
    main()