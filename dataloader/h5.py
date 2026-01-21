import os
from typing import Optional, Dict, Any

import h5py
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyTorch is required to use H5Dataset. Install with `pip install torch`."
    ) from exc


class H5Dataset(Dataset):
    """
    PyTorch Dataset for the 22644_0921_time_shift.h5 file format.

    Output per event:
      - sig:   Tensor float32 shape (2, L)   -> [npe, firstTime]
      - geo:   Tensor float32 shape (3, L)   -> [x, y, z]
      - label: Tensor float32 shape (6,)     -> [Energy, Zenith, Azimuth, X, Y, Z]

    Notes:
      - Geometry (x/y/z) is static; it can be preloaded into memory (default) to
        avoid repeated disk reads.
      - HDF5 handles are re-opened lazily in each worker process to remain
        DataLoader-safe when using num_workers > 0.
    """

    def __init__(
        self,
        h5_path: str,
        signal_key: str = "input",
        label_key: str = "label",
        x_key: str = "xpmt",
        y_key: str = "ypmt",
        z_key: str = "zpmt",
        preload_geometry: bool = True,
    ) -> None:
        super().__init__()
        self.h5_path = os.fspath(h5_path)
        self.signal_key = signal_key
        self.label_key = label_key
        self.x_key = x_key
        self.y_key = y_key
        self.z_key = z_key
        self.preload_geometry = preload_geometry

        # Lazily opened file/datasets for worker safety.
        self._file: Optional[h5py.File] = None
        self._signal_ds = None
        self._label_ds = None
        self._x_ds = None
        self._y_ds = None
        self._z_ds = None

        # Discover shapes without keeping the file open forever.
        with h5py.File(self.h5_path, "r") as f:
            self.length = f[self.signal_key].shape[0]
            self.waveform_len = f[self.signal_key].shape[2]
            if self.preload_geometry:
                self._geo_cache = np.stack(
                    [f[self.x_key][...], f[self.y_key][...], f[self.z_key][...]],
                    axis=0,
                ).astype(np.float32, copy=False)
            else:
                self._geo_cache = None

    def __len__(self) -> int:
        return self.length

    # --- internal helpers -------------------------------------------------
    def _ensure_open(self) -> None:
        """Open HDF5 handles on first use or after worker fork."""
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
            self._signal_ds = self._file[self.signal_key]
            self._label_ds = self._file[self.label_key]
            if not self.preload_geometry:
                self._x_ds = self._file[self.x_key]
                self._y_ds = self._file[self.y_key]
                self._z_ds = self._file[self.z_key]

    def _get_geo(self) -> np.ndarray:
        if self._geo_cache is not None:
            return self._geo_cache
        return np.stack(
            [self._x_ds[...], self._y_ds[...], self._z_ds[...]], axis=0
        ).astype(np.float32, copy=False)

    # --- main API ---------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._ensure_open()

        sig_np = self._signal_ds[idx]  # shape (2, L)
        lbl_np = self._label_ds[idx]   # shape (6,)
        geo_np = self._get_geo()       # shape (3, L)

        sig = torch.from_numpy(np.asarray(sig_np, dtype=np.float32))
        geo = torch.from_numpy(geo_np)
        label = torch.from_numpy(np.asarray(lbl_np, dtype=np.float32))

        return {"sig": sig, "geo": geo, "label": label}



if __name__ == "__main__":
    dataset = H5Dataset(h5_path="/home/work/GENESIS/0121/data/22644_0921_time_shift.h5")
    print(dataset[0])
