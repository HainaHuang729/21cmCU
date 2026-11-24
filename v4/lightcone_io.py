"""Fast HDF5 lightcone writer reusing v3 GPU/parallel shuffle+LZF compressors."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from time import perf_counter

import h5py
import numpy as np
from py21cmfast import __version__
from py21cmfast.io import h5

_SHARED = str(Path(__file__).resolve().parent.parent / "lightcone")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)


def _default_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 3:
        return (min(64, shape[0]), min(64, shape[1]), min(16, shape[2]))
    if len(shape) == 2:
        return (min(4096, shape[0]), min(16, shape[1]))
    return shape


def save_lightcone(
    cone,
    path: str | Path,
    *,
    clobber: bool = False,
    compression: str = "device-lzf",
    lowz_buffer_pixels: int = 0,
    highz_buffer_pixels: int = 0,
    fsync: bool = True,
    workers: int | None = None,
    chunks: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Save a 4.3.2 LightCone using GPU or parallel LZF compression, matching native HDF5 layout."""
    if compression not in ("device-lzf", "parallel-lzf", "none"):
        raise ValueError(f"Unsupported lightcone compression mode: {compression}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "w" if clobber else "a"

    device_fields = []
    parallel_fields = []
    plain_fields = []
    chunks_written = 0
    skipped_lzf_chunks = 0

    t0 = perf_counter()
    with h5py.File(path, file_mode) as fl:
        fl.attrs["lightcone"] = True
        fl.attrs["last_completed_node"] = cone._last_completed_node
        fl.attrs["last_completed_lcidx"] = cone._last_completed_lcidx
        fl.attrs["lowz_buffer_pixels"] = lowz_buffer_pixels
        fl.attrs["highz_buffer_pixels"] = highz_buffer_pixels
        fl.attrs["__version__"] = __version__

        grp = fl.create_group("photon_nonconservation_data")
        for k, v in cone.photon_nonconservation_data.items():
            grp[k] = v

        boxes = fl.create_group("lightcones")
        for k, val in cone.lightcones.items():
            arr = np.asarray(val)
            chunk_shape = chunks if chunks is not None else _default_chunks(arr.shape)
            if (
                compression == "device-lzf"
                and arr.ndim == 3
                and arr.dtype == np.float32
            ):
                device_fields.append((boxes, k, arr, chunk_shape))
            elif compression in ("device-lzf", "parallel-lzf") and arr.dtype in (
                np.float32,
                np.float64,
            ):
                parallel_fields.append((boxes, k, arr, chunk_shape))
            else:
                plain_fields.append((boxes, k, arr))

        if device_fields:
            from device_lzf import DeviceLZFWriter

            with DeviceLZFWriter() as writer:
                writer.write(device_fields)
            chunks_written += writer.chunks
            skipped_lzf_chunks += writer.skipped_lzf_chunks

        if parallel_fields:
            from parallel_lzf import ParallelLZFWriter

            n_workers = workers or min(8, os.cpu_count() or 8)
            with ParallelLZFWriter(workers=n_workers) as writer:
                writer.write(parallel_fields)
            chunks_written += writer.chunks
            skipped_lzf_chunks += writer.skipped_lzf_chunks

        for group, name, arr in plain_fields:
            if compression == "none":
                group.create_dataset(name, data=arr)
            else:
                group.create_dataset(name, data=arr, compression="lzf", shuffle=True)

        global_q = fl.create_group("global_quantities")
        for k, v in cone.global_quantities.items():
            global_q[k] = v

        fl["lightcone_distances"] = cone.lightcone_distances.to_value("Mpc")
        fl.flush()

    h5._write_inputs_to_group(cone.inputs, path)
    write_seconds = perf_counter() - t0

    fsync_seconds = 0.0
    if fsync:
        t_fs = perf_counter()
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        fsync_seconds = perf_counter() - t_fs

    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "compression": compression,
        "chunks_written": chunks_written,
        "skipped_lzf_chunks": skipped_lzf_chunks,
        "write_seconds": write_seconds,
        "fsync_seconds": fsync_seconds,
    }
