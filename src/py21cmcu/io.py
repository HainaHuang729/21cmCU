"""GPU-accelerated HDF5 lightcone I/O with native CUDA DeviceLZF compression."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from py21cmfast._cfg import config
from py21cmfast.c_21cmfast import ffi, lib

_VALID_COMPRESSIONS = ("device-lzf", "lzf", "none")


def _write_dataset_device_lzf(
    group: h5py.Group,
    name: str,
    arr: np.ndarray,
    *,
    shuffle: bool,
    chunk_z: int | None,
) -> tuple[int, int, bool]:
    """Write a 3D float32 array to HDF5 using CUDA DeviceLZF chunk compression."""
    nx, ny, nz = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
    slice_elems = nx * ny
    if chunk_z is None:
        cz = max(1, min(nz, (512 * 1024) // max(1, slice_elems)))
    else:
        cz = max(1, min(nz, int(chunk_z)))

    dset = group.create_dataset(
        name,
        shape=(nx, ny, nz),
        dtype=np.float32,
        chunks=(nx, ny, cz),
        compression="lzf",
        shuffle=shuffle,
    )

    chunk_elems = slice_elems * cz
    chunk_bytes = chunk_elems * 4
    n_full_chunks = nz // cz
    batch_cap = max(1, (32 * 1024 * 1024) // max(1, chunk_bytes))

    raw_bytes = int(arr.nbytes)
    comp_bytes = 0
    has_nonfinite = False
    filter_mask_raw = 3 if shuffle else 1

    for start_c in range(0, n_full_chunks, batch_cap):
        batch_n = min(batch_cap, n_full_chunks - start_c)
        z0 = start_c * cz
        z1 = z0 + batch_n * cz
        # Transpose chunks so each chunk (nx, ny, cz) is C-contiguous: (batch_n, nx, ny, cz).
        slab = np.ascontiguousarray(arr[:, :, z0:z1], dtype=np.float32)
        chunks_4d = np.ascontiguousarray(
            slab.reshape(nx, ny, batch_n, cz).transpose(2, 0, 1, 3)
        )
        out_buf = np.empty((batch_n, chunk_bytes), dtype=np.uint8)
        out_sizes = ffi.new("size_t[]", batch_n)
        nf_ptr = ffi.new("int *", 0)

        rc = lib.p21_cuda_lzf_compress_batch_f32(
            ffi.cast("const float *", chunks_4d.ctypes.data),
            batch_n,
            chunk_elems,
            1 if shuffle else 0,
            ffi.cast("unsigned char *", out_buf.ctypes.data),
            chunk_bytes,
            out_sizes,
            nf_ptr,
        )
        if rc != 0:
            raise RuntimeError("p21_cuda_lzf_compress_batch_f32 failed on device.")
        if int(nf_ptr[0]) != 0:
            has_nonfinite = True

        raw_mv = memoryview(chunks_4d.view(np.uint8).ravel())
        out_mv = memoryview(out_buf.ravel())
        for i in range(batch_n):
            sz = int(out_sizes[i])
            cz_offset = (start_c + i) * cz
            if sz > 0:
                payload = bytes(out_mv[i * chunk_bytes : i * chunk_bytes + sz])
                dset.id.write_direct_chunk((0, 0, cz_offset), payload, filter_mask=0)
                comp_bytes += sz
            else:
                payload = bytes(raw_mv[i * chunk_bytes : (i + 1) * chunk_bytes])
                dset.id.write_direct_chunk(
                    (0, 0, cz_offset), payload, filter_mask=filter_mask_raw
                )
                comp_bytes += chunk_bytes

    tail_z0 = n_full_chunks * cz
    if tail_z0 < nz:
        tail = np.ascontiguousarray(arr[:, :, tail_z0:nz], dtype=np.float32)
        if not bool(np.all(np.isfinite(tail))):
            has_nonfinite = True
        dset[:, :, tail_z0:nz] = tail
        comp_bytes += int(tail.nbytes)

    return raw_bytes, comp_bytes, has_nonfinite


def save_lightcone(
    lightcone: Any,
    path: str | Path,
    *,
    compression: str = "device-lzf",
    shuffle: bool = True,
    chunk_z: int | None = None,
    fields: list[str] | tuple[str, ...] | None = None,
    attrs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a lightcone object or field dictionary to HDF5 with optional GPU DeviceLZF compression.

    Parameters
    ----------
    lightcone : LightCone or Mapping[str, np.ndarray]
        LightCone output returned by ``run_lightcone`` / ``generate_lightcone`` or a
        mapping of field names to numpy arrays.
    path : str or Path
        Destination HDF5 file path.
    compression : {'device-lzf', 'lzf', 'none'}, default 'device-lzf'
        Compression backend. ``'device-lzf'`` compresses 3D ``float32`` chunks on the GPU
        via native CUDA ``liblzf`` and falls back cleanly to CPU ``h5py`` LZF when CUDA
        is disabled or unavailable.
    shuffle : bool, default True
        Whether to enable HDF5 byte-shuffle before LZF compression.
    chunk_z : int, optional
        Z-thickness of ``(nx, ny, chunk_z)`` HDF5 chunks. Chosen automatically (~2 MB raw
        chunks) when omitted.
    fields : sequence of str, optional
        Subset of field names to save (defaults to all lightcone fields).
    attrs : dict, optional
        Optional HDF5 root attributes to store on the output file.

    Returns
    -------
    dict[str, Any]
        Telemetry dictionary including ``path``, ``compression_used``, ``fields``,
        ``raw_bytes``, ``compressed_bytes``, ``ratio``, ``has_nonfinite``, and ``wall_s``.
    """
    if compression not in _VALID_COMPRESSIONS:
        raise ValueError(
            f"Unsupported compression {compression!r}; expected one of {_VALID_COMPRESSIONS}."
        )

    t0 = time.perf_counter()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(lightcone, dict):
        field_map = dict(lightcone)
        lc_redshifts = None
        lc_distances = None
        global_quantities = None
    else:
        field_map = dict(lightcone.lightcones)
        lc_redshifts = getattr(lightcone, "lightcone_redshifts", None)
        lc_distances = getattr(lightcone, "lightcone_distances", None)
        global_quantities = getattr(lightcone, "global_quantities", None)

    selected_names = list(fields) if fields is not None else list(field_map.keys())
    use_cuda_lzf = (
        compression == "device-lzf"
        and bool(config["USE_CUDA"])
        and bool(lib.p21_cuda_is_compiled())
        and bool(lib.p21_cuda_is_available())
    )
    effective_comp = "device-lzf" if use_cuda_lzf else ("lzf" if compression == "device-lzf" else compression)

    total_raw = 0
    total_comp = 0
    any_nonfinite = False
    saved_fields: list[str] = []

    with h5py.File(out_path, "w") as h5f:
        lc_grp = h5f.create_group("lightcones")
        for name in selected_names:
            if name not in field_map:
                raise KeyError(f"Field {name!r} not found in lightcone.")
            arr = np.ascontiguousarray(field_map[name])
            saved_fields.append(name)

            if use_cuda_lzf and arr.ndim == 3 and arr.dtype == np.float32 and arr.size > 0:
                rb, cb, nf = _write_dataset_device_lzf(
                    lc_grp, name, arr, shuffle=shuffle, chunk_z=chunk_z
                )
                total_raw += rb
                total_comp += cb
                any_nonfinite = any_nonfinite or nf
            else:
                if arr.ndim == 3:
                    nx, ny, nz = arr.shape
                    cz = (
                        max(1, min(nz, int(chunk_z)))
                        if chunk_z is not None
                        else max(1, min(nz, (512 * 1024) // max(1, nx * ny)))
                    )
                    chunks = (nx, ny, cz)
                else:
                    chunks = None

                if effective_comp == "lzf":
                    dset = lc_grp.create_dataset(
                        name, data=arr, chunks=chunks, compression="lzf", shuffle=shuffle
                    )
                else:
                    dset = lc_grp.create_dataset(name, data=arr, chunks=chunks)
                if np.issubdtype(arr.dtype, np.floating) and not bool(np.all(np.isfinite(arr))):
                    any_nonfinite = True
                total_raw += int(arr.nbytes)
                dset_size = dset.id.get_storage_size()
                total_comp += int(dset_size) if dset_size > 0 else int(arr.nbytes)

        if lc_redshifts is not None:
            h5f.create_dataset("lightcone_redshifts", data=np.asarray(lc_redshifts))
        if lc_distances is not None:
            dist_vals = (
                lc_distances.value if hasattr(lc_distances, "value") else np.asarray(lc_distances)
            )
            h5f.create_dataset("lightcone_distances", data=dist_vals)
        if isinstance(global_quantities, dict) and global_quantities:
            gq = h5f.create_group("global_quantities")
            for k, v in global_quantities.items():
                gq.create_dataset(k, data=np.asarray(v))

        h5f.attrs["compression"] = effective_comp
        h5f.attrs["shuffle"] = bool(shuffle)
        h5f.attrs["has_nonfinite"] = bool(any_nonfinite)
        if attrs:
            for k, v in attrs.items():
                h5f.attrs[k] = v

    wall_s = time.perf_counter() - t0
    ratio = (float(total_raw) / float(total_comp)) if total_comp > 0 else 1.0
    return {
        "path": str(out_path),
        "compression_used": effective_comp,
        "fields": saved_fields,
        "raw_bytes": total_raw,
        "compressed_bytes": total_comp,
        "ratio": ratio,
        "has_nonfinite": any_nonfinite,
        "wall_s": wall_s,
    }


__all__ = ["save_lightcone"]
