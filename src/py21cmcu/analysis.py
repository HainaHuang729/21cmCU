"""GPU-accelerated spectral and lightcone analysis utilities for 21cmCU."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from py21cmfast._cfg import config
from py21cmfast.c_21cmfast import ffi as _ffi
from py21cmfast.c_21cmfast import lib as _lib

if TYPE_CHECKING:
    from py21cmfast.drivers.lightcone import LightCone


def _cuda_ready(use_cuda: bool | None) -> bool:
    enabled = bool(config["USE_CUDA"]) if use_cuda is None else bool(use_cuda)
    return bool(
        enabled
        and _lib.p21_cuda_is_compiled()
        and _lib.p21_cuda_is_available()
    )


def _normalize_box_length(
    box_length: float | tuple[float, float, float],
) -> tuple[float, float, float]:
    if np.ndim(box_length) == 0:
        L = float(box_length)
        return (L, L, L)
    vals = tuple(float(v) for v in box_length)
    if len(vals) != 3:
        raise ValueError(f"box_length must be scalar or 3-tuple, got {box_length!r}")
    return vals


def _default_k_edges(
    nx: int,
    ny: int,
    nz: int,
    lx: float,
    ly: float,
    lz: float,
    n_bins: int,
    k_min: float | None,
    k_max: float | None,
    log_bins: bool,
) -> np.ndarray:
    k_lo = (2.0 * np.pi / max(lx, ly, lz)) if k_min is None else float(k_min)
    k_hi = (
        (np.pi * min(nx / lx, ny / ly, nz / lz) * (1.0 + 1e-6))
        if k_max is None
        else float(k_max)
    )
    if log_bins:
        if k_lo <= 0.0:
            raise ValueError("k_min must be > 0 when log_bins=True")
        return np.geomspace(k_lo, k_hi, n_bins + 1, dtype=np.float64)
    return np.linspace(k_lo, k_hi, n_bins + 1, dtype=np.float64)


def _numpy_power_spectrum_1d(
    field_a: np.ndarray,
    field_b: np.ndarray | None,
    lx: float,
    ly: float,
    lz: float,
    k_edges: np.ndarray,
    skip_dc: bool,
) -> dict[str, np.ndarray]:
    nx, ny, nz = field_a.shape
    n_tot = nx * ny * nz
    fa = np.fft.rfftn(field_a.astype(np.float64, copy=False))
    if field_b is None or field_b is field_a:
        raw_p = (fa.real**2 + fa.imag**2)
    else:
        fb = np.fft.rfftn(field_b.astype(np.float64, copy=False))
        raw_p = fa.real * fb.real + fa.imag * fb.imag

    norm = (lx * ly * lz) / float(n_tot * n_tot)
    raw_p *= norm

    kx = np.fft.fftfreq(nx, d=lx / nx) * (2.0 * np.pi)
    ky = np.fft.fftfreq(ny, d=ly / ny) * (2.0 * np.pi)
    kz = np.fft.rfftfreq(nz, d=lz / nz) * (2.0 * np.pi)
    k_mag = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)

    mult = np.full((1, 1, kz.size), 2, dtype=np.uint64)
    mult[0, 0, 0] = 1
    if nz % 2 == 0:
        mult[0, 0, -1] = 1
    mult = np.broadcast_to(mult, k_mag.shape)

    if skip_dc:
        mask = np.ones_like(k_mag, dtype=bool)
        mask[0, 0, 0] = False
        k_flat = k_mag[mask]
        p_flat = raw_p[mask]
        m_flat = mult[mask]
    else:
        k_flat = k_mag.ravel()
        p_flat = raw_p.ravel()
        m_flat = mult.ravel()

    n_bins = k_edges.size - 1
    bin_idx = np.searchsorted(k_edges, k_flat, side="right") - 1
    valid = (k_flat >= k_edges[0]) & (k_flat < k_edges[-1]) & (bin_idx >= 0) & (bin_idx < n_bins)

    bi = bin_idx[valid]
    mv = m_flat[valid].astype(np.float64)
    counts = np.bincount(bi, weights=m_flat[valid], minlength=n_bins).astype(np.uint64)
    k_sum = np.bincount(bi, weights=mv * k_flat[valid], minlength=n_bins)
    pk_sum = np.bincount(bi, weights=mv * p_flat[valid], minlength=n_bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        k_mean = np.where(counts > 0, k_sum / counts, np.nan)
        pk = np.where(counts > 0, pk_sum / counts, np.nan)
        delta_sq = (k_mean**3) * pk / (2.0 * np.pi**2)

    return {
        "k": k_mean,
        "pk": pk,
        "delta_sq": delta_sq,
        "counts": counts,
        "k_edges": k_edges.copy(),
    }


def _numpy_power_spectrum_2d(
    field_a: np.ndarray,
    field_b: np.ndarray | None,
    lx: float,
    ly: float,
    lz: float,
    los_axis: int,
    kperp_edges: np.ndarray,
    kpara_edges: np.ndarray,
    skip_dc: bool,
) -> dict[str, np.ndarray]:
    nx, ny, nz = field_a.shape
    n_tot = nx * ny * nz
    fa = np.fft.rfftn(field_a.astype(np.float64, copy=False))
    if field_b is None or field_b is field_a:
        raw_p = fa.real**2 + fa.imag**2
    else:
        fb = np.fft.rfftn(field_b.astype(np.float64, copy=False))
        raw_p = fa.real * fb.real + fa.imag * fb.imag

    raw_p *= (lx * ly * lz) / float(n_tot * n_tot)

    kx = np.fft.fftfreq(nx, d=lx / nx) * (2.0 * np.pi)
    ky = np.fft.fftfreq(ny, d=ly / ny) * (2.0 * np.pi)
    kz = np.fft.rfftfreq(nz, d=lz / nz) * (2.0 * np.pi)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

    if los_axis == 0:
        k_para = np.abs(KX)
        k_perp = np.sqrt(KY**2 + KZ**2)
    elif los_axis == 1:
        k_para = np.abs(KY)
        k_perp = np.sqrt(KX**2 + KZ**2)
    else:
        k_para = np.abs(KZ)
        k_perp = np.sqrt(KX**2 + KY**2)

    mult = np.full((1, 1, kz.size), 2, dtype=np.uint64)
    mult[0, 0, 0] = 1
    if nz % 2 == 0:
        mult[0, 0, -1] = 1
    mult = np.broadcast_to(mult, k_para.shape)

    mask = np.ones_like(k_para, dtype=bool)
    if skip_dc:
        mask[0, 0, 0] = False
    kp_flat = k_perp[mask]
    kz_flat = k_para[mask]
    p_flat = raw_p[mask]
    m_flat = mult[mask]

    n_kp = kperp_edges.size - 1
    n_kz = kpara_edges.size - 1
    ip = np.searchsorted(kperp_edges, kp_flat, side="right") - 1
    iz = np.searchsorted(kpara_edges, kz_flat, side="right") - 1
    valid = (
        (kp_flat >= kperp_edges[0])
        & (kp_flat < kperp_edges[-1])
        & (kz_flat >= kpara_edges[0])
        & (kz_flat < kpara_edges[-1])
        & (ip >= 0)
        & (ip < n_kp)
        & (iz >= 0)
        & (iz < n_kz)
    )

    flat_bin = ip[valid] * n_kz + iz[valid]
    mv = m_flat[valid].astype(np.float64)
    n_bins = n_kp * n_kz
    counts = np.bincount(flat_bin, weights=m_flat[valid], minlength=n_bins).astype(np.uint64)
    kp_sum = np.bincount(flat_bin, weights=mv * kp_flat[valid], minlength=n_bins)
    kz_sum = np.bincount(flat_bin, weights=mv * kz_flat[valid], minlength=n_bins)
    pk_sum = np.bincount(flat_bin, weights=mv * p_flat[valid], minlength=n_bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        k_perp_mean = np.where(counts > 0, kp_sum / counts, np.nan).reshape(n_kp, n_kz)
        k_para_mean = np.where(counts > 0, kz_sum / counts, np.nan).reshape(n_kp, n_kz)
        pk = np.where(counts > 0, pk_sum / counts, np.nan).reshape(n_kp, n_kz)

    return {
        "k_perp": k_perp_mean,
        "k_para": k_para_mean,
        "pk": pk,
        "counts": counts.reshape(n_kp, n_kz),
        "kperp_edges": kperp_edges.copy(),
        "kpara_edges": kpara_edges.copy(),
    }


def power_spectrum_1d(
    field: np.ndarray,
    box_length: float | tuple[float, float, float],
    *,
    field_b: np.ndarray | None = None,
    n_bins: int = 20,
    k_min: float | None = None,
    k_max: float | None = None,
    k_edges: np.ndarray | None = None,
    log_bins: bool = True,
    remove_mean: bool = False,
    use_cuda: bool | None = None,
) -> dict[str, np.ndarray]:
    """Compute the 1D spherical auto- or cross-power spectrum on GPU (cuFFT) or CPU."""
    arr_a = np.asarray(field)
    if arr_a.ndim != 3:
        raise ValueError(f"field must be 3D, got shape {arr_a.shape}")
    if field_b is not None:
        arr_b = np.asarray(field_b)
        if arr_b.shape != arr_a.shape:
            raise ValueError(
                f"field_b shape {arr_b.shape} does not match field shape {arr_a.shape}"
            )
    else:
        arr_b = None

    nx, ny, nz = arr_a.shape
    lx, ly, lz = _normalize_box_length(box_length)
    edges = (
        np.asarray(k_edges, dtype=np.float64)
        if k_edges is not None
        else _default_k_edges(nx, ny, nz, lx, ly, lz, n_bins, k_min, k_max, log_bins)
    )
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("k_edges must be 1D with at least 2 entries")
    edges = np.ascontiguousarray(edges, dtype=np.float64)
    actual_bins = edges.size - 1

    if _cuda_ready(use_cuda):
        a32 = np.ascontiguousarray(arr_a, dtype=np.float32)
        b32 = (
            None
            if arr_b is None
            else (a32 if arr_b is arr_a else np.ascontiguousarray(arr_b, dtype=np.float32))
        )
        k_mean = np.empty(actual_bins, dtype=np.float64)
        pk = np.empty(actual_bins, dtype=np.float64)
        delta_sq = np.empty(actual_bins, dtype=np.float64)
        counts = np.empty(actual_bins, dtype=np.uint64)

        rc = _lib.p21_cuda_power_spectrum_1d(
            _ffi.cast("const float *", _ffi.from_buffer(a32)),
            _ffi.NULL if b32 is None else _ffi.cast("const float *", _ffi.from_buffer(b32)),
            int(nx),
            int(ny),
            int(nz),
            float(lx),
            float(ly),
            float(lz),
            int(actual_bins),
            _ffi.cast("const double *", _ffi.from_buffer(edges)),
            int(bool(remove_mean)),
            _ffi.cast("double *", _ffi.from_buffer(k_mean)),
            _ffi.cast("double *", _ffi.from_buffer(pk)),
            _ffi.cast("double *", _ffi.from_buffer(delta_sq)),
            _ffi.cast("unsigned long long *", _ffi.from_buffer(counts)),
        )
        if rc == 0:
            return {
                "k": k_mean,
                "pk": pk,
                "delta_sq": delta_sq,
                "counts": counts,
                "k_edges": edges.copy(),
            }

    return _numpy_power_spectrum_1d(
        arr_a, arr_b, lx, ly, lz, edges, skip_dc=bool(remove_mean)
    )


def power_spectrum_2d(
    field: np.ndarray,
    box_length: float | tuple[float, float, float],
    *,
    field_b: np.ndarray | None = None,
    los_axis: int = 2,
    n_kperp: int = 12,
    n_kpara: int = 12,
    kperp_edges: np.ndarray | None = None,
    kpara_edges: np.ndarray | None = None,
    log_bins: bool = True,
    remove_mean: bool = False,
    use_cuda: bool | None = None,
) -> dict[str, np.ndarray]:
    """Compute the 2D cylindrical power spectrum P(k_perp, k_para) on GPU (cuFFT) or CPU."""
    arr_a = np.asarray(field)
    if arr_a.ndim != 3:
        raise ValueError(f"field must be 3D, got shape {arr_a.shape}")
    if los_axis not in (0, 1, 2):
        raise ValueError(f"los_axis must be 0, 1, or 2, got {los_axis}")
    if field_b is not None:
        arr_b = np.asarray(field_b)
        if arr_b.shape != arr_a.shape:
            raise ValueError(
                f"field_b shape {arr_b.shape} does not match field shape {arr_a.shape}"
            )
    else:
        arr_b = None

    nx, ny, nz = arr_a.shape
    lx, ly, lz = _normalize_box_length(box_length)
    dims = (nx, ny, nz)
    lens = (lx, ly, lz)
    perp_axes = [ax for ax in (0, 1, 2) if ax != los_axis]

    if kperp_edges is None:
        kp_lo = 2.0 * np.pi / max(lens[ax] for ax in perp_axes)
        kp_hi = np.pi * min(dims[ax] / lens[ax] for ax in perp_axes) * (1.0 + 1e-6)
        kp_edges = (
            np.geomspace(kp_lo, kp_hi, n_kperp + 1, dtype=np.float64)
            if log_bins
            else np.linspace(kp_lo, kp_hi, n_kperp + 1, dtype=np.float64)
        )
    else:
        kp_edges = np.ascontiguousarray(kperp_edges, dtype=np.float64)

    if kpara_edges is None:
        kz_lo = 2.0 * np.pi / lens[los_axis]
        kz_hi = np.pi * (dims[los_axis] / lens[los_axis]) * (1.0 + 1e-6)
        kz_edges = (
            np.geomspace(kz_lo, kz_hi, n_kpara + 1, dtype=np.float64)
            if log_bins
            else np.linspace(kz_lo, kz_hi, n_kpara + 1, dtype=np.float64)
        )
    else:
        kz_edges = np.ascontiguousarray(kpara_edges, dtype=np.float64)

    n_kp = kp_edges.size - 1
    n_kz = kz_edges.size - 1

    if _cuda_ready(use_cuda):
        a32 = np.ascontiguousarray(arr_a, dtype=np.float32)
        b32 = (
            None
            if arr_b is None
            else (a32 if arr_b is arr_a else np.ascontiguousarray(arr_b, dtype=np.float32))
        )
        kp_mean = np.empty((n_kp, n_kz), dtype=np.float64)
        kz_mean = np.empty((n_kp, n_kz), dtype=np.float64)
        pk = np.empty((n_kp, n_kz), dtype=np.float64)
        counts = np.empty((n_kp, n_kz), dtype=np.uint64)

        rc = _lib.p21_cuda_power_spectrum_2d(
            _ffi.cast("const float *", _ffi.from_buffer(a32)),
            _ffi.NULL if b32 is None else _ffi.cast("const float *", _ffi.from_buffer(b32)),
            int(nx),
            int(ny),
            int(nz),
            float(lx),
            float(ly),
            float(lz),
            int(los_axis),
            int(n_kp),
            _ffi.cast("const double *", _ffi.from_buffer(kp_edges)),
            int(n_kz),
            _ffi.cast("const double *", _ffi.from_buffer(kz_edges)),
            int(bool(remove_mean)),
            _ffi.cast("double *", _ffi.from_buffer(kp_mean)),
            _ffi.cast("double *", _ffi.from_buffer(kz_mean)),
            _ffi.cast("double *", _ffi.from_buffer(pk)),
            _ffi.cast("unsigned long long *", _ffi.from_buffer(counts)),
        )
        if rc == 0:
            return {
                "k_perp": kp_mean,
                "k_para": kz_mean,
                "pk": pk,
                "counts": counts,
                "kperp_edges": kp_edges.copy(),
                "kpara_edges": kz_edges.copy(),
            }

    return _numpy_power_spectrum_2d(
        arr_a,
        arr_b,
        lx,
        ly,
        lz,
        los_axis,
        kp_edges,
        kz_edges,
        skip_dc=bool(remove_mean),
    )


def _lightcone_chunk_slices(
    lightcone: LightCone,
    quantity: str,
    cross_quantity: str | None,
    n_chunks: int | None,
    chunk_depth: int | None,
) -> tuple[np.ndarray, np.ndarray | None, float, float, float, np.ndarray, np.ndarray]:
    lc_box = np.asarray(lightcone.lightcones[quantity])
    if lc_box.ndim != 3:
        raise ValueError(
            f"lightcone_chunk_power_spectra requires a 3D rectilinear lightcone, got {lc_box.shape}"
        )
    lc_box_b = np.asarray(lightcone.lightcones[cross_quantity]) if cross_quantity else None
    if lc_box_b is not None and lc_box_b.shape != lc_box.shape:
        raise ValueError(
            f"cross_quantity {cross_quantity!r} shape {lc_box_b.shape} != {lc_box.shape}"
        )

    nx, ny, nz = lc_box.shape
    cell_mpc = float(lightcone.inputs.simulation_options.cell_size.value)
    lx = float(nx) * cell_mpc
    ly = float(ny) * cell_mpc

    if chunk_depth is None:
        if n_chunks is None:
            chunk_depth = max(min(nx, ny, nz), 16)
        else:
            chunk_depth = max(nz // max(int(n_chunks), 1), 4)
    n_c = nz // int(chunk_depth) if n_chunks is None else int(n_chunks)
    if n_c <= 0:
        raise ValueError(f"Lightcone LoS length {nz} smaller than chunk_depth {chunk_depth}")

    lc_z = np.asarray(lightcone.lightcone_redshifts, dtype=np.float64)
    edges = np.linspace(0, nz, n_c + 1, dtype=int)
    return lc_box, lc_box_b, cell_mpc, lx, ly, lc_z, edges


def lightcone_chunk_power_spectra(
    lightcone: LightCone,
    quantity: str = "brightness_temp",
    *,
    cross_quantity: str | None = None,
    n_chunks: int | None = None,
    chunk_depth: int | None = None,
    n_bins: int = 16,
    use_cuda: bool | None = None,
    **ps_kwargs,
) -> dict[str, object]:
    """Split a rectilinear lightcone along its LoS axis and compute 1D auto/cross PS per chunk."""
    lc_box, lc_box_b, cell_mpc, lx, ly, lc_z, edges = _lightcone_chunk_slices(
        lightcone, quantity, cross_quantity, n_chunks, chunk_depth
    )

    chunks = []
    chunk_z = []
    for c in range(edges.size - 1):
        lo, hi = int(edges[c]), int(edges[c + 1])
        if hi <= lo:
            continue
        sub_a = np.ascontiguousarray(lc_box[:, :, lo:hi], dtype=np.float32)
        sub_b = (
            np.ascontiguousarray(lc_box_b[:, :, lo:hi], dtype=np.float32)
            if lc_box_b is not None
            else None
        )
        lz = float(hi - lo) * cell_mpc
        ps = power_spectrum_1d(
            sub_a,
            (lx, ly, lz),
            field_b=sub_b,
            n_bins=n_bins,
            use_cuda=use_cuda,
            **ps_kwargs,
        )
        chunks.append(ps)
        chunk_z.append(float(np.mean(lc_z[lo:hi])))

    return {
        "chunk_redshifts": np.asarray(chunk_z, dtype=np.float64),
        "chunks": chunks,
    }


def lightcone_chunk_power_spectra_2d(
    lightcone: LightCone,
    quantity: str = "brightness_temp",
    *,
    cross_quantity: str | None = None,
    n_chunks: int | None = None,
    chunk_depth: int | None = None,
    n_kperp: int = 10,
    n_kpara: int = 10,
    use_cuda: bool | None = None,
    **ps2d_kwargs,
) -> dict[str, object]:
    """Split a rectilinear lightcone along its LoS axis and compute 2D cylindrical PS per chunk."""
    lc_box, lc_box_b, cell_mpc, lx, ly, lc_z, edges = _lightcone_chunk_slices(
        lightcone, quantity, cross_quantity, n_chunks, chunk_depth
    )

    chunks = []
    chunk_z = []
    for c in range(edges.size - 1):
        lo, hi = int(edges[c]), int(edges[c + 1])
        if hi <= lo:
            continue
        sub_a = np.ascontiguousarray(lc_box[:, :, lo:hi], dtype=np.float32)
        sub_b = (
            np.ascontiguousarray(lc_box_b[:, :, lo:hi], dtype=np.float32)
            if lc_box_b is not None
            else None
        )
        lz = float(hi - lo) * cell_mpc
        ps2d = power_spectrum_2d(
            sub_a,
            (lx, ly, lz),
            field_b=sub_b,
            los_axis=2,
            n_kperp=n_kperp,
            n_kpara=n_kpara,
            use_cuda=use_cuda,
            **ps2d_kwargs,
        )
        chunks.append(ps2d)
        chunk_z.append(float(np.mean(lc_z[lo:hi])))

    return {
        "chunk_redshifts": np.asarray(chunk_z, dtype=np.float64),
        "chunks": chunks,
    }
