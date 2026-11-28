"""Higher-order non-Gaussian and morphological topology statistics (Phase 10, v0.9.0-dev)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from py21cmfast.c_21cmfast import ffi as _ffi
from py21cmfast.c_21cmfast import lib as _lib

from ._internal import cuda_ready as _cuda_ready
from ._internal import normalize_box_length as _normalize_box_length
from ._results import BispectrumResult, MFPResult, MinkowskiResult

BispectrumConfig = Literal["equilateral", "squeezed", "folded", "isosceles"]


def _resolve_triangle_specs(
    nx: int,
    ny: int,
    nz: int,
    lx: float,
    ly: float,
    lz: float,
    *,
    config: BispectrumConfig,
    k_values: Sequence[float] | np.ndarray | None,
    triangles: Sequence[tuple[float, float, float]] | None,
    dk: float | None,
    n_bins: int,
    k_min: float | None,
    k_max: float | None,
    k1_fixed: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k_fund = 2.0 * np.pi / max(lx, ly, lz)
    k_nyq = np.pi * min(nx / lx, ny / ly, nz / lz)

    if triangles is not None:
        tri_arr = np.asarray(triangles, dtype=np.float64)
        if tri_arr.ndim != 2 or tri_arr.shape[1] != 3 or tri_arr.shape[0] == 0:
            raise ValueError(f"triangles must have shape (M, 3) with M >= 1, got {tri_arr.shape}")
        k1 = tri_arr[:, 0]
        k2 = tri_arr[:, 1]
        k3 = tri_arr[:, 2]
        dk_val = np.full(
            tri_arr.shape[0],
            float(dk) if dk is not None else max(k_fund * 2.0, 0.05),
            dtype=np.float64,
        )
        half = 0.5 * dk_val
        return (
            k1,
            k2,
            k3,
            dk_val,
            np.maximum(0.0, k1 - half),
            k1 + half,
            np.maximum(0.0, k2 - half),
            k2 + half,
            np.maximum(0.0, k3 - half),
            k3 + half,
        )

    k_lo_def = max(k_fund * 2.0, float(k_min) if k_min is not None else k_fund * 2.5)
    k_hi_def = float(k_max) if k_max is not None else k_nyq * 0.75
    if k_hi_def <= k_lo_def:
        k_hi_def = k_lo_def * 2.0

    if k_values is not None:
        k_vals = np.asarray(k_values, dtype=np.float64).ravel()
        if k_vals.size == 0:
            raise ValueError("k_values must not be empty")
    else:
        if int(n_bins) < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        k_vals = np.geomspace(k_lo_def, k_hi_def, int(n_bins), dtype=np.float64)

    k_fix = float(k1_fixed) if k1_fixed is not None else float(k_lo_def)
    if config == "equilateral":
        k1 = k_vals.copy()
        k2 = k_vals.copy()
        k3 = k_vals.copy()
    elif config == "squeezed":
        k1 = np.full_like(k_vals, k_fix)
        k2 = k_vals.copy()
        k3 = k_vals.copy()
    elif config == "folded":
        k1 = k_vals.copy()
        k2 = 0.5 * k_vals
        k3 = 0.5 * k_vals
    elif config == "isosceles":
        k1 = k_vals.copy()
        k2 = k_vals.copy()
        k3 = np.full_like(k_vals, k_fix)
    else:
        raise ValueError(
            f"Unsupported config {config!r}; expected 'equilateral', 'squeezed', 'folded', or 'isosceles'"
        )

    def _half_width(k_arr: np.ndarray) -> np.ndarray:
        if dk is not None:
            return np.full_like(k_arr, 0.5 * float(dk))
        return np.maximum(0.5 * np.maximum(k_fund * 2.2, 0.25 * k_arr), k_fund * 1.25)

    h1 = _half_width(k1)
    h2 = _half_width(k2)
    h3 = _half_width(k3)
    dk_arr = 2.0 * h1
    return (
        k1,
        k2,
        k3,
        dk_arr,
        np.maximum(0.0, k1 - h1),
        k1 + h1,
        np.maximum(0.0, k2 - h2),
        k2 + h2,
        np.maximum(0.0, k3 - h3),
        k3 + h3,
    )


def _numpy_bispectrum_3d(
    field: np.ndarray,
    lx: float,
    ly: float,
    lz: float,
    remove_mean: bool,
    k1_lo: np.ndarray,
    k1_hi: np.ndarray,
    k2_lo: np.ndarray,
    k2_hi: np.ndarray,
    k3_lo: np.ndarray,
    k3_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = field.shape
    n_tot = nx * ny * nz
    vol = lx * ly * lz
    arr64 = field.astype(np.float64, copy=False)
    f_src = np.fft.rfftn(arr64)
    if remove_mean:
        f_src[0, 0, 0] = 0.0 + 0.0j

    kx = np.fft.fftfreq(nx, d=lx / nx) * (2.0 * np.pi)
    ky = np.fft.fftfreq(ny, d=ly / ny) * (2.0 * np.pi)
    kz = np.fft.rfftfreq(nz, d=lz / nz) * (2.0 * np.pi)
    k_mag = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)

    shell_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}

    def _get_shell(lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
        key = (float(lo), float(hi))
        cached = shell_cache.get(key)
        if cached is not None:
            return cached
        mask = (k_mag >= lo) & (k_mag < hi)
        if remove_mean:
            mask[0, 0, 0] = False
        d_box = np.fft.irfftn(np.where(mask, f_src, 0.0), s=(nx, ny, nz), axes=(0, 1, 2))
        i_box = np.fft.irfftn(mask.astype(np.float64), s=(nx, ny, nz), axes=(0, 1, 2))
        if len(shell_cache) >= 6:
            shell_cache.pop(next(iter(shell_cache)))
        shell_cache[key] = (d_box, i_box)
        return d_box, i_box

    m_tri = k1_lo.size
    b_out = np.empty(m_tri, dtype=np.float64)
    q_out = np.empty(m_tri, dtype=np.float64)
    pk1_out = np.empty(m_tri, dtype=np.float64)
    pk2_out = np.empty(m_tri, dtype=np.float64)
    pk3_out = np.empty(m_tri, dtype=np.float64)
    ntri_out = np.empty(m_tri, dtype=np.float64)

    for t in range(m_tri):
        d1, w1 = _get_shell(k1_lo[t], k1_hi[t])
        d2, w2 = _get_shell(k2_lo[t], k2_hi[t])
        d3, w3 = _get_shell(k3_lo[t], k3_hi[t])

        s_d = float(np.sum(d1 * d2 * d3))
        s_i = float(np.sum(w1 * w2 * w3))
        ntri = float(n_tot) ** 2 * s_i
        b_val = ((vol**2) / (float(n_tot) ** 3)) * (s_d / s_i) if s_i > 0.0 else np.nan

        def _pk(d_arr: np.ndarray, w_arr: np.ndarray) -> float:
            wsq = float(np.sum(w_arr * w_arr))
            if wsq <= 0.0:
                return float("nan")
            return (vol / (float(n_tot) ** 2)) * (float(np.sum(d_arr * d_arr)) / wsq)

        p1 = _pk(d1, w1)
        p2 = _pk(d2, w2)
        p3 = _pk(d3, w3)
        denom = p1 * p2 + p2 * p3 + p3 * p1
        q_val = (
            b_val / denom
            if np.isfinite(b_val) and np.isfinite(denom) and abs(denom) > 0.0
            else np.nan
        )
        ntri_out[t] = ntri
        b_out[t] = b_val
        pk1_out[t] = p1
        pk2_out[t] = p2
        pk3_out[t] = p3
        q_out[t] = q_val

    return b_out, q_out, pk1_out, pk2_out, pk3_out, ntri_out


def bispectrum_3d(
    field: np.ndarray,
    box_length: float | tuple[float, float, float],
    *,
    config: BispectrumConfig = "equilateral",
    k_values: Sequence[float] | np.ndarray | None = None,
    triangles: Sequence[tuple[float, float, float]] | None = None,
    dk: float | None = None,
    n_bins: int = 8,
    k_min: float | None = None,
    k_max: float | None = None,
    k1_fixed: float | None = None,
    remove_mean: bool = True,
    use_cuda: bool | None = None,
) -> dict[str, Any]:
    """Compute the 3D FFT bispectrum $B(k_1, k_2, k_3)$ and reduced bispectrum $Q$.

    Uses Scoccimarro (2000) / Watkinson et al. (2017) FFT shell-product estimators on GPU
    (when available) or NumPy CPU fallback.
    """
    raw = np.asarray(field)
    if raw.ndim != 3:
        raise ValueError(f"Expected 3D field array, got shape {raw.shape}")
    if raw.size == 0:
        raise ValueError("Cannot compute bispectrum of an empty array")
    lengths = _normalize_box_length(box_length)
    nx, ny, nz = raw.shape
    lx, ly, lz = lengths

    k1, k2, k3, dk_arr, k1_lo, k1_hi, k2_lo, k2_hi, k3_lo, k3_hi = _resolve_triangle_specs(
        nx,
        ny,
        nz,
        lx,
        ly,
        lz,
        config=config,
        k_values=k_values,
        triangles=triangles,
        dk=dk,
        n_bins=n_bins,
        k_min=k_min,
        k_max=k_max,
        k1_fixed=k1_fixed,
    )

    n_tri = k1.size
    if _cuda_ready(use_cuda):
        arr_f32 = np.ascontiguousarray(raw, dtype=np.float32)
        b_out = np.empty(n_tri, dtype=np.float64)
        q_out = np.empty(n_tri, dtype=np.float64)
        pk1_out = np.empty(n_tri, dtype=np.float64)
        pk2_out = np.empty(n_tri, dtype=np.float64)
        pk3_out = np.empty(n_tri, dtype=np.float64)
        ntri_out = np.empty(n_tri, dtype=np.float64)
        rc = _lib.p21_cuda_bispectrum_3d(
            _ffi.cast("const float *", _ffi.from_buffer(arr_f32)),
            int(nx),
            int(ny),
            int(nz),
            float(lx),
            float(ly),
            float(lz),
            1 if remove_mean else 0,
            int(n_tri),
            _ffi.cast("const double *", _ffi.from_buffer(k1_lo)),
            _ffi.cast("const double *", _ffi.from_buffer(k1_hi)),
            _ffi.cast("const double *", _ffi.from_buffer(k2_lo)),
            _ffi.cast("const double *", _ffi.from_buffer(k2_hi)),
            _ffi.cast("const double *", _ffi.from_buffer(k3_lo)),
            _ffi.cast("const double *", _ffi.from_buffer(k3_hi)),
            _ffi.cast("double *", _ffi.from_buffer(b_out)),
            _ffi.cast("double *", _ffi.from_buffer(q_out)),
            _ffi.cast("double *", _ffi.from_buffer(pk1_out)),
            _ffi.cast("double *", _ffi.from_buffer(pk2_out)),
            _ffi.cast("double *", _ffi.from_buffer(pk3_out)),
            _ffi.cast("double *", _ffi.from_buffer(ntri_out)),
        )
        if rc == 0:
            return BispectrumResult(
                k1=k1,
                k2=k2,
                k3=k3,
                B=b_out,
                Q=q_out,
                pk1=pk1_out,
                pk2=pk2_out,
                pk3=pk3_out,
                n_triangles=ntri_out,
                dk=dk_arr,
                config=config,
            )

    b_out, q_out, pk1_out, pk2_out, pk3_out, ntri_out = _numpy_bispectrum_3d(
        raw,
        lx,
        ly,
        lz,
        remove_mean,
        k1_lo,
        k1_hi,
        k2_lo,
        k2_hi,
        k3_lo,
        k3_hi,
    )
    return BispectrumResult(
        k1=k1,
        k2=k2,
        k3=k3,
        B=b_out,
        Q=q_out,
        pk1=pk1_out,
        pk2=pk2_out,
        pk3=pk3_out,
        n_triangles=ntri_out,
        dk=dk_arr,
        config=config,
    )


def _numpy_mfp_bsd(
    field: np.ndarray,
    lx: float,
    ly: float,
    lz: float,
    threshold: float,
    below_threshold: bool,
    n_rays: int,
    step_cells: float,
    max_dist: float,
    r_edges: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float, float, int]:
    nx, ny, nz = field.shape
    mask = (field < threshold) if below_threshold else (field >= threshold)
    ion_coords = np.argwhere(mask)
    n_ion = int(ion_coords.shape[0])
    n_bins = r_edges.size - 1
    if n_ion == 0:
        return np.zeros(n_bins, dtype=np.uint64), 0.0, 0.0, 0

    dx = lx / nx
    dy = ly / ny
    dz = lz / nz
    ds = float(step_cells) * min(dx, dy, dz)
    max_steps = max(1, int(np.ceil(max_dist / ds)))

    rng = np.random.default_rng(int(seed))
    chunk = 16384
    all_r = np.empty(n_rays, dtype=np.float64)
    offset = 0
    while offset < n_rays:
        cur = min(chunk, n_rays - offset)
        picks = rng.integers(0, n_ion, size=cur)
        coords = ion_coords[picks]
        cx = (coords[:, 0].astype(np.float64) + 0.5) * dx
        cy = (coords[:, 1].astype(np.float64) + 0.5) * dy
        cz = (coords[:, 2].astype(np.float64) + 0.5) * dz

        cos_t = rng.uniform(-1.0, 1.0, size=cur)
        sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
        phi = rng.uniform(0.0, 2.0 * np.pi, size=cur)
        vx = sin_t * np.cos(phi)
        vy = sin_t * np.sin(phi)
        vz = cos_t

        r_cur = np.full(cur, max_dist, dtype=np.float64)
        active = np.arange(cur, dtype=np.int64)
        for s in range(1, max_steps + 1):
            if active.size == 0:
                break
            cx[active] += vx[active] * ds
            cy[active] += vy[active] * ds
            cz[active] += vz[active] * ds
            gx = np.floor(cx[active] / dx).astype(np.int64) % nx
            gy = np.floor(cy[active] / dy).astype(np.int64) % ny
            gz = np.floor(cz[active] / dz).astype(np.int64) % nz
            hit = ~mask[gx, gy, gz]
            if np.any(hit):
                r_cur[active[hit]] = (float(s) - 0.5) * ds
                active = active[~hit]
        all_r[offset : offset + cur] = np.minimum(r_cur, max_dist)
        offset += cur

    counts, _ = np.histogram(all_r, bins=r_edges)
    mean_r = float(np.mean(all_r))
    std_r = float(np.std(all_r))
    return counts.astype(np.uint64), mean_r, std_r, n_ion


def bubble_size_distribution_mfp(
    field: np.ndarray,
    box_length: float | tuple[float, float, float],
    *,
    threshold: float = 0.5,
    below_threshold: bool = True,
    n_rays: int = 100_000,
    n_bins: int = 32,
    r_min: float | None = None,
    r_max: float | None = None,
    log_bins: bool = True,
    step_cells: float = 0.5,
    seed: int = 42,
    use_cuda: bool | None = None,
) -> dict[str, Any]:
    """Compute Mean-Free-Path ionized Bubble Size Distribution (MFP BSD) via ray tracing."""
    raw = np.asarray(field)
    if raw.ndim != 3:
        raise ValueError(f"Expected 3D field array, got shape {raw.shape}")
    if raw.size == 0:
        raise ValueError("Cannot compute MFP BSD of an empty array")
    if n_rays < 1:
        raise ValueError(f"n_rays must be >= 1, got {n_rays}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if step_cells <= 0.0:
        raise ValueError(f"step_cells must be > 0, got {step_cells}")

    lengths = _normalize_box_length(box_length)
    nx, ny, nz = raw.shape
    lx, ly, lz = lengths
    dx, dy, dz = lx / nx, ly / ny, lz / nz

    r_lo = float(r_min) if r_min is not None else 0.25 * min(dx, dy, dz)
    r_hi = float(r_max) if r_max is not None else max(lengths)
    if r_hi <= r_lo:
        raise ValueError(f"r_max ({r_hi}) must be > r_min ({r_lo})")
    if log_bins:
        if r_lo <= 0.0:
            raise ValueError(f"r_min must be > 0 when log_bins=True, got {r_lo}")
        r_edges = np.geomspace(r_lo, r_hi, int(n_bins) + 1, dtype=np.float64)
        r_centers = np.sqrt(r_edges[:-1] * r_edges[1:])
    else:
        r_edges = np.linspace(r_lo, r_hi, int(n_bins) + 1, dtype=np.float64)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    counts: np.ndarray | None = None
    mean_r = 0.0
    std_r = 0.0
    n_ion = 0

    if _cuda_ready(use_cuda):
        arr_f32 = np.ascontiguousarray(raw, dtype=np.float32)
        hist_buf = np.zeros(int(n_bins), dtype=np.uint64)
        mean_out = _ffi.new("double *")
        std_out = _ffi.new("double *")
        nion_out = _ffi.new("unsigned long long *")
        rc = _lib.p21_cuda_mfp_bsd(
            _ffi.cast("const float *", _ffi.from_buffer(arr_f32)),
            int(nx),
            int(ny),
            int(nz),
            float(lx),
            float(ly),
            float(lz),
            float(threshold),
            1 if below_threshold else 0,
            int(n_rays),
            float(step_cells),
            float(r_hi),
            int(n_bins),
            float(r_lo),
            float(r_hi),
            1 if log_bins else 0,
            int(seed) & 0xFFFFFFFFFFFFFFFF,
            _ffi.cast("unsigned long long *", _ffi.from_buffer(hist_buf)),
            mean_out,
            std_out,
            nion_out,
        )
        if rc == 0:
            counts = hist_buf
            mean_r = float(mean_out[0])
            std_r = float(std_out[0])
            n_ion = int(nion_out[0])

    if counts is None:
        counts, mean_r, std_r, n_ion = _numpy_mfp_bsd(
            raw,
            lx,
            ly,
            lz,
            float(threshold),
            bool(below_threshold),
            int(n_rays),
            float(step_cells),
            float(r_hi),
            r_edges,
            int(seed),
        )

    tot_binned = float(np.sum(counts))
    dr = np.diff(r_edges)
    pdf = counts.astype(np.float64) / (tot_binned * dr) if tot_binned > 0.0 else np.zeros_like(dr)
    rdp_dr = r_centers * pdf
    return MFPResult(
        r_edges=r_edges,
        r_centers=r_centers,
        counts=counts,
        pdf=pdf,
        rdp_dr=rdp_dr,
        mean_r=mean_r,
        std_r=std_r,
        ionized_fraction=float(n_ion) / float(nx * ny * nz),
        n_rays=int(n_rays),
    )


def _numpy_minkowski_3d(
    field: np.ndarray,
    lx: float,
    ly: float,
    lz: float,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = field.shape
    n_tot = float(nx * ny * nz)
    a = float(np.cbrt((lx / nx) * (ly / ny) * (lz / nz)))
    vol = lx * ly * lz

    m = thresholds.size
    v0 = np.empty(m, dtype=np.float64)
    v1 = np.empty(m, dtype=np.float64)
    v2 = np.empty(m, dtype=np.float64)
    v3 = np.empty(m, dtype=np.float64)
    euler = np.empty(m, dtype=np.int64)

    for i, thr in enumerate(thresholds):
        b = (field >= thr).astype(np.uint8)
        bx = np.roll(b, -1, axis=0)
        by = np.roll(b, -1, axis=1)
        bz = np.roll(b, -1, axis=2)
        bxy = np.roll(by, -1, axis=0)
        bxz = np.roll(bz, -1, axis=0)
        byz = np.roll(bz, -1, axis=1)
        bxyz = np.roll(byz, -1, axis=0)

        n3 = int(b.sum(dtype=np.int64))
        n2 = int((b | bx).sum(dtype=np.int64) + (b | by).sum(dtype=np.int64) + (b | bz).sum(dtype=np.int64))
        n1 = int(
            (b | by | bz | byz).sum(dtype=np.int64)
            + (b | bx | bz | bxz).sum(dtype=np.int64)
            + (b | bx | by | bxy).sum(dtype=np.int64)
        )
        n0 = int((b | bx | by | bz | bxy | bxz | byz | bxyz).sum(dtype=np.int64))

        chi = n0 - n1 + n2 - n3
        v0[i] = n3 / n_tot
        v1[i] = (2.0 / (9.0 * a * n_tot)) * (n2 - 3 * n3)
        v2[i] = (2.0 / (9.0 * a * a * n_tot)) * (n1 - 2 * n2 + 3 * n3)
        euler[i] = chi
        v3[i] = float(chi) / vol

    return v0, v1, v2, v3, euler


def minkowski_functionals_3d(
    field: np.ndarray,
    box_length: float | tuple[float, float, float],
    *,
    thresholds: Sequence[float] | np.ndarray | None = None,
    n_thresholds: int = 21,
    standardize: bool = False,
    use_cuda: bool | None = None,
) -> dict[str, Any]:
    """Compute exact 3D Minkowski Functionals $(V_0, V_1, V_2, V_3)$ and Euler characteristic."""
    raw = np.asarray(field)
    if raw.ndim != 3:
        raise ValueError(f"Expected 3D field array, got shape {raw.shape}")
    if raw.size == 0:
        raise ValueError("Cannot compute Minkowski functionals of an empty array")
    lengths = _normalize_box_length(box_length)
    nx, ny, nz = raw.shape
    lx, ly, lz = lengths

    arr = np.ascontiguousarray(raw, dtype=np.float32)
    if standardize:
        arr64 = arr.astype(np.float64)
        mu = float(np.mean(arr64))
        sig = float(np.std(arr64))
        if sig > 0.0:
            arr = ((arr64 - mu) / sig).astype(np.float32)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

    if thresholds is not None:
        thr = np.ascontiguousarray(np.asarray(thresholds, dtype=np.float64).ravel(), dtype=np.float64)
        if thr.size == 0:
            raise ValueError("thresholds must not be empty")
    else:
        if int(n_thresholds) < 1:
            raise ValueError(f"n_thresholds must be >= 1, got {n_thresholds}")
        if standardize:
            thr = np.linspace(-3.0, 3.0, int(n_thresholds), dtype=np.float64)
        else:
            lo = float(np.min(arr))
            hi = float(np.max(arr))
            thr = np.linspace(lo, hi, int(n_thresholds), dtype=np.float64)

    if _cuda_ready(use_cuda):
        v0 = np.empty(thr.size, dtype=np.float64)
        v1 = np.empty(thr.size, dtype=np.float64)
        v2 = np.empty(thr.size, dtype=np.float64)
        v3 = np.empty(thr.size, dtype=np.float64)
        euler = np.empty(thr.size, dtype=np.int64)
        rc = _lib.p21_cuda_minkowski_3d(
            _ffi.cast("const float *", _ffi.from_buffer(arr)),
            int(nx),
            int(ny),
            int(nz),
            float(lx),
            float(ly),
            float(lz),
            int(thr.size),
            _ffi.cast("const double *", _ffi.from_buffer(thr)),
            _ffi.cast("double *", _ffi.from_buffer(v0)),
            _ffi.cast("double *", _ffi.from_buffer(v1)),
            _ffi.cast("double *", _ffi.from_buffer(v2)),
            _ffi.cast("double *", _ffi.from_buffer(v3)),
            _ffi.cast("long long *", _ffi.from_buffer(euler)),
        )
        if rc == 0:
            return MinkowskiResult(
                thresholds=thr,
                V0=v0,
                V1=v1,
                V2=v2,
                V3=v3,
                euler=euler,
                standardized=bool(standardize),
            )

    v0, v1, v2, v3, euler = _numpy_minkowski_3d(arr, lx, ly, lz, thr)
    return MinkowskiResult(
        thresholds=thr,
        V0=v0,
        V1=v1,
        V2=v2,
        V3=v3,
        euler=euler,
        standardized=bool(standardize),
    )


__all__ = [
    "bispectrum_3d",
    "bubble_size_distribution_mfp",
    "minkowski_functionals_3d",
]
