"""Foreground wedge filtering, thermal noise injection, and Sobol sampling for 21cm SBI."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18
from scipy.stats import qmc

from py21cmfast.c_21cmfast import ffi as _ffi
from py21cmfast.c_21cmfast import lib as _lib

from .analysis import _cuda_ready, _normalize_box_length

_C_KMS = 299792.458
_REST_FREQ_21CM_MHZ = 1420.40575177


def horizon_slope(
    redshift: float,
    *,
    cosmo: Any = None,
    mode: Literal["horizon", "fov"] = "horizon",
    fov_rad: float | None = None,
) -> float:
    """Compute the 21cm foreground wedge slope m(z) = H(z) D_M(z) / [c (1 + z)]."""
    cobj = Planck18 if cosmo is None else cosmo
    z = float(redshift)
    if z < 0.0:
        raise ValueError(f"redshift must be >= 0, got {redshift}")
    hz = float(cobj.H(z).to_value(u.km / u.s / u.Mpc))
    dm = float(cobj.comoving_transverse_distance(z).to_value(u.Mpc))
    slope = (hz * dm) / (_C_KMS * (1.0 + z))
    if mode == "fov":
        if fov_rad is None or fov_rad <= 0.0:
            raise ValueError("fov_rad > 0 is required when mode='fov'")
        slope *= float(np.sin(float(fov_rad)))
    elif mode != "horizon":
        raise ValueError(f"Unknown wedge mode {mode!r}; expected 'horizon' or 'fov'")
    return float(slope)


def horizon_wedge_mask(
    k_perp: np.ndarray | float,
    k_para: np.ndarray | float,
    redshift: float,
    *,
    cosmo: Any = None,
    buffer_h_mpc: float = 0.0,
    mode: Literal["horizon", "fov"] = "horizon",
    fov_rad: float | None = None,
) -> np.ndarray:
    """Return a boolean EoR-window mask (True = clean, False = inside foreground wedge)."""
    cobj = Planck18 if cosmo is None else cosmo
    slope = horizon_slope(redshift, cosmo=cobj, mode=mode, fov_rad=fov_rad)
    buffer_mpc = float(buffer_h_mpc) * float(cobj.h)
    kp = np.asarray(k_perp, dtype=np.float64)
    kz = np.asarray(k_para, dtype=np.float64)
    return np.abs(kz) >= (slope * kp + buffer_mpc)


def apply_wedge_filter_3d(
    field: np.ndarray,
    box_length: float | tuple[float, float, float],
    redshift: float,
    *,
    los_axis: int = 2,
    buffer_h_mpc: float = 0.0,
    mode: Literal["horizon", "fov"] = "horizon",
    fov_rad: float | None = None,
    cosmo: Any = None,
    use_cuda: bool | None = None,
) -> np.ndarray:
    """Zero out Fourier modes inside the 21cm foreground horizon wedge on a 3D box."""
    raw = np.asarray(field)
    if raw.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {raw.shape}")
    if los_axis not in (0, 1, 2):
        raise ValueError(f"los_axis must be 0, 1, or 2, got {los_axis}")

    lengths = _normalize_box_length(box_length)
    cobj = Planck18 if cosmo is None else cosmo

    if _cuda_ready(use_cuda):
        arr_f32 = np.ascontiguousarray(raw, dtype=np.float32)
        out_f32 = np.empty_like(arr_f32, dtype=np.float32)
        slope = horizon_slope(redshift, cosmo=cobj, mode=mode, fov_rad=fov_rad)
        buffer_mpc = float(buffer_h_mpc) * float(cobj.h)
        nx, ny, nz = arr_f32.shape
        rc = _lib.p21_cuda_wedge_filter_3d(
            _ffi.cast("const float *", _ffi.from_buffer(arr_f32)),
            _ffi.cast("float *", _ffi.from_buffer(out_f32)),
            int(nx),
            int(ny),
            int(nz),
            float(lengths[0]),
            float(lengths[1]),
            float(lengths[2]),
            int(los_axis),
            float(slope),
            float(buffer_mpc),
        )
        if rc == 0:
            return out_f32

    arr = np.asarray(raw, dtype=np.float64)
    work = np.moveaxis(arr, los_axis, 2)
    l_perm = tuple(
        lengths[i]
        for i in ([0, 1, 2] if los_axis == 2 else ([1, 2, 0] if los_axis == 0 else [0, 2, 1]))
    )
    nx, ny, nz = work.shape
    lx, ly, lz = l_perm

    kx = np.fft.fftfreq(nx, d=lx / nx) * (2.0 * np.pi)
    ky = np.fft.fftfreq(ny, d=ly / ny) * (2.0 * np.pi)
    kz = np.fft.rfftfreq(nz, d=lz / nz) * (2.0 * np.pi)

    k_perp = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2)
    k_para = np.abs(kz)[None, None, :]
    keep = horizon_wedge_mask(
        k_perp,
        k_para,
        redshift,
        cosmo=cobj,
        buffer_h_mpc=buffer_h_mpc,
        mode=mode,
        fov_rad=fov_rad,
    )

    kspace = np.fft.rfftn(work)
    kspace *= keep
    filtered = np.fft.irfftn(kspace, s=(nx, ny, nz), axes=(0, 1, 2))
    return np.moveaxis(filtered, 2, los_axis).astype(np.float32)


def add_thermal_noise_3d(
    field: np.ndarray,
    box_length: float | tuple[float, float, float],
    *,
    sigma_mk: float | None = None,
    redshift: float = 8.0,
    t_int_hours: float = 1000.0,
    t_sys_k: float | None = None,
    n_ant: int = 350,
    seed: int = 0,
    use_cuda: bool | None = None,
) -> np.ndarray:
    """Add deterministic zero-mean Gaussian thermal noise [mK] to a 3D 21cm brightness field."""
    raw = np.asarray(field)
    if raw.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {raw.shape}")

    if sigma_mk is None:
        lengths = _normalize_box_length(box_length)
        if t_int_hours <= 0.0 or n_ant < 2:
            raise ValueError("t_int_hours > 0 and n_ant >= 2 are required for noise estimation")
        tsys = (
            float(t_sys_k)
            if t_sys_k is not None
            else 100.0 + 60.0 * ((1.0 + float(redshift)) / 4.73) ** 2.55
        )
        # Compact baseline scaling anchored near HERA-350 1000 h at 2 Mpc cells
        cell_vol = (lengths[0] / raw.shape[0]) * (lengths[1] / raw.shape[1]) * (
            lengths[2] / raw.shape[2]
        )
        rms = (
            2.5
            * (tsys / 280.0)
            * np.sqrt((1000.0 / float(t_int_hours)) * (350.0 / float(n_ant)) * (8.0 / max(cell_vol, 1e-3)))
        )
    else:
        rms = float(sigma_mk)
        if rms < 0.0:
            raise ValueError(f"sigma_mk must be >= 0, got {sigma_mk}")

    if _cuda_ready(use_cuda):
        arr_f32 = np.ascontiguousarray(raw, dtype=np.float32)
        out_f32 = np.empty_like(arr_f32, dtype=np.float32)
        rc = _lib.p21_cuda_add_thermal_noise_3d(
            _ffi.cast("const float *", _ffi.from_buffer(arr_f32)),
            _ffi.cast("float *", _ffi.from_buffer(out_f32)),
            int(arr_f32.size),
            float(rms),
            int(seed) & 0xFFFFFFFFFFFFFFFF,
        )
        if rc == 0:
            return out_f32

    arr = np.asarray(raw, dtype=np.float32)
    if rms == 0.0:
        return arr.copy()
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(loc=0.0, scale=rms, size=arr.shape).astype(np.float32)
    return (arr + noise).astype(np.float32)


def field_moments_and_pdf(
    field: np.ndarray,
    *,
    n_bins: int = 32,
    vmin: float | None = None,
    vmax: float | None = None,
    use_cuda: bool | None = None,
) -> dict[str, Any]:
    """Compute mean, variance, skewness, excess kurtosis, and binned 1-point PDF on GPU or CPU."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    arr = np.ascontiguousarray(field, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Cannot compute moments of an empty array")

    v_lo = float(np.min(arr)) if vmin is None else float(vmin)
    v_hi = float(np.max(arr)) if vmax is None else float(vmax)
    if v_hi <= v_lo:
        v_lo -= 0.5
        v_hi += 0.5
    bin_edges = np.linspace(v_lo, v_hi, int(n_bins) + 1, dtype=np.float64)
    bin_width = float(bin_edges[1] - bin_edges[0])

    if _cuda_ready(use_cuda):
        mean_out = _ffi.new("double *")
        var_out = _ffi.new("double *")
        skew_out = _ffi.new("double *")
        kurt_out = _ffi.new("double *")
        hist = np.zeros(int(n_bins), dtype=np.uint64)
        rc = _lib.p21_cuda_field_moments_and_pdf(
            _ffi.cast("const float *", _ffi.from_buffer(arr)),
            int(arr.size),
            int(n_bins),
            v_lo,
            v_hi,
            mean_out,
            var_out,
            skew_out,
            kurt_out,
            _ffi.cast("unsigned long long *", _ffi.from_buffer(hist)),
        )
        if rc == 0:
            mean = float(mean_out[0])
            var = max(0.0, float(var_out[0]))
            skew = float(skew_out[0])
            kurt = float(kurt_out[0])
            total_in_range = float(np.sum(hist))
            pdf = (
                hist.astype(np.float64) / (total_in_range * bin_width)
                if total_in_range > 0.0
                else np.zeros(int(n_bins), dtype=np.float64)
            )
            return {
                "mean": mean,
                "std": float(np.sqrt(var)),
                "variance": var,
                "skewness": skew,
                "kurtosis": kurt,
                "hist": hist,
                "bin_edges": bin_edges,
                "pdf": pdf,
            }

    flat = arr.astype(np.float64, copy=False).ravel()
    mean = float(np.mean(flat))
    diff = flat - mean
    m2 = float(np.mean(diff * diff))
    if m2 > 0.0:
        m3 = float(np.mean(diff * diff * diff))
        m4 = float(np.mean((diff * diff) ** 2))
        skew = m3 / (np.sqrt(m2) * m2)
        kurt = (m4 / (m2 * m2)) - 3.0
    else:
        skew = 0.0
        kurt = 0.0
    hist_counts, _ = np.histogram(flat, bins=bin_edges)
    hist = hist_counts.astype(np.uint64)
    total_in_range = float(np.sum(hist))
    pdf = (
        hist.astype(np.float64) / (total_in_range * bin_width)
        if total_in_range > 0.0
        else np.zeros(int(n_bins), dtype=np.float64)
    )
    return {
        "mean": mean,
        "std": float(np.sqrt(max(0.0, m2))),
        "variance": m2,
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "hist": hist,
        "bin_edges": bin_edges,
        "pdf": pdf,
    }


def generate_sobol_parameter_grid(
    bounds: dict[str, tuple[float, float]],
    n_samples: int,
    *,
    seed: int = 42,
    base_seed: int = 1000,
) -> list[dict[str, float | int]]:
    """Generate a low-discrepancy scrambled Sobol parameter ensemble for SBI sweeps."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if not bounds:
        return [{"random_seed": int(base_seed + i)} for i in range(n_samples)]

    names = list(bounds.keys())
    lo = np.array([float(bounds[k][0]) for k in names], dtype=np.float64)
    hi = np.array([float(bounds[k][1]) for k in names], dtype=np.float64)
    if np.any(hi < lo):
        raise ValueError("All parameter bounds must satisfy upper >= lower")

    sampler = qmc.Sobol(d=len(names), scramble=True, seed=int(seed))
    m = (n_samples - 1).bit_length()
    unit = sampler.random_base2(m)[:n_samples]
    scaled = qmc.scale(unit, lo, hi)

    rows: list[dict[str, float | int]] = []
    for i in range(n_samples):
        row: dict[str, float | int] = {
            name: float(scaled[i, j]) for j, name in enumerate(names)
        }
        if "random_seed" not in row:
            row["random_seed"] = int(base_seed + i)
        else:
            row["random_seed"] = round(row["random_seed"])
        rows.append(row)
    return rows


__all__ = [
    "add_thermal_noise_3d",
    "apply_wedge_filter_3d",
    "field_moments_and_pdf",
    "generate_sobol_parameter_grid",
    "horizon_slope",
    "horizon_wedge_mask",
]
