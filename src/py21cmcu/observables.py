"""Foreground wedge filtering, thermal noise injection, and Sobol sampling for 21cm SBI."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18
from scipy.stats import qmc

from .analysis import _normalize_box_length

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
    del use_cuda  # FFT filtering runs in float64 to preserve conjugate symmetry
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    if los_axis not in (0, 1, 2):
        raise ValueError(f"los_axis must be 0, 1, or 2, got {los_axis}")

    lengths = _normalize_box_length(box_length)
    work = np.moveaxis(arr, los_axis, 2)
    l_perm = tuple(lengths[i] for i in ([0, 1, 2] if los_axis == 2 else ([1, 2, 0] if los_axis == 0 else [0, 2, 1])))
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
        cosmo=cosmo,
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
) -> np.ndarray:
    """Add deterministic zero-mean Gaussian thermal noise [mK] to a 3D 21cm brightness field."""
    arr = np.asarray(field, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

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
        cell_vol = (lengths[0] / arr.shape[0]) * (lengths[1] / arr.shape[1]) * (
            lengths[2] / arr.shape[2]
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

    if rms == 0.0:
        return arr.copy()
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(loc=0.0, scale=rms, size=arr.shape).astype(np.float32)
    return (arr + noise).astype(np.float32)


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
    unit = sampler.random(n_samples)
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
    "generate_sobol_parameter_grid",
    "horizon_slope",
    "horizon_wedge_mask",
]
