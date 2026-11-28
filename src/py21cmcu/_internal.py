"""Shared internal helpers for ``py21cmcu`` modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from py21cmfast._cfg import config
from py21cmfast.c_21cmfast import lib as _lib

PARAM_ALIAS_MAP: dict[str, str] = {
    "F_STAR10": "F_STAR10_ACG",
    "F_ESC10": "F_ESC10_ACG",
    "F_STAR7_MINI": "F_STAR7_MCG",
    "F_ESC7_MINI": "F_ESC7_MCG",
    "ALPHA_STAR": "ALPHA_STAR_ACG",
    "ALPHA_STAR_MINI": "ALPHA_STAR_MCG",
    "L_X": "LX_OVER_SFR_ACG",
    "L_X_MINI": "LX_OVER_SFR_MCG",
}


def cuda_ready(use_cuda: bool | None) -> bool:
    """Return True if CUDA execution is enabled, compiled, and a device is available."""
    enabled = bool(config["USE_CUDA"]) if use_cuda is None else bool(use_cuda)
    return bool(
        enabled
        and _lib.p21_cuda_is_compiled()
        and _lib.p21_cuda_is_available()
    )


def normalize_box_length(
    box_length: float | tuple[float, float, float],
) -> tuple[float, float, float]:
    """Normalize a scalar or 3-tuple simulation box length into ``(lx, ly, lz)`` in Mpc."""
    if np.ndim(box_length) == 0:
        length = float(box_length)
        return (length, length, length)
    vals = tuple(float(v) for v in box_length)
    if len(vals) != 3:
        raise ValueError(f"box_length must be scalar or 3-tuple, got {box_length!r}")
    return vals


def default_k_edges(
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
    """Construct default 1D isotropic wavenumber bin edges for a 3D periodic volume."""
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


def evolve_from_vector(
    base_inputs: Any,
    param_names: Sequence[str],
    theta: Sequence[float] | np.ndarray,
    *,
    seed_offset: int = 0,
) -> Any:
    """Clone ``base_inputs`` and apply parameter overrides mapped through ``PARAM_ALIAS_MAP``."""
    vals = np.asarray(theta, dtype=np.float64).ravel()
    if vals.size != len(param_names):
        raise ValueError(
            f"Expected {len(param_names)} parameters ({list(param_names)}), got shape {vals.shape}"
        )
    overrides = {
        PARAM_ALIAS_MAP.get(name, name): float(v)
        for name, v in zip(param_names, vals, strict=True)
    }
    if "random_seed" in overrides:
        seed = round(overrides.pop("random_seed"))
        cloned = base_inputs.clone(random_seed=seed)
    elif seed_offset:
        cloned = base_inputs.clone(random_seed=int(base_inputs.random_seed) + int(seed_offset))
    else:
        cloned = base_inputs
    return cloned.evolve_input_structs(**overrides) if overrides else cloned


__all__ = [
    "PARAM_ALIAS_MAP",
    "cuda_ready",
    "default_k_edges",
    "evolve_from_vector",
    "normalize_box_length",
]
