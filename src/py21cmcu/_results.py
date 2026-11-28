"""Typed Mapping-compatible result dataclasses for ``py21cmcu`` science APIs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np


class ResultMapping(Mapping[str, Any]):
    """Read-only Mapping mixin allowing both attribute and ``dict`` key access."""

    def __getitem__(self, key: str) -> Any:
        for f in fields(self):
            if f.name == key:
                return getattr(self, f.name)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        for f in fields(self):
            yield f.name

    def __len__(self) -> int:
        return len(fields(self))

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow dictionary copy mapping each field name to its value."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class PowerSpectrum1DResult(ResultMapping):
    """Result container for :func:`py21cmcu.power_spectrum_1d`."""

    k: np.ndarray
    pk: np.ndarray
    delta_sq: np.ndarray
    counts: np.ndarray
    k_edges: np.ndarray


@dataclass(frozen=True)
class PowerSpectrum2DResult(ResultMapping):
    """Result container for :func:`py21cmcu.power_spectrum_2d`."""

    k_perp: np.ndarray
    k_para: np.ndarray
    pk: np.ndarray
    counts: np.ndarray
    kperp_edges: np.ndarray
    kpara_edges: np.ndarray


@dataclass(frozen=True)
class BispectrumResult(ResultMapping):
    """Result container for :func:`py21cmcu.bispectrum_3d`."""

    k1: np.ndarray
    k2: np.ndarray
    k3: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    pk1: np.ndarray
    pk2: np.ndarray
    pk3: np.ndarray
    n_triangles: np.ndarray
    dk: np.ndarray
    config: str


@dataclass(frozen=True)
class MFPResult(ResultMapping):
    """Result container for :func:`py21cmcu.bubble_size_distribution_mfp`."""

    r_edges: np.ndarray
    r_centers: np.ndarray
    counts: np.ndarray
    pdf: np.ndarray
    rdp_dr: np.ndarray
    mean_r: float
    std_r: float
    ionized_fraction: float
    n_rays: int


@dataclass(frozen=True)
class MinkowskiResult(ResultMapping):
    """Result container for :func:`py21cmcu.minkowski_functionals_3d`."""

    thresholds: np.ndarray
    V0: np.ndarray
    V1: np.ndarray
    V2: np.ndarray
    V3: np.ndarray
    euler: np.ndarray
    standardized: bool


@dataclass(frozen=True)
class CrossPowerResult(ResultMapping):
    """Result container for :func:`py21cmcu.cross_power_21cm_galaxy`."""

    k: np.ndarray
    k_edges: np.ndarray
    counts: np.ndarray
    pk_21: np.ndarray
    delta_sq_21: np.ndarray
    pk_gal: np.ndarray
    pk_gal_shot_sub: np.ndarray
    delta_sq_gal: np.ndarray
    pk_cross: np.ndarray
    delta_sq_cross: np.ndarray
    shot_noise_pk: float
    r_k: np.ndarray
    r_k_shot_sub: np.ndarray
    galaxy_field: np.ndarray
    n_galaxies: int


@dataclass(frozen=True)
class KSZMapResult(ResultMapping):
    """Result container for :func:`py21cmcu.ksz_momentum_map`."""

    delta_T_uK: np.ndarray
    q_los: np.ndarray
    rms_uK: float
    mean_uK: float
    ell: np.ndarray
    ell_edges: np.ndarray
    cl_uK2: np.ndarray
    dl_uK2: np.ndarray
    counts: np.ndarray
    redshift: float
    theta_box_rad: tuple[float, float]


__all__ = [
    "BispectrumResult",
    "CrossPowerResult",
    "KSZMapResult",
    "MFPResult",
    "MinkowskiResult",
    "PowerSpectrum1DResult",
    "PowerSpectrum2DResult",
    "ResultMapping",
]
