"""Multi-probe 21cm cross-correlations, GPU/CPU 3D CIC gridding, gravityCU halo bridge, and kSZ maps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck18

from py21cmfast.c_21cmfast import ffi as _ffi
from py21cmfast.c_21cmfast import lib as _lib

from .analysis import _cuda_ready, _normalize_box_length, power_spectrum_1d


def _cpu_cic_deposit_3d(
    pos: np.ndarray,
    weights: np.ndarray | None,
    nx: int,
    ny: int,
    nz: int,
    lx: float,
    ly: float,
    lz: float,
    *,
    overdensity: bool,
) -> np.ndarray:
    grid = np.zeros((nx, ny, nz), dtype=np.float64)
    n_part = int(pos.shape[0])
    if n_part == 0:
        return grid.astype(np.float32)

    ux = np.mod(pos[:, 0].astype(np.float64) * (float(nx) / lx), float(nx))
    uy = np.mod(pos[:, 1].astype(np.float64) * (float(ny) / ly), float(ny))
    uz = np.mod(pos[:, 2].astype(np.float64) * (float(nz) / lz), float(nz))

    fx_fl = np.floor(ux)
    fy_fl = np.floor(uy)
    fz_fl = np.floor(uz)
    ix0 = np.mod(fx_fl.astype(np.int64), nx)
    iy0 = np.mod(fy_fl.astype(np.int64), ny)
    iz0 = np.mod(fz_fl.astype(np.int64), nz)
    ix1 = (ix0 + 1) % nx
    iy1 = (iy0 + 1) % ny
    iz1 = (iz0 + 1) % nz

    fx = ux - fx_fl
    fy = uy - fy_fl
    fz = uz - fz_fl
    wx0 = 1.0 - fx
    wy0 = 1.0 - fy
    wz0 = 1.0 - fz
    w = np.ones(n_part, dtype=np.float64) if weights is None else weights.astype(np.float64)

    w00 = w * wx0 * wy0
    w01 = w * wx0 * fy
    w10 = w * fx * wy0
    w11 = w * fx * fy

    np.add.at(grid, (ix0, iy0, iz0), w00 * wz0)
    np.add.at(grid, (ix0, iy0, iz1), w00 * fz)
    np.add.at(grid, (ix0, iy1, iz0), w01 * wz0)
    np.add.at(grid, (ix0, iy1, iz1), w01 * fz)
    np.add.at(grid, (ix1, iy0, iz0), w10 * wz0)
    np.add.at(grid, (ix1, iy0, iz1), w10 * fz)
    np.add.at(grid, (ix1, iy1, iz0), w11 * wz0)
    np.add.at(grid, (ix1, iy1, iz1), w11 * fz)

    if overdensity:
        mean_w = float(np.sum(w)) / float(nx * ny * nz)
        if mean_w > 0.0:
            grid = grid / mean_w - 1.0
    return grid.astype(np.float32)


def _deconvolve_cic_3d(grid: np.ndarray) -> np.ndarray:
    nx, ny, nz = grid.shape
    fx = np.fft.fftfreq(nx)[:, None, None]
    fy = np.fft.fftfreq(ny)[None, :, None]
    fz = np.fft.rfftfreq(nz)[None, None, :]
    win2 = np.maximum((np.sinc(fx) * np.sinc(fy) * np.sinc(fz)) ** 2, 0.1)
    kspace = np.fft.rfftn(grid.astype(np.float64, copy=False))
    return np.fft.irfftn(kspace / win2, s=(nx, ny, nz), axes=(0, 1, 2)).astype(np.float32)


def cic_deposit_3d(
    positions: np.ndarray,
    shape: tuple[int, int, int],
    box_length: float | tuple[float, float, float],
    *,
    weights: np.ndarray | None = None,
    overdensity: bool = True,
    deconvolve: bool = False,
    use_cuda: bool | None = None,
) -> np.ndarray:
    """Assign 3D particle/halo coordinates onto a periodic grid via Cloud-in-Cell (CIC)."""
    if len(shape) != 3 or any(int(s) <= 0 for s in shape):
        raise ValueError(f"shape must be 3 positive integers, got {shape!r}")
    nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))
    lx, ly, lz = _normalize_box_length(box_length)

    pos = np.asarray(positions, dtype=np.float32)
    if pos.size == 0:
        pos = pos.reshape(0, 3)
    elif pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")
    pos = np.ascontiguousarray(pos, dtype=np.float32)

    w_arr: np.ndarray | None = None
    if weights is not None:
        w_arr = np.ascontiguousarray(np.asarray(weights, dtype=np.float32).ravel())
        if w_arr.size != pos.shape[0]:
            raise ValueError(
                f"weights length {w_arr.size} does not match positions count {pos.shape[0]}"
            )

    if _cuda_ready(use_cuda) and hasattr(_lib, "p21_cuda_cic_deposit_3d"):
        grid_out = np.empty((nx, ny, nz), dtype=np.float32)
        pos_ptr = (
            _ffi.NULL
            if pos.shape[0] == 0
            else _ffi.cast("const float *", _ffi.from_buffer(pos))
        )
        w_ptr = (
            _ffi.NULL
            if w_arr is None or w_arr.size == 0
            else _ffi.cast("const float *", _ffi.from_buffer(w_arr))
        )
        rc = _lib.p21_cuda_cic_deposit_3d(
            pos_ptr,
            w_ptr,
            int(pos.shape[0]),
            nx,
            ny,
            nz,
            float(lx),
            float(ly),
            float(lz),
            int(bool(overdensity)),
            _ffi.cast("float *", _ffi.from_buffer(grid_out)),
        )
        if rc == 0:
            return _deconvolve_cic_3d(grid_out) if deconvolve else grid_out

    grid = _cpu_cic_deposit_3d(
        pos, w_arr, nx, ny, nz, lx, ly, lz, overdensity=bool(overdensity)
    )
    return _deconvolve_cic_3d(grid) if deconvolve else grid


def _find_first_hdf5(fl: h5py.File, keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        if key in fl and isinstance(fl[key], h5py.Dataset):
            return np.asarray(fl[key])
    return None


def _find_first_npz(npz: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        if key in npz:
            return np.asarray(npz[key])
    return None


def _box_scalar_from_raw(raw: Any) -> float | None:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64).ravel()
    if arr.size == 0:
        return None
    return float(arr[0])


def load_gravitycu_halos(
    path: str | Path,
    *,
    mass_min: float | None = None,
    mass_max: float | None = None,
    box_length: float | None = None,
    h_little: float | None = None,
    units: Literal["Mpc", "Mpc/h", "kpc", "kpc/h"] = "Mpc",
) -> dict[str, Any]:
    """Load an HDF5 or NPZ halo/galaxy catalog from ``gravityCU`` / Gadget / FoF snapshots."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Catalog file not found: {p}")

    pos_keys = (
        "positions",
        "pos",
        "Coordinates",
        "GroupPos",
        "halos/positions",
        "halos/pos",
        "Group/GroupPos",
        "Subhalo/SubhaloPos",
    )
    mass_keys = (
        "masses",
        "mass",
        "Mass",
        "GroupMass",
        "m_halo",
        "halos/masses",
        "halos/mass",
        "Group/GroupMass",
    )
    vel_keys = (
        "velocities",
        "vel",
        "Velocities",
        "GroupVel",
        "halos/velocities",
        "halos/vel",
        "Group/GroupVel",
    )

    attrs: dict[str, Any] = {}
    if p.suffix.lower() in (".h5", ".hdf5"):
        with h5py.File(p, "r") as fl:
            for k, v in fl.attrs.items():
                attrs[str(k)] = v
            if "Header" in fl:
                for k, v in fl["Header"].attrs.items():
                    attrs.setdefault(str(k), v)
            pos = _find_first_hdf5(fl, pos_keys)
            masses = _find_first_hdf5(fl, mass_keys)
            velocities = _find_first_hdf5(fl, vel_keys)
            inferred_box = None
            for bkey in ("box_length", "BOX_LEN", "BoxSize", "box_size"):
                if bkey in attrs:
                    inferred_box = _box_scalar_from_raw(attrs[bkey])
                    break
    elif p.suffix.lower() == ".npz":
        with np.load(p, allow_pickle=False) as npz:
            pos = _find_first_npz(npz, pos_keys)
            masses = _find_first_npz(npz, mass_keys)
            velocities = _find_first_npz(npz, vel_keys)
            inferred_box = None
            for bkey in ("box_length", "BOX_LEN", "BoxSize", "box_size"):
                if bkey in npz:
                    inferred_box = _box_scalar_from_raw(npz[bkey])
                    break
            if "h_little" in npz:
                attrs["HubbleParam"] = float(np.asarray(npz["h_little"]).ravel()[0])
    else:
        raise ValueError(f"Unsupported catalog extension {p.suffix!r}; expected .h5/.hdf5 or .npz")

    if pos is None:
        raise ValueError(f"Could not find halo/particle positions dataset in {p}")
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected positions shape (N, 3) in {p}, got {pos.shape}")

    h_val = float(h_little) if h_little is not None else float(attrs.get("HubbleParam", 0.6766))
    if h_val <= 0.0:
        raise ValueError(f"h_little must be > 0, got {h_val}")

    scale = 1.0
    if units == "Mpc":
        scale = 1.0
    elif units == "Mpc/h":
        scale = 1.0 / h_val
    elif units == "kpc":
        scale = 1e-3
    elif units == "kpc/h":
        scale = 1e-3 / h_val
    else:
        raise ValueError(f"units must be 'Mpc', 'Mpc/h', 'kpc', or 'kpc/h', got {units!r}")

    pos_mpc = (pos * scale).astype(np.float32)
    eff_box = (
        float(box_length)
        if box_length is not None
        else (float(inferred_box * scale) if inferred_box is not None else None)
    )

    masses_out = np.asarray(masses, dtype=np.float32).ravel() if masses is not None else None
    vel_out = np.asarray(velocities, dtype=np.float32) if velocities is not None else None

    if mass_min is not None or mass_max is not None:
        if masses_out is None:
            raise ValueError("mass_min/mass_max specified, but catalog contains no mass dataset")
        mask = np.ones(masses_out.shape[0], dtype=bool)
        if mass_min is not None:
            mask &= masses_out >= float(mass_min)
        if mass_max is not None:
            mask &= masses_out <= float(mass_max)
        pos_mpc = np.ascontiguousarray(pos_mpc[mask])
        masses_out = np.ascontiguousarray(masses_out[mask])
        if vel_out is not None:
            vel_out = np.ascontiguousarray(vel_out[mask])

    return {
        "positions": pos_mpc,
        "masses": masses_out,
        "velocities": vel_out,
        "box_length": eff_box,
        "n_halos": int(pos_mpc.shape[0]),
        "metadata": {
            "path": str(p),
            "units_input": units,
            "h_little": h_val,
            "scale_to_mpc": scale,
        },
    }


def cross_power_21cm_galaxy(
    tb_field: np.ndarray,
    galaxy_tracer: np.ndarray | Mapping[str, Any],
    box_length: float | tuple[float, float, float],
    *,
    weights: np.ndarray | None = None,
    n_bins: int = 20,
    k_min: float | None = None,
    k_max: float | None = None,
    k_edges: np.ndarray | None = None,
    log_bins: bool = True,
    deconvolve_cic: bool = True,
    subtract_shot_noise: bool = True,
    use_cuda: bool | None = None,
) -> dict[str, Any]:
    """Compute 1D 21cm–galaxy cross-power spectrum P_{21xg}(k) and correlation coefficient r(k)."""
    tb = np.asarray(tb_field, dtype=np.float32)
    if tb.ndim != 3:
        raise ValueError(f"tb_field must be 3D, got shape {tb.shape}")
    lx, ly, lz = _normalize_box_length(box_length)
    vol = lx * ly * lz

    n_gal: int | None = None
    if isinstance(galaxy_tracer, Mapping):
        if "positions" not in galaxy_tracer:
            raise ValueError("Mapping galaxy_tracer must contain a 'positions' array")
        pos = np.asarray(galaxy_tracer["positions"], dtype=np.float32)
        eff_w = weights if weights is not None else galaxy_tracer.get("masses")
        n_gal = int(pos.shape[0])
        gal = cic_deposit_3d(
            pos,
            tb.shape,
            (lx, ly, lz),
            weights=eff_w,
            overdensity=True,
            deconvolve=deconvolve_cic,
            use_cuda=use_cuda,
        )
    else:
        tr = np.asarray(galaxy_tracer, dtype=np.float32)
        if tr.ndim == 2 and tr.shape[1] == 3:
            n_gal = int(tr.shape[0])
            gal = cic_deposit_3d(
                tr,
                tb.shape,
                (lx, ly, lz),
                weights=weights,
                overdensity=True,
                deconvolve=deconvolve_cic,
                use_cuda=use_cuda,
            )
        elif tr.shape == tb.shape:
            gal = tr
        else:
            raise ValueError(
                f"galaxy_tracer must be a 3D grid matching tb_field {tb.shape} or (N, 3) positions, "
                f"got shape {tr.shape}"
            )

    ps_21 = power_spectrum_1d(
        tb,
        (lx, ly, lz),
        n_bins=n_bins,
        k_min=k_min,
        k_max=k_max,
        k_edges=k_edges,
        log_bins=log_bins,
        remove_mean=True,
        use_cuda=use_cuda,
    )
    edges = ps_21["k_edges"]
    ps_gal = power_spectrum_1d(
        gal,
        (lx, ly, lz),
        k_edges=edges,
        remove_mean=True,
        use_cuda=use_cuda,
    )
    ps_cross = power_spectrum_1d(
        tb,
        (lx, ly, lz),
        field_b=gal,
        k_edges=edges,
        remove_mean=True,
        use_cuda=use_cuda,
    )

    pk_21 = np.asarray(ps_21["pk"], dtype=np.float64)
    pk_gal = np.asarray(ps_gal["pk"], dtype=np.float64)
    pk_cross = np.asarray(ps_cross["pk"], dtype=np.float64)

    shot_noise = float(vol / n_gal) if (subtract_shot_noise and n_gal is not None and n_gal > 0) else 0.0
    pk_gal_sub = pk_gal - shot_noise

    with np.errstate(divide="ignore", invalid="ignore"):
        denom_raw = np.sqrt(np.maximum(pk_21 * pk_gal, 0.0))
        r_k = np.where((pk_21 > 0.0) & (pk_gal > 0.0) & (denom_raw > 0.0), pk_cross / denom_raw, np.nan)
        r_k = np.where(np.isfinite(r_k), np.clip(r_k, -1.0, 1.0), np.nan)

        denom_sub = np.sqrt(np.maximum(pk_21 * pk_gal_sub, 0.0))
        r_k_sub = np.where(
            (pk_21 > 0.0) & (pk_gal_sub > 0.0) & (denom_sub > 0.0), pk_cross / denom_sub, np.nan
        )
        r_k_sub = np.where(np.isfinite(r_k_sub), np.clip(r_k_sub, -1.0, 1.0), np.nan)

    return {
        "k": ps_21["k"],
        "k_edges": edges,
        "counts": ps_21["counts"],
        "pk_21": pk_21,
        "delta_sq_21": ps_21["delta_sq"],
        "pk_gal": pk_gal,
        "pk_gal_shot_sub": pk_gal_sub,
        "delta_sq_gal": ps_gal["delta_sq"],
        "pk_cross": pk_cross,
        "delta_sq_cross": ps_cross["delta_sq"],
        "shot_noise_pk": shot_noise,
        "r_k": r_k,
        "r_k_shot_sub": r_k_sub,
        "galaxy_field": gal,
        "n_galaxies": n_gal,
    }


def cross_power_21cm_galaxy_error(
    pk_21: np.ndarray,
    pk_gal: np.ndarray,
    pk_cross: np.ndarray,
    n_modes: np.ndarray,
    *,
    thermal_noise_21: np.ndarray | float = 0.0,
    shot_noise_gal: np.ndarray | float = 0.0,
    n_fields: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Forecast 1-sigma uncertainty and SNR on 1D 21cm–galaxy cross-power spectrum."""
    if float(n_fields) <= 0.0:
        raise ValueError("n_fields must be > 0")
    p21 = np.asarray(pk_21, dtype=np.float64)
    pg = np.asarray(pk_gal, dtype=np.float64)
    pc = np.asarray(pk_cross, dtype=np.float64)
    modes = np.asarray(n_modes, dtype=np.float64)
    pn21 = np.asarray(thermal_noise_21, dtype=np.float64)
    pshot = np.asarray(shot_noise_gal, dtype=np.float64)

    indep_modes = np.maximum(0.5 * modes, 0.0)
    num = pc**2 + (p21 + pn21) * (pg + pshot)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.where(
            (indep_modes > 0.0) & np.isfinite(num) & (num >= 0.0),
            np.sqrt(num / (2.0 * indep_modes * float(n_fields))),
            np.inf,
        )
        snr = np.where(np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(pc), np.abs(pc) / sigma, 0.0)
    return {
        "sigma_pk_cross": sigma,
        "snr": snr,
        "total_snr": float(np.sqrt(np.sum(snr**2))),
    }


def ksz_momentum_map(
    coeval_or_xH: Any,
    box_length: float | tuple[float, float, float] | None = None,
    redshift: float | None = None,
    *,
    density: np.ndarray | None = None,
    los_velocity: np.ndarray | None = None,
    los_axis: int = 2,
    velocity_units: Literal["comoving_mpc_s", "km_s"] = "comoving_mpc_s",
    cosmo: Any = None,
    n_ell_bins: int = 16,
    ell_min: float = 500.0,
    ell_max: float = 10000.0,
) -> dict[str, Any]:
    """Compute a 2D patchy kSZ brightness temperature map (in uK) and flat-sky C_ell / D_ell."""
    if los_axis not in (0, 1, 2):
        raise ValueError(f"los_axis must be 0, 1, or 2, got {los_axis}")

    if hasattr(coeval_or_xH, "neutral_fraction") and hasattr(coeval_or_xH, "density"):
        cv = coeval_or_xH
        xh = np.asarray(cv.neutral_fraction, dtype=np.float64)
        dens = np.asarray(cv.density, dtype=np.float64)
        vel_name = ("velocity_x", "velocity_y", "velocity_z")[los_axis]
        vel = np.asarray(cv.perturbed_field.get(vel_name), dtype=np.float64)
        if vel is None:
            raise ValueError(
                f"Coeval object missing {vel_name}; enable KEEP_3D_VELOCITIES or pass los_velocity"
            )
        if box_length is None:
            sim_opts = cv.inputs.simulation_options
            box_length = (
                float(sim_opts.BOX_LEN),
                float(sim_opts.BOX_LEN),
                float(sim_opts.NON_CUBIC_FACTOR * sim_opts.BOX_LEN),
            )
        if redshift is None:
            redshift = float(cv.redshift)
        if cosmo is None:
            cosmo = cv.inputs.cosmo_params.cosmo
    else:
        if density is None or los_velocity is None or box_length is None or redshift is None:
            raise ValueError(
                "Pass either a Coeval object or explicit neutral_fraction, density, los_velocity, "
                "box_length, and redshift."
            )
        xh = np.asarray(coeval_or_xH, dtype=np.float64)
        dens = np.asarray(density, dtype=np.float64)
        vel = np.asarray(los_velocity, dtype=np.float64)

    if xh.ndim != 3 or dens.shape != xh.shape or vel.shape != xh.shape:
        raise ValueError(
            f"neutral_fraction {xh.shape}, density {dens.shape}, and los_velocity {vel.shape} "
            "must all be matching 3D arrays"
        )

    lx, ly, lz = _normalize_box_length(box_length)
    z = float(redshift)
    cobj = Planck18 if cosmo is None else cosmo

    if velocity_units == "comoving_mpc_s":
        c_mpc_s = float(const.c.to_value(u.Mpc / u.s))
        v_over_c = vel / ((1.0 + z) * c_mpc_s)
    elif velocity_units == "km_s":
        c_km_s = float(const.c.to_value(u.km / u.s))
        v_over_c = vel / c_km_s
    else:
        raise ValueError(f"velocity_units must be 'comoving_mpc_s' or 'km_s', got {velocity_units!r}")

    q_los = (1.0 - np.clip(xh, 0.0, 1.0)) * np.maximum(1.0 + dens, 0.0) * v_over_c

    y_he = 0.245
    t_cmb_uk = float(cobj.Tcmb0.to_value(u.K)) * 1e6 if hasattr(cobj, "Tcmb0") else 2.7255e6
    rho_c0 = float(cobj.critical_density0.to_value(u.g / (u.cm**3)))
    m_p = float(const.m_p.to_value(u.g))
    sigma_t = float(const.sigma_T.to_value(u.cm**2))
    mpc_cm = float((1.0 * u.Mpc).to_value(u.cm))
    ob0 = float(cobj.Ob0) if getattr(cobj, "Ob0", None) is not None else 0.049
    n_e0 = (1.0 - 0.75 * y_he) * ob0 * rho_c0 / m_p

    shape = xh.shape
    lengths = (lx, ly, lz)
    dx_los = lengths[los_axis] / float(shape[los_axis])
    prefactor_uk = -t_cmb_uk * sigma_t * n_e0 * mpc_cm * ((1.0 + z) ** 2) * dx_los
    delta_t_uk = prefactor_uk * np.sum(q_los, axis=los_axis)

    perp_axes = [i for i in (0, 1, 2) if i != los_axis]
    n0, n1 = shape[perp_axes[0]], shape[perp_axes[1]]
    l0, l1 = lengths[perp_axes[0]], lengths[perp_axes[1]]

    dm_mpc = float(cobj.comoving_transverse_distance(z).to_value(u.Mpc))
    theta0 = l0 / dm_mpc
    theta1 = l1 / dm_mpc
    omega_box = theta0 * theta1

    fluct = delta_t_uk - np.mean(delta_t_uk)
    fft2 = np.fft.rfft2(fluct)
    raw_cl = ((fft2.real**2 + fft2.imag**2) * omega_box) / float((n0 * n1) ** 2)

    ell_x = np.fft.fftfreq(n0, d=theta0 / n0) * (2.0 * np.pi)
    ell_y = np.fft.rfftfreq(n1, d=theta1 / n1) * (2.0 * np.pi)
    ell_mag = np.sqrt(ell_x[:, None] ** 2 + ell_y[None, :] ** 2)

    mult = np.full((1, ell_y.size), 2.0, dtype=np.float64)
    mult[0, 0] = 1.0
    if n1 % 2 == 0:
        mult[0, -1] = 1.0
    mult = np.broadcast_to(mult, ell_mag.shape)

    mask = np.ones_like(ell_mag, dtype=bool)
    mask[0, 0] = False
    ell_f = ell_mag[mask]
    cl_f = raw_cl[mask]
    m_f = mult[mask]

    eff_ell_min = max(float(ell_min), float(2.0 * np.pi / max(theta0, theta1)))
    eff_ell_max = min(float(ell_max), float(np.pi * min(n0 / theta0, n1 / theta1) * (1.0 + 1e-6)))
    if eff_ell_max <= eff_ell_min:
        eff_ell_min = float(np.min(ell_f))
        eff_ell_max = float(np.max(ell_f) * (1.0 + 1e-6))

    ell_edges = np.geomspace(eff_ell_min, eff_ell_max, int(n_ell_bins) + 1, dtype=np.float64)
    nb = ell_edges.size - 1
    bidx = np.searchsorted(ell_edges, ell_f, side="right") - 1
    valid = (ell_f >= ell_edges[0]) & (ell_f < ell_edges[-1]) & (bidx >= 0) & (bidx < nb)

    bi = bidx[valid]
    mv = m_f[valid]
    counts = np.bincount(bi, weights=mv, minlength=nb).astype(np.uint64)
    ell_sum = np.bincount(bi, weights=mv * ell_f[valid], minlength=nb)
    cl_sum = np.bincount(bi, weights=mv * cl_f[valid], minlength=nb)

    with np.errstate(divide="ignore", invalid="ignore"):
        ell_mean = np.where(counts > 0, ell_sum / counts, np.nan)
        cl_mean = np.where(counts > 0, cl_sum / counts, np.nan)
        dl_mean = ell_mean * (ell_mean + 1.0) * cl_mean / (2.0 * np.pi)

    return {
        "delta_T_uK": delta_t_uk.astype(np.float32),
        "q_los": q_los.astype(np.float32),
        "rms_uK": float(np.sqrt(np.mean(delta_t_uk**2))),
        "mean_uK": float(np.mean(delta_t_uk)),
        "ell": ell_mean,
        "ell_edges": ell_edges,
        "cl_uK2": cl_mean,
        "dl_uK2": dl_mean,
        "counts": counts,
        "redshift": z,
        "theta_box_rad": (theta0, theta1),
    }


__all__ = [
    "cic_deposit_3d",
    "cross_power_21cm_galaxy",
    "cross_power_21cm_galaxy_error",
    "ksz_momentum_map",
    "load_gravitycu_halos",
]
