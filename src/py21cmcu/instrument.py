"""Radio interferometer mock observation and thermal-noise sensitivity forecasting."""

from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
from typing import Any, Literal

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18

from py21cmfast.c_21cmfast import ffi as _ffi
from py21cmfast.c_21cmfast import lib as _lib

from .analysis import (
    _cuda_ready,
    _default_k_edges,
    _normalize_box_length,
    power_spectrum_1d,
    power_spectrum_2d,
)
from .observables import horizon_slope

_C_MS = 299_792_458.0
_REST_FREQ_21CM_HZ = 1_420_405_751.77


@dataclass(frozen=True)
class InterferometerConfig:
    """Physical and observing configuration for a 21cm radio interferometer."""

    name: str
    n_ant: int
    d_ant_m: float
    b_min_m: float
    b_max_m: float
    b_core_m: float
    core_fraction: float
    channel_width_hz: float
    t_int_hours: float
    n_pol: int = 2
    n_fields: int = 1
    t_rx_k: float = 40.0
    t_sky_ref_k: float = 60.0
    t_sky_freq_mhz: float = 300.0
    t_sky_index: float = -2.55
    beam_factor: float = 1.03
    psf_fwhm_scale: float = 1.0
    default_wedge_mode: Literal["horizon", "fov"] = "horizon"
    default_buffer_h_mpc: float = 0.05
    antenna_xy_m: np.ndarray | None = None

    def replace(self, **kwargs: Any) -> InterferometerConfig:
        """Return a modified copy of this interferometer configuration."""
        return _dc_replace(self, **kwargs)

    def obs_freq_hz(self, redshift: float) -> float:
        """Return observed 21cm frequency in Hz at ``redshift``."""
        z = float(redshift)
        if z < 0.0:
            raise ValueError(f"redshift must be >= 0, got {redshift}")
        return _REST_FREQ_21CM_HZ / (1.0 + z)

    def obs_freq_mhz(self, redshift: float) -> float:
        """Return observed 21cm frequency in MHz at ``redshift``."""
        return self.obs_freq_hz(redshift) * 1e-6

    def wavelength_m(self, redshift: float) -> float:
        """Return observed redshifted 21cm wavelength in meters."""
        return _C_MS / self.obs_freq_hz(redshift)

    def system_temp_k(self, redshift: float) -> float:
        """Return system temperature $T_{\\mathrm{sys}}(z) = T_{\\mathrm{rx}} + T_{\\mathrm{sky}}(z)$ in K."""
        nu_mhz = self.obs_freq_mhz(redshift)
        t_sky = self.t_sky_ref_k * (nu_mhz / self.t_sky_freq_mhz) ** self.t_sky_index
        return float(self.t_rx_k + t_sky)

    def primary_beam_fwhm_rad(self, redshift: float) -> float:
        """Return primary beam FWHM in radians ($\\theta_{\\mathrm{FWHM}} \\approx \\mathrm{beam\\_factor}\\,\\lambda / D_{\\mathrm{ant}}$)."""
        return float(self.beam_factor * self.wavelength_m(redshift) / self.d_ant_m)

    def primary_beam_omega_sr(self, redshift: float) -> float:
        """Return primary beam solid angle $\\Omega_p$ in steradians."""
        fwhm = self.primary_beam_fwhm_rad(redshift)
        return float(np.pi * fwhm * fwhm / (4.0 * np.log(2.0)))

    def effective_omega_sr(self, redshift: float) -> float:
        """Return effective power-spectrum beam solid angle $\\Omega_{\\mathrm{eff}} = \\Omega_p^2 / \\Omega_{pp} = 2\\Omega_p$."""
        return 2.0 * self.primary_beam_omega_sr(redshift)

    def psf_fwhm_rad(self, redshift: float) -> float:
        """Return synthesized beam (PSF) angular FWHM in radians ($\\sim \\lambda / B_{\\max}$)."""
        return float(self.psf_fwhm_scale * self.wavelength_m(redshift) / self.b_max_m)

    def comoving_conversions(
        self, redshift: float, *, cosmo: Any = None
    ) -> tuple[float, float, float]:
        """Return ``(X_dm_mpc, Y_mpc_per_hz, x2y_mpc3_per_sr_hz)`` at ``redshift``."""
        cobj = Planck18 if cosmo is None else cosmo
        z = float(redshift)
        dm_mpc = float(cobj.comoving_transverse_distance(z).to_value(u.Mpc))
        hz_inv_s = float(cobj.H(z).to_value(u.km / u.s / u.Mpc)) * (1000.0 / 3.085677581491367e22)
        # d(r_comoving)/d(nu_obs) = c (1 + z)^2 / (nu_21 H(z))
        y_mpc_per_hz = (
            (_C_MS / 3.085677581491367e22) * ((1.0 + z) ** 2) / (_REST_FREQ_21CM_HZ * hz_inv_s)
        )
        x2y = (dm_mpc**2) * y_mpc_per_hz
        return dm_mpc, float(y_mpc_per_hz), float(x2y)


INSTRUMENT_PRESETS: dict[str, InterferometerConfig] = {
    "SKA1-Low": InterferometerConfig(
        name="SKA1-Low",
        n_ant=512,
        d_ant_m=38.0,
        b_min_m=35.0,
        b_max_m=65_000.0,
        b_core_m=600.0,
        core_fraction=0.75,
        channel_width_hz=80_000.0,
        t_int_hours=1000.0,
        n_pol=2,
        t_rx_k=40.0,
        default_wedge_mode="horizon",
        default_buffer_h_mpc=0.05,
    ),
    "HERA-350": InterferometerConfig(
        name="HERA-350",
        n_ant=350,
        d_ant_m=14.0,
        b_min_m=14.6,
        b_max_m=876.0,
        b_core_m=180.0,
        core_fraction=0.92,
        channel_width_hz=97_656.25,
        t_int_hours=1000.0,
        n_pol=2,
        t_rx_k=100.0,
        default_wedge_mode="horizon",
        default_buffer_h_mpc=0.05,
    ),
    "MWA-II": InterferometerConfig(
        name="MWA-II",
        n_ant=128,
        d_ant_m=4.0,
        b_min_m=7.7,
        b_max_m=5_300.0,
        b_core_m=140.0,
        core_fraction=0.80,
        channel_width_hz=80_000.0,
        t_int_hours=1000.0,
        n_pol=2,
        t_rx_k=50.0,
        default_wedge_mode="fov",
        default_buffer_h_mpc=0.05,
    ),
    "LOFAR": InterferometerConfig(
        name="LOFAR",
        n_ant=48,
        d_ant_m=30.75,
        b_min_m=68.0,
        b_max_m=80_000.0,
        b_core_m=1_500.0,
        core_fraction=0.65,
        channel_width_hz=195_312.5,
        t_int_hours=1000.0,
        n_pol=2,
        t_rx_k=140.0,
        default_wedge_mode="fov",
        default_buffer_h_mpc=0.05,
    ),
}

_PRESET_ALIASES = {
    "ska": "SKA1-Low",
    "ska1": "SKA1-Low",
    "ska1-low": "SKA1-Low",
    "ska-low": "SKA1-Low",
    "hera": "HERA-350",
    "hera-350": "HERA-350",
    "hera350": "HERA-350",
    "mwa": "MWA-II",
    "mwa-ii": "MWA-II",
    "mwa2": "MWA-II",
    "lofar": "LOFAR",
}


def get_instrument_preset(
    instrument: str | InterferometerConfig,
    **overrides: Any,
) -> InterferometerConfig:
    """Resolve an interferometer preset name or ``InterferometerConfig`` with optional overrides."""
    if isinstance(instrument, InterferometerConfig):
        cfg = instrument
    elif isinstance(instrument, str):
        key = _PRESET_ALIASES.get(instrument.strip().lower(), instrument.strip())
        if key not in INSTRUMENT_PRESETS:
            valid = ", ".join(sorted(INSTRUMENT_PRESETS))
            raise ValueError(f"Unknown instrument {instrument!r}; expected one of: {valid}")
        cfg = INSTRUMENT_PRESETS[key]
    else:
        raise TypeError(
            f"instrument must be str or InterferometerConfig, got {type(instrument).__name__}"
        )
    return cfg.replace(**overrides) if overrides else cfg


def _baseline_uv_from_positions(
    u_arr: np.ndarray,
    xy_m: np.ndarray,
    lam_m: float,
    u_min: float,
    u_max: float,
    n_bins: int = 96,
) -> np.ndarray:
    coords = np.asarray(xy_m, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 2:
        raise ValueError(f"antenna_xy_m must have shape (n_ant, 2), got {coords.shape}")
    diff = coords[:, None, :] - coords[None, :, :]
    iu = np.triu_indices(coords.shape[0], k=1)
    bl_u = np.hypot(diff[iu[0], iu[1], 0], diff[iu[0], iu[1], 1]) / lam_m
    bl_u = bl_u[(bl_u >= u_min) & (bl_u <= u_max)]
    out = np.zeros_like(u_arr, dtype=np.float64)
    if bl_u.size == 0 or u_max <= u_min:
        return out

    edges = np.linspace(max(u_min, 1e-6), u_max * (1.0 + 1e-6), n_bins + 1)
    hist, _ = np.histogram(bl_u, bins=edges)
    annulus_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    density = np.where(annulus_area > 0.0, hist / annulus_area, 0.0)
    centers = 0.5 * (edges[1:] + edges[:-1])
    in_range = (u_arr >= u_min) & (u_arr <= u_max)
    out[in_range] = np.interp(u_arr[in_range], centers, density, left=density[0], right=0.0)
    return out


def baseline_density_uv(
    u_lambda: np.ndarray | float,
    instrument: str | InterferometerConfig,
    redshift: float,
) -> np.ndarray | float:
    """Evaluate azimuthally averaged UV baseline number density $n(u)$ [$\\lambda^{-2}$].

    Normalized so $\\int_{u_{\\min}}^{u_{\\max}} 2\\pi u\\, n(u)\\, du = N_{\\mathrm{ant}}(N_{\\mathrm{ant}}-1)/2$.
    """
    cfg = get_instrument_preset(instrument)
    lam = cfg.wavelength_m(redshift)
    u_min = max(cfg.b_min_m / lam, 1e-3)
    u_max = cfg.b_max_m / lam
    u_core = max(cfg.b_core_m / lam, u_min * 1.1)

    scalar = np.ndim(u_lambda) == 0
    u_arr = np.atleast_1d(np.asarray(u_lambda, dtype=np.float64))

    if cfg.antenna_xy_m is not None:
        out = _baseline_uv_from_positions(u_arr, cfg.antenna_xy_m, lam, u_min, u_max)
    else:
        n_bl = 0.5 * cfg.n_ant * max(cfg.n_ant - 1, 0)
        f_core = float(np.clip(cfg.core_fraction, 0.0, 1.0))

        # Analytic core Gaussian density: integral 2*pi*u * exp(-u^2/(2*sigma^2)) du
        sigma_u = u_core / np.sqrt(2.0 * np.log(2.0))
        norm_core = 2.0 * np.pi * (sigma_u**2) * (
            np.exp(-0.5 * (u_min / sigma_u) ** 2) - np.exp(-0.5 * (u_max / sigma_u) ** 2)
        )
        # Log-flat extended halo density ~ u^-2 across [u_core, u_max]
        u_halo_lo = max(u_core, u_min)
        if u_max > u_halo_lo and (1.0 - f_core) > 0.0:
            norm_halo = 2.0 * np.pi * np.log(u_max / u_halo_lo)
        else:
            norm_halo = 0.0

        in_range = (u_arr >= u_min) & (u_arr <= u_max)
        u_safe = np.where(in_range, np.maximum(u_arr, u_min), 1.0)
        core_term = (
            (f_core * n_bl / norm_core) * np.exp(-0.5 * (u_safe / sigma_u) ** 2)
            if norm_core > 0.0
            else 0.0
        )
        halo_active = in_range & (u_arr >= u_halo_lo)
        halo_term = (
            np.where(
                halo_active,
                ((1.0 - f_core) * n_bl / norm_halo) / (u_safe**2),
                0.0,
            )
            if norm_halo > 0.0
            else 0.0
        )
        out = np.where(in_range, core_term + halo_term, 0.0)

    return float(out[0]) if scalar else out


def thermal_noise_power_spectrum(
    k_perp: np.ndarray | float,
    redshift: float,
    instrument: str | InterferometerConfig,
    *,
    cosmo: Any = None,
) -> np.ndarray | float:
    """Compute interferometer thermal noise power $P_N(k_\\perp, z)$ in [$\\mathrm{mK}^2\\,\\mathrm{Mpc}^3$]."""
    cfg = get_instrument_preset(instrument)
    if cfg.t_int_hours <= 0.0 or cfg.n_pol <= 0:
        raise ValueError("t_int_hours > 0 and n_pol > 0 are required")

    dm_mpc, _, x2y = cfg.comoving_conversions(redshift, cosmo=cosmo)
    omega_p = cfg.primary_beam_omega_sr(redshift)
    omega_eff = cfg.effective_omega_sr(redshift)
    t_sys_mk = cfg.system_temp_k(redshift) * 1000.0
    t_sec = float(cfg.t_int_hours) * 3600.0

    scalar = np.ndim(k_perp) == 0
    kp = np.atleast_1d(np.asarray(k_perp, dtype=np.float64))
    u_arr = kp * dm_mpc / (2.0 * np.pi)
    n_uv = np.asarray(baseline_density_uv(u_arr, cfg, redshift), dtype=np.float64)

    prefactor = x2y * ((omega_p * omega_eff) / (float(cfg.n_pol) * t_sec)) * (t_sys_mk**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        pn = np.where(n_uv > 0.0, prefactor / n_uv, np.inf)
    return float(pn[0]) if scalar else pn


def _resolved_wedge_and_beam_params(
    cfg: InterferometerConfig,
    redshift: float,
    *,
    apply_beam: bool,
    apply_channel: bool,
    channel_response: Literal["boxcar", "gaussian", "none"],
    apply_wedge: bool,
    wedge_mode: Literal["horizon", "fov"] | None,
    wedge_buffer_h_mpc: float | None,
    cosmo: Any,
) -> dict[str, Any]:
    cobj = Planck18 if cosmo is None else cosmo
    dm_mpc, y_mpc_per_hz, _ = cfg.comoving_conversions(redshift, cosmo=cobj)
    lam = cfg.wavelength_m(redshift)
    k_perp_min = 2.0 * np.pi * (cfg.b_min_m / lam) / dm_mpc
    k_perp_max = 2.0 * np.pi * (cfg.b_max_m / lam) / dm_mpc

    sigma_perp_mpc = (
        (cfg.psf_fwhm_rad(redshift) * dm_mpc) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        if apply_beam
        else 0.0
    )

    if not apply_channel or channel_response == "none":
        ch_mode = 0
        channel_dr_mpc = 0.0
    elif channel_response == "boxcar":
        ch_mode = 1
        channel_dr_mpc = cfg.channel_width_hz * y_mpc_per_hz
    elif channel_response == "gaussian":
        ch_mode = 2
        channel_dr_mpc = (cfg.channel_width_hz * y_mpc_per_hz) / np.sqrt(12.0)
    else:
        raise ValueError(
            f"channel_response must be 'boxcar', 'gaussian', or 'none', got {channel_response!r}"
        )

    eff_wedge_mode = cfg.default_wedge_mode if wedge_mode is None else wedge_mode
    fov_rad = (0.5 * cfg.primary_beam_fwhm_rad(redshift)) if eff_wedge_mode == "fov" else None
    w_slope = (
        horizon_slope(redshift, cosmo=cobj, mode=eff_wedge_mode, fov_rad=fov_rad)
        if apply_wedge
        else 0.0
    )
    buf_h = cfg.default_buffer_h_mpc if wedge_buffer_h_mpc is None else float(wedge_buffer_h_mpc)
    w_buf_mpc = (buf_h * float(cobj.h)) if apply_wedge else 0.0

    return {
        "cosmo": cobj,
        "dm_mpc": dm_mpc,
        "y_mpc_per_hz": y_mpc_per_hz,
        "k_perp_min": float(k_perp_min),
        "k_perp_max": float(k_perp_max),
        "sigma_perp_mpc": float(sigma_perp_mpc),
        "channel_mode": int(ch_mode),
        "channel_dr_mpc": float(channel_dr_mpc),
        "wedge_slope": float(w_slope),
        "wedge_buffer_mpc": float(w_buf_mpc),
        "wedge_mode": eff_wedge_mode,
        "wedge_buffer_h_mpc": buf_h,
    }


def _k_grid_and_mask(
    nx: int,
    ny: int,
    nz: int,
    lx: float,
    ly: float,
    lz: float,
    los_axis: int,
    params: dict[str, Any],
    *,
    apply_uv_mask: bool,
    apply_wedge: bool,
    need_kmag: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Compute Fourier ``(k_perp, k_para, k_mag)``, R2C multiplicity, and survival mask."""
    kx = np.fft.fftfreq(nx, d=lx / nx) * (2.0 * np.pi)
    ky = np.fft.fftfreq(ny, d=ly / ny) * (2.0 * np.pi)
    kz = np.fft.rfftfreq(nz, d=lz / nz) * (2.0 * np.pi)
    kx3, ky3, kz3 = np.meshgrid(kx, ky, kz, indexing="ij")
    if los_axis == 0:
        k_para = np.abs(kx3)
        k_perp = np.sqrt(ky3**2 + kz3**2)
    elif los_axis == 1:
        k_para = np.abs(ky3)
        k_perp = np.sqrt(kx3**2 + kz3**2)
    else:
        k_para = np.abs(kz3)
        k_perp = np.sqrt(kx3**2 + ky3**2)

    k_mag = np.sqrt(kx3**2 + ky3**2 + kz3**2) if need_kmag else None

    mult = np.full((1, 1, kz.size), 2.0, dtype=np.float64)
    mult[0, 0, 0] = 1.0
    if nz % 2 == 0:
        mult[0, 0, -1] = 1.0
    mult = np.broadcast_to(mult, k_perp.shape)

    keep = np.ones(k_perp.shape, dtype=bool)
    if apply_uv_mask:
        keep &= (k_perp >= params["k_perp_min"]) & (k_perp <= params["k_perp_max"])
    if apply_wedge:
        keep &= k_para >= (params["wedge_slope"] * k_perp + params["wedge_buffer_mpc"])
    return k_perp, k_para, k_mag, mult, keep


def mock_observation_3d(
    field: np.ndarray,
    box_length: float | tuple[float, float, float],
    redshift: float,
    instrument: str | InterferometerConfig = "SKA1-Low",
    *,
    los_axis: int = 2,
    apply_beam: bool = True,
    apply_channel: bool = True,
    channel_response: Literal["boxcar", "gaussian", "none"] = "boxcar",
    apply_uv_mask: bool = True,
    apply_wedge: bool = True,
    wedge_mode: Literal["horizon", "fov"] | None = None,
    wedge_buffer_h_mpc: float | None = None,
    add_noise: bool = True,
    seed: int = 0,
    use_cuda: bool | None = None,
    cosmo: Any = None,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Simulate a 3D interferometer mock observation (UV coverage, PSF, channel width, wedge, thermal noise).

    Dispatches to the single-pass GPU cuFFT+Philox kernel ``p21_cuda_mock_observation_3d`` when
    CUDA is available, falling back cleanly to an equivalent NumPy implementation on CPU.
    """
    raw = np.asarray(field)
    if raw.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {raw.shape}")
    if los_axis not in (0, 1, 2):
        raise ValueError(f"los_axis must be 0, 1, or 2, got {los_axis}")

    cfg = get_instrument_preset(instrument)
    lengths = _normalize_box_length(box_length)
    lx, ly, lz = lengths
    nx, ny, nz = raw.shape
    vol = lx * ly * lz
    n_tot = nx * ny * nz

    params = _resolved_wedge_and_beam_params(
        cfg,
        redshift,
        apply_beam=apply_beam,
        apply_channel=apply_channel,
        channel_response=channel_response,
        apply_wedge=apply_wedge,
        wedge_mode=wedge_mode,
        wedge_buffer_h_mpc=wedge_buffer_h_mpc,
        cosmo=cosmo,
    )

    k_nyq = np.pi * np.sqrt(2.0) * max(nx / lx, ny / ly, nz / lz)
    noise_kperp_max = float(max(k_nyq * 1.05, params["k_perp_min"] * 1.5, 1e-3))

    meta = {
        "instrument": cfg.name,
        "redshift": float(redshift),
        "t_sys_k": cfg.system_temp_k(redshift),
        "primary_beam_fwhm_deg": float(np.degrees(cfg.primary_beam_fwhm_rad(redshift))),
        "psf_fwhm_arcmin": float(np.degrees(cfg.psf_fwhm_rad(redshift)) * 60.0),
        "channel_dr_mpc": params["channel_dr_mpc"],
        "k_perp_min": params["k_perp_min"],
        "k_perp_max": params["k_perp_max"],
        "wedge_slope": params["wedge_slope"],
        "wedge_buffer_mpc": params["wedge_buffer_mpc"],
        "backend": "cpu",
    }

    if _cuda_ready(use_cuda):
        arr_f32 = np.ascontiguousarray(raw, dtype=np.float32)
        out_f32 = np.empty_like(arr_f32, dtype=np.float32)
        if add_noise:
            n_table = 512
            kp_grid = np.linspace(0.0, noise_kperp_max, n_table, dtype=np.float64)
            pn_grid = np.asarray(
                thermal_noise_power_spectrum(kp_grid, redshift, cfg, cosmo=params["cosmo"]),
                dtype=np.float64,
            )
            sigma_re_table = np.where(
                np.isfinite(pn_grid) & (pn_grid > 0.0),
                np.sqrt(0.5 * (float(n_tot) ** 2 / vol) * pn_grid),
                0.0,
            ).astype(np.float32)
            tbl_ptr = _ffi.cast("const float *", _ffi.from_buffer(sigma_re_table))
        else:
            n_table = 0
            sigma_re_table = np.empty(0, dtype=np.float32)
            tbl_ptr = _ffi.NULL

        rc = _lib.p21_cuda_mock_observation_3d(
            _ffi.cast("const float *", _ffi.from_buffer(arr_f32)),
            _ffi.cast("float *", _ffi.from_buffer(out_f32)),
            int(nx),
            int(ny),
            int(nz),
            float(lx),
            float(ly),
            float(lz),
            int(los_axis),
            float(params["sigma_perp_mpc"]),
            float(params["channel_dr_mpc"]),
            int(params["channel_mode"]),
            1 if apply_uv_mask else 0,
            float(params["k_perp_min"]),
            float(params["k_perp_max"]),
            1 if apply_wedge else 0,
            float(params["wedge_slope"]),
            float(params["wedge_buffer_mpc"]),
            1 if add_noise else 0,
            int(n_table),
            float(noise_kperp_max),
            tbl_ptr,
            int(seed) & 0xFFFFFFFFFFFFFFFF,
        )
        if rc == 0:
            meta["backend"] = "cuda"
            return (out_f32, meta) if return_metadata else out_f32

    k_perp, k_para, _, _, keep = _k_grid_and_mask(
        nx,
        ny,
        nz,
        lx,
        ly,
        lz,
        los_axis,
        params,
        apply_uv_mask=apply_uv_mask,
        apply_wedge=apply_wedge,
        need_kmag=False,
    )

    resp = np.ones(k_perp.shape, dtype=np.float64)
    if params["sigma_perp_mpc"] > 0.0:
        resp *= np.exp(-0.5 * (k_perp * params["sigma_perp_mpc"]) ** 2)
    if params["channel_dr_mpc"] > 0.0 and params["channel_mode"] != 0:
        if params["channel_mode"] == 1:
            x = 0.5 * k_para * params["channel_dr_mpc"]
            resp *= np.where(x > 1e-8, np.sin(np.maximum(x, 1e-12)) / np.maximum(x, 1e-12), 1.0)
        else:
            resp *= np.exp(-0.5 * (k_para * params["channel_dr_mpc"]) ** 2)

    kspace = np.fft.rfftn(raw.astype(np.float64, copy=False))
    kspace = np.where(keep, kspace * resp, 0.0)

    if add_noise:
        rng = np.random.default_rng(int(seed))
        white_real = rng.normal(0.0, 1.0, size=(nx, ny, nz))
        fn = np.fft.rfftn(white_real)
        fn[0, 0, 0] = 0.0
        pn = np.asarray(
            thermal_noise_power_spectrum(k_perp, redshift, cfg, cosmo=params["cosmo"]),
            dtype=np.float64,
        )
        scale = np.where(
            keep & np.isfinite(pn) & (pn > 0.0),
            np.sqrt((float(n_tot) / vol) * pn),
            0.0,
        )
        kspace = kspace + fn * scale

    out = np.fft.irfftn(kspace, s=(nx, ny, nz), axes=(0, 1, 2)).astype(np.float32)
    return (out, meta) if return_metadata else out


def forecast_power_spectrum_error(
    box_shape: tuple[int, int, int],
    box_length: float | tuple[float, float, float],
    redshift: float,
    instrument: str | InterferometerConfig = "SKA1-Low",
    *,
    signal_field: np.ndarray | None = None,
    los_axis: int = 2,
    n_bins: int = 16,
    k_min: float | None = None,
    k_max: float | None = None,
    log_bins: bool = True,
    include_sample_variance: bool = True,
    apply_uv_mask: bool = True,
    apply_wedge: bool = True,
    wedge_mode: Literal["horizon", "fov"] | None = None,
    wedge_buffer_h_mpc: float | None = None,
    n_fields: int | None = None,
    use_cuda: bool | None = None,
    cosmo: Any = None,
) -> dict[str, Any]:
    """Forecast 1D 21cm power-spectrum thermal + sample-variance sensitivity ``sigma_P(k)``."""
    if len(box_shape) != 3 or any(int(s) <= 0 for s in box_shape):
        raise ValueError(f"box_shape must be 3 positive integers, got {box_shape!r}")
    if los_axis not in (0, 1, 2):
        raise ValueError(f"los_axis must be 0, 1, or 2, got {los_axis}")

    cfg = get_instrument_preset(instrument)
    lengths = _normalize_box_length(box_length)
    lx, ly, lz = lengths
    nx, ny, nz = (int(box_shape[0]), int(box_shape[1]), int(box_shape[2]))
    k_edges = _default_k_edges(nx, ny, nz, lx, ly, lz, n_bins, k_min, k_max, log_bins)
    params = _resolved_wedge_and_beam_params(
        cfg,
        redshift,
        apply_beam=False,
        apply_channel=False,
        channel_response="none",
        apply_wedge=apply_wedge,
        wedge_mode=wedge_mode,
        wedge_buffer_h_mpc=wedge_buffer_h_mpc,
        cosmo=cosmo,
    )

    sig_pk_vals: np.ndarray | None = None
    if signal_field is not None:
        arr = np.asarray(signal_field, dtype=np.float32)
        if arr.shape != (nx, ny, nz):
            raise ValueError(
                f"signal_field shape {arr.shape} does not match box_shape {(nx, ny, nz)}"
            )
        ps = power_spectrum_1d(
            arr,
            lengths,
            n_bins=n_bins,
            k_edges=k_edges,
            remove_mean=True,
            use_cuda=use_cuda,
        )
        sig_pk_vals = ps["pk"]

    k_perp, _, k_mag, mult, clean = _k_grid_and_mask(
        nx,
        ny,
        nz,
        lx,
        ly,
        lz,
        los_axis,
        params,
        apply_uv_mask=apply_uv_mask,
        apply_wedge=apply_wedge,
        need_kmag=True,
    )
    clean[0, 0, 0] = False

    pn_grid = np.asarray(
        thermal_noise_power_spectrum(k_perp, redshift, cfg, cosmo=params["cosmo"]),
        dtype=np.float64,
    )
    clean &= np.isfinite(pn_grid) & (pn_grid > 0.0)

    k_flat = k_mag[clean]
    pn_flat = pn_grid[clean]
    indep_modes = 0.5 * mult[clean]  # unique real Fourier mode pairs per R2C entry

    bin_idx = np.searchsorted(k_edges, k_flat, side="right") - 1
    in_bins = (k_flat >= k_edges[0]) & (k_flat < k_edges[-1]) & (bin_idx >= 0) & (bin_idx < n_bins)

    bi = bin_idx[in_bins]
    wf = indep_modes[in_bins]
    pn_f = pn_flat[in_bins]

    eff_fields = float(cfg.n_fields if n_fields is None else n_fields)
    if eff_fields <= 0.0:
        raise ValueError("n_fields must be > 0")

    p_sig_mode = np.zeros_like(pn_f)
    if include_sample_variance and sig_pk_vals is not None:
        raw_sig = sig_pk_vals[bi]
        p_sig_mode = np.where(np.isfinite(raw_sig) & (raw_sig > 0.0), raw_sig, 0.0)

    inv_var = wf * eff_fields / ((p_sig_mode + pn_f) ** 2)
    sum_inv_var = np.bincount(bi, weights=inv_var, minlength=n_bins)
    sum_k = np.bincount(bi, weights=wf * k_flat[in_bins], minlength=n_bins)
    sum_modes = np.bincount(bi, weights=wf, minlength=n_bins)
    sum_pn = np.bincount(bi, weights=wf * pn_f, minlength=n_bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        k_mean = np.where(sum_modes > 0.0, sum_k / sum_modes, np.nan)
        sigma_pk = np.where(sum_inv_var > 0.0, 1.0 / np.sqrt(sum_inv_var), np.inf)
        sigma_delta_sq = np.where(
            np.isfinite(sigma_pk) & np.isfinite(k_mean),
            (k_mean**3) * sigma_pk / (2.0 * np.pi**2),
            np.inf,
        )
        pn_mean = np.where(sum_modes > 0.0, sum_pn / sum_modes, np.inf)

    out: dict[str, Any] = {
        "instrument": cfg.name,
        "redshift": float(redshift),
        "k": k_mean,
        "k_edges": k_edges,
        "sigma_pk": sigma_pk,
        "sigma_delta_sq": sigma_delta_sq,
        "thermal_noise_pk": pn_mean,
        "n_clean_modes": sum_modes,
    }
    if sig_pk_vals is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            signal_delta_sq = np.where(
                np.isfinite(sig_pk_vals) & np.isfinite(k_mean),
                (k_mean**3) * sig_pk_vals / (2.0 * np.pi**2),
                np.nan,
            )
            snr = np.where(
                np.isfinite(sig_pk_vals) & np.isfinite(sigma_pk) & (sigma_pk > 0.0),
                np.abs(sig_pk_vals) / sigma_pk,
                0.0,
            )
        out["signal_pk"] = sig_pk_vals
        out["signal_delta_sq"] = signal_delta_sq
        out["snr"] = snr
        out["total_snr"] = float(np.sqrt(np.sum(snr**2)))
    return out


def forecast_power_spectrum_2d_error(
    box_shape: tuple[int, int, int],
    box_length: float | tuple[float, float, float],
    redshift: float,
    instrument: str | InterferometerConfig = "SKA1-Low",
    *,
    signal_field: np.ndarray | None = None,
    los_axis: int = 2,
    n_kperp: int = 12,
    n_kpara: int = 12,
    kperp_edges: np.ndarray | None = None,
    kpara_edges: np.ndarray | None = None,
    include_sample_variance: bool = True,
    apply_uv_mask: bool = True,
    apply_wedge: bool = True,
    wedge_mode: Literal["horizon", "fov"] | None = None,
    wedge_buffer_h_mpc: float | None = None,
    n_fields: int | None = None,
    use_cuda: bool | None = None,
    cosmo: Any = None,
) -> dict[str, Any]:
    """Forecast 2D cylindrical $(k_\\perp, k_\\parallel)$ 21cm power-spectrum sensitivity."""
    if len(box_shape) != 3 or any(int(s) <= 0 for s in box_shape):
        raise ValueError(f"box_shape must be 3 positive integers, got {box_shape!r}")
    if los_axis not in (0, 1, 2):
        raise ValueError(f"los_axis must be 0, 1, or 2, got {los_axis}")

    cfg = get_instrument_preset(instrument)
    lengths = _normalize_box_length(box_length)
    lx, ly, lz = lengths
    nx, ny, nz = (int(box_shape[0]), int(box_shape[1]), int(box_shape[2]))

    if kperp_edges is None:
        kp_edges = _default_k_edges(nx, ny, nz, lx, ly, lz, n_kperp, None, None, True)
    else:
        kp_edges = np.asarray(kperp_edges, dtype=np.float64)
        n_kperp = kp_edges.size - 1
    if kpara_edges is None:
        kz_edges = _default_k_edges(nx, ny, nz, lx, ly, lz, n_kpara, None, None, True)
    else:
        kz_edges = np.asarray(kpara_edges, dtype=np.float64)
        n_kpara = kz_edges.size - 1

    params = _resolved_wedge_and_beam_params(
        cfg,
        redshift,
        apply_beam=False,
        apply_channel=False,
        channel_response="none",
        apply_wedge=apply_wedge,
        wedge_mode=wedge_mode,
        wedge_buffer_h_mpc=wedge_buffer_h_mpc,
        cosmo=cosmo,
    )

    sig_pk_2d: np.ndarray | None = None
    if signal_field is not None:
        arr = np.asarray(signal_field, dtype=np.float32)
        if arr.shape != (nx, ny, nz):
            raise ValueError(
                f"signal_field shape {arr.shape} does not match box_shape {(nx, ny, nz)}"
            )
        ps2d = power_spectrum_2d(
            arr,
            lengths,
            los_axis=los_axis,
            kperp_edges=kp_edges,
            kpara_edges=kz_edges,
            remove_mean=True,
            use_cuda=use_cuda,
        )
        sig_pk_2d = ps2d["pk"]

    k_perp, k_para, _, mult, clean = _k_grid_and_mask(
        nx,
        ny,
        nz,
        lx,
        ly,
        lz,
        los_axis,
        params,
        apply_uv_mask=apply_uv_mask,
        apply_wedge=apply_wedge,
        need_kmag=False,
    )
    clean[0, 0, 0] = False

    pn_grid = np.asarray(
        thermal_noise_power_spectrum(k_perp, redshift, cfg, cosmo=params["cosmo"]),
        dtype=np.float64,
    )
    clean &= np.isfinite(pn_grid) & (pn_grid > 0.0)

    kp_f = k_perp[clean]
    kz_f = k_para[clean]
    pn_f = pn_grid[clean]
    wf = 0.5 * mult[clean]

    ip = np.searchsorted(kp_edges, kp_f, side="right") - 1
    iz = np.searchsorted(kz_edges, kz_f, side="right") - 1
    in_bins = (
        (kp_f >= kp_edges[0])
        & (kp_f < kp_edges[-1])
        & (kz_f >= kz_edges[0])
        & (kz_f < kz_edges[-1])
        & (ip >= 0)
        & (ip < n_kperp)
        & (iz >= 0)
        & (iz < n_kpara)
    )

    flat_idx = ip[in_bins] * n_kpara + iz[in_bins]
    wf_b = wf[in_bins]
    pn_b = pn_f[in_bins]
    n_bins_tot = n_kperp * n_kpara
    eff_fields = float(cfg.n_fields if n_fields is None else n_fields)

    p_sig_mode = np.zeros_like(pn_b)
    if include_sample_variance and sig_pk_2d is not None:
        raw_sig = sig_pk_2d[ip[in_bins], iz[in_bins]]
        p_sig_mode = np.where(np.isfinite(raw_sig) & (raw_sig > 0.0), raw_sig, 0.0)

    inv_var = wf_b * eff_fields / ((p_sig_mode + pn_b) ** 2)
    sum_inv_var = np.bincount(flat_idx, weights=inv_var, minlength=n_bins_tot).reshape(
        n_kperp, n_kpara
    )
    sum_modes = np.bincount(flat_idx, weights=wf_b, minlength=n_bins_tot).reshape(n_kperp, n_kpara)

    kp_centers = 0.5 * (kp_edges[1:] + kp_edges[:-1])
    kz_centers = 0.5 * (kz_edges[1:] + kz_edges[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_pk = np.where(sum_inv_var > 0.0, 1.0 / np.sqrt(sum_inv_var), np.inf)

    out: dict[str, Any] = {
        "instrument": cfg.name,
        "redshift": float(redshift),
        "k_perp": kp_centers,
        "k_para": kz_centers,
        "k_perp_edges": kp_edges,
        "k_para_edges": kz_edges,
        "sigma_pk": sigma_pk,
        "n_clean_modes": sum_modes,
    }
    if sig_pk_2d is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(
                np.isfinite(sig_pk_2d) & np.isfinite(sigma_pk) & (sigma_pk > 0.0),
                np.abs(sig_pk_2d) / sigma_pk,
                0.0,
            )
        out["signal_pk"] = sig_pk_2d
        out["snr"] = snr
        out["total_snr"] = float(np.sqrt(np.sum(snr**2)))
    return out
