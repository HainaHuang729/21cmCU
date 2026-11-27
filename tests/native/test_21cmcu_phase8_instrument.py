"""Phase 8 (Direction S) interferometer mock observation & sensitivity forecast tests."""

from __future__ import annotations

import numpy as np
import pytest

import py21cmcu

requires_cuda_runtime = pytest.mark.skipif(
    not (py21cmcu.cuda_compiled() and py21cmcu.cuda_available()),
    reason="CUDA runtime not available on this host",
)


def _synthetic_eor_field(shape: tuple[int, int, int], box_len: float, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    kx = np.fft.fftfreq(nx, d=box_len / nx) * (2.0 * np.pi)
    ky = np.fft.fftfreq(ny, d=box_len / ny) * (2.0 * np.pi)
    kz = np.fft.rfftfreq(nz, d=box_len / nz) * (2.0 * np.pi)
    kmag = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)
    scale = np.where(kmag > 0.0, 25.0 / np.sqrt(1.0 + (kmag / 0.15) ** 3), 0.0)
    white = np.fft.rfftn(rng.normal(0.0, 1.0, size=shape))
    return np.fft.irfftn(white * scale, s=shape, axes=(0, 1, 2)).astype(np.float32) * 18.0


def test_presets_baseline_density_and_thermal_noise_physics() -> None:
    """Verify instrument presets, T_sys scaling, UV baseline normalization, and P_N physics."""
    ska = py21cmcu.get_instrument_preset("ska")
    hera = py21cmcu.get_instrument_preset("HERA-350")
    assert ska.name == "SKA1-Low"
    assert hera.name == "HERA-350"
    assert ska.system_temp_k(10.0) > ska.system_temp_k(7.0)

    for cfg in (ska, hera):
        lam = cfg.wavelength_m(8.0)
        u_min = cfg.b_min_m / lam
        u_max = cfg.b_max_m / lam
        u_grid = np.geomspace(u_min, u_max, 4096)
        n_uv = py21cmcu.baseline_density_uv(u_grid, cfg, 8.0)
        n_bl_int = float(np.trapezoid(2.0 * np.pi * u_grid * n_uv, u_grid))
        n_bl_expected = 0.5 * cfg.n_ant * (cfg.n_ant - 1)
        assert np.isclose(n_bl_int, n_bl_expected, rtol=0.03)

    # Custom 2D antenna coordinates produce finite P_N in range and inf outside baselines
    rng = np.random.default_rng(7)
    xy = rng.normal(0.0, 180.0, size=(64, 2))
    custom = ska.replace(n_ant=64, antenna_xy_m=xy, b_min_m=15.0, b_max_m=700.0)
    kp = np.array([1e-4, 0.06, 0.12, 10.0])
    pn = py21cmcu.thermal_noise_power_spectrum(kp, 8.0, custom)
    assert np.isinf(pn[0]) and np.isinf(pn[-1])
    assert np.all(np.isfinite(pn[1:3]) & (pn[1:3] > 0.0))


@pytest.mark.parametrize("los_axis", [0, 1, 2])
@pytest.mark.parametrize("channel_response", ["boxcar", "gaussian"])
@requires_cuda_runtime
def test_mock_observation_signal_only_gpu_cpu_parity(
    los_axis: int, channel_response: str
) -> None:
    """Verify signal-only mock_observation_3d GPU/CPU parity across axes and channel shapes."""
    field = _synthetic_eor_field((32, 32, 32), 256.0, seed=19 + los_axis)
    out_gpu, meta_gpu = py21cmcu.mock_observation_3d(
        field,
        box_length=256.0,
        redshift=8.0,
        instrument="SKA1-Low",
        los_axis=los_axis,
        channel_response=channel_response,
        add_noise=False,
        use_cuda=True,
        return_metadata=True,
    )
    out_cpu, meta_cpu = py21cmcu.mock_observation_3d(
        field,
        box_length=256.0,
        redshift=8.0,
        instrument="SKA1-Low",
        los_axis=los_axis,
        channel_response=channel_response,
        add_noise=False,
        use_cuda=False,
        return_metadata=True,
    )
    assert meta_gpu["backend"] == "cuda"
    assert meta_cpu["backend"] == "cpu"
    assert out_gpu.dtype == np.float32 and out_gpu.shape == field.shape
    np.testing.assert_allclose(out_gpu, out_cpu, rtol=1e-4, atol=2e-3)
    py21cmcu.cleanup_cuda()


def test_mock_observation_noise_power_spectrum_cpu_and_gpu() -> None:
    """Verify deterministic Hermitian thermal noise matches analytic P_N(k_perp) and wedge zeroing."""
    shape = (48, 48, 48)
    box_len = 256.0
    zeros = np.zeros(shape, dtype=np.float32)
    cfg = py21cmcu.get_instrument_preset("HERA-350", t_int_hours=200.0)

    use_backends = [False]
    if py21cmcu.cuda_compiled() and py21cmcu.cuda_available():
        use_backends.append(True)

    kp_edges = np.array([0.04, 0.075, 0.11], dtype=np.float64)
    kz_edges = np.array([0.15, 0.55], dtype=np.float64)
    kz_wedge_edges = np.array([0.02, 0.08], dtype=np.float64)

    for use_cuda in use_backends:
        # Realize multiple seeds with wedge off to check unbiased P_N(k_perp) normalization
        measured = []
        kperp_measured = []
        for s in range(6):
            n_box = py21cmcu.mock_observation_3d(
                zeros,
                box_length=box_len,
                redshift=8.0,
                instrument=cfg,
                apply_beam=False,
                apply_channel=False,
                apply_uv_mask=True,
                apply_wedge=False,
                add_noise=True,
                seed=500 + s,
                use_cuda=use_cuda,
            )
            assert abs(float(np.mean(n_box))) < 1e-3
            ps2d = py21cmcu.power_spectrum_2d(
                n_box,
                box_len,
                los_axis=2,
                kperp_edges=kp_edges,
                kpara_edges=kz_edges,
                use_cuda=use_cuda,
            )
            measured.append(ps2d["pk"][:, 0])
            kperp_measured.append(ps2d["k_perp"][:, 0])

        # With wedge on, deep-wedge modes are strictly zeroed
        w_box = py21cmcu.mock_observation_3d(
            zeros,
            box_length=box_len,
            redshift=8.0,
            instrument=cfg,
            apply_beam=False,
            apply_channel=False,
            apply_uv_mask=True,
            apply_wedge=True,
            add_noise=True,
            seed=500,
            use_cuda=use_cuda,
        )
        ps_wedge = py21cmcu.power_spectrum_2d(
            w_box,
            box_len,
            los_axis=2,
            kperp_edges=kp_edges,
            kpara_edges=kz_wedge_edges,
            use_cuda=use_cuda,
        )
        assert float(np.max(ps_wedge["pk"])) < 1e-6

        # Exact reproducibility check
        n_rep = py21cmcu.mock_observation_3d(
            zeros,
            box_length=box_len,
            redshift=8.0,
            instrument=cfg,
            apply_beam=False,
            apply_channel=False,
            apply_uv_mask=True,
            apply_wedge=True,
            add_noise=True,
            seed=500,
            use_cuda=use_cuda,
        )
        np.testing.assert_array_equal(n_rep, w_box)

        mean_pk = np.mean(measured, axis=0)
        mean_kp = np.mean(kperp_measured, axis=0)
        analytic_pn = py21cmcu.thermal_noise_power_spectrum(mean_kp, 8.0, cfg)
        ratio = mean_pk / analytic_pn
        assert np.all((ratio > 0.75) & (ratio < 1.25)), f"use_cuda={use_cuda}, ratio={ratio}"

    if py21cmcu.cuda_compiled() and py21cmcu.cuda_available():
        py21cmcu.cleanup_cuda()


def test_forecast_power_spectrum_1d_and_2d_scaling() -> None:
    """Verify 1D/2D forecast scaling with t_int, n_fields, and signal SNR."""
    shape = (32, 32, 32)
    box_len = 256.0
    sig = _synthetic_eor_field(shape, box_len, seed=99)

    fc_1000 = py21cmcu.forecast_power_spectrum_error(
        shape,
        box_len,
        8.0,
        py21cmcu.get_instrument_preset("SKA1-Low", t_int_hours=1000.0),
        signal_field=sig,
        include_sample_variance=False,
        n_bins=8,
    )
    fc_2000 = py21cmcu.forecast_power_spectrum_error(
        shape,
        box_len,
        8.0,
        py21cmcu.get_instrument_preset("SKA1-Low", t_int_hours=2000.0),
        signal_field=sig,
        include_sample_variance=False,
        n_bins=8,
    )
    fc_2fields = py21cmcu.forecast_power_spectrum_error(
        shape,
        box_len,
        8.0,
        py21cmcu.get_instrument_preset("SKA1-Low", t_int_hours=1000.0),
        signal_field=sig,
        include_sample_variance=False,
        n_bins=8,
        n_fields=2,
    )

    finite = np.isfinite(fc_1000["sigma_pk"])
    assert np.any(finite)
    assert fc_1000["total_snr"] > 0.0
    # Thermal sensitivity scales as 1 / t_int and 1 / sqrt(n_fields)
    np.testing.assert_allclose(fc_1000["sigma_pk"][finite] / fc_2000["sigma_pk"][finite], 2.0)
    np.testing.assert_allclose(
        fc_1000["sigma_pk"][finite] / fc_2fields["sigma_pk"][finite], np.sqrt(2.0)
    )

    fc_2d = py21cmcu.forecast_power_spectrum_2d_error(
        shape, box_len, 8.0, "HERA-350", signal_field=sig, n_kperp=6, n_kpara=6
    )
    assert fc_2d["sigma_pk"].shape == (6, 6)
    assert fc_2d["snr"].shape == (6, 6)
    assert fc_2d["total_snr"] > 0.0
