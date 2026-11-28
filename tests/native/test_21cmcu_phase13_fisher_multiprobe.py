"""Phase 13 tests (v0.9.0 Phase 11): CRN Fisher forecaster, CIC/gravityCU bridge, 21cm x galaxy, & kSZ."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

import py21cmcu
from py21cmfast.c_21cmfast import lib

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


def test_cic_deposit_mass_conservation_periodic_and_deconvolution() -> None:
    """Verify 3D CIC conservation, exact 8-cell vertex weights, periodic wrap, and deconvolution."""
    shape = (8, 8, 8)
    box_len = 16.0
    dx = 2.0

    # Empty catalog returns zeros cleanly
    empty = py21cmcu.cic_deposit_3d(np.zeros((0, 3)), shape, box_len, overdensity=False, use_cuda=False)
    assert empty.shape == shape
    assert np.all(empty == 0.0)

    # Single particle at (0.25*dx, 0.50*dx, 0.75*dx) -> exact trilinear vertex weights
    pos = np.array([[0.25 * dx, 0.50 * dx, 0.75 * dx]], dtype=np.float32)
    grid = py21cmcu.cic_deposit_3d(
        pos, shape, box_len, weights=np.array([4.0]), overdensity=False, use_cuda=False
    )
    assert np.isclose(np.sum(grid), 4.0, rtol=1e-6)
    np.testing.assert_allclose(grid[0, 0, 0], 4.0 * 0.75 * 0.5 * 0.25, rtol=1e-6)
    np.testing.assert_allclose(grid[1, 1, 1], 4.0 * 0.25 * 0.5 * 0.75, rtol=1e-6)

    # Periodic wrap with negative and >box_len coordinates
    pos_wrap = np.array([[-0.5 * dx, 8.25 * dx, 16.5 * dx]], dtype=np.float32)
    grid_wrap = py21cmcu.cic_deposit_3d(pos_wrap, shape, box_len, overdensity=True, use_cuda=False)
    assert abs(float(np.mean(grid_wrap))) < 1e-6
    assert grid_wrap[7, 0, 0] > 0.0

    # CIC deconvolution boosts Nyquist-adjacent Fourier amplitude back toward analytic density
    n = 32
    L = 64.0
    rng = np.random.default_rng(1301)
    k_mode = 2.0 * np.pi * 6.0 / L  # mode index 6 along x
    u = rng.uniform(0.0, L, size=(80_000, 3))
    w = 1.0 + 0.5 * np.cos(k_mode * u[:, 0])
    raw = py21cmcu.cic_deposit_3d(u, (n, n, n), L, weights=w, overdensity=True, deconvolve=False, use_cuda=False)
    dec = py21cmcu.cic_deposit_3d(u, (n, n, n), L, weights=w, overdensity=True, deconvolve=True, use_cuda=False)
    amp_raw = 2.0 * abs(float(np.fft.rfftn(raw)[6, 0, 0])) / raw.size
    amp_dec = 2.0 * abs(float(np.fft.rfftn(dec)[6, 0, 0])) / dec.size
    assert abs(amp_dec - 0.5) < abs(amp_raw - 0.5)
    assert abs(amp_dec - 0.5) < 0.03


def test_load_gravitycu_halos_hdf5_and_npz_schemas(tmp_path: Path) -> None:
    """Verify gravityCU / Gadget HDF5 and NPZ halo bridge, unit scaling, and mass cuts."""
    h5_path = tmp_path / "snap_halos.h5"
    pos = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]], dtype=np.float32)
    mass = np.array([1e10, 5e11, 2e12], dtype=np.float32)
    vel = np.array([[100.0, 0.0, -50.0], [0.0, 200.0, 0.0], [-10.0, 20.0, 30.0]], dtype=np.float32)
    with h5py.File(h5_path, "w") as fl:
        hdr = fl.create_group("Header")
        hdr.attrs["BoxSize"] = 100.0
        hdr.attrs["HubbleParam"] = 0.7
        grp = fl.create_group("Group")
        grp.create_dataset("GroupPos", data=pos)
        grp.create_dataset("GroupMass", data=mass)
        grp.create_dataset("GroupVel", data=vel)

    cat = py21cmcu.load_gravitycu_halos(h5_path, mass_min=1e11, mass_max=1e12, units="Mpc/h")
    assert cat["n_halos"] == 1
    np.testing.assert_allclose(cat["box_length"], 100.0 / 0.7, rtol=1e-6)
    np.testing.assert_allclose(cat["positions"][0], np.array([40.0, 50.0, 60.0]) / 0.7, rtol=1e-5)
    np.testing.assert_allclose(cat["masses"], [5e11], rtol=1e-6)
    assert cat["velocities"] is not None and cat["velocities"].shape == (1, 3)

    npz_path = tmp_path / "halos.npz"
    np.savez(npz_path, positions=pos * 1000.0, masses=mass, box_length=np.array([100_000.0]))
    cat_npz = py21cmcu.load_gravitycu_halos(npz_path, units="kpc")
    assert cat_npz["n_halos"] == 3
    np.testing.assert_allclose(cat_npz["box_length"], 100.0, rtol=1e-6)
    np.testing.assert_allclose(cat_npz["positions"], pos, rtol=1e-5)


def test_cross_power_21cm_galaxy_analytic_and_error() -> None:
    """Verify r(k) for +1/-1/uncorrelated fields, shot noise, and cross-power error scaling."""
    rng = np.random.default_rng(1302)
    shape = (32, 32, 32)
    L = 128.0
    g1 = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    g2 = rng.normal(0.0, 1.0, size=shape).astype(np.float32)

    c_pos = py21cmcu.cross_power_21cm_galaxy(20.0 * g1, g1, L, n_bins=8, use_cuda=False)
    c_neg = py21cmcu.cross_power_21cm_galaxy(-20.0 * g1, g1, L, n_bins=8, use_cuda=False)
    c_unc = py21cmcu.cross_power_21cm_galaxy(20.0 * g1, g2, L, n_bins=8, use_cuda=False)

    valid = np.isfinite(c_pos["r_k"]) & (c_pos["counts"] >= 20)
    np.testing.assert_allclose(c_pos["r_k"][valid], 1.0, atol=1e-6)
    np.testing.assert_allclose(c_neg["r_k"][valid], -1.0, atol=1e-6)
    assert np.all(np.abs(c_unc["r_k"][valid]) < 0.25)

    # Particle catalog computes exact Poisson shot noise L^3 / N_gal
    pos = rng.uniform(0.0, L, size=(8192, 3)).astype(np.float32)
    c_cat = py21cmcu.cross_power_21cm_galaxy(20.0 * g1, pos, L, n_bins=8, use_cuda=False)
    assert np.isclose(c_cat["shot_noise_pk"], (L**3) / 8192.0, rtol=1e-6)
    np.testing.assert_allclose(c_cat["pk_gal_shot_sub"], c_cat["pk_gal"] - c_cat["shot_noise_pk"])

    e1 = py21cmcu.cross_power_21cm_galaxy_error(
        c_pos["pk_21"], c_pos["pk_gal"], c_pos["pk_cross"], c_pos["counts"], n_fields=1.0
    )
    e4 = py21cmcu.cross_power_21cm_galaxy_error(
        c_pos["pk_21"], c_pos["pk_gal"], c_pos["pk_cross"], c_pos["counts"], n_fields=4.0
    )
    np.testing.assert_allclose(e4["sigma_pk_cross"], 0.5 * e1["sigma_pk_cross"], rtol=1e-6)
    assert e4["total_snr"] > e1["total_snr"]


def test_ksz_momentum_map_physics_and_coeval(tmp_path: Path) -> None:
    """Verify kSZ physics (zero when xH=1, linear velocity scaling, Parseval) and Coeval input."""
    n = 24
    L = 96.0
    xh_neutral = np.ones((n, n, n), dtype=np.float32)
    dens = np.zeros((n, n, n), dtype=np.float32)
    vel_kms = np.full((n, n, n), 300.0, dtype=np.float32)

    ksz0 = py21cmcu.ksz_momentum_map(
        xh_neutral, L, 8.0, density=dens, los_velocity=vel_kms, velocity_units="km_s"
    )
    assert np.all(ksz0["delta_T_uK"] == 0.0) and ksz0["rms_uK"] == 0.0

    x = np.arange(n, dtype=np.float64) * (L / n)
    xx, _, _ = np.meshgrid(x, x, x, indexing="ij")
    vel_wave = (300.0 * np.sin(2.0 * np.pi * 2.0 * xx / L)).astype(np.float32)
    xh_ion = np.full((n, n, n), 0.2, dtype=np.float32)

    ksz1 = py21cmcu.ksz_momentum_map(
        xh_ion, L, 8.0, density=dens, los_velocity=vel_wave, velocity_units="km_s"
    )
    ksz2 = py21cmcu.ksz_momentum_map(
        xh_ion, L, 8.0, density=dens, los_velocity=2.0 * vel_wave, velocity_units="km_s"
    )
    assert ksz1["rms_uK"] > 0.0
    np.testing.assert_allclose(ksz2["delta_T_uK"], 2.0 * ksz1["delta_T_uK"], rtol=1e-5)
    valid_ell = np.isfinite(ksz1["cl_uK2"])
    np.testing.assert_allclose(ksz2["cl_uK2"][valid_ell], 4.0 * ksz1["cl_uK2"][valid_ell], rtol=1e-5)

    # Real Coeval integration with comoving Mpc/s velocities
    with py21cmcu.config.use(direc=str(tmp_path / "ksz")):
        inputs = py21cmcu.InputParameters(
            random_seed=1303, node_redshifts=[8.0]
        ).evolve_input_structs(
            DIM=24,
            HII_DIM=12,
            BOX_LEN=48.0,
            N_THREADS=2,
            SOURCE_MODEL="E-INTEGRAL",
            USE_TS_FLUCT=False,
            RECOMB_MODEL="none",
            USE_EXP_FILTER=False,
            USE_UPPER_STELLAR_TURNOVER=False,
            KEEP_3D_VELOCITIES=True,
        )
        cv = py21cmcu.run_coeval(inputs=inputs, out_redshifts=[8.0], write=False)[0]
        ksz_cv = py21cmcu.ksz_momentum_map(cv, n_ell_bins=6)
        assert ksz_cv["delta_T_uK"].shape == (12, 12)
        assert np.isfinite(ksz_cv["rms_uK"]) and ksz_cv["rms_uK"] > 0.0


def test_fisher_forecaster_analytic_crn_prior_combine_and_contours(tmp_path: Path) -> None:
    """Verify exact analytic Fisher recovery, CRN noise cancellation, Richardson, combine, & plot."""
    rng = np.random.default_rng(1304)
    A = rng.normal(0.0, 1.0, size=(8, 2))
    raw = rng.normal(0.0, 0.3, size=(8, 8))
    cov = np.eye(8) * 0.5 + 0.05 * (raw @ raw.T)
    noise_bank = rng.normal(0.0, 2.0, size=(10, 8))

    def noisy_linear(theta: np.ndarray, seed: int) -> np.ndarray:
        return A @ theta + noise_bank[seed]

    fc = py21cmcu.FisherForecaster.from_callable(
        noisy_linear,
        param_names=("p0", "p1"),
        fiducial=(1.2, -0.7),
        step_sizes=(0.05, 0.05),
        covariance=cov,
        n_seeds=4,
    )
    res = fc.run()
    f_exact = A.T @ np.linalg.solve(cov, A)
    c_exact = np.linalg.inv(f_exact)
    np.testing.assert_allclose(res.fisher_matrix, f_exact, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res.covariance_matrix, c_exact, rtol=1e-10, atol=1e-10)

    # CRN prior and combine check
    res_p = res.with_prior({"p0": 0.4})
    f_p_exact = f_exact + np.diag([1.0 / (0.4**2), 0.0])
    np.testing.assert_allclose(res_p.fisher_matrix, f_p_exact, rtol=1e-10)
    assert res_p.sigma_marginalized[0] < res.sigma_marginalized[0]

    res_comb = res.combine(res)
    np.testing.assert_allclose(res_comb.fisher_matrix, 2.0 * f_exact, rtol=1e-10)
    np.testing.assert_allclose(res_comb.sigma_marginalized, res.sigma_marginalized / np.sqrt(2.0), rtol=1e-10)

    # Richardson extrapolation eliminates O(h^2) truncation error on cubic model
    def cubic_model(theta: np.ndarray, seed: int) -> np.ndarray:
        del seed
        return np.array([theta[0] ** 3, 2.0 * (theta[1] ** 3)], dtype=np.float64)

    fc_c = py21cmcu.FisherForecaster.from_callable(
        cubic_model, ("x", "y"), fiducial=(2.0, -1.5), step_sizes=(0.4, 0.4), data_sigma=np.ones(2), order="central"
    ).run()
    fc_r = py21cmcu.FisherForecaster.from_callable(
        cubic_model, ("x", "y"), fiducial=(2.0, -1.5), step_sizes=(0.4, 0.4), data_sigma=np.ones(2), order="richardson"
    ).run()
    exact_j = np.diag([3.0 * 4.0, 2.0 * 3.0 * 2.25])
    assert np.max(np.abs(fc_r.jacobian - exact_j)) < 1e-11
    assert np.max(np.abs(fc_c.jacobian - exact_j)) > 0.1

    out_png = tmp_path / "fisher_contours.png"
    py21cmcu.plot_fisher_contours([res, res_p], labels=["Base", "+Prior"], out_path=out_png)
    assert out_png.exists() and out_png.stat().st_size > 2000


def test_fisher_forecaster_21cm_eor_step_convergence_and_multiprobe(tmp_path: Path) -> None:
    """Verify EoR CRN Fisher forecast step convergence and large-scale 21cm x galaxy anti-correlation."""
    use_cuda = bool(py21cmcu.cuda_compiled() and py21cmcu.cuda_available())
    with py21cmcu.config.use(direc=str(tmp_path / "eor_p11")):
        inputs = py21cmcu.InputParameters(
            random_seed=1305, node_redshifts=[8.0]
        ).evolve_input_structs(
            DIM=24,
            HII_DIM=12,
            BOX_LEN=48.0,
            N_THREADS=2,
            SOURCE_MODEL="E-INTEGRAL",
            USE_TS_FLUCT=False,
            RECOMB_MODEL="none",
            USE_EXP_FILTER=False,
            USE_UPPER_STELLAR_TURNOVER=False,
        )
        forecaster = py21cmcu.FisherForecaster(
            base_inputs=inputs,
            param_names=("F_STAR10_ACG", "F_ESC10_ACG"),
            fiducial=(-1.3, -1.0),
            step_sizes=(0.04, 0.04),
            out_redshifts=(8.0,),
            instrument="SKA1-Low",
            summary="ps1d",
            ps_n_bins=6,
            model_error_frac=0.1,
            use_cuda=use_cuda,
        )
        conv = forecaster.check_step_convergence(step_scales=(0.5, 1.0))
        res = conv["results"][1]
        assert res.fisher_matrix.shape == (2, 2)
        assert np.all(np.isfinite(res.sigma_marginalized))
        assert np.all(res.sigma_marginalized > 0.0)
        assert conv["max_sigma_rel_diff"] < 0.15

        # Inside-out EoR: ionized bubbles around dense galaxy regions yield negative 21cm x galaxy r(k)
        cv = py21cmcu.run_coeval(inputs=inputs, out_redshifts=[8.0], write=False)[0]
        dens = np.asarray(cv.density, dtype=np.float64)
        prob = np.maximum(1.0 + 2.5 * dens, 0.0).ravel()
        prob /= np.sum(prob)
        rng = np.random.default_rng(1306)
        idx = rng.choice(dens.size, size=10_000, replace=True, p=prob)
        ix, iy, iz = np.unravel_index(idx, dens.shape)
        pos = (np.column_stack([ix, iy, iz]) + rng.uniform(0.0, 1.0, size=(10_000, 3))) * (
            48.0 / dens.shape[0]
        )
        cross = py21cmcu.cross_power_21cm_galaxy(
            cv.brightness_temp, pos, 48.0, n_bins=5, use_cuda=use_cuda
        )
        valid = np.isfinite(cross["r_k"])
        assert np.any(valid)
        assert float(cross["r_k"][valid][0]) < -0.1

    if use_cuda:
        py21cmcu.cleanup_cuda()


@requires_cuda_runtime
def test_phase11_cic_and_cross_power_gpu_cpu_parity() -> None:
    """Verify 86th CUDA API p21_cuda_cic_deposit_3d and cross_power_21cm_galaxy GPU vs CPU parity."""
    shape = (64, 64, 64)
    box_len = (128.0, 128.0, 160.0)
    rng = np.random.default_rng(1307)
    pos = rng.uniform(-10.0, 170.0, size=(100_000, 3)).astype(np.float32)
    weights = rng.uniform(0.5, 3.0, size=100_000).astype(np.float32)

    # Empty catalog on GPU
    g_empty = py21cmcu.cic_deposit_3d(np.zeros((0, 3)), shape, box_len, overdensity=False, use_cuda=True)
    assert g_empty.shape == shape and np.all(g_empty == 0.0)

    for w in (None, weights):
        for od in (False, True):
            cpu_g = py21cmcu.cic_deposit_3d(
                pos, shape, box_len, weights=w, overdensity=od, deconvolve=False, use_cuda=False
            )
            gpu_g = py21cmcu.cic_deposit_3d(
                pos, shape, box_len, weights=w, overdensity=od, deconvolve=False, use_cuda=True
            )
            np.testing.assert_allclose(gpu_g, cpu_g, rtol=5e-5, atol=5e-5)

    tb = rng.normal(0.0, 15.0, size=shape).astype(np.float32)
    c_cpu = py21cmcu.cross_power_21cm_galaxy(tb, pos, box_len, weights=weights, n_bins=8, use_cuda=False)
    c_gpu = py21cmcu.cross_power_21cm_galaxy(tb, pos, box_len, weights=weights, n_bins=8, use_cuda=True)
    valid = np.isfinite(c_cpu["r_k"])
    np.testing.assert_allclose(c_gpu["pk_cross"][valid], c_cpu["pk_cross"][valid], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(c_gpu["r_k"][valid], c_cpu["r_k"][valid], rtol=1e-4, atol=1e-4)
    py21cmcu.cleanup_cuda()
