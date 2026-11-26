"""Direction D (Phase 3) tests for GPU RSDs, rectilinear slicing, and 1D/2D power spectra."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import py21cmcu
import pytest
from py21cmfast.c_21cmfast import lib
from py21cmfast.rsds import apply_rsds, include_dvdr_in_tau21, rsds_shift

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


def _assert_close(gpu: np.ndarray, ref: np.ndarray, label: str, *, rtol: float = 1e-4, atol: float = 1e-4) -> None:
    assert gpu.shape == ref.shape, f"{label} shape mismatch: {gpu.shape} vs {ref.shape}"
    assert np.all(np.isfinite(gpu)), f"{label} has non-finite GPU outputs"
    assert np.all(np.isfinite(ref)), f"{label} has non-finite CPU outputs"
    np.testing.assert_allclose(gpu.astype(np.float64), ref.astype(np.float64), rtol=rtol, atol=atol, err_msg=label)


@requires_cuda_runtime
@pytest.mark.parametrize("use_ts", [False, True])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("dim_z", [24, 25])
def test_include_dvdr_in_tau21_parity(
    tmp_path: Path, use_ts: bool, periodic: bool, dim_z: int
) -> None:
    """Verify GPU include_dvdr_in_tau21 matches CPU across TS flags, periodicities, and even/odd N."""
    rng = np.random.default_rng(3101 + int(use_ts) * 10 + int(periodic) * 100 + dim_z)
    hii_dim = dim_z
    kw = {"node_redshifts": [7.0, 8.5, 10.0]} if use_ts else {}
    extra = {"Z_HEAT_MAX": 10.0} if use_ts else {}
    inputs = py21cmcu.InputParameters(random_seed=3101, **kw).evolve_input_structs(
        DIM=hii_dim,
        HII_DIM=hii_dim,
        BOX_LEN=120.0,
        USE_TS_FLUCT=use_ts,
        **extra,
    )
    shape = (hii_dim, hii_dim, dim_z)
    tb = rng.normal(loc=-20.0, scale=15.0, size=shape).astype(np.float32)
    vel = rng.normal(loc=0.0, scale=2.0e-15, size=shape).astype(np.float32)
    tau21 = np.abs(rng.normal(loc=0.02, scale=0.02, size=shape)).astype(np.float32)
    tau21[0, 0, 0] = 1e-12  # Exercise tiny-tau branch
    redshifts = 8.0 if periodic else np.linspace(7.5, 9.0, dim_z)

    with py21cmcu.use_mode("strict", use_cuda=False, direc=str(tmp_path / "cpu")):
        cpu_out = include_dvdr_in_tau21(
            brightness_temp=tb,
            los_velocity=vel,
            redshifts=redshifts,
            inputs=inputs,
            periodic=periodic,
            tau_21=tau21 if use_ts else None,
        )

    py21cmcu.cleanup_cuda()
    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "gpu")):
        gpu_out = include_dvdr_in_tau21(
            brightness_temp=tb,
            los_velocity=vel,
            redshifts=redshifts,
            inputs=inputs,
            periodic=periodic,
            tau_21=tau21 if use_ts else None,
        )

    rtol = 1e-3 if (use_ts and periodic) else 1e-4
    _assert_close(
        gpu_out,
        cpu_out,
        f"dvdr(ts={use_ts}, periodic={periodic}, nz={dim_z})",
        rtol=rtol,
    )
    if use_ts and periodic:
        rms = max(float(np.sqrt(np.mean(cpu_out.astype(np.float64) ** 2))), 1e-30)
        rel_rms = float(
            np.sqrt(np.mean((gpu_out.astype(np.float64) - cpu_out.astype(np.float64)) ** 2))
        ) / rms
        assert rel_rms < 5e-4
    py21cmcu.cleanup_cuda()


@requires_cuda_runtime
def test_include_dvdr_2d_angular_parity(tmp_path: Path) -> None:
    """Verify GPU include_dvdr_in_tau21 matches CPU for 2D angular lightcone shapes."""
    rng = np.random.default_rng(3201)
    inputs = py21cmcu.InputParameters(
        random_seed=3201, node_redshifts=[7.0, 8.5, 10.0]
    ).evolve_input_structs(
        DIM=32,
        HII_DIM=32,
        BOX_LEN=100.0,
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=10.0,
    )
    n_los, nz = 200, 48
    tb = rng.normal(loc=-15.0, scale=12.0, size=(n_los, nz)).astype(np.float32)
    vel = rng.normal(loc=0.0, scale=1.5e-15, size=(n_los, nz)).astype(np.float32)
    tau21 = np.abs(rng.normal(loc=0.05, scale=0.03, size=(n_los, nz))).astype(np.float32)
    redshifts = np.linspace(7.0, 10.0, nz)

    with py21cmcu.use_mode("strict", use_cuda=False, direc=str(tmp_path / "cpu")):
        cpu_out = include_dvdr_in_tau21(
            tb, vel, redshifts, inputs, periodic=False, tau_21=tau21
        )
    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "gpu")):
        gpu_out = include_dvdr_in_tau21(
            tb, vel, redshifts, inputs, periodic=False, tau_21=tau21
        )
    _assert_close(gpu_out, cpu_out, "dvdr_2d_angular")
    py21cmcu.cleanup_cuda()


@requires_cuda_runtime
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("n_rsd_subcells", [1, 4])
def test_rsds_shift_and_apply_rsds_parity(
    tmp_path: Path, periodic: bool, n_rsd_subcells: int
) -> None:
    """Verify GPU rsds_shift and apply_rsds match CPU & conserve total weight when periodic."""
    rng = np.random.default_rng(3301 + int(periodic) * 10 + n_rsd_subcells)
    inputs = py21cmcu.InputParameters(random_seed=3301).evolve_input_structs(
        DIM=24, HII_DIM=24, BOX_LEN=96.0
    )

    # 2D rsds_shift check
    field2d = rng.normal(loc=-10.0, scale=8.0, size=(36, 40)).astype(np.float32)
    disp2d = rng.normal(loc=0.0, scale=1.4, size=(36, 40)).astype(np.float32)
    with py21cmcu.use_mode("strict", use_cuda=False, direc=str(tmp_path / "cpu")):
        cpu_2d = rsds_shift(field2d, disp2d, n_rsd_subcells=n_rsd_subcells, periodic=periodic)
    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "gpu")):
        gpu_2d = rsds_shift(field2d, disp2d, n_rsd_subcells=n_rsd_subcells, periodic=periodic)
    _assert_close(gpu_2d, cpu_2d, f"rsds_shift_2d(periodic={periodic}, S={n_rsd_subcells})")
    if periodic:
        np.testing.assert_allclose(gpu_2d.sum(), field2d.sum(), rtol=1e-4)

    # 3D apply_rsds check
    field3d = rng.normal(loc=-20.0, scale=15.0, size=(24, 24, 30)).astype(np.float32)
    vel3d = rng.normal(loc=0.0, scale=1.5e-15, size=(24, 24, 30)).astype(np.float32)
    redshifts = 8.5 if periodic else np.linspace(7.2, 8.8, 30)
    with py21cmcu.use_mode("strict", use_cuda=False, direc=str(tmp_path / "cpu")):
        cpu_3d = apply_rsds(
            field3d, vel3d, redshifts, inputs, periodic=periodic, n_rsd_subcells=n_rsd_subcells
        )
    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "gpu")):
        gpu_3d = apply_rsds(
            field3d, vel3d, redshifts, inputs, periodic=periodic, n_rsd_subcells=n_rsd_subcells
        )
    _assert_close(gpu_3d, cpu_3d, f"apply_rsds_3d(periodic={periodic}, S={n_rsd_subcells})")
    py21cmcu.cleanup_cuda()


@requires_cuda_runtime
def test_rectilinear_lightcone_with_rsds_strict_parity(tmp_path: Path) -> None:
    """Verify full GPU RectilinearLightconer slicing + dvdr + apply_rsds matches CPU in strict mode."""
    inputs = py21cmcu.InputParameters(
        random_seed=3401, node_redshifts=[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    ).evolve_input_structs(
        DIM=48,
        HII_DIM=24,
        BOX_LEN=72.0,
        N_THREADS=2,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=11.0,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    lightconer = py21cmcu.RectilinearLightconer.between_redshifts(
        min_redshift=7.5,
        max_redshift=9.0,
        resolution=inputs.simulation_options.cell_size,
        quantities=("brightness_temp", "neutral_fraction", "z_reion"),
        cosmo=inputs.cosmo_params.cosmo,
    )

    with py21cmcu.use_mode("strict", use_cuda=False, direc=str(tmp_path / "cpu")):
        lc_cpu = py21cmcu.run_lightcone(
            lightconer=lightconer,
            inputs=inputs,
            include_dvdr_in_tau21=True,
            apply_rsds=True,
            n_rsd_subcells=4,
            regenerate=True,
            write=False,
        )

    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "gpu")):
        lc_gpu = py21cmcu.run_lightcone(
            lightconer=lightconer,
            inputs=inputs,
            include_dvdr_in_tau21=True,
            apply_rsds=True,
            n_rsd_subcells=4,
            regenerate=True,
            write=False,
        )

    for key in ("brightness_temp", "brightness_temp_with_rsds", "neutral_fraction", "z_reion_with_rsds"):
        g = np.asarray(lc_gpu.lightcones[key], dtype=np.float64)
        c = np.asarray(lc_cpu.lightcones[key], dtype=np.float64)
        rms = max(float(np.sqrt(np.mean(c**2))), 1e-30)
        rel_rms = float(np.sqrt(np.mean((g - c) ** 2))) / rms
        assert rel_rms < 5e-4, f"lightcone[{key}] rel_rms={rel_rms:.3e}"
    py21cmcu.cleanup_cuda()


@requires_cuda_runtime
def test_rectilinear_slicer_batch_parity(tmp_path: Path) -> None:
    """Verify RectilinearLightconer.make_lightcone_slices GPU batch matches CPU across mean & mean_max."""
    inputs = py21cmcu.InputParameters(
        random_seed=3402, node_redshifts=[7.0, 8.0, 9.0, 10.0]
    ).evolve_input_structs(
        DIM=48,
        HII_DIM=24,
        BOX_LEN=72.0,
        N_THREADS=2,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=10.0,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    lightconer = py21cmcu.RectilinearLightconer.between_redshifts(
        min_redshift=7.2,
        max_redshift=9.8,
        resolution=inputs.simulation_options.cell_size,
        quantities=("brightness_temp", "z_reion", "los_velocity"),
        cosmo=inputs.cosmo_params.cosmo,
    )
    with py21cmcu.use_mode("strict", use_cuda=False, direc=str(tmp_path / "coeval")):
        c2, c1 = py21cmcu.run_coeval(
            out_redshifts=[7.5, 9.0], inputs=inputs, regenerate=True, write=False
        )
        cpu_slices = list(lightconer.make_lightcone_slices(c1, c2))

    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "coeval")):
        gpu_slices = list(lightconer.make_lightcone_slices(c1, c2))

    assert len(gpu_slices) == len(cpu_slices)
    assert len(gpu_slices) > 0
    for (gq, gidx, garr), (cq, cidx, carr) in zip(gpu_slices, cpu_slices, strict=True):
        assert gq == cq
        assert int(gidx) == int(cidx)
        _assert_close(garr, carr, f"rectilinear_slice({gq}, idx={gidx})", rtol=1e-5, atol=1e-5)
    py21cmcu.cleanup_cuda()


@requires_cuda_runtime
@pytest.mark.parametrize("shape,box_len,los_axis", [
    ((32, 32, 32), 128.0, 2),
    ((32, 32, 35), (128.0, 128.0, 140.0), 0),
    ((24, 32, 48), (96.0, 128.0, 192.0), 1),
])
def test_gpu_power_spectra_1d_and_2d(
    shape: tuple[int, int, int], box_len: float | tuple[float, float, float], los_axis: int
) -> None:
    """Verify GPU 1D/2D auto & cross power spectra match NumPy (and powerbox when installed)."""
    rng = np.random.default_rng(3501 + sum(shape) + los_axis)
    a = rng.normal(loc=2.5, scale=4.0, size=shape).astype(np.float32)
    b = (0.7 * a + rng.normal(scale=2.0, size=shape)).astype(np.float32)

    ps1_gpu = py21cmcu.power_spectrum_1d(a, box_len, n_bins=12, use_cuda=True)
    ps1_cpu = py21cmcu.power_spectrum_1d(a, box_len, n_bins=12, use_cuda=False)
    np.testing.assert_array_equal(ps1_gpu["counts"], ps1_cpu["counts"])
    mask1 = ps1_cpu["counts"] > 0
    np.testing.assert_allclose(ps1_gpu["k"][mask1], ps1_cpu["k"][mask1], rtol=1e-6)
    np.testing.assert_allclose(ps1_gpu["pk"][mask1], ps1_cpu["pk"][mask1], rtol=1e-5)
    np.testing.assert_allclose(ps1_gpu["delta_sq"][mask1], ps1_cpu["delta_sq"][mask1], rtol=1e-5)

    # Cross power + remove_mean check
    ps1x_gpu = py21cmcu.power_spectrum_1d(a, box_len, field_b=b, n_bins=10, remove_mean=True, use_cuda=True)
    ps1x_cpu = py21cmcu.power_spectrum_1d(a, box_len, field_b=b, n_bins=10, remove_mean=True, use_cuda=False)
    mask1x = ps1x_cpu["counts"] > 0
    np.testing.assert_allclose(ps1x_gpu["pk"][mask1x], ps1x_cpu["pk"][mask1x], rtol=1e-5)

    # 2D cylindrical power spectrum check
    ps2_gpu = py21cmcu.power_spectrum_2d(
        a, box_len, field_b=b, los_axis=los_axis, n_kperp=6, n_kpara=6, use_cuda=True
    )
    ps2_cpu = py21cmcu.power_spectrum_2d(
        a, box_len, field_b=b, los_axis=los_axis, n_kperp=6, n_kpara=6, use_cuda=False
    )
    np.testing.assert_array_equal(ps2_gpu["counts"], ps2_cpu["counts"])
    mask2 = ps2_cpu["counts"] > 0
    np.testing.assert_allclose(ps2_gpu["k_perp"][mask2], ps2_cpu["k_perp"][mask2], rtol=1e-6)
    np.testing.assert_allclose(ps2_gpu["k_para"][mask2], ps2_cpu["k_para"][mask2], rtol=1e-6)
    np.testing.assert_allclose(ps2_gpu["pk"][mask2], ps2_cpu["pk"][mask2], rtol=1e-5)

    # Compare 1D spherical P(k) against powerbox on cubic array if available
    if shape[0] == shape[1] == shape[2] and isinstance(box_len, float):
        powerbox = pytest.importorskip("powerbox")
        k_edges = ps1_gpu["k_edges"]
        pb_res = powerbox.get_power(
            a, box_len, bins=k_edges, res_ndim=3, bins_upto_boxlen=False
        )
        pk_pb = np.asarray(pb_res.power)
        k_pb = np.asarray(pb_res.bin_avg)
        valid = np.isfinite(pk_pb) & (ps1_gpu["counts"] > 0)
        np.testing.assert_allclose(ps1_gpu["pk"][valid], pk_pb[valid], rtol=1e-4)
        np.testing.assert_allclose(ps1_gpu["k"][valid], k_pb[valid], rtol=1e-4)

    py21cmcu.cleanup_cuda()


@requires_cuda_runtime
def test_gpu_power_spectrum_residency_zero_h2d(tmp_path: Path) -> None:
    """Verify power_spectrum_1d reuses GPU-resident coeval fields with zero extra H2D copy."""
    inputs = py21cmcu.InputParameters(random_seed=3601).evolve_input_structs(
        DIM=48, HII_DIM=32, BOX_LEN=120.0, N_THREADS=2, USE_TS_FLUCT=False
    )
    py21cmcu.cleanup_cuda()
    with py21cmcu.use_mode("fast", use_cuda=True, direc=str(tmp_path / "gpu")):
        cv = py21cmcu.run_coeval(out_redshifts=8.0, inputs=inputs, regenerate=True, write=False)[0]
        tb = cv.brightness_temp
        py21cmcu.reset_cuda_stats()
        ps1 = py21cmcu.power_spectrum_1d(tb, inputs.simulation_options.BOX_LEN, n_bins=12, use_cuda=True)
        stats = py21cmcu.cuda_stats()
        assert np.any(ps1["counts"] > 0)
        assert stats["registry_hits"] >= 1
        assert stats["registry_h2d_bytes"] == 0
    py21cmcu.cleanup_cuda()
