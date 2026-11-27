"""Direction G (Phase 5) tests for multi-GPU slab domain decomposition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import py21cmcu
import pytest
from py21cmfast._cfg import ConfigurationError, config
from py21cmfast.c_21cmfast import ffi, lib

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)

requires_multi_gpu = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0
    or lib.p21_cuda_is_available() == 0
    or lib.p21_cuda_device_count() < 2,
    reason="Requires at least 2 CUDA devices.",
)

requires_three_plus_gpus = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0
    or lib.p21_cuda_is_available() == 0
    or lib.p21_cuda_device_count() < 3,
    reason="Requires at least 3 CUDA devices.",
)


def _gpu_counts_to_test() -> list[int]:
    n_avail = int(py21cmcu.cuda_device_count())
    counts = []
    if n_avail >= 2:
        counts.append(2)
    if n_avail >= 4:
        counts.append(4)
    elif n_avail == 3:
        counts.append(3)
    return counts


def test_mgpu_device_config_and_api() -> None:
    """Verify CUDA_DEVICES config, validation, get/set helpers, stats keys, and use_mode(devices=...)."""
    orig_cuda = config["USE_CUDA"]
    orig_mode = config["CUDA_MODE"]
    orig_devices = config["CUDA_DEVICES"]
    try:
        n_total = py21cmcu.cuda_device_count()
        assert isinstance(n_total, int)
        assert n_total >= 0

        stats = py21cmcu.cuda_stats()
        assert "mgpu_fft_runs" in stats
        assert "mgpu_ic_runs" in stats

        # String parsing ("all", comma-separated, list/tuple, empty -> None)
        config["CUDA_DEVICES"] = "0, 1, 0"
        assert config["CUDA_DEVICES"] == (0, 1)
        if py21cmcu.cuda_compiled():
            assert py21cmcu.get_cuda_devices() == (0, 1)
        else:
            assert py21cmcu.get_cuda_devices() == ()

        config["CUDA_DEVICES"] = (2, 0)
        if py21cmcu.cuda_compiled():
            assert py21cmcu.get_cuda_devices() == (2, 0)
        else:
            assert py21cmcu.get_cuda_devices() == ()

        py21cmcu.set_cuda_devices("all")
        expected_all = tuple(range(n_total)) if n_total > 0 else (0,)
        assert config["CUDA_DEVICES"] == expected_all

        config["CUDA_DEVICES"] = " "
        assert config["CUDA_DEVICES"] is None

        with pytest.raises(ConfigurationError):
            config["CUDA_DEVICES"] = [-1]
        with pytest.raises(ConfigurationError):
            config["CUDA_DEVICES"] = "abc"

        with py21cmcu.use_mode("fast", use_cuda=True, devices=[0, 1]):
            assert config["CUDA_DEVICES"] == (0, 1)
            if py21cmcu.cuda_compiled():
                assert lib.p21_cuda_get_num_devices() >= 1
            else:
                assert lib.p21_cuda_get_num_devices() == 0

        # use_mode restores prior device selection cleanly
        assert config["CUDA_DEVICES"] is None
    finally:
        config["CUDA_DEVICES"] = orig_devices
        config["CUDA_MODE"] = orig_mode
        config["USE_CUDA"] = orig_cuda


@requires_multi_gpu
@pytest.mark.parametrize("shape", [(32, 32, 32), (24, 30, 35), (48, 48, 64)])
def test_mgpu_fft_r2c_c2r_parity(shape: tuple[int, int, int]) -> None:
    """Verify multi-GPU 3D R2C/C2R slab FFT matches NumPy on cubic and non-cubic grids."""
    nx, ny, nz = shape
    hz = nz // 2 + 1
    rng = np.random.default_rng(5101 + nx * 13 + ny * 7 + nz)
    real_in = rng.normal(loc=0.5, scale=2.0, size=shape).astype(np.float32)
    ref_cplx = np.fft.rfftn(real_in.astype(np.float64)).astype(np.complex64)

    for n_gpus in _gpu_counts_to_test():
        devs = list(range(n_gpus))
        py21cmcu.reset_cuda_stats()
        with py21cmcu.use_mode("fast", use_cuda=True, devices=devs):
            assert list(py21cmcu.get_cuda_devices()[:n_gpus]) == devs
            assert int(lib.p21_cuda_get_num_devices()) == n_gpus

            cplx_out = np.empty((nx, ny, hz), dtype=np.complex64)
            rc = lib.p21_cuda_mgpu_fft_r2c_host(
                ffi.cast("const float *", real_in.ctypes.data),
                ffi.cast("float *", cplx_out.ctypes.data),
                nx,
                ny,
                nz,
            )
            assert rc == 0
            np.testing.assert_allclose(cplx_out, ref_cplx, rtol=1e-4, atol=1e-3)

            real_rt = np.empty(shape, dtype=np.float32)
            rc2 = lib.p21_cuda_mgpu_fft_c2r_host(
                ffi.cast("const float *", cplx_out.ctypes.data),
                ffi.cast("float *", real_rt.ctypes.data),
                nx,
                ny,
                nz,
            )
            assert rc2 == 0
            real_rt /= float(nx * ny * nz)
            np.testing.assert_allclose(real_rt, real_in, rtol=1e-4, atol=1e-4)

            st = py21cmcu.cuda_stats()
            assert st["mgpu_fft_runs"] >= 2

        py21cmcu.cleanup_cuda()


@requires_multi_gpu
@pytest.mark.parametrize(
    "perturb_algo,perturb_hires,ps_model",
    [
        ("ZELDOVICH", False, "EH"),
        ("2LPT", False, "EH"),
        ("2LPT", True, "CLASS"),
    ],
)
def test_mgpu_ic_fast_mode_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    perturb_algo: str,
    perturb_hires: bool,
    ps_model: str,
) -> None:
    """Verify multi-GPU compute_initial_conditions (fast Philox) matches 1-GPU parity."""
    v_cb_model = "FLUCTS" if ps_model == "CLASS" else "NONE"
    extra = (
        {
            "K_MAX_FOR_CLASS": 1.0,
            "Z_HEAT_MAX": 15.0,
            "ZPRIME_STEP_FACTOR": 1.2,
            "USE_MCGS": True,
            "USE_TS_FLUCT": True,
            "RECOMB_MODEL": "inhomogeneous",
            "R_BUBBLE_MAX": 50.0,
            "M_TURN_STELLAR_FEEDBACK": 6.0,
        }
        if ps_model == "CLASS"
        else {}
    )
    seed_inputs = (
        py21cmcu.InputParameters(random_seed=7701).with_logspaced_redshifts(
            zmin=10.0, zmax=15.0, step=1.2
        )
        if ps_model == "CLASS"
        else py21cmcu.InputParameters(random_seed=7701)
    )
    inputs = seed_inputs.evolve_input_structs(
        DIM=48,
        HII_DIM=24,
        BOX_LEN=96.0,
        NON_CUBIC_FACTOR=1.25,
        PERTURB_ALGORITHM=perturb_algo,
        PERTURB_ON_HIGH_RES=perturb_hires,
        POWER_SPECTRUM=ps_model,
        V_CB_MODEL=v_cb_model,
        **extra,
    )

    monkeypatch.setenv("P21_CUDA_FORCE_MGPU", "1")
    with py21cmcu.use_mode("fast", use_cuda=True, devices=[0], direc=str(tmp_path / "gpu1")):
        ic_ref = py21cmcu.compute_initial_conditions(inputs=inputs, regenerate=True, write=False)

    fields = [
        "lowres_density",
        "hires_density",
    ]
    if perturb_hires:
        fields.extend(["hires_vx", "hires_vy", "hires_vz"])
        if perturb_algo == "2LPT":
            fields.extend(["hires_vx_2LPT", "hires_vy_2LPT", "hires_vz_2LPT"])
    else:
        fields.extend(["lowres_vx", "lowres_vy", "lowres_vz"])
        if perturb_algo == "2LPT":
            fields.extend(["lowres_vx_2LPT", "lowres_vy_2LPT", "lowres_vz_2LPT"])
    if v_cb_model == "FLUCTS":
        fields.append("lowres_vcb")

    for n_gpus in _gpu_counts_to_test():
        py21cmcu.reset_cuda_stats()
        with py21cmcu.use_mode(
            "fast",
            use_cuda=True,
            devices=list(range(n_gpus)),
            direc=str(tmp_path / f"gpu{n_gpus}"),
        ):
            ic_mgpu = py21cmcu.compute_initial_conditions(
                inputs=inputs, regenerate=True, write=False
            )
            st = py21cmcu.cuda_stats()
            assert st["mgpu_ic_runs"] >= 1
            for name in fields:
                arr_m = np.asarray(getattr(ic_mgpu, name), dtype=np.float64)
                arr_r = np.asarray(getattr(ic_ref, name), dtype=np.float64)
                np.testing.assert_allclose(arr_m, arr_r, rtol=1e-4, atol=1e-4, err_msg=name)
        py21cmcu.cleanup_cuda()


@requires_multi_gpu
def test_mgpu_ic_initial_density_parity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify multi-GPU compute_initial_conditions handles user-provided initial_density."""
    inputs = py21cmcu.InputParameters(random_seed=7801).evolve_input_structs(
        DIM=48,
        HII_DIM=24,
        BOX_LEN=96.0,
        PERTURB_ALGORITHM="2LPT",
        PERTURB_ON_HIGH_RES=False,
    )
    rng = np.random.default_rng(7802)
    init_dens = rng.normal(loc=0.0, scale=1e-3, size=(48, 48, 48)).astype(np.float32)

    monkeypatch.setenv("P21_CUDA_FORCE_MGPU", "1")
    with py21cmcu.use_mode("strict", use_cuda=True, devices=[0], direc=str(tmp_path / "g1")):
        ic_ref = py21cmcu.compute_initial_conditions(
            inputs=inputs, initial_density=init_dens.copy(), regenerate=True, write=False
        )

    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("strict", use_cuda=True, devices=[0, 1], direc=str(tmp_path / "g2")):
        ic_mgpu = py21cmcu.compute_initial_conditions(
            inputs=inputs, initial_density=init_dens.copy(), regenerate=True, write=False
        )
        assert py21cmcu.cuda_stats()["mgpu_ic_runs"] >= 1
        for name in (
            "lowres_density",
            "lowres_vx",
            "lowres_vy",
            "lowres_vz",
            "lowres_vx_2LPT",
            "lowres_vy_2LPT",
            "lowres_vz_2LPT",
        ):
            np.testing.assert_allclose(
                np.asarray(getattr(ic_mgpu, name)),
                np.asarray(getattr(ic_ref, name)),
                rtol=1e-4,
                atol=1e-4,
                err_msg=name,
            )
    py21cmcu.cleanup_cuda()


@requires_multi_gpu
@pytest.mark.parametrize("los_axis", [0, 1, 2])
def test_mgpu_power_spectra_parity(monkeypatch: pytest.MonkeyPatch, los_axis: int) -> None:
    """Verify multi-GPU 1D and 2D auto/cross power spectra match 1-GPU across all LOS axes."""
    shape = (36, 40, 48)
    box_len = (72.0, 80.0, 96.0)
    rng = np.random.default_rng(7901 + los_axis)
    a = rng.normal(loc=1.5, scale=3.0, size=shape).astype(np.float32)
    b = (0.65 * a + rng.normal(scale=1.5, size=shape)).astype(np.float32)

    with py21cmcu.use_mode("fast", use_cuda=True, devices=[0]):
        ps1_ref = py21cmcu.power_spectrum_1d(a, box_len, field_b=b, n_bins=12, remove_mean=True, use_cuda=True)
        ps2_ref = py21cmcu.power_spectrum_2d(
            a, box_len, field_b=b, los_axis=los_axis, n_kperp=6, n_kpara=6, remove_mean=True, use_cuda=True
        )

    monkeypatch.setenv("P21_CUDA_FORCE_MGPU", "1")
    for n_gpus in _gpu_counts_to_test():
        with py21cmcu.use_mode("fast", use_cuda=True, devices=list(range(n_gpus))):
            ps1_m = py21cmcu.power_spectrum_1d(
                a, box_len, field_b=b, n_bins=12, remove_mean=True, use_cuda=True
            )
            ps2_m = py21cmcu.power_spectrum_2d(
                a,
                box_len,
                field_b=b,
                los_axis=los_axis,
                n_kperp=6,
                n_kpara=6,
                remove_mean=True,
                use_cuda=True,
            )
            np.testing.assert_array_equal(ps1_m["counts"], ps1_ref["counts"])
            m1 = ps1_ref["counts"] > 0
            np.testing.assert_allclose(ps1_m["k"][m1], ps1_ref["k"][m1], rtol=1e-6)
            np.testing.assert_allclose(ps1_m["pk"][m1], ps1_ref["pk"][m1], rtol=1e-5)
            np.testing.assert_allclose(ps1_m["delta_sq"][m1], ps1_ref["delta_sq"][m1], rtol=1e-5)

            np.testing.assert_array_equal(ps2_m["counts"], ps2_ref["counts"])
            m2 = ps2_ref["counts"] > 0
            np.testing.assert_allclose(ps2_m["k_perp"][m2], ps2_ref["k_perp"][m2], rtol=1e-6)
            np.testing.assert_allclose(ps2_m["k_para"][m2], ps2_ref["k_para"][m2], rtol=1e-6)
            np.testing.assert_allclose(ps2_m["pk"][m2], ps2_ref["pk"][m2], rtol=1e-5)
        py21cmcu.cleanup_cuda()


@requires_multi_gpu
@pytest.mark.parametrize("use_init_dens", [False, True])
def test_mgpu_ic_single_slab_mode_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_init_dens: bool
) -> None:
    """Verify Direction K single-work-slab multi-GPU mode matches 1-GPU across 2LPT + CLASS."""
    inputs = py21cmcu.InputParameters(random_seed=7951).with_logspaced_redshifts(
        zmin=10.0, zmax=15.0, step=1.2
    ).evolve_input_structs(
        DIM=48,
        HII_DIM=24,
        BOX_LEN=96.0,
        PERTURB_ALGORITHM="2LPT",
        PERTURB_ON_HIGH_RES=False,
        POWER_SPECTRUM="CLASS",
        V_CB_MODEL="FLUCTS",
        K_MAX_FOR_CLASS=1.0,
        Z_HEAT_MAX=15.0,
        ZPRIME_STEP_FACTOR=1.2,
        USE_MCGS=True,
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
        R_BUBBLE_MAX=50.0,
        M_TURN_STELLAR_FEEDBACK=6.0,
    )
    init_kw = {}
    if use_init_dens:
        rng = np.random.default_rng(7952)
        init_kw["initial_density"] = rng.normal(loc=0.0, scale=1e-3, size=(48, 48, 48)).astype(
            np.float32
        )

    mode = "strict" if use_init_dens else "fast"
    with py21cmcu.use_mode(mode, use_cuda=True, devices=[0], direc=str(tmp_path / "g1")):
        ic_ref = py21cmcu.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False, **init_kw
        )

    monkeypatch.setenv("P21_CUDA_FORCE_MGPU", "1")
    monkeypatch.setenv("P21_CUDA_MGPU_SINGLE_SLAB", "1")
    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode(mode, use_cuda=True, devices=[0, 1], direc=str(tmp_path / "g2")):
        ic_ss = py21cmcu.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False, **init_kw
        )
        assert py21cmcu.cuda_stats()["mgpu_ic_runs"] >= 1
        for name in (
            "lowres_density",
            "hires_density",
            "lowres_vx",
            "lowres_vy",
            "lowres_vz",
            "lowres_vx_2LPT",
            "lowres_vy_2LPT",
            "lowres_vz_2LPT",
            "lowres_vcb",
        ):
            np.testing.assert_allclose(
                np.asarray(getattr(ic_ss, name), dtype=np.float64),
                np.asarray(getattr(ic_ref, name), dtype=np.float64),
                rtol=1e-4,
                atol=1e-4,
                err_msg=name,
            )
    py21cmcu.cleanup_cuda()


@requires_three_plus_gpus
def test_mgpu_large_1536_e2e(tmp_path: Path) -> None:
    """Smoke-test real 1536^3 IC + perturb_field + 1D P(k) across 3+ GPUs without forced env."""
    inputs = py21cmcu.InputParameters(random_seed=8001).evolve_input_structs(
        DIM=1536,
        HII_DIM=256,
        BOX_LEN=512.0,
        PERTURB_ALGORITHM="ZELDOVICH",
        PERTURB_ON_HIGH_RES=False,
    )
    devs = list(range(min(int(py21cmcu.cuda_device_count()), 4)))
    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("fast", use_cuda=True, devices=devs, direc=str(tmp_path)):
        ic = py21cmcu.compute_initial_conditions(inputs=inputs, regenerate=True, write=False)
        st = py21cmcu.cuda_stats()
        assert st["mgpu_ic_runs"] >= 1
        pf = py21cmcu.perturb_field(redshift=8.0, initial_conditions=ic, regenerate=True, write=False)
        dens = np.asarray(pf.density, dtype=np.float32)
        assert dens.shape == (256, 256, 256)
        assert np.all(np.isfinite(dens))
        ps1 = py21cmcu.power_spectrum_1d(dens, 512.0, n_bins=16, use_cuda=True)
        valid = ps1["counts"] > 0
        assert np.any(valid)
        assert np.all(np.isfinite(ps1["pk"][valid]))
    py21cmcu.cleanup_cuda()
