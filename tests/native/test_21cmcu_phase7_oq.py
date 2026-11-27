"""Directions O & Q tests: GPU observables, moments/PDF, 2D chunk PS, DLPack, SBI & 21CMMC."""

from __future__ import annotations

import ctypes
import gc
from pathlib import Path

import numpy as np
import pytest

import py21cmcu
from py21cmfast.c_21cmfast import lib

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


@requires_cuda_runtime
@pytest.mark.parametrize("los_axis", [0, 1, 2])
def test_gpu_wedge_filter_parity_all_axes(los_axis: int) -> None:
    """Verify GPU cuFFT horizon wedge filter matches CPU float64 NumPy across all LOS axes."""
    rng = np.random.default_rng(7100 + los_axis)
    field = rng.normal(loc=-12.0, scale=15.0, size=(24, 28, 32)).astype(np.float32)
    box_len = (96.0, 112.0, 128.0)

    cpu_out = py21cmcu.apply_wedge_filter_3d(
        field,
        box_len,
        redshift=8.0,
        los_axis=los_axis,
        buffer_h_mpc=0.05,
        use_cuda=False,
    )
    gpu_out = py21cmcu.apply_wedge_filter_3d(
        field,
        box_len,
        redshift=8.0,
        los_axis=los_axis,
        buffer_h_mpc=0.05,
        use_cuda=True,
    )
    assert gpu_out.dtype == np.float32
    assert gpu_out.shape == field.shape
    np.testing.assert_allclose(gpu_out, cpu_out, rtol=1e-3, atol=2e-3)
    py21cmcu.cleanup_cuda()


@requires_cuda_runtime
def test_gpu_thermal_noise_determinism_and_statistics() -> None:
    """Verify GPU Philox thermal noise is deterministic, zero-mean Gaussian, and seed-independent."""
    base = np.zeros((32, 32, 32), dtype=np.float32)
    n1 = py21cmcu.add_thermal_noise_3d(base, 128.0, sigma_mk=3.2, seed=42, use_cuda=True)
    n2 = py21cmcu.add_thermal_noise_3d(base, 128.0, sigma_mk=3.2, seed=42, use_cuda=True)
    n3 = py21cmcu.add_thermal_noise_3d(base, 128.0, sigma_mk=3.2, seed=43, use_cuda=True)

    np.testing.assert_array_equal(n1, n2)
    assert abs(float(np.mean(n1))) < 0.12
    assert abs(float(np.std(n1)) - 3.2) < 0.12
    corr = float(np.corrcoef(n1.ravel(), n3.ravel())[0, 1])
    assert abs(corr) < 0.04

    n_zero = py21cmcu.add_thermal_noise_3d(n1, 128.0, sigma_mk=0.0, seed=1, use_cuda=True)
    np.testing.assert_array_equal(n_zero, n1)
    py21cmcu.cleanup_cuda()


def test_field_moments_and_pdf_cpu_and_gpu_parity() -> None:
    """Verify CPU and (if available) GPU 1-point moments and histogram/PDF match."""
    rng = np.random.default_rng(7201)
    # Positively skewed synthetic EoR field
    x = rng.normal(loc=0.0, scale=1.0, size=(24, 24, 24))
    arr = (np.exp(0.45 * x) - 1.2).astype(np.float32)

    cpu_res = py21cmcu.field_moments_and_pdf(arr, n_bins=20, use_cuda=False)
    assert cpu_res["hist"].sum() == arr.size
    bin_w = float(cpu_res["bin_edges"][1] - cpu_res["bin_edges"][0])
    np.testing.assert_allclose(float(np.sum(cpu_res["pdf"]) * bin_w), 1.0, rtol=1e-6)
    assert cpu_res["skewness"] > 0.5

    if py21cmcu.cuda_compiled() and py21cmcu.cuda_available():
        gpu_res = py21cmcu.field_moments_and_pdf(arr, n_bins=20, use_cuda=True)
        for key in ("mean", "std", "variance", "skewness", "kurtosis"):
            np.testing.assert_allclose(gpu_res[key], cpu_res[key], rtol=1e-5, atol=1e-5, err_msg=key)
        np.testing.assert_array_equal(gpu_res["hist"], cpu_res["hist"])
        np.testing.assert_allclose(gpu_res["pdf"], cpu_res["pdf"], rtol=1e-6)
        py21cmcu.cleanup_cuda()


@requires_cuda_runtime
def test_lightcone_chunk_power_spectra_2d_and_cross(tmp_path: Path) -> None:
    """Verify 2D cylindrical chunk PS and cross-PS match direct power_spectrum_1d/2d."""
    inputs = py21cmcu.InputParameters(
        random_seed=7301, node_redshifts=[7.0, 8.0, 9.0, 10.0]
    ).evolve_input_structs(
        DIM=32,
        HII_DIM=16,
        BOX_LEN=48.0,
        N_THREADS=2,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=9.5,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    lc_def = py21cmcu.RectilinearLightconer.between_redshifts(
        min_redshift=7.2,
        max_redshift=9.0,
        resolution=inputs.simulation_options.cell_size,
        quantities=("brightness_temp", "neutral_fraction"),
        cosmo=inputs.cosmo_params.cosmo,
    )
    with py21cmcu.use_mode("fast", use_cuda=True, lazy_d2h=False, direc=str(tmp_path / "lc")):
        lc = py21cmcu.run_lightcone(
            lightconer=lc_def,
            inputs=inputs,
            include_dvdr_in_tau21=True,
            apply_rsds=False,
            regenerate=True,
            write=False,
        )
        cross_1d = py21cmcu.lightcone_chunk_power_spectra(
            lc,
            quantity="brightness_temp",
            cross_quantity="neutral_fraction",
            n_chunks=2,
            n_bins=8,
            use_cuda=True,
        )
        chunk_2d = py21cmcu.lightcone_chunk_power_spectra_2d(
            lc,
            quantity="brightness_temp",
            n_chunks=2,
            n_kperp=5,
            n_kpara=5,
            use_cuda=True,
        )
        assert len(cross_1d["chunks"]) == 2
        assert len(chunk_2d["chunks"]) == 2
        assert np.any(np.isfinite(cross_1d["chunks"][0]["pk"]))
        assert chunk_2d["chunks"][0]["pk"].shape == (5, 5)
        py21cmcu.cleanup_cuda()


@requires_cuda_runtime
def test_cuda_field_view_dlpack_and_zero_pcie_export(tmp_path: Path) -> None:
    """Verify CudaFieldView zero-D2H export from lazy residency, DLPack capsule, and PyTorch."""
    inputs = py21cmcu.InputParameters(
        random_seed=7401, node_redshifts=[8.0]
    ).evolve_input_structs(
        DIM=32,
        HII_DIM=16,
        BOX_LEN=48.0,
        N_THREADS=2,
        USE_TS_FLUCT=False,
        RECOMB_MODEL="none",
    )
    with py21cmcu.use_mode("fast", use_cuda=True, lazy_d2h=True, direc=str(tmp_path / "lazy")):
        cv = py21cmcu.run_coeval(inputs=inputs, out_redshifts=[8.0], write=False)[0]
        tb = cv.brightness_temp

        py21cmcu.reset_cuda_stats()
        with py21cmcu.to_cuda_view(tb) as view:
            st = py21cmcu.cuda_stats()
            assert st["registry_h2d_bytes"] == 0
            assert view.shape == tb.shape
            cai = view.__cuda_array_interface__
            assert cai["typestr"] == "<f4" and cai["data"][0] == view.ptr
            assert view.__dlpack_device__() == (2, view.device_id)
            roundtrip = view.to_numpy()
            np.testing.assert_allclose(roundtrip, np.asarray(tb), rtol=1e-6, atol=1e-6)

        # Unconsumed DLPack capsule safely releases on GC
        view2 = py21cmcu.to_cuda_view(tb)
        cap = view2.__dlpack__()
        assert ctypes.pythonapi.PyCapsule_IsValid(ctypes.py_object(cap), b"dltensor") == 1
        del cap
        gc.collect()

        # Optional PyTorch zero-copy DLPack check
        try:
            import torch

            tensor = py21cmcu.to_torch(tb)
            torch.cuda.synchronize()
            assert tensor.is_cuda and tuple(tensor.shape) == tb.shape
            np.testing.assert_allclose(tensor.detach().cpu().numpy(), roundtrip, rtol=1e-6)
        except ImportError:
            pass
        py21cmcu.cleanup_cuda()


def test_likelihood_21cm_and_sbi_simulator_cpu_and_gpu(tmp_path: Path) -> None:
    """Verify Likelihood21cm and SBISimulator run cleanly on CPU and GPU."""
    use_gpu = bool(py21cmcu.cuda_compiled() and py21cmcu.cuda_available())
    inputs = py21cmcu.InputParameters(
        random_seed=7501, node_redshifts=[8.0]
    ).evolve_input_structs(
        DIM=24,
        HII_DIM=12,
        BOX_LEN=36.0,
        N_THREADS=2,
        USE_TS_FLUCT=False,
        RECOMB_MODEL="none",
    )
    with py21cmcu.config.use(direc=str(tmp_path / "sim")):
        sim_sbi = py21cmcu.SBISimulator(
            base_inputs=inputs,
            param_names=("F_STAR10_ACG", "F_ESC10_ACG"),
            mode="coeval",
            out_redshift=8.0,
            summary="stats",
            ps_n_bins=6,
            wedge_filter=True,
            noise_sigma_mk=1.0,
            use_cuda=use_gpu,
        )
        out_vec = sim_sbi(np.array([-1.3, -1.0]))
        assert out_vec.shape == (11,)
        assert np.all(np.isfinite(out_vec))

        dummy_like = py21cmcu.Likelihood21cm(
            base_inputs=inputs,
            param_names=("F_STAR10_ACG",),
            data_pk=np.ones(6),
            sigma_pk=np.ones(6),
            mode="coeval",
            out_redshift=8.0,
            use_cuda=use_gpu,
        )
        truth = dummy_like.simulate([-1.3])
        like = py21cmcu.Likelihood21cm(
            base_inputs=inputs,
            param_names=("F_STAR10_ACG",),
            data_pk=truth["delta_sq"],
            sigma_pk=np.maximum(np.abs(truth["delta_sq"]) * 0.2, 1.0),
            mode="coeval",
            out_redshift=8.0,
            use_cuda=use_gpu,
        )
        ll_true = like([-1.3])
        ll_off = like([-2.0])
        assert abs(ll_true) < 1e-6
        assert ll_off <= ll_true
    if use_gpu:
        py21cmcu.cleanup_cuda()
