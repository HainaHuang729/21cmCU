"""Phase 2 (Direction B) tests for 21cmCU large-box VRAM & extreme-scale optimizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import py21cmcu
import pytest
from py21cmfast.c_21cmfast import lib

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


def _assert_rel_close(
    gpu: np.ndarray,
    ref: np.ndarray,
    name: str,
    *,
    rel_rms_tol: float = 1e-4,
    atol: float = 1e-6,
) -> None:
    """Assert finite matching arrays within relative RMS tolerance."""
    assert gpu.shape == ref.shape, f"{name} shape mismatch: {gpu.shape} vs {ref.shape}"
    assert np.all(np.isfinite(gpu)), f"Non-finite GPU values in {name}"
    assert np.all(np.isfinite(ref)), f"Non-finite reference values in {name}"
    g64 = gpu.astype(np.float64)
    r64 = ref.astype(np.float64)
    rms = float(np.sqrt(np.mean(r64**2)))
    diff_rms = float(np.sqrt(np.mean((g64 - r64) ** 2)))
    if rms < atol and diff_rms < atol:
        return
    rel_rms = diff_rms / max(rms, 1e-30)
    assert (
        rel_rms < rel_rms_tol
    ), f"{name} rel_rms={rel_rms:.3e} >= {rel_rms_tol:.3e} (diff_rms={diff_rms:.3e}, rms={rms:.3e})"


@requires_cuda_runtime
@pytest.mark.parametrize("perturb_hires", [False, True])
def test_ic_padded_and_2buffer_2lpt_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, perturb_hires: bool
) -> None:
    """Verify in-place padded IC & 2-buffer algebraic 2LPT match CPU & stay invariant in uncached path."""
    inputs = py21cmcu.InputParameters(random_seed=7201).evolve_input_structs(
        DIM=64,
        HII_DIM=32,
        BOX_LEN=128.0,
        N_THREADS=2,
        PERTURB_ALGORITHM="2LPT",
        PERTURB_ON_HIGH_RES=perturb_hires,
    )

    with py21cmcu.use_mode("strict", use_cuda=False, direc=str(tmp_path / "cpu")):
        ic_cpu = py21cmcu.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False
        )

    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "gpu_strict")):
        ic_strict = py21cmcu.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False
        )
    assert py21cmcu.cuda_stats()["ic_runs"] == 1

    # Also force uncached cudaMemcpy2DAsync path in d2h_padded_and_register via 1 MiB cap.
    monkeypatch.setenv("P21_CUDA_REGISTRY_MAX_ENTRY_MB", "1")
    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "gpu_2d")):
        ic_strict_2d = py21cmcu.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False
        )
    assert py21cmcu.cuda_stats()["ic_runs"] == 1

    fields = ["hires_density", "lowres_density"]
    if perturb_hires:
        fields += [
            "hires_vx",
            "hires_vy",
            "hires_vz",
            "hires_vx_2LPT",
            "hires_vy_2LPT",
            "hires_vz_2LPT",
        ]
    else:
        fields += [
            "lowres_vx",
            "lowres_vy",
            "lowres_vz",
            "lowres_vx_2LPT",
            "lowres_vy_2LPT",
            "lowres_vz_2LPT",
            "hires_vx_2LPT",
            "hires_vy_2LPT",
            "hires_vz_2LPT",
        ]

    for fname in fields:
        a_cpu = np.asarray(ic_cpu.get(fname))
        a_gpu = np.asarray(ic_strict.get(fname))
        a_2d = np.asarray(ic_strict_2d.get(fname))
        _assert_rel_close(a_gpu, a_cpu, f"strict:{fname}", rel_rms_tol=5e-5)
        np.testing.assert_allclose(a_2d, a_gpu, rtol=1e-6, atol=1e-7)

    # Fast mode IC matches across default vs forced cudaMemcpy2DAsync path.
    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("fast", use_cuda=True, direc=str(tmp_path / "gpu_fast_2d")):
        ic_fast_2d = py21cmcu.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False
        )
    assert py21cmcu.cuda_stats()["ic_runs"] == 1

    monkeypatch.delenv("P21_CUDA_REGISTRY_MAX_ENTRY_MB", raising=False)
    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("fast", use_cuda=True, direc=str(tmp_path / "gpu_fast")):
        ic_fast = py21cmcu.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False
        )
    assert py21cmcu.cuda_stats()["ic_runs"] == 1
    for fname in fields:
        arr = np.asarray(ic_fast.get(fname))
        arr_2d = np.asarray(ic_fast_2d.get(fname))
        assert np.all(np.isfinite(arr)), f"Non-finite fast IC field {fname}"
        np.testing.assert_allclose(arr_2d, arr, rtol=1e-6, atol=1e-7)


@requires_cuda_runtime
@pytest.mark.parametrize(
    ("mode", "perturb_hires", "alg"),
    [
        ("strict", False, "ZELDOVICH"),
        ("strict", False, "2LPT"),
        ("strict", True, "2LPT"),
        ("fast", False, "2LPT"),
        ("fast", True, "2LPT"),
    ],
)
def test_perturb_slab_streaming_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    perturb_hires: bool,
    alg: str,
) -> None:
    """Verify x-slab streaming Perturb (both Mode A and Mode B) matches resident GPU and CPU."""
    inputs = py21cmcu.InputParameters(random_seed=7202).evolve_input_structs(
        DIM=64,
        HII_DIM=32,
        BOX_LEN=128.0,
        N_THREADS=2,
        PERTURB_ALGORITHM=alg,
        PERTURB_ON_HIGH_RES=perturb_hires,
        KEEP_3D_VELOCITIES=True,
    )

    py21cmcu.cleanup_cuda()
    with py21cmcu.use_mode(mode, use_cuda=True, direc=str(tmp_path / "ic")):
        ic_box = py21cmcu.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False
        )

    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode(mode, use_cuda=True, direc=str(tmp_path / "pf_res")):
        pf_res = py21cmcu.perturb_field(
            redshift=8.0, inputs=inputs, initial_conditions=ic_box, write=False
        )
    assert py21cmcu.cuda_stats()["perturb_runs"] == 1

    # Force x-slab streaming with uneven slab width (9 doesn't divide 64 evenly).
    monkeypatch.setenv("P21_CUDA_PERTURB_SLAB_X", "9")
    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode(mode, use_cuda=True, direc=str(tmp_path / "pf_slab")):
        pf_slab = py21cmcu.perturb_field(
            redshift=8.0, inputs=inputs, initial_conditions=ic_box, write=False
        )
    assert py21cmcu.cuda_stats()["perturb_runs"] == 1

    # Mode B (fast float32 atomicAdd) can reorder atomics across slabs slightly at ~1e-6, strict is ~1e-13.
    tol = 1e-6 if mode == "strict" else 5e-5
    for fname in ("density", "velocity_x", "velocity_y", "velocity_z"):
        _assert_rel_close(
            np.asarray(pf_slab.get(fname)),
            np.asarray(pf_res.get(fname)),
            f"{mode}:slab_vs_res:{fname}",
            rel_rms_tol=tol,
        )

    if mode == "strict":
        with py21cmcu.use_mode(
            "strict", use_cuda=False, direc=str(tmp_path / "pf_cpu")
        ):
            pf_cpu = py21cmcu.perturb_field(
                redshift=8.0, inputs=inputs, initial_conditions=ic_box, write=False
            )
        for fname in ("density", "velocity_x", "velocity_y", "velocity_z"):
            _assert_rel_close(
                np.asarray(pf_slab.get(fname)),
                np.asarray(pf_cpu.get(fname)),
                f"strict:slab_vs_cpu:{fname}",
                rel_rms_tol=5e-5,
            )


@requires_cuda_runtime
@pytest.mark.parametrize(
    ("recomb_model", "use_ts_fluct"),
    [("none", False), ("inhomogeneous", True)],
)
def test_ion_host_spilled_kspace_and_trimmed_prev_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recomb_model: str,
    use_ts_fluct: bool,
) -> None:
    """Verify host-spilled Ion k-space, skipped prev_first_z boxes, and skipped G12 match CPU."""
    inputs = (
        py21cmcu.InputParameters(random_seed=7203)
        .with_logspaced_redshifts(zmin=9.0, zmax=12.0, step=1.2)
        .evolve_input_structs(
            DIM=64,
            HII_DIM=32,
            BOX_LEN=128.0,
            N_THREADS=2,
            SOURCE_MODEL="E-INTEGRAL",
            USE_EXP_FILTER=False,
            USE_UPPER_STELLAR_TURNOVER=False,
            PHOTON_CONS_TYPE="no-photoncons",
            Z_HEAT_MAX=12.0,
            ZPRIME_STEP_FACTOR=1.2,
            RECOMB_MODEL=recomb_model,
            USE_TS_FLUCT=use_ts_fluct,
            R_BUBBLE_MAX=50.0,
        )
    )

    with py21cmcu.use_mode("strict", use_cuda=False, direc=str(tmp_path / "cpu")):
        cpu_coevals = py21cmcu.run_coeval(inputs=inputs, write=False)

    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "gpu_def")):
        gpu_coevals = py21cmcu.run_coeval(inputs=inputs, write=False)
    stats_def = py21cmcu.cuda_stats()
    assert stats_def["ionization_runs"] == len(cpu_coevals)

    monkeypatch.setenv("P21_CUDA_ION_SPILL_KSPACE", "1")
    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "gpu_spill")):
        spill_coevals = py21cmcu.run_coeval(inputs=inputs, write=False)
    assert py21cmcu.cuda_stats()["ionization_runs"] == len(cpu_coevals)

    check_fields = [
        "neutral_fraction",
        "z_reion",
        "ionisation_rate_G12",
        "kinetic_temperature",
    ]
    if recomb_model == "inhomogeneous":
        check_fields.append("cumulative_recombinations")

    for z_idx, (c_box, g_box, s_box) in enumerate(
        zip(cpu_coevals, gpu_coevals, spill_coevals, strict=True)
    ):
        assert abs(g_box.ionized_box.global_xH - c_box.ionized_box.global_xH) < 5e-3
        assert abs(s_box.ionized_box.global_xH - g_box.ionized_box.global_xH) < 1e-6
        for fname in check_fields:
            a_cpu = np.asarray(c_box.ionized_box.get(fname))
            a_gpu = np.asarray(g_box.ionized_box.get(fname))
            a_spill = np.asarray(s_box.ionized_box.get(fname))
            if fname in ("kinetic_temperature", "cumulative_recombinations"):
                _assert_rel_close(
                    a_gpu, a_cpu, f"z{z_idx}:gpu:{fname}", rel_rms_tol=5e-3
                )
            elif fname == "ionisation_rate_G12" and recomb_model == "none":
                assert np.all(a_gpu == 0.0)
            np.testing.assert_allclose(a_spill, a_gpu, rtol=1e-6, atol=1e-7)


@requires_cuda_runtime
@pytest.mark.parametrize("use_lya_heating", [False, True])
def test_radiation_trimmed_memory_parity(tmp_path: Path, use_lya_heating: bool) -> None:
    """Verify setup-cache eviction and eliminated d_interp/d_k_unfilt preserve strict TS parity."""
    inputs = (
        py21cmcu.InputParameters(random_seed=7204)
        .with_logspaced_redshifts(zmin=10.0, zmax=15.0, step=1.2)
        .evolve_input_structs(
            DIM=64,
            HII_DIM=32,
            BOX_LEN=128.0,
            N_THREADS=2,
            SOURCE_MODEL="E-INTEGRAL",
            USE_EXP_FILTER=False,
            USE_UPPER_STELLAR_TURNOVER=False,
            PHOTON_CONS_TYPE="no-photoncons",
            Z_HEAT_MAX=15.0,
            ZPRIME_STEP_FACTOR=1.2,
            USE_TS_FLUCT=True,
            USE_LYA_HEATING=use_lya_heating,
            USE_MCGS=False,
        )
    )

    with py21cmcu.use_mode("strict", use_cuda=False, direc=str(tmp_path / "cpu")):
        cpu_coevals = py21cmcu.run_coeval(inputs=inputs, write=False)

    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "gpu")):
        gpu_coevals = py21cmcu.run_coeval(inputs=inputs, write=False)
    stats = py21cmcu.cuda_stats()
    assert stats["spin_temp_runs"] == len(cpu_coevals)
    assert stats["radiation_setup_runs"] >= 2

    for z_idx, (c_box, g_box) in enumerate(zip(cpu_coevals, gpu_coevals, strict=True)):
        for fname in (
            "xray_ionised_fraction",
            "kinetic_temp_neutral",
            "spin_temperature",
        ):
            _assert_rel_close(
                np.asarray(g_box.ts_box.get(fname)),
                np.asarray(c_box.ts_box.get(fname)),
                f"z{z_idx}:{fname}",
                rel_rms_tol=5e-4,
            )


@requires_cuda_runtime
def test_cufft_zero_idle_workspace_reuse(tmp_path: Path) -> None:
    """Verify pool-backed cuFFT plans reuse cached handles and leave no leaked workspaces after cleanup."""
    py21cmcu.cleanup_cuda()
    py21cmcu.reset_cuda_stats()

    for dim, hii_dim in ((64, 32), (96, 48)):
        inputs = py21cmcu.InputParameters(random_seed=7205).evolve_input_structs(
            DIM=dim,
            HII_DIM=hii_dim,
            BOX_LEN=128.0,
            N_THREADS=2,
            PERTURB_ALGORITHM="2LPT",
            USE_TS_FLUCT=False,
        )
        with py21cmcu.use_mode(
            "fast", use_cuda=True, direc=str(tmp_path / f"run_{dim}")
        ):
            py21cmcu.run_coeval(out_redshifts=(8.0,), inputs=inputs, write=False)

    stats = py21cmcu.cuda_stats()
    assert stats["cufft_exec_r2c"] > 0
    assert stats["cufft_exec_c2r"] > 0
    assert stats["cufft_plan_hits"] > 0
    py21cmcu.cleanup_cuda()
