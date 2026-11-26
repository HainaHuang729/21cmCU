"""Phase 1 tests for 21cmCU: fast/strict parallel IC RNG, GPU chunk hashing, and GPU emissivity tables."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import py21cmcu
import pytest
from powerbox import get_power
from py21cmfast.c_21cmfast import lib
from py21cmfast.drivers import single_field as sf

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


def _assert_rel_close(
    gpu: np.ndarray,
    cpu: np.ndarray,
    name: str,
    *,
    rel_rms_tol: float = 1e-3,
) -> None:
    """Assert finite matching shapes and relative RMS tolerance between GPU and CPU arrays."""
    assert gpu.shape == cpu.shape, f"{name} shape mismatch: {gpu.shape} vs {cpu.shape}"
    assert np.all(np.isfinite(gpu)), f"Non-finite GPU values in {name}"
    assert np.all(np.isfinite(cpu)), f"Non-finite CPU values in {name}"
    g64 = gpu.astype(np.float64)
    c64 = cpu.astype(np.float64)
    rms = float(np.sqrt(np.mean(c64**2)))
    diff_rms = float(np.sqrt(np.mean((g64 - c64) ** 2)))
    rel_rms = diff_rms / max(rms, 1e-30)
    assert (
        rel_rms < rel_rms_tol
    ), f"{name} rel_rms={rel_rms:.3e} >= {rel_rms_tol:.3e} (rms={rms:.3e})"


@requires_cuda_runtime
@pytest.mark.parametrize("ps_model", ["EH", "CLASS"])
def test_fast_ic_power_spectrum_and_thread_invariance(
    tmp_path: Path, ps_model: str
) -> None:
    """Verify fast Philox IC RNG is thread-invariant and matches strict P(k) & v_cb RMS."""
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
        py21cmcu.InputParameters(random_seed=7101).with_logspaced_redshifts(
            zmin=10.0, zmax=15.0, step=1.2
        )
        if ps_model == "CLASS"
        else py21cmcu.InputParameters(random_seed=7101)
    )
    base_inputs = seed_inputs.evolve_input_structs(
        DIM=128,
        HII_DIM=32,
        BOX_LEN=128.0,
        N_THREADS=1,
        POWER_SPECTRUM=ps_model,
        V_CB_MODEL=v_cb_model,
        **extra,
    )

    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "strict")):
        ic_strict = py21cmcu.compute_initial_conditions(
            inputs=base_inputs, regenerate=True, write=False
        )

    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("fast", use_cuda=True, direc=str(tmp_path / "fast1")):
        ic_fast_1t = py21cmcu.compute_initial_conditions(
            inputs=base_inputs, regenerate=True, write=False
        )
    stats_fast = py21cmcu.cuda_stats()
    assert stats_fast["ic_runs"] == 1

    inputs_4t = base_inputs.evolve_input_structs(N_THREADS=4)
    with py21cmcu.use_mode("fast", use_cuda=True, direc=str(tmp_path / "fast4")):
        ic_fast_4t = py21cmcu.compute_initial_conditions(
            inputs=inputs_4t, regenerate=True, write=False
        )

    # Bitwise thread-count invariance under Philox fast IC.
    np.testing.assert_array_equal(
        np.asarray(ic_fast_1t.get("hires_density")),
        np.asarray(ic_fast_4t.get("hires_density")),
    )
    np.testing.assert_array_equal(
        np.asarray(ic_fast_1t.get("lowres_density")),
        np.asarray(ic_fast_4t.get("lowres_density")),
    )
    np.testing.assert_array_equal(
        np.asarray(ic_fast_1t.get("hires_vx")),
        np.asarray(ic_fast_4t.get("hires_vx")),
    )

    # Different seeds produce distinct boxes.
    diff_seed = base_inputs.clone(random_seed=7102)
    with py21cmcu.use_mode("fast", use_cuda=True, direc=str(tmp_path / "fast_s2")):
        ic_other = py21cmcu.compute_initial_conditions(
            inputs=diff_seed, regenerate=True, write=False
        )
    assert not np.allclose(
        np.asarray(ic_fast_1t.get("lowres_density")),
        np.asarray(ic_other.get("lowres_density")),
    )

    # Compare spherically averaged 3D power spectra P(k) of lowres_density.
    lo_strict = np.asarray(ic_strict.get("lowres_density"), dtype=np.float64)
    lo_fast = np.asarray(ic_fast_1t.get("lowres_density"), dtype=np.float64)
    p_s = get_power(lo_strict, 128.0, bins_upto_boxlen=True).power
    p_f = get_power(lo_fast, 128.0, bins_upto_boxlen=True).power
    mask = np.isfinite(p_s) & (p_s > 0)
    rel_diff = np.abs(p_f[mask] - p_s[mask]) / p_s[mask]
    assert float(np.median(rel_diff)) < 0.20

    if ps_model == "CLASS":
        vcb_s = np.asarray(ic_strict.get("lowres_vcb"), dtype=np.float64)
        vcb_f = np.asarray(ic_fast_1t.get("lowres_vcb"), dtype=np.float64)
        assert np.all(np.isfinite(vcb_f)) and np.all(vcb_f >= 0.0)
        rms_s = float(np.sqrt(np.mean(vcb_s**2)))
        rms_f = float(np.sqrt(np.mean(vcb_f**2)))
        assert abs(rms_f - rms_s) / max(rms_s, 1e-12) < 0.08

    assert "cupy" not in sys.modules


@requires_cuda_runtime
def test_chunk_hash_aba_detection_large_buffer(tmp_path: Path) -> None:
    """Verify chunk-tree GPU/CPU fingerprint detects mid-buffer ABA mutations (> 1 MB)."""
    inputs = py21cmcu.InputParameters(random_seed=7201).evolve_input_structs(
        DIM=96,
        HII_DIM=32,
        BOX_LEN=128.0,
        N_THREADS=2,
        PERTURB_ON_HIGH_RES=True,
    )
    with py21cmcu.use_mode("fast", use_cuda=True, direc=str(tmp_path / "aba")):
        ics = sf.compute_initial_conditions(inputs=inputs, regenerate=True, write=False)
        hi_arr = ics.get("hires_density")
        assert hi_arr.nbytes >= 2 * 1024 * 1024

        py21cmcu.reset_cuda_stats()
        pf1 = sf.perturb_field(
            redshift=10.0,
            initial_conditions=ics,
            inputs=inputs,
            regenerate=True,
            write=False,
        )
        s1 = py21cmcu.cuda_stats()

        pf2 = sf.perturb_field(
            redshift=10.0,
            initial_conditions=ics,
            inputs=inputs,
            regenerate=True,
            write=False,
        )
        s2 = py21cmcu.cuda_stats()
        assert s2["registry_hits"] > s1["registry_hits"]
        assert s2["registry_misses"] == s1["registry_misses"]
        np.testing.assert_array_equal(
            np.asarray(pf1.get("density")), np.asarray(pf2.get("density"))
        )

        # Mutate one interior element beyond chunk 0 and verify invalidation + updated output.
        mid_i = hi_arr.shape[0] // 2
        hi_arr[mid_i, mid_i, mid_i] += 0.25
        pf3 = sf.perturb_field(
            redshift=10.0,
            initial_conditions=ics,
            inputs=inputs,
            regenerate=True,
            write=False,
        )
        s3 = py21cmcu.cuda_stats()
        assert s3["registry_invalidations"] > s2["registry_invalidations"]
        assert s3["registry_misses"] > s2["registry_misses"]
        assert not np.allclose(
            np.asarray(pf2.get("density")), np.asarray(pf3.get("density"))
        )


@requires_cuda_runtime
@pytest.mark.parametrize(
    "source_model",
    ["CHMF-SAMPLER", "L-INTEGRAL", "E-INTEGRAL"],
)
def test_gpu_conditional_and_unconditional_tables_parity(
    tmp_path: Path, source_model: str
) -> None:
    """Verify GPU conditional/unconditional emissivity table builders match CPU across models."""
    extra: dict[str, object] = {}
    if source_model in ("L-INTEGRAL", "E-INTEGRAL"):
        extra["USE_UPPER_STELLAR_TURNOVER"] = False
    if source_model == "E-INTEGRAL":
        extra["USE_EXP_FILTER"] = False

    inputs = (
        py21cmcu.InputParameters(random_seed=7301)
        .with_logspaced_redshifts(zmin=10.0, zmax=15.0, step=1.2)
        .evolve_input_structs(
            DIM=64,
            HII_DIM=16,
            BOX_LEN=64.0,
            N_THREADS=2,
            Z_HEAT_MAX=15.0,
            ZPRIME_STEP_FACTOR=1.2,
            SOURCE_MODEL=source_model,
            POWER_SPECTRUM="CLASS",
            K_MAX_FOR_CLASS=1.0,
            V_CB_MODEL="FLUCTS",
            USE_MCGS=True,
            USE_TS_FLUCT=True,
            USE_METALLICITY=True,
            USE_REIONIZATION_PHOTOHEATING_FEEDBACK=True,
            RECOMB_MODEL="inhomogeneous",
            R_BUBBLE_MAX=50.0,
            M_TURN_STELLAR_FEEDBACK=6.0,
            **extra,
        )
    )

    with py21cmcu.use_mode("strict", use_cuda=False, direc=str(tmp_path / "cpu")):
        cpu_coevs = py21cmcu.run_coeval(inputs=inputs, regenerate=True, write=True)

    if source_model == "CHMF-SAMPLER":
        # Isolate GPU conditional tables & Lagrangian deposition from DexM/Poisson halo count shifts.
        z1 = inputs.node_redshifts[1]
        with py21cmcu.use_mode(
            "strict", use_cuda=False, direc=str(tmp_path / "cpu_hc")
        ):
            ics = sf.compute_initial_conditions(
                inputs=inputs, regenerate=True, write=False
            )
            desc = None
            hc1 = None
            for z in reversed(inputs.node_redshifts):
                desc = sf.determine_halo_catalog(
                    redshift=z,
                    initial_conditions=ics,
                    descendant_halos=desc,
                    inputs=inputs,
                    regenerate=True,
                    write=False,
                )
                if z == z1:
                    hc1 = desc
                    break
            assert hc1 is not None
            ef1_cpu = sf.compute_emissivity_fields(
                redshift=z1,
                halo_catalog=hc1,
                initial_conditions=ics,
                previous_spin_temp=cpu_coevs[0].ts_box,
                previous_ionize_box=cpu_coevs[0].ionized_box,
                inputs=inputs,
                regenerate=True,
                write=False,
            )

        py21cmcu.reset_cuda_stats()
        with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "gpu_ef")):
            ef1_gpu = sf.compute_emissivity_fields(
                redshift=z1,
                halo_catalog=hc1,
                initial_conditions=ics,
                previous_spin_temp=cpu_coevs[0].ts_box,
                previous_ionize_box=cpu_coevs[0].ionized_box,
                inputs=inputs,
                regenerate=True,
                write=False,
            )
        stats = py21cmcu.cuda_stats()
        assert stats["cond_table_runs"] == 3
        assert stats["emissivity_runs"] == 1
        assert "cupy" not in sys.modules

        for field_name in ("n_ion", "sfrd_acg", "sfrd_mcg", "xray_emissivity"):
            _assert_rel_close(
                np.asarray(ef1_gpu.get(field_name)),
                np.asarray(ef1_cpu.get(field_name)),
                f"{source_model}:{field_name}",
                rel_rms_tol=1e-4,
            )
        return

    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "gpu")):
        gpu_coevs = py21cmcu.run_coeval(inputs=inputs, regenerate=True, write=True)
    stats = py21cmcu.cuda_stats()

    assert stats["cond_table_runs"] >= 1
    assert stats["emissivity_runs"] == len(inputs.node_redshifts)
    assert "cupy" not in sys.modules

    for cpu_c, gpu_c in zip(cpu_coevs, gpu_coevs, strict=True):
        _assert_rel_close(
            np.asarray(gpu_c.emissivity_fields.get("n_ion")),
            np.asarray(cpu_c.emissivity_fields.get("n_ion")),
            f"{source_model}:n_ion",
            rel_rms_tol=1e-3,
        )
        _assert_rel_close(
            np.asarray(gpu_c.emissivity_fields.get("sfrd_acg")),
            np.asarray(cpu_c.emissivity_fields.get("sfrd_acg")),
            f"{source_model}:sfrd_acg",
            rel_rms_tol=1e-3,
        )
        _assert_rel_close(
            np.asarray(gpu_c.emissivity_fields.get("sfrd_mcg")),
            np.asarray(cpu_c.emissivity_fields.get("sfrd_mcg")),
            f"{source_model}:sfrd_mcg",
            rel_rms_tol=1e-3,
        )
        _assert_rel_close(
            np.asarray(gpu_c.emissivity_fields.get("xray_emissivity")),
            np.asarray(cpu_c.emissivity_fields.get("xray_emissivity")),
            f"{source_model}:xray",
            rel_rms_tol=1e-3,
        )
        _assert_rel_close(
            np.asarray(gpu_c.neutral_fraction),
            np.asarray(cpu_c.neutral_fraction),
            f"{source_model}:xHI",
            rel_rms_tol=1e-3,
        )
        _assert_rel_close(
            np.asarray(gpu_c.brightness_temp),
            np.asarray(cpu_c.brightness_temp),
            f"{source_model}:Tb",
            rel_rms_tol=2e-3,
        )


@requires_cuda_runtime
def test_chmf_mcg_class_fast_mode_end_to_end(tmp_path: Path) -> None:
    """Verify end-to-end ``CHMF-SAMPLER`` + ``MCG`` + ``CLASS`` coeval run in fast mode."""
    inputs = (
        py21cmcu.InputParameters(random_seed=7401)
        .with_logspaced_redshifts(zmin=10.0, zmax=15.0, step=1.2)
        .evolve_input_structs(
            DIM=64,
            HII_DIM=24,
            BOX_LEN=64.0,
            N_THREADS=4,
            Z_HEAT_MAX=15.0,
            ZPRIME_STEP_FACTOR=1.2,
            SOURCE_MODEL="CHMF-SAMPLER",
            POWER_SPECTRUM="CLASS",
            K_MAX_FOR_CLASS=1.0,
            V_CB_MODEL="FLUCTS",
            USE_MCGS=True,
            USE_TS_FLUCT=True,
            USE_METALLICITY=True,
            RECOMB_MODEL="inhomogeneous",
            R_BUBBLE_MAX=50.0,
            M_TURN_STELLAR_FEEDBACK=6.0,
        )
    )

    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("fast", use_cuda=True, direc=str(tmp_path / "fast")):
        coevs = py21cmcu.run_coeval(inputs=inputs, regenerate=True, write=True)
    stats = py21cmcu.cuda_stats()

    assert stats["ic_runs"] == 1
    assert stats["halo_sampler_runs"] >= 2
    assert stats["cond_table_runs"] >= 1
    assert stats["emissivity_runs"] == len(inputs.node_redshifts)
    assert stats["radiation_setup_runs"] >= 1
    assert "cupy" not in sys.modules

    for c in coevs:
        xhi = np.asarray(c.neutral_fraction, dtype=np.float64)
        tb = np.asarray(c.brightness_temp, dtype=np.float64)
        assert np.all(np.isfinite(xhi)) and np.all((xhi >= 0.0) & (xhi <= 1.0))
        assert np.all(np.isfinite(tb))
