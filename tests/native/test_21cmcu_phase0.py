"""Phase 0 tests for 21cmCU: packaging, strict/fast modes, CLI, and upstream bug fixes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import py21cmcu
import py21cmcu.cli as cu_cli
import pytest
from py21cmfast import cli
from py21cmfast._cfg import ConfigurationError, config
from py21cmfast.c_21cmfast import lib
from py21cmfast.drivers import single_field
from py21cmfast.wrapper import photoncons

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


def _stats() -> dict[str, int]:
    """Return a snapshot of runtime counters via ``py21cmcu.cuda_stats``."""
    return py21cmcu.cuda_stats()


def test_py21cmcu_exports_and_no_cupy() -> None:
    """Verify ``py21cmcu`` exports core APIs and helpers without importing ``cupy``."""
    assert py21cmcu.__version__ == "0.2.0"
    for name in (
        "InputParameters",
        "RectilinearLightconer",
        "compute_initial_conditions",
        "run_coeval",
        "run_lightcone",
        "config",
        "cuda_compiled",
        "cuda_available",
        "cuda_stats",
        "reset_cuda_stats",
        "cleanup_cuda",
        "get_cuda_mode",
        "set_cuda_mode",
        "use_mode",
    ):
        assert hasattr(py21cmcu, name), f"Missing export: {name}"
    assert py21cmcu.cuda_compiled() == bool(lib.p21_cuda_is_compiled())
    assert py21cmcu.cuda_available() == bool(lib.p21_cuda_is_available())
    assert "cupy" not in sys.modules


def test_cuda_mode_config_and_runtime_sync() -> None:
    """Verify ``CUDA_MODE`` strict/fast synchronization and context restoration."""
    orig_cuda = config["USE_CUDA"]
    orig_mode = config["CUDA_MODE"]
    orig_fast_halo = config["CUDA_FAST_HALO_SAMPLER"]
    try:
        assert py21cmcu.get_cuda_mode() in ("strict", "fast")

        with pytest.raises(ConfigurationError, match="Invalid CUDA_MODE"):
            config["CUDA_MODE"] = "bogus"

        py21cmcu.set_cuda_mode("fast")
        assert config["CUDA_MODE"] == "fast"
        assert config["CUDA_FAST_HALO_SAMPLER"] is True
        assert lib.p21_cuda_get_mode() == 1
        assert lib.p21_cuda_get_fast_ic() == 1
        assert lib.p21_cuda_get_fast_halo_sampler() == 1

        py21cmcu.set_cuda_mode("strict")
        assert config["CUDA_MODE"] == "strict"
        assert config["CUDA_FAST_HALO_SAMPLER"] is False
        assert lib.p21_cuda_get_mode() == 0
        assert lib.p21_cuda_get_fast_ic() == 0
        assert lib.p21_cuda_get_fast_halo_sampler() == 0

        # Explicit CUDA_FAST_HALO_SAMPLER=False must win over CUDA_MODE="fast",
        # and both must cleanly restore upon exiting the context manager.
        with config.use(CUDA_MODE="fast", CUDA_FAST_HALO_SAMPLER=False):
            assert config["CUDA_MODE"] == "fast"
            assert config["CUDA_FAST_HALO_SAMPLER"] is False
            assert lib.p21_cuda_get_mode() == 1
            assert lib.p21_cuda_get_fast_ic() == 1
            assert lib.p21_cuda_get_fast_halo_sampler() == 0

        assert config["CUDA_MODE"] == "strict"
        assert config["CUDA_FAST_HALO_SAMPLER"] is False
        assert lib.p21_cuda_get_mode() == 0
        assert lib.p21_cuda_get_fast_ic() == 0
        assert lib.p21_cuda_get_fast_halo_sampler() == 0

        with py21cmcu.use_mode("fast"):
            assert config["USE_CUDA"] is True
            assert config["CUDA_MODE"] == "fast"
            assert config["CUDA_FAST_HALO_SAMPLER"] is True
            assert lib.p21_cuda_get_enabled() == 1
            assert lib.p21_cuda_get_mode() == 1
            assert lib.p21_cuda_get_fast_ic() == 1
            assert lib.p21_cuda_get_fast_halo_sampler() == 1
    finally:
        config["USE_CUDA"] = orig_cuda
        config["CUDA_MODE"] = orig_mode
        config["CUDA_FAST_HALO_SAMPLER"] = orig_fast_halo


def test_cli_cuda_mode_option_and_restore(tmp_path: Path, monkeypatch) -> None:
    """Verify ``py21cmcu.cli`` parses ``--cuda-mode`` and restores config on exit."""
    observed: list[tuple[bool, str, bool, int, int, int, int]] = []

    def _fake_generate_coeval(**kwargs):
        observed.append(
            (
                config["USE_CUDA"],
                config["CUDA_MODE"],
                config["CUDA_FAST_HALO_SAMPLER"],
                lib.p21_cuda_get_enabled(),
                lib.p21_cuda_get_mode(),
                lib.p21_cuda_get_fast_ic(),
                lib.p21_cuda_get_fast_halo_sampler(),
            )
        )
        return iter(())

    monkeypatch.setattr(cli, "generate_coeval", _fake_generate_coeval)

    with config.use(USE_CUDA=False, CUDA_MODE="strict", CUDA_FAST_HALO_SAMPLER=False):
        cu_cli.app(
            "run coeval -z 10 --template simple tiny "
            f"--cachedir {tmp_path} --out {tmp_path} "
            "--use-cuda --cuda-mode fast --no-cuda-fast-halo-sampler --no-show",
            result_action="return_value",
        )
        assert observed == [(True, "fast", False, 1, 1, 1, 0)]
        assert config["USE_CUDA"] is False
        assert config["CUDA_MODE"] == "strict"
        assert config["CUDA_FAST_HALO_SAMPLER"] is False
        assert lib.p21_cuda_get_enabled() == 0
        assert lib.p21_cuda_get_mode() == 0
        assert lib.p21_cuda_get_fast_ic() == 0
        assert lib.p21_cuda_get_fast_halo_sampler() == 0


def test_photoncons_calibration_array_lengths(monkeypatch) -> None:
    """Verify photon-conservation calibration passes matching redshift/xHI lengths."""
    inputs = py21cmcu.InputParameters(
        random_seed=404,
        node_redshifts=(10.0, 8.0),
    ).evolve_input_structs(
        DIM=16,
        HII_DIM=8,
        BOX_LEN=32.0,
        SOURCE_MODEL="E-INTEGRAL",
        PHOTON_CONS_TYPE="z-photoncons",
        USE_EXP_FILTER=False,
        USE_UPPER_STELLAR_TURNOVER=False,
        R_BUBBLE_MAX=15.0,
        USE_TS_FLUCT=False,
        RECOMB_MODEL="none",
    )
    dummy_ic = py21cmcu.InitialConditions.new(inputs=inputs)
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        photoncons, "_init_photon_conservation_correction", lambda **kw: None
    )
    monkeypatch.setattr(photoncons, "_calc_zstart_photon_cons", lambda: 6.2)
    monkeypatch.setattr(
        single_field,
        "perturb_field",
        lambda **kw: SimpleNamespace(redshift=kw["redshift"]),
    )
    monkeypatch.setattr(
        single_field,
        "compute_ionization_field",
        lambda **kw: SimpleNamespace(get=lambda _k: np.full((8, 8, 8), 0.5)),
    )

    def _capture_calibrate(*, redshifts_estimate, nf_estimate, NSpline) -> None:
        recorded["z"] = redshifts_estimate
        recorded["nf"] = nf_estimate
        recorded["n"] = NSpline

    monkeypatch.setattr(
        photoncons, "_calibrate_photon_conservation_correction", _capture_calibrate
    )

    photoncons.calibrate_photon_cons(
        inputs=inputs, initial_conditions=dummy_ic, write=False
    )

    z_arr = recorded["z"]
    nf_arr = recorded["nf"]
    assert isinstance(z_arr, np.ndarray)
    assert isinstance(nf_arr, np.ndarray)
    assert z_arr.size >= 4
    assert z_arr.shape == nf_arr.shape
    assert recorded["n"] == len(z_arr) == len(nf_arr)
    assert np.all(np.diff(z_arr) > 0.0)


@requires_cuda_runtime
def test_partition_sampler_cpu_and_gpu(tmp_path: Path) -> None:
    """Verify ST ``PARTITION`` progenitor sampling succeeds on both CPU and GPU fast mode."""
    inputs = py21cmcu.InputParameters(
        random_seed=5102,
        node_redshifts=(11.5, 10.5),
    ).evolve_input_structs(
        DIM=48,
        HII_DIM=16,
        BOX_LEN=64.0,
        N_THREADS=2,
        SOURCE_MODEL="CHMF-SAMPLER",
        HMF="ST",
        SAMPLE_METHOD="PARTITION",
        SAMPLER_MIN_MASS=5e8,
        USE_TS_FLUCT=False,
        RECOMB_MODEL="none",
    )

    with py21cmcu.use_mode("strict", use_cuda=False, direc=str(tmp_path / "cpu")):
        cpu_ics = py21cmcu.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False
        )
        cpu_desc = py21cmcu.determine_halo_catalog(
            redshift=10.5,
            initial_conditions=cpu_ics,
            inputs=inputs,
            regenerate=True,
            write=False,
        )
        cpu_prog = py21cmcu.determine_halo_catalog(
            redshift=11.5,
            initial_conditions=cpu_ics,
            descendant_halos=cpu_desc,
            inputs=inputs,
            regenerate=True,
            write=False,
        )

    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("fast", use_cuda=True, direc=str(tmp_path / "gpu")):
        gpu_ics = py21cmcu.compute_initial_conditions(
            inputs=inputs, regenerate=True, write=False
        )
        gpu_desc = py21cmcu.determine_halo_catalog(
            redshift=10.5,
            initial_conditions=gpu_ics,
            inputs=inputs,
            regenerate=True,
            write=False,
        )
        gpu_prog = py21cmcu.determine_halo_catalog(
            redshift=11.5,
            initial_conditions=gpu_ics,
            descendant_halos=gpu_desc,
            inputs=inputs,
            regenerate=True,
            write=False,
        )

    stats = _stats()
    assert stats["halo_sampler_runs"] == 2
    assert cpu_desc.n_halos > 50 and cpu_prog.n_halos > 20
    assert gpu_desc.n_halos > 50 and gpu_prog.n_halos > 20
    m_min = inputs.simulation_options.SAMPLER_MIN_MASS * (1.0 - 1e-6)
    assert np.all(cpu_prog.get("halo_masses")[: cpu_prog.n_halos] >= m_min)
    assert np.all(gpu_prog.get("halo_masses")[: gpu_prog.n_halos] >= m_min)
    assert "cupy" not in sys.modules


@requires_cuda_runtime
def test_e_integral_mcgs_without_photoheating_feedback(tmp_path: Path) -> None:
    """Verify ``E-INTEGRAL`` + ``USE_MCGS=True`` without photoheating matches CPU/GPU."""
    inputs = py21cmcu.InputParameters(
        random_seed=6103,
        node_redshifts=(12.0, 10.0),
    ).evolve_input_structs(
        DIM=48,
        HII_DIM=16,
        BOX_LEN=50.0,
        N_THREADS=2,
        Z_HEAT_MAX=12.0,
        SOURCE_MODEL="E-INTEGRAL",
        USE_EXP_FILTER=False,
        USE_UPPER_STELLAR_TURNOVER=False,
        V_CB_MODEL="AVG-DEBUG",
        USE_TS_FLUCT=True,
        USE_MCGS=True,
        USE_REIONIZATION_PHOTOHEATING_FEEDBACK=False,
        M_TURN_STELLAR_FEEDBACK=6.0,
        RECOMB_MODEL="inhomogeneous",
        R_BUBBLE_MAX=50.0,
    )

    with py21cmcu.use_mode("strict", use_cuda=False, direc=str(tmp_path / "cpu")):
        cpu_coevs = py21cmcu.run_coeval(inputs=inputs, regenerate=True, write=False)

    py21cmcu.reset_cuda_stats()
    with py21cmcu.use_mode("strict", use_cuda=True, direc=str(tmp_path / "gpu")):
        gpu_coevs = py21cmcu.run_coeval(inputs=inputs, regenerate=True, write=False)

    stats = _stats()
    assert stats["ionization_runs"] == 2
    assert stats["brightness_runs"] == 2
    assert "cupy" not in sys.modules

    for cpu_c, gpu_c in zip(cpu_coevs, gpu_coevs, strict=True):
        cpu_xhi = np.asarray(cpu_c.neutral_fraction, dtype=np.float64)
        gpu_xhi = np.asarray(gpu_c.neutral_fraction, dtype=np.float64)
        assert np.isfinite(cpu_xhi).all() and np.isfinite(gpu_xhi).all()
        assert np.all((cpu_xhi >= 0.0) & (cpu_xhi <= 1.0))
        assert np.all((gpu_xhi >= 0.0) & (gpu_xhi <= 1.0))
        rms = float(np.sqrt(np.mean(cpu_xhi**2)))
        rel_rms = float(np.sqrt(np.mean((gpu_xhi - cpu_xhi) ** 2))) / max(rms, 1e-12)
        assert rel_rms < 1e-3

        cpu_tb = np.asarray(cpu_c.brightness_temp, dtype=np.float64)
        gpu_tb = np.asarray(gpu_c.brightness_temp, dtype=np.float64)
        tb_rms = float(np.sqrt(np.mean(cpu_tb**2)))
        tb_rel_rms = float(np.sqrt(np.mean((gpu_tb - cpu_tb) ** 2))) / max(
            tb_rms, 1e-12
        )
        assert tb_rel_rms < 1e-3
