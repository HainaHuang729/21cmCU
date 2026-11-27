"""Directions L & M tests: SBI observables, resumable batch, CLI batch/doctor."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import py21cmcu
import py21cmcu.cli as cu_cli
import pytest
from py21cmfast.c_21cmfast import lib

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


def test_horizon_wedge_mask_and_3d_filter() -> None:
    """Verify foreground wedge mask removes inside-wedge modes while preserving EoR window."""
    nx, ny, nz = 24, 24, 24
    box_len = (120.0, 120.0, 120.0)
    z = 8.0

    # Pure high-k_parallel plane wave (outside wedge) vs pure high-k_perp plane wave (inside wedge)
    x = np.arange(nx)[:, None, None]
    zz = np.arange(nz)[None, None, :]
    in_wedge = np.cos(2.0 * np.pi * 4.0 * x / nx).astype(np.float32) * np.ones(
        (1, ny, nz), dtype=np.float32
    )
    in_window = np.cos(2.0 * np.pi * 6.0 * zz / nz).astype(np.float32) * np.ones(
        (nx, ny, 1), dtype=np.float32
    )
    combined = (in_wedge + in_window).astype(np.float32)

    filtered = py21cmcu.apply_wedge_filter_3d(
        combined, box_len, z, los_axis=2, buffer_h_mpc=0.02
    )
    # Inside-wedge mode (kz=0) is zeroed; window mode (kperp=0, |kz|>0) survives intact
    np.testing.assert_allclose(filtered, in_window, rtol=1e-4, atol=1e-4)

    mask = py21cmcu.horizon_wedge_mask(
        np.array([0.5, 0.5]), np.array([0.05, 5.0]), z, buffer_h_mpc=0.05
    )
    assert list(mask) == [False, True]


def test_add_thermal_noise_and_sobol_grid() -> None:
    """Verify thermal noise RMS, deterministic seeding, and Sobol parameter bounds."""
    base = np.zeros((32, 32, 32), dtype=np.float32)
    n1 = py21cmcu.add_thermal_noise_3d(base, 128.0, sigma_mk=3.5, seed=99)
    n2 = py21cmcu.add_thermal_noise_3d(base, 128.0, sigma_mk=3.5, seed=99)
    np.testing.assert_array_equal(n1, n2)
    assert abs(float(np.std(n1)) - 3.5) < 0.15

    # Physical radiometer formula branch (sigma_mk=None)
    n_phys = py21cmcu.add_thermal_noise_3d(base, 128.0, redshift=8.0, t_int_hours=500.0, seed=7)
    assert np.all(np.isfinite(n_phys))
    assert float(np.std(n_phys)) > 0.0

    grid = py21cmcu.generate_sobol_parameter_grid(
        {"F_STAR10": (-2.0, -0.5), "M_TURN": (8.0, 9.5)}, n_samples=6, seed=12, base_seed=500
    )
    assert len(grid) == 6
    assert [row["random_seed"] for row in grid] == [500, 501, 502, 503, 504, 505]
    for row in grid:
        assert -2.0 <= float(row["F_STAR10"]) <= -0.5
        assert 8.0 <= float(row["M_TURN"]) <= 9.5


@requires_cuda_runtime
def test_resumable_batch_checkpoint_and_observables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify BatchTask noise+wedge pipeline and run_batch checkpoint resume skips done tasks."""
    tasks = []
    for i in range(3):
        inp = py21cmcu.InputParameters(
            random_seed=6100 + i, node_redshifts=[8.0]
        ).evolve_input_structs(
            DIM=32,
            HII_DIM=16,
            BOX_LEN=64.0,
            N_THREADS=2,
            USE_TS_FLUCT=False,
            RECOMB_MODEL="none",
        )
        tasks.append(
            py21cmcu.BatchTask(
                task_id=f"sbi_{i}",
                inputs=inp,
                mode="coeval",
                out_redshifts=(8.0,),
                cuda_mode="fast",
                ps_n_bins=8,
                wedge_filter=True,
                wedge_buffer=0.05,
                noise_sigma_mk=1.5,
            )
        )

    ckpt = tmp_path / "sbi_ckpt.h5"
    # Pre-run only first 2 tasks into checkpoint
    res_part = py21cmcu.run_batch(
        tasks[:2],
        devices=(0,),
        workers_per_gpu=1,
        checkpoint_h5=ckpt,
        use_cuda=True,
        in_process=True,
    )
    assert all(r.ok for r in res_part)
    with h5py.File(ckpt, "r") as fl:
        assert set(fl["tasks"].keys()) == {"sbi_0", "sbi_1"}

    executed: list[str] = []
    orig_exec = py21cmcu.batch._execute_single_task

    def _spy_exec(task, **kw):
        executed.append(task.task_id)
        return orig_exec(task, **kw)

    monkeypatch.setattr(py21cmcu.batch, "_execute_single_task", _spy_exec)
    res_full = py21cmcu.run_batch(
        tasks,
        devices=(0,),
        workers_per_gpu=1,
        output_h5=ckpt,
        checkpoint_h5=ckpt,
        resume=True,
        use_cuda=True,
        in_process=True,
    )
    assert executed == ["sbi_2"]
    assert [r.task_id for r in res_full] == ["sbi_0", "sbi_1", "sbi_2"]
    assert all(r.ok for r in res_full)
    with h5py.File(ckpt, "r") as fl:
        assert set(fl["tasks"].keys()) == {"sbi_0", "sbi_1", "sbi_2"}


def test_cli_doctor_and_batch_cpu(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify doctor() API, `21cmcu doctor --as-json`, and CPU `21cmcu batch --sobol-samples` CLI."""
    rep = py21cmcu.doctor(as_json=False)
    assert rep["versions"]["py21cmcu"] == py21cmcu.__version__
    assert "cuda" in rep and "grid_sizing" in rep

    cu_cli.app(["doctor", "--as-json"], result_action="return_value")
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["versions"]["py21cmcu"] == py21cmcu.__version__

    out_h5 = tmp_path / "cli_batch.h5"
    cu_cli.app(
        [
            "batch",
            "--sobol-samples",
            "2",
            "--out",
            str(out_h5),
            "--dim",
            "16",
            "--hii-dim",
            "8",
            "--box-len",
            "32.0",
            "--no-use-cuda",
            "--in-process",
            "--wedge-filter",
            "--noise-sigma-mk",
            "1.0",
        ],
        result_action="return_value",
    )
    assert out_h5.exists()
    with h5py.File(out_h5, "r") as fl:
        assert set(fl["tasks"].keys()) == {"task_0000", "task_0001"}
        assert bool(fl["tasks/task_0000"].attrs["ok"])
