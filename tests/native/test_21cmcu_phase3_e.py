"""Direction E tests for spawned multi-worker batch runner and summary persistence."""

from __future__ import annotations

import h5py
import numpy as np
import py21cmcu
import pytest
from py21cmfast.c_21cmfast import lib

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


@requires_cuda_runtime
def test_run_batch_in_process_and_spawned_coeval(tmp_path) -> None:
    """Verify run_batch matches in-process vs spawned workers_per_gpu=2 and writes summary HDF5."""
    tasks = []
    for i, seed in enumerate((5101, 5102)):
        inp = py21cmcu.InputParameters(
            random_seed=seed, node_redshifts=[8.0]
        ).evolve_input_structs(
            DIM=48,
            HII_DIM=24,
            BOX_LEN=96.0,
            N_THREADS=2,
            USE_TS_FLUCT=False,
            RECOMB_MODEL="none",
        )
        tasks.append(
            py21cmcu.BatchTask(
                task_id=f"cv_{i}",
                inputs=inp,
                mode="coeval",
                out_redshifts=(8.0,),
                cuda_mode="fast",
                ps_n_bins=8,
                compute_ps_2d=True,
                ps_2d_n_kperp=4,
                ps_2d_n_kpara=4,
            )
        )

    inline_res = py21cmcu.run_batch(tasks, in_process=True, use_cuda=True)
    h5_out = tmp_path / "batch_summary.h5"
    spawn_res = py21cmcu.run_batch(
        tasks,
        devices=(0,),
        workers_per_gpu=2,
        threads_per_worker=2,
        output_h5=h5_out,
        use_cuda=True,
    )

    assert len(inline_res) == 2 and len(spawn_res) == 2
    for r_in, r_sp in zip(inline_res, spawn_res, strict=True):
        assert r_in.ok and r_sp.ok, f"Errors: {r_in.error} / {r_sp.error}"
        assert r_in.task_id == r_sp.task_id
        np.testing.assert_allclose(r_in.global_xH, r_sp.global_xH, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(r_in.global_Tb, r_sp.global_Tb, rtol=1e-5, atol=1e-5)
        sp_in = r_in.ps_1d["spectra"][0]
        sp_sp = r_sp.ps_1d["spectra"][0]
        mask = sp_in["counts"] > 0
        np.testing.assert_allclose(sp_in["delta_sq"][mask], sp_sp["delta_sq"][mask], rtol=1e-5)
        assert r_sp.ps_2d is not None and r_sp.ps_2d["spectra"][0]["pk"].shape == (4, 4)

    with h5py.File(h5_out, "r") as fl:
        assert "cv_0" in fl["tasks"] and "cv_1" in fl["tasks"]
        assert fl["tasks/cv_0"].attrs["ok"]
        np.testing.assert_allclose(fl["tasks/cv_0/global_xH"][:], spawn_res[0].global_xH)
        assert "delta_sq" in fl["tasks/cv_0/ps_1d/z_0"]


@requires_cuda_runtime
def test_run_batch_lightcone_and_error_isolation() -> None:
    """Verify lightcone BatchTask + error isolation across concurrent tasks."""
    good_inp = py21cmcu.InputParameters(
        random_seed=5201, node_redshifts=[7.0, 8.0, 9.0]
    ).evolve_input_structs(
        DIM=48,
        HII_DIM=24,
        BOX_LEN=72.0,
        N_THREADS=2,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        Z_HEAT_MAX=9.0,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    bad_task = py21cmcu.BatchTask(
        task_id="bad_mode",
        inputs=good_inp,
        mode="invalid_mode",  # type: ignore[arg-type]
    )
    good_task = py21cmcu.BatchTask(
        task_id="lc_good",
        inputs=good_inp,
        mode="lightcone",
        min_redshift=7.2,
        max_redshift=8.8,
        cuda_mode="fast",
        apply_rsds=True,
        include_dvdr_in_tau21=True,
        ps_n_bins=8,
        lightcone_chunk_depth=12,
    )

    res = py21cmcu.run_batch(
        [bad_task, good_task],
        devices=(0,),
        workers_per_gpu=1,
        in_process=False,
        use_cuda=True,
    )
    assert len(res) == 2
    assert not res[0].ok and res[0].error is not None
    assert res[1].ok, res[1].error
    assert len(res[1].ps_1d["spectra"]) >= 1
