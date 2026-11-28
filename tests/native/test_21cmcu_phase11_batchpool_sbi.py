"""Phase 11 tests: Persistent multi-GPU cuFFT cache, warm BatchPool, and OnlineSBIDataset."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import py21cmcu
from py21cmfast.c_21cmfast import ffi, lib

requires_cuda_runtime = pytest.mark.skipif(
    lib.p21_cuda_is_compiled() == 0 or lib.p21_cuda_is_available() == 0,
    reason="Native CUDA runtime is not compiled or no CUDA device is available.",
)


def _small_inputs(seed: int = 8100, *, two_z: bool = False) -> py21cmcu.InputParameters:
    redshifts = [7.5, 8.5, 9.5] if two_z else [8.0]
    kw = dict(
        DIM=24,
        HII_DIM=12,
        BOX_LEN=48.0,
        N_THREADS=1,
        USE_UPPER_STELLAR_TURNOVER=False,
        PHOTON_CONS_TYPE="no-photoncons",
    )
    if two_z:
        kw.update(SOURCE_MODEL="L-INTEGRAL", USE_TS_FLUCT=True, Z_HEAT_MAX=9.2)
    else:
        kw.update(
            SOURCE_MODEL="E-INTEGRAL",
            USE_TS_FLUCT=False,
            RECOMB_MODEL="none",
            USE_EXP_FILTER=False,
        )
    return py21cmcu.InputParameters(random_seed=seed, node_redshifts=redshifts).evolve_input_structs(
        **kw
    )


def test_batch_pool_persistent_workers_and_imap(tmp_path: Path) -> None:
    """Verify BatchPool reuses worker PIDs across rounds, supports imap, and resumes checkpoints."""
    with py21cmcu.config.use(direc=str(tmp_path / "cache")):
        round1 = [
            py21cmcu.BatchTask(
                task_id="r1_ok_0",
                inputs=_small_inputs(8101),
                mode="coeval",
                out_redshifts=(8.0,),
                ps_n_bins=6,
                compute_moments=True,
            ),
            py21cmcu.BatchTask(
                task_id="r1_bad",
                inputs=_small_inputs(8102),
                mode="invalid_mode",  # type: ignore[arg-type]
            ),
            py21cmcu.BatchTask(
                task_id="r1_ok_1",
                inputs=_small_inputs(8103),
                mode="coeval",
                out_redshifts=(8.0,),
                ps_n_bins=6,
                compute_moments=True,
            ),
        ]
        round2 = [
            py21cmcu.BatchTask(
                task_id=f"r2_{i}",
                inputs=_small_inputs(8200 + i),
                mode="coeval",
                out_redshifts=(8.0,),
                ps_n_bins=6,
            )
            for i in range(2)
        ]

        ckpt = tmp_path / "pool_ckpt.h5"
        with py21cmcu.BatchPool(
            devices=(0,),
            workers_per_gpu=2,
            threads_per_worker=1,
            use_cuda=False,
            in_process=False,
            clear_registry_every=2,
        ) as pool:
            assert pool.n_workers == 2
            res1 = pool.map(round1, checkpoint_h5=ckpt)
            assert res1[0].ok and not res1[1].ok and res1[2].ok
            pids_r1 = {r.worker_pid for r in res1}
            assert all(pid > 0 and pid != os.getpid() for pid in pids_r1)
            assert res1[0].moments is not None

            res2_unordered = list(pool.imap(round2, ordered=False))
            assert {r.task_id for r in res2_unordered} == {"r2_0", "r2_1"}
            assert all(r.ok and r.worker_pid in pids_r1 for r in res2_unordered)

            res2_ordered = list(pool.imap(round2, ordered=True))
            assert [r.task_id for r in res2_ordered] == ["r2_0", "r2_1"]
            assert all(r.ok and r.worker_pid in pids_r1 for r in res2_ordered)
            assert pool.tasks_executed == 7

            # Resume skips completed r1_ok_* tasks from checkpoint and re-runs only r1_bad
            res_resume = pool.map(round1, checkpoint_h5=ckpt, resume=True)
            assert res_resume[0].ok and res_resume[2].ok and not res_resume[1].ok
            assert pool.tasks_executed == 8

        assert pool.is_closed
        with pytest.raises(RuntimeError):
            pool.map(round2)


def test_batch_task_instrument_moments_and_oblique_lightcone(tmp_path: Path) -> None:
    """Verify BatchTask SKA1-Low mock, moments persistence, and oblique lightcone path."""
    use_cuda = bool(py21cmcu.cuda_compiled() and py21cmcu.cuda_available())
    with py21cmcu.config.use(direc=str(tmp_path / "lc")):
        t_cv = py21cmcu.BatchTask(
            task_id="cv_ska",
            inputs=_small_inputs(8301),
            mode="coeval",
            out_redshifts=(8.0,),
            ps_n_bins=6,
            instrument="SKA1-Low",
            compute_moments=True,
            moments_n_bins=12,
        )
        t_lc = py21cmcu.BatchTask(
            task_id="lc_hera_oblique",
            inputs=_small_inputs(8302, two_z=True),
            mode="lightcone",
            min_redshift=7.7,
            max_redshift=9.0,
            apply_rsds=True,
            ps_n_bins=6,
            lightcone_chunk_depth=8,
            instrument="HERA-350",
            compute_moments=True,
            oblique_lightcone=True,
        )
        out_h5 = tmp_path / "inst_moments.h5"
        results = py21cmcu.run_batch(
            [t_cv, t_lc],
            in_process=True,
            use_cuda=use_cuda,
            output_h5=out_h5,
        )
        assert all(r.ok for r in results), [r.error for r in results]
        assert isinstance(results[0].moments, list)
        assert np.isfinite(results[0].moments[0]["skewness"])
        assert isinstance(results[1].moments, dict)
        assert np.isfinite(results[1].moments["kurtosis"])
        assert len(results[1].ps_1d["spectra"]) >= 1

        # Verify HDF5 roundtrip of moments via resume
        resumed = py21cmcu.run_batch(
            [t_cv, t_lc],
            checkpoint_h5=out_h5,
            resume=True,
            in_process=True,
            use_cuda=use_cuda,
        )
        assert isinstance(resumed[0].moments, list)
        np.testing.assert_allclose(
            resumed[0].moments[0]["skewness"], results[0].moments[0]["skewness"]
        )
        assert isinstance(resumed[1].moments, dict)
        np.testing.assert_allclose(resumed[1].moments["kurtosis"], results[1].moments["kurtosis"])


def test_sbi_simulator_with_batch_pool_and_online_dataset(tmp_path: Path) -> None:
    """Verify SBISimulator BatchPool parity and finite/infinite OnlineSBIDataset streaming."""
    with py21cmcu.config.use(direc=str(tmp_path / "sbi")):
        base_inputs = _small_inputs(8400)
        bounds = {"F_STAR10_ACG": (-1.5, -1.1), "F_ESC10_ACG": (-1.3, -0.8)}

        with pytest.raises(ValueError):
            py21cmcu.OnlineSBIDataset(
                py21cmcu.SBISimulator(
                    base_inputs=base_inputs,
                    param_names=("F_STAR10_ACG", "F_ESC10_ACG"),
                    use_cuda=False,
                ),
                {"F_STAR10_ACG": (-1.5, -1.1)},
            )

        with py21cmcu.BatchPool(devices=(0,), workers_per_gpu=1, use_cuda=False, in_process=True) as pool:
            sim_inline = py21cmcu.SBISimulator(
                base_inputs=base_inputs,
                param_names=("F_STAR10_ACG", "F_ESC10_ACG"),
                mode="coeval",
                out_redshift=8.0,
                summary="stats",
                ps_n_bins=6,
                instrument="SKA1-Low",
                use_cuda=False,
            )
            sim_pool = py21cmcu.SBISimulator(
                base_inputs=base_inputs,
                param_names=("F_STAR10_ACG", "F_ESC10_ACG"),
                mode="coeval",
                out_redshift=8.0,
                summary="stats",
                ps_n_bins=6,
                instrument="SKA1-Low",
                use_cuda=False,
                batch_pool=pool,
            )

            thetas = np.array([[-1.3, -1.0], [-1.4, -1.1], [-1.2, -0.9]], dtype=np.float64)
            out_pool = sim_pool(thetas)
            out_ref = np.stack([sim_inline._simulate_single_stats(thetas[i], idx=i) for i in range(3)], axis=0)
            np.testing.assert_allclose(out_pool, out_ref, rtol=1e-5, atol=1e-5)
            assert pool.tasks_executed == 3

            ds_sobol = py21cmcu.OnlineSBIDataset(
                sim_pool,
                bounds,
                n_samples=5,
                batch_size=2,
                sampling="sobol",
                seed=77,
                as_torch=False,
            )
            assert len(ds_sobol) == 3
            batches_e1 = list(ds_sobol)
            assert [b[0].shape[0] for b in batches_e1] == [2, 2, 1]
            assert batches_e1[0][0].dtype == np.float32 and batches_e1[0][1].dtype == np.float32
            assert batches_e1[0][1].shape == (2, 11)

            batches_e2 = list(ds_sobol)
            assert not np.allclose(batches_e1[0][0], batches_e2[0][0])

            ds_inf = py21cmcu.OnlineSBIDataset(
                sim_pool,
                bounds,
                n_samples=None,
                batch_size=2,
                sampling="uniform",
                seed=99,
            )
            with pytest.raises(TypeError):
                len(ds_inf)
            it = iter(ds_inf)
            b0 = next(it)
            b1 = next(it)
            assert b0[0].shape == (2, 2) and b1[1].shape == (2, 11)
            assert not np.allclose(b0[0], b1[0])

            try:
                import torch

                ds_torch = py21cmcu.OnlineSBIDataset(
                    sim_pool,
                    bounds,
                    n_samples=2,
                    batch_size=2,
                    sampling="sobol",
                    seed=77,
                    as_torch=True,
                )
                th_t, x_t = next(iter(ds_torch))
                assert isinstance(th_t, torch.Tensor) and isinstance(x_t, torch.Tensor)
                assert th_t.dtype == torch.float32 and x_t.dtype == torch.float32
                assert x_t.shape == (2, 11)
            except ImportError:
                pass


@requires_cuda_runtime
def test_mgpu_persistent_cufft_plan_cache_and_parity() -> None:
    """Verify multi-GPU cuFFT plan cache records hits on repeated transforms and resets on cleanup."""
    shape = (24, 24, 24)
    rng = np.random.default_rng(8501)
    field = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    hz = shape[2] // 2 + 1
    cplx = np.empty((shape[0], shape[1], hz, 2), dtype=np.float32)
    rec = np.empty(shape, dtype=np.float32)

    orig_devs = py21cmcu.get_cuda_devices()
    n_avail = py21cmcu.cuda_device_count()
    use_devs = (0, 1) if n_avail >= 2 else (0,)
    try:
        py21cmcu.cleanup_cuda()
        py21cmcu.set_cuda_devices(use_devs)
        py21cmcu.reset_cuda_stats()

        rc1 = lib.p21_cuda_mgpu_fft_r2c_host(
            ffi.cast("const float *", ffi.from_buffer(field)),
            ffi.cast("float *", ffi.from_buffer(cplx)),
            *shape,
        )
        rc2 = lib.p21_cuda_mgpu_fft_c2r_host(
            ffi.cast("const float *", ffi.from_buffer(cplx)),
            ffi.cast("float *", ffi.from_buffer(rec)),
            *shape,
        )
        assert rc1 == 0 and rc2 == 0
        np.testing.assert_allclose(rec / float(np.prod(shape)), field, rtol=1e-4, atol=1e-4)
        st_first = py21cmcu.cuda_stats()

        py21cmcu.reset_cuda_stats()
        rc3 = lib.p21_cuda_mgpu_fft_r2c_host(
            ffi.cast("const float *", ffi.from_buffer(field)),
            ffi.cast("float *", ffi.from_buffer(cplx)),
            *shape,
        )
        rc4 = lib.p21_cuda_mgpu_fft_c2r_host(
            ffi.cast("const float *", ffi.from_buffer(cplx)),
            ffi.cast("float *", ffi.from_buffer(rec)),
            *shape,
        )
        assert rc3 == 0 and rc4 == 0
        st_second = py21cmcu.cuda_stats()
        assert st_second["cufft_plan_hits"] > 0
        assert st_second["cufft_plan_misses"] == 0
        assert st_first["cufft_plan_misses"] > 0

        # cleanup_cuda clears multi-GPU plan cache cleanly so the next call misses again
        py21cmcu.cleanup_cuda()
        py21cmcu.reset_cuda_stats()
        rc5 = lib.p21_cuda_mgpu_fft_r2c_host(
            ffi.cast("const float *", ffi.from_buffer(field)),
            ffi.cast("float *", ffi.from_buffer(cplx)),
            *shape,
        )
        assert rc5 == 0
        assert py21cmcu.cuda_stats()["cufft_plan_misses"] > 0
    finally:
        py21cmcu.set_cuda_devices(orig_devs or (0,))
        py21cmcu.cleanup_cuda()


@requires_cuda_runtime
def test_gpu_warm_batchpool_and_online_torch_field_stream(tmp_path: Path) -> None:
    """Verify warm GPU BatchPool across rounds and zero-disk PyTorch GPU tensor streaming."""
    cache_dir = tmp_path / "gpu_cache"
    with py21cmcu.config.use(direc=str(cache_dir)):
        base_inputs = _small_inputs(8601)
        tasks_r1 = [
            py21cmcu.BatchTask(
                task_id=f"g1_{i}",
                inputs=_small_inputs(8610 + i),
                mode="coeval",
                out_redshifts=(8.0,),
                ps_n_bins=6,
                instrument="SKA1-Low",
                compute_moments=True,
            )
            for i in range(2)
        ]
        tasks_r2 = [
            py21cmcu.BatchTask(
                task_id=f"g2_{i}",
                inputs=_small_inputs(8620 + i),
                mode="coeval",
                out_redshifts=(8.0,),
                ps_n_bins=6,
                instrument="SKA1-Low",
                compute_moments=True,
            )
            for i in range(2)
        ]
        with py21cmcu.BatchPool(
            devices=(0,),
            workers_per_gpu=1,
            threads_per_worker=1,
            use_cuda=True,
            in_process=False,
        ) as pool:
            res1 = pool.map(tasks_r1)
            res2 = pool.map(tasks_r2)
            assert all(r.ok for r in res1 + res2)
            assert {r.worker_pid for r in res2} <= {r.worker_pid for r in res1}

        try:
            import torch

            if torch.cuda.is_available():
                sim_field = py21cmcu.SBISimulator(
                    base_inputs=base_inputs,
                    param_names=("F_STAR10_ACG", "F_ESC10_ACG"),
                    mode="coeval",
                    out_redshift=8.0,
                    summary="field",
                    instrument="SKA1-Low",
                    use_cuda=True,
                    lazy_d2h=True,
                    as_torch=True,
                )
                ds = py21cmcu.OnlineSBIDataset(
                    sim_field,
                    {"F_STAR10_ACG": (-1.5, -1.1), "F_ESC10_ACG": (-1.3, -0.8)},
                    n_samples=2,
                    batch_size=2,
                    sampling="sobol",
                    seed=17,
                    as_torch=True,
                )
                th_t, x_t = next(iter(ds))
                torch.cuda.synchronize()
                assert th_t.is_cuda and x_t.is_cuda
                assert th_t.dtype == torch.float32 and x_t.dtype == torch.float32
                assert tuple(x_t.shape) == (2, 12, 12, 12)
                assert not list(cache_dir.glob("*.h5"))
                py21cmcu.cleanup_cuda()
        except ImportError:
            pass
