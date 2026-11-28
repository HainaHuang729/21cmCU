"""High-throughput multi-GPU / concurrent batch runner and persistent pool for 21cmCU."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np


@dataclass(slots=True)
class BatchTask:
    """Specification of a single coeval or lightcone simulation + analysis task."""

    task_id: str
    inputs: Any
    mode: Literal["coeval", "lightcone"] = "coeval"
    out_redshifts: float | Sequence[float] | None = None
    min_redshift: float | None = None
    max_redshift: float | None = None
    cuda_mode: Literal["strict", "fast"] = "fast"
    apply_rsds: bool = False
    include_dvdr_in_tau21: bool = True
    n_rsd_subcells: int = 4
    quantities: tuple[str, ...] = ("brightness_temp",)
    ps_n_bins: int = 16
    compute_ps_2d: bool = False
    ps_2d_n_kperp: int = 10
    ps_2d_n_kpara: int = 10
    lightcone_chunk_depth: int | None = None
    extra_ps_kwargs: dict[str, Any] = field(default_factory=dict)
    wedge_filter: bool = False
    wedge_buffer: float = 0.0
    noise_sigma_mk: float | None = None
    lightcone_out_path: str | None = None
    lightcone_compression: str = "device-lzf"
    instrument: str | Any | None = None
    instrument_kwargs: dict[str, Any] = field(default_factory=dict)
    lazy_d2h: bool = True
    compute_moments: bool = False
    moments_n_bins: int = 16
    oblique_lightcone: bool = False
    oblique_direction: tuple[float, float, float] | None = None


@dataclass(slots=True)
class BatchResult:
    """In-memory scientific summary returned for one completed or failed BatchTask."""

    task_id: str
    ok: bool
    error: str | None
    wall_s: float
    device_id: int
    redshifts: np.ndarray
    global_xH: np.ndarray
    global_Tb: np.ndarray
    global_Ts: np.ndarray | None
    ps_1d: dict[str, Any]
    ps_2d: dict[str, Any] | None
    cuda_stats: dict[str, int]
    moments: dict[str, Any] | list[dict[str, Any]] | None = None
    worker_pid: int = 0


def _box_length_3d(inputs: Any) -> tuple[float, float, float]:
    L = float(inputs.simulation_options.BOX_LEN)
    ncf = float(inputs.simulation_options.NON_CUBIC_FACTOR)
    return (L, L, L * ncf)


def _apply_postprocessing(
    field: np.ndarray,
    box_len: tuple[float, float, float],
    redshift: float,
    task: BatchTask,
    *,
    cosmo: Any,
    seed: int,
    use_cuda: bool,
) -> np.ndarray:
    import py21cmcu

    out = field
    if task.instrument is not None:
        return py21cmcu.mock_observation_3d(
            out,
            box_len,
            redshift=redshift,
            instrument=task.instrument,
            los_axis=2,
            seed=seed,
            cosmo=cosmo,
            use_cuda=use_cuda,
            **task.instrument_kwargs,
        )
    if task.noise_sigma_mk is not None:
        out = py21cmcu.add_thermal_noise_3d(
            out,
            box_len,
            sigma_mk=task.noise_sigma_mk,
            redshift=redshift,
            seed=seed,
            use_cuda=use_cuda,
        )
    if task.wedge_filter:
        out = py21cmcu.apply_wedge_filter_3d(
            out,
            box_len,
            redshift,
            los_axis=2,
            buffer_h_mpc=task.wedge_buffer,
            cosmo=cosmo,
            use_cuda=use_cuda,
        )
    return out


def _execute_single_task(
    task: BatchTask,
    *,
    device_id: int,
    use_cuda: bool,
    threads_per_worker: int | None,
) -> BatchResult:
    import py21cmcu

    t0 = time.perf_counter()
    py21cmcu.reset_cuda_stats()
    inputs = task.inputs
    if threads_per_worker is not None:
        inputs = inputs.evolve_input_structs(N_THREADS=int(threads_per_worker))

    try:
        with py21cmcu.use_mode(task.cuda_mode, use_cuda=use_cuda, lazy_d2h=task.lazy_d2h):
            moments_out: dict[str, Any] | list[dict[str, Any]] | None = None
            if task.mode == "coeval":
                if task.out_redshifts is not None:
                    out_z = task.out_redshifts
                elif inputs.node_redshifts:
                    out_z = tuple(inputs.node_redshifts)
                else:
                    out_z = (8.0,)

                coevals = py21cmcu.run_coeval(
                    out_redshifts=out_z,
                    inputs=inputs,
                    regenerate=True,
                    write=False,
                )
                redshifts = np.array([float(c.redshift) for c in coevals], dtype=np.float64)
                global_xH = np.array(
                    [float(np.mean(c.neutral_fraction)) for c in coevals],
                    dtype=np.float64,
                )
                global_Tb = np.array(
                    [float(np.mean(c.brightness_temp)) for c in coevals],
                    dtype=np.float64,
                )
                if inputs.astro_options.USE_TS_FLUCT:
                    global_Ts = np.array(
                        [float(np.mean(c.ts_box.spin_temperature)) for c in coevals],
                        dtype=np.float64,
                    )
                else:
                    global_Ts = None

                box_len = _box_length_3d(inputs)
                ps1_list = []
                ps2_list = []
                mom_list = []
                for c in coevals:
                    z_c = float(c.redshift)
                    tb = np.asarray(c.brightness_temp, dtype=np.float32)
                    seed_c = int(inputs.random_seed) * 31 + int(round(z_c * 100))
                    tb = _apply_postprocessing(
                        tb,
                        box_len,
                        z_c,
                        task,
                        cosmo=inputs.cosmo_params.cosmo,
                        seed=seed_c,
                        use_cuda=use_cuda,
                    )
                    ps1 = py21cmcu.power_spectrum_1d(
                        tb,
                        box_len,
                        n_bins=task.ps_n_bins,
                        use_cuda=use_cuda,
                        **task.extra_ps_kwargs,
                    )
                    ps1_list.append(ps1)
                    if task.compute_ps_2d:
                        ps2 = py21cmcu.power_spectrum_2d(
                            tb,
                            box_len,
                            n_kperp=task.ps_2d_n_kperp,
                            n_kpara=task.ps_2d_n_kpara,
                            use_cuda=use_cuda,
                        )
                        ps2_list.append(ps2)
                    if task.compute_moments:
                        mom_list.append(
                            py21cmcu.field_moments_and_pdf(
                                tb, n_bins=task.moments_n_bins, use_cuda=use_cuda
                            )
                        )

                ps_1d: dict[str, Any] = {
                    "redshifts": redshifts,
                    "spectra": ps1_list,
                }
                ps_2d = (
                    {"redshifts": redshifts, "spectra": ps2_list}
                    if task.compute_ps_2d
                    else None
                )
                if task.compute_moments:
                    moments_out = mom_list
            elif task.mode == "lightcone":
                min_z = task.min_redshift
                max_z = task.max_redshift
                if min_z is None or max_z is None:
                    nz = sorted(float(z) for z in inputs.node_redshifts)
                    if len(nz) < 2:
                        raise ValueError(
                            "Lightcone BatchTask needs min_redshift/max_redshift or >=2 node_redshifts"
                        )
                    min_z = min_z if min_z is not None else nz[0] + 0.05
                    max_z = max_z if max_z is not None else nz[-1] - 0.05

                quantities = tuple(task.quantities)
                if task.oblique_lightcone:
                    if (task.apply_rsds or task.include_dvdr_in_tau21) and not inputs.matter_options.KEEP_3D_VELOCITIES:
                        inputs = inputs.evolve_input_structs(KEEP_3D_VELOCITIES=True)
                    ob_kwargs: dict[str, Any] = {}
                    if task.oblique_direction is not None:
                        ob_kwargs["direction"] = tuple(task.oblique_direction)
                    lightconer = py21cmcu.ObliqueRectilinearLightconer.between_redshifts(
                        min_redshift=float(min_z),
                        max_redshift=float(max_z),
                        resolution=inputs.simulation_options.cell_size,
                        quantities=quantities,
                        cosmo=inputs.cosmo_params.cosmo,
                        **ob_kwargs,
                    )
                else:
                    lightconer = py21cmcu.RectilinearLightconer.between_redshifts(
                        min_redshift=float(min_z),
                        max_redshift=float(max_z),
                        resolution=inputs.simulation_options.cell_size,
                        quantities=quantities,
                        cosmo=inputs.cosmo_params.cosmo,
                    )
                lc = py21cmcu.run_lightcone(
                    lightconer=lightconer,
                    inputs=inputs,
                    include_dvdr_in_tau21=task.include_dvdr_in_tau21,
                    apply_rsds=task.apply_rsds,
                    n_rsd_subcells=task.n_rsd_subcells,
                    regenerate=True,
                    write=False,
                )
                redshifts = np.asarray(lc.node_redshifts, dtype=np.float64)
                global_xH = np.asarray(
                    lc.global_quantities.get("neutral_fraction", np.array([])),
                    dtype=np.float64,
                )
                global_Tb = np.asarray(
                    lc.global_quantities.get("brightness_temp", np.array([])),
                    dtype=np.float64,
                )
                global_Ts = (
                    np.asarray(lc.global_quantities["spin_temperature"], dtype=np.float64)
                    if "spin_temperature" in lc.global_quantities
                    else None
                )
                ps_qty = (
                    "brightness_temp_with_rsds"
                    if task.apply_rsds and "brightness_temp_with_rsds" in lc.lightcones
                    else "brightness_temp"
                )
                if task.lightcone_out_path is not None:
                    py21cmcu.save_lightcone(
                        lc.lightcones,
                        task.lightcone_out_path,
                        compression=task.lightcone_compression,
                    )

                lc_arr = np.asarray(lc.lightcones[ps_qty], dtype=np.float32)
                cell = float(inputs.simulation_options.cell_size.value)
                lx = lc_arr.shape[0] * cell
                ly = lc_arr.shape[1] * cell
                mean_z = float(np.mean(lc.lightcone_redshifts))
                needs_postproc = (
                    task.instrument is not None
                    or task.noise_sigma_mk is not None
                    or task.wedge_filter
                )
                lc_arr = _apply_postprocessing(
                    lc_arr,
                    (lx, ly, lc_arr.shape[2] * cell),
                    mean_z,
                    task,
                    cosmo=inputs.cosmo_params.cosmo,
                    seed=int(inputs.random_seed) * 31 + 7,
                    use_cuda=use_cuda,
                )

                if not needs_postproc:
                    chunk_res = py21cmcu.lightcone_chunk_power_spectra(
                        lc,
                        quantity=ps_qty,
                        chunk_depth=task.lightcone_chunk_depth,
                        n_bins=task.ps_n_bins,
                        use_cuda=use_cuda,
                        **task.extra_ps_kwargs,
                    )
                    ps_1d = {
                        "redshifts": chunk_res["chunk_redshifts"],
                        "spectra": chunk_res["chunks"],
                    }
                else:
                    nz_tot = lc_arr.shape[2]
                    depth = (
                        int(task.lightcone_chunk_depth)
                        if task.lightcone_chunk_depth is not None
                        else max(1, nz_tot // 6)
                    )
                    chunk_z = []
                    chunk_specs = []
                    lc_redshifts = np.asarray(lc.lightcone_redshifts, dtype=np.float64)
                    for z0 in range(0, nz_tot, depth):
                        z1 = min(nz_tot, z0 + depth)
                        if z1 - z0 < 4:
                            continue
                        sub = np.ascontiguousarray(lc_arr[:, :, z0:z1])
                        chunk_z.append(float(np.mean(lc_redshifts[z0:z1])))
                        chunk_specs.append(
                            py21cmcu.power_spectrum_1d(
                                sub,
                                (lx, ly, (z1 - z0) * cell),
                                n_bins=task.ps_n_bins,
                                use_cuda=use_cuda,
                                **task.extra_ps_kwargs,
                            )
                        )
                    ps_1d = {
                        "redshifts": np.asarray(chunk_z, dtype=np.float64),
                        "spectra": chunk_specs,
                    }

                ps_2d = None
                if task.compute_ps_2d:
                    lz = lc_arr.shape[2] * cell
                    ps_2d = {
                        "redshifts": np.array([mean_z], dtype=np.float64),
                        "spectra": [
                            py21cmcu.power_spectrum_2d(
                                lc_arr,
                                (lx, ly, lz),
                                los_axis=2,
                                n_kperp=task.ps_2d_n_kperp,
                                n_kpara=task.ps_2d_n_kpara,
                                use_cuda=use_cuda,
                            )
                        ],
                    }
                if task.compute_moments:
                    moments_out = py21cmcu.field_moments_and_pdf(
                        lc_arr, n_bins=task.moments_n_bins, use_cuda=use_cuda
                    )
            else:
                raise ValueError(f"Unsupported BatchTask mode: {task.mode!r}")

            stats = py21cmcu.cuda_stats()
            wall_s = time.perf_counter() - t0
            return BatchResult(
                task_id=task.task_id,
                ok=True,
                error=None,
                wall_s=wall_s,
                device_id=device_id,
                redshifts=redshifts,
                global_xH=global_xH,
                global_Tb=global_Tb,
                global_Ts=global_Ts,
                ps_1d=ps_1d,
                ps_2d=ps_2d,
                cuda_stats=stats,
                moments=moments_out,
                worker_pid=os.getpid(),
            )
    except Exception:
        wall_s = time.perf_counter() - t0
        return BatchResult(
            task_id=task.task_id,
            ok=False,
            error=traceback.format_exc(),
            wall_s=wall_s,
            device_id=device_id,
            redshifts=np.array([], dtype=np.float64),
            global_xH=np.array([], dtype=np.float64),
            global_Tb=np.array([], dtype=np.float64),
            global_Ts=None,
            ps_1d={},
            ps_2d=None,
            cuda_stats={},
            moments=None,
            worker_pid=os.getpid(),
        )


def _persistent_worker_main(
    device_id: int,
    use_cuda: bool,
    threads_per_worker: int | None,
    clear_registry_every: int,
    task_queue: Any,
    result_queue: Any,
) -> None:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    import py21cmcu
    from py21cmfast.c_21cmfast import lib as _lib

    local_count = 0
    try:
        while True:
            item = task_queue.get()
            if item is None:
                break
            job_id, task = item
            res = _execute_single_task(
                task,
                device_id=device_id,
                use_cuda=use_cuda,
                threads_per_worker=threads_per_worker,
            )
            local_count += 1
            if use_cuda and clear_registry_every > 0 and (local_count % clear_registry_every == 0):
                _lib.p21_cuda_registry_clear()
            result_queue.put((job_id, res))
    finally:
        if use_cuda:
            py21cmcu.cleanup_cuda()


def _write_moments_group(grp: h5py.Group, mom: dict[str, Any]) -> None:
    for k in ("mean", "std", "variance", "skewness", "kurtosis"):
        if k in mom:
            grp.attrs[k] = float(mom[k])


def _read_moments_group(grp: h5py.Group) -> dict[str, Any]:
    return {
        k: float(grp.attrs[k])
        for k in ("mean", "std", "variance", "skewness", "kurtosis")
        if k in grp.attrs
    }


def save_batch_h5(path: str | Path, results: Sequence[BatchResult]) -> None:
    """Persist compact global histories, moments, and 1D/2D power spectra from BatchResult list."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as fl:
        tasks_grp = fl.create_group("tasks")
        for res in results:
            tg = tasks_grp.create_group(res.task_id)
            tg.attrs["ok"] = bool(res.ok)
            tg.attrs["wall_s"] = float(res.wall_s)
            tg.attrs["device_id"] = int(res.device_id)
            tg.attrs["worker_pid"] = int(res.worker_pid)
            if res.error is not None:
                tg.attrs["error"] = res.error
            tg.create_dataset("redshifts", data=res.redshifts)
            tg.create_dataset("global_xH", data=res.global_xH)
            tg.create_dataset("global_Tb", data=res.global_Tb)
            if res.global_Ts is not None:
                tg.create_dataset("global_Ts", data=res.global_Ts)

            if res.ps_1d and "spectra" in res.ps_1d:
                ps1_grp = tg.create_group("ps_1d")
                ps1_grp.create_dataset("redshifts", data=res.ps_1d["redshifts"])
                for i, spec in enumerate(res.ps_1d["spectra"]):
                    sg = ps1_grp.create_group(f"z_{i}")
                    for k, v in spec.items():
                        sg.create_dataset(k, data=v)

            if res.ps_2d and "spectra" in res.ps_2d:
                ps2_grp = tg.create_group("ps_2d")
                ps2_grp.create_dataset("redshifts", data=res.ps_2d["redshifts"])
                for i, spec in enumerate(res.ps_2d["spectra"]):
                    sg = ps2_grp.create_group(f"z_{i}")
                    for k, v in spec.items():
                        sg.create_dataset(k, data=v)

            if res.moments is not None:
                mg = tg.create_group("moments")
                if isinstance(res.moments, list):
                    mg.attrs["is_list"] = True
                    for i, mom in enumerate(res.moments):
                        _write_moments_group(mg.create_group(f"z_{i}"), mom)
                else:
                    mg.attrs["is_list"] = False
                    _write_moments_group(mg, res.moments)


def _load_batch_h5(path: Path) -> dict[str, BatchResult]:
    """Load completed BatchResult entries from a summary HDF5 checkpoint."""
    loaded: dict[str, BatchResult] = {}
    if not path.exists():
        return loaded
    with h5py.File(path, "r") as fl:
        if "tasks" not in fl:
            return loaded
        tgrp = fl["tasks"]
        for tid in tgrp:
            tg = tgrp[tid]
            ok = bool(tg.attrs.get("ok", False))
            if not ok:
                continue
            ps_1d: dict[str, Any] = {}
            if "ps_1d" in tg:
                p1g = tg["ps_1d"]
                specs = []
                i = 0
                while f"z_{i}" in p1g:
                    sg = p1g[f"z_{i}"]
                    specs.append({k: sg[k][:] for k in sg})
                    i += 1
                ps_1d = {"redshifts": p1g["redshifts"][:], "spectra": specs}
            ps_2d: dict[str, Any] | None = None
            if "ps_2d" in tg:
                p2g = tg["ps_2d"]
                specs2 = []
                i = 0
                while f"z_{i}" in p2g:
                    sg = p2g[f"z_{i}"]
                    specs2.append({k: sg[k][:] for k in sg})
                    i += 1
                ps_2d = {"redshifts": p2g["redshifts"][:], "spectra": specs2}
            moments: dict[str, Any] | list[dict[str, Any]] | None = None
            if "moments" in tg:
                mg = tg["moments"]
                if bool(mg.attrs.get("is_list", False)):
                    moms = []
                    i = 0
                    while f"z_{i}" in mg:
                        moms.append(_read_moments_group(mg[f"z_{i}"]))
                        i += 1
                    moments = moms
                else:
                    moments = _read_moments_group(mg)
            loaded[tid] = BatchResult(
                task_id=str(tid),
                ok=True,
                error=None,
                wall_s=float(tg.attrs.get("wall_s", 0.0)),
                device_id=int(tg.attrs.get("device_id", 0)),
                redshifts=tg["redshifts"][:],
                global_xH=tg["global_xH"][:],
                global_Tb=tg["global_Tb"][:],
                global_Ts=tg["global_Ts"][:] if "global_Ts" in tg else None,
                ps_1d=ps_1d,
                ps_2d=ps_2d,
                cuda_stats={},
                moments=moments,
                worker_pid=int(tg.attrs.get("worker_pid", 0)),
            )
    return loaded


class BatchPool:
    """Warm persistent multi-GPU / multi-worker pool for repeated SBI & ML sweeps.

    Keeps worker processes, CUDA contexts, ``cuFFT`` plan caches, and ``CLASS`` tables
    resident across multiple ``map`` / ``imap`` rounds without re-spawning processes.
    """

    def __init__(
        self,
        *,
        devices: Sequence[int] = (0,),
        workers_per_gpu: int = 1,
        threads_per_worker: int | None = None,
        use_cuda: bool = True,
        in_process: bool | None = None,
        clear_registry_every: int = 32,
    ) -> None:
        if workers_per_gpu < 1:
            raise ValueError(f"workers_per_gpu must be >= 1, got {workers_per_gpu}")
        if clear_registry_every < 0:
            raise ValueError(f"clear_registry_every must be >= 0, got {clear_registry_every}")

        dev_tuple = tuple(int(d) for d in devices) if devices else (0,)
        self.devices = dev_tuple
        self.workers_per_gpu = int(workers_per_gpu)
        self.threads_per_worker = threads_per_worker
        self.use_cuda = bool(use_cuda)
        self.clear_registry_every = int(clear_registry_every)
        self._worker_devices = [int(d) for d in dev_tuple for _ in range(self.workers_per_gpu)]
        self._in_process = (
            (len(self._worker_devices) == 1 and (not self.use_cuda or self._worker_devices[0] == 0))
            if in_process is None
            else bool(in_process)
        )
        self._closed = False
        self._tasks_executed = 0
        self._next_job_id = 0
        self._procs: list[mp.Process] = []
        self._task_q: Any = None
        self._result_q: Any = None

        if not self._in_process:
            ctx = mp.get_context("spawn")
            self._task_q = ctx.Queue()
            self._result_q = ctx.Queue()
            for dev in self._worker_devices:
                proc = ctx.Process(
                    target=_persistent_worker_main,
                    args=(
                        dev,
                        self.use_cuda,
                        self.threads_per_worker,
                        self.clear_registry_every,
                        self._task_q,
                        self._result_q,
                    ),
                )
                proc.start()
                self._procs.append(proc)

    @property
    def n_workers(self) -> int:
        return 1 if self._in_process else len(self._procs)

    @property
    def tasks_executed(self) -> int:
        return self._tasks_executed

    @property
    def is_closed(self) -> bool:
        return self._closed

    def imap(self, tasks: Sequence[BatchTask], *, ordered: bool = True) -> Iterator[BatchResult]:
        """Execute ``tasks`` on the warm pool and stream :class:`BatchResult` objects."""
        if self._closed:
            raise RuntimeError("BatchPool has already been closed")
        task_list = list(tasks)
        if not task_list:
            return

        if self._in_process:
            from py21cmfast.c_21cmfast import lib as _lib

            dev = self._worker_devices[0] if self._worker_devices else 0
            for t in task_list:
                res = _execute_single_task(
                    t,
                    device_id=dev,
                    use_cuda=self.use_cuda,
                    threads_per_worker=self.threads_per_worker,
                )
                self._tasks_executed += 1
                if (
                    self.use_cuda
                    and self.clear_registry_every > 0
                    and (self._tasks_executed % self.clear_registry_every == 0)
                ):
                    _lib.p21_cuda_registry_clear()
                yield res
            return

        job_ids: list[int] = []
        for t in task_list:
            jid = self._next_job_id
            self._next_job_id += 1
            job_ids.append(jid)
            self._task_q.put((jid, t))

        if not ordered:
            pending_set = set(job_ids)
            while pending_set:
                jid, res = self._result_q.get()
                pending_set.discard(jid)
                self._tasks_executed += 1
                yield res
            return

        ready: dict[int, BatchResult] = {}
        next_idx = 0
        while next_idx < len(job_ids):
            target_jid = job_ids[next_idx]
            while target_jid not in ready:
                jid, res = self._result_q.get()
                ready[jid] = res
            yield ready.pop(target_jid)
            self._tasks_executed += 1
            next_idx += 1

    def map(
        self,
        tasks: Sequence[BatchTask],
        *,
        output_h5: str | Path | None = None,
        checkpoint_h5: str | Path | None = None,
        resume: bool = False,
    ) -> list[BatchResult]:
        """Execute ``tasks`` on the warm pool with optional resumable HDF5 checkpointing."""
        if self._closed:
            raise RuntimeError("BatchPool has already been closed")
        ckpt = (
            Path(checkpoint_h5)
            if checkpoint_h5 is not None
            else (Path(output_h5) if output_h5 is not None else None)
        )
        prior: dict[str, BatchResult] = _load_batch_h5(ckpt) if (resume and ckpt is not None) else {}
        pending_tasks = [t for t in tasks if t.task_id not in prior]

        ran = list(self.imap(pending_tasks, ordered=True)) if pending_tasks else []
        by_id = dict(prior)
        for res in ran:
            by_id[res.task_id] = res
        ordered = [by_id[t.task_id] for t in tasks]

        if checkpoint_h5 is not None:
            save_batch_h5(checkpoint_h5, ordered)
        if output_h5 is not None and (checkpoint_h5 is None or Path(output_h5) != ckpt):
            save_batch_h5(output_h5, ordered)
        return ordered

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._procs:
            for _ in self._procs:
                self._task_q.put(None)
            for proc in self._procs:
                proc.join()
            self._procs.clear()
        elif self.use_cuda:
            import py21cmcu

            py21cmcu.cleanup_cuda()

    def __enter__(self) -> BatchPool:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def run_batch(
    tasks: Sequence[BatchTask],
    *,
    devices: Sequence[int] = (0,),
    workers_per_gpu: int = 1,
    threads_per_worker: int | None = None,
    output_h5: str | Path | None = None,
    checkpoint_h5: str | Path | None = None,
    resume: bool = False,
    use_cuda: bool = True,
    in_process: bool | None = None,
    pool: BatchPool | None = None,
) -> list[BatchResult]:
    """Execute a sequence of BatchTasks concurrently across one or more GPUs."""
    if pool is not None:
        return pool.map(
            tasks,
            output_h5=output_h5,
            checkpoint_h5=checkpoint_h5,
            resume=resume,
        )

    with BatchPool(
        devices=devices,
        workers_per_gpu=workers_per_gpu,
        threads_per_worker=threads_per_worker,
        use_cuda=use_cuda,
        in_process=in_process,
    ) as local_pool:
        return local_pool.map(
            tasks,
            output_h5=output_h5,
            checkpoint_h5=checkpoint_h5,
            resume=resume,
        )


__all__ = ["BatchPool", "BatchResult", "BatchTask", "run_batch", "save_batch_h5"]
