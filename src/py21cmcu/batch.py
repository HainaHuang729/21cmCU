"""High-throughput multi-GPU / concurrent batch runner for 21cmCU parameter sweeps and SBI."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Sequence
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


def _box_length_3d(inputs: Any) -> tuple[float, float, float]:
    L = float(inputs.simulation_options.BOX_LEN)
    ncf = float(inputs.simulation_options.NON_CUBIC_FACTOR)
    return (L, L, L * ncf)


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
        with py21cmcu.use_mode(task.cuda_mode, use_cuda=use_cuda):
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
                for c in coevals:
                    tb = c.brightness_temp
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

                ps_1d: dict[str, Any] = {
                    "redshifts": redshifts,
                    "spectra": ps1_list,
                }
                ps_2d = (
                    {"redshifts": redshifts, "spectra": ps2_list}
                    if task.compute_ps_2d
                    else None
                )
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
                ps_2d = None
                if task.compute_ps_2d:
                    lc_arr = np.asarray(lc.lightcones[ps_qty], dtype=np.float32)
                    cell = float(inputs.simulation_options.cell_size.value)
                    lx = lc_arr.shape[0] * cell
                    ly = lc_arr.shape[1] * cell
                    lz = lc_arr.shape[2] * cell
                    ps_2d = {
                        "redshifts": np.array(
                            [float(np.mean(lc.lightcone_redshifts))], dtype=np.float64
                        ),
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
        )


def _gpu_worker_main(
    device_id: int,
    use_cuda: bool,
    threads_per_worker: int | None,
    task_queue: Any,
    result_queue: Any,
) -> None:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    import py21cmcu

    try:
        while True:
            item = task_queue.get()
            if item is None:
                break
            idx, task = item
            res = _execute_single_task(
                task,
                device_id=device_id,
                use_cuda=use_cuda,
                threads_per_worker=threads_per_worker,
            )
            result_queue.put((idx, res))
    finally:
        py21cmcu.cleanup_cuda()


def save_batch_h5(path: str | Path, results: Sequence[BatchResult]) -> None:
    """Persist compact global histories and 1D/2D power spectra from BatchResult list to HDF5."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as fl:
        tasks_grp = fl.create_group("tasks")
        for res in results:
            tg = tasks_grp.create_group(res.task_id)
            tg.attrs["ok"] = bool(res.ok)
            tg.attrs["wall_s"] = float(res.wall_s)
            tg.attrs["device_id"] = int(res.device_id)
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


def run_batch(
    tasks: Sequence[BatchTask],
    *,
    devices: Sequence[int] = (0,),
    workers_per_gpu: int = 1,
    threads_per_worker: int | None = None,
    output_h5: str | Path | None = None,
    use_cuda: bool = True,
    in_process: bool | None = None,
) -> list[BatchResult]:
    """Execute a sequence of BatchTasks concurrently across one or more GPUs."""
    if not tasks:
        if output_h5 is not None:
            save_batch_h5(output_h5, [])
        return []
    if workers_per_gpu < 1:
        raise ValueError(f"workers_per_gpu must be >= 1, got {workers_per_gpu}")
    if not devices:
        devices = (0,)

    worker_devices = [int(d) for d in devices for _ in range(workers_per_gpu)]
    run_inline = (
        (len(worker_devices) == 1 and len(tasks) <= 1)
        if in_process is None
        else bool(in_process)
    )

    if run_inline:
        import py21cmcu

        try:
            ordered = [
                _execute_single_task(
                    t,
                    device_id=worker_devices[0],
                    use_cuda=use_cuda,
                    threads_per_worker=threads_per_worker,
                )
                for t in tasks
            ]
        finally:
            py21cmcu.cleanup_cuda()
    else:
        ctx = mp.get_context("spawn")
        task_q = ctx.Queue()
        res_q = ctx.Queue()
        for idx, t in enumerate(tasks):
            task_q.put((idx, t))
        for _ in worker_devices:
            task_q.put(None)

        procs = [
            ctx.Process(
                target=_gpu_worker_main,
                args=(dev, use_cuda, threads_per_worker, task_q, res_q),
            )
            for dev in worker_devices
        ]
        for p in procs:
            p.start()

        slots: list[BatchResult | None] = [None] * len(tasks)
        for _ in range(len(tasks)):
            idx, res = res_q.get()
            slots[idx] = res

        for p in procs:
            p.join()
        ordered = [r for r in slots if r is not None]

    if output_h5 is not None:
        save_batch_h5(output_h5, ordered)
    return ordered
