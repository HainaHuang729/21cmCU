"""Concurrent single- or multi-GPU batch runner for 21cmFAST 4.3.2 lightcones."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from time import perf_counter

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

DEFAULT_QUANTITIES = (
    "brightness_temp",
    "neutral_fraction",
    "xray_ionised_fraction",
    "kinetic_temp_neutral",
    "spin_temperature",
    "cumulative_recombinations",
)
NONTHERMAL_QUANTITIES = ("brightness_temp", "neutral_fraction")


def build_arg_parser():
    """Construct CLI parser for concurrent 4.3.2 GPU/CPU/verify lightcone batches."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/v4_batch_out"))
    p.add_argument("--work", type=Path, default=Path("runs/v4_batch_work"))
    p.add_argument("--mode", choices=("gpu", "cpu", "verify"), default="gpu")
    p.add_argument(
        "--source-model",
        choices=(
            "E-INTEGRAL",
            "CONST-ION-EFF",
            "CHMF-SAMPLER",
            "DEXM-ESF",
            "L-INTEGRAL",
        ),
        default="E-INTEGRAL",
    )
    p.add_argument(
        "--power-spectrum",
        choices=("EH", "BBKS", "EFSTATHIOU", "PEEBLES", "WHITE", "CLASS"),
        default="EH",
    )
    p.add_argument(
        "--filter",
        dest="matter_filter",
        choices=("spherical-tophat", "gaussian"),
        default="spherical-tophat",
    )
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--hii-dim", type=int, default=64)
    p.add_argument("--box-len", type=float, default=200.0)
    p.add_argument("--min-z", type=float, default=6.0)
    p.add_argument("--max-z", type=float, default=15.0)
    p.add_argument(
        "--perturb-on-high-res",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--smooth-density",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--thermal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--recomb-model",
        choices=("inhomogeneous", "homogeneous", "none"),
        default="inhomogeneous",
    )
    p.add_argument("--tasks-json", type=Path, help="JSON list of task overrides")
    p.add_argument("--seeds", help="Comma-separated seeds")
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--seed-start", type=int, default=424242)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--gpus", help="Comma-separated CUDA device IDs to round-robin")
    p.add_argument("--threads-per-worker", type=int, default=2)
    p.add_argument(
        "--compression",
        choices=("device-lzf", "parallel-lzf", "none"),
        default=None,
        help="HDF5 compression for GPU outputs (defaults to device-lzf for gpu, none for cpu)",
    )
    p.add_argument("--skip-prewarm", action="store_true")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_task-json", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_task-out", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_worker-scratch", type=Path, help=argparse.SUPPRESS)
    return p


def _load_tasks(args):
    if args.tasks_json is not None:
        raw = json.loads(args.tasks_json.resolve().read_text())
        if not isinstance(raw, list) or not raw:
            raise ValueError("--tasks-json must be a non-empty JSON list")
        tasks = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"Task {idx} must be a JSON object")
            entry = dict(item)
            entry.setdefault("name", f"task_{idx:03d}")
            entry.setdefault("seed", args.seed_start + idx)
            tasks.append(entry)
        return tasks

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if not seeds:
            raise ValueError("No valid seeds in --seeds")
    else:
        if args.count <= 0:
            raise ValueError("--count must be positive")
        seeds = [args.seed_start + i for i in range(args.count)]
    return [{"name": f"seed_{s}", "seed": s} for s in seeds]


def _query_vram_mib(gpu_targets):
    cmd = ["nvidia-smi"]
    if gpu_targets:
        cmd.extend(["-i", ",".join(gpu_targets)])
    cmd.extend(
        ["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    )
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, check=False
        )
    except OSError:
        return None, None
    if res.returncode != 0:
        return None, None
    used_list, total_list = [], []
    for line in res.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            used_list.append(int(parts[0]))
            total_list.append(int(parts[1]))
        except ValueError:
            continue
    return (max(used_list), max(total_list)) if used_list else (None, None)


def _start_vram_monitor(gpu_targets):
    if not shutil.which("nvidia-smi"):
        return None, {}
    stats = {"peak_vram_mib": 0, "total_vram_mib": 0}
    stop_event = threading.Event()

    def _poll():
        while not stop_event.is_set():
            used, total = _query_vram_mib(gpu_targets)
            if used is not None:
                stats["peak_vram_mib"] = max(stats["peak_vram_mib"], used)
                stats["total_vram_mib"] = max(stats["total_vram_mib"], total)
            stop_event.wait(0.25)

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    return (stop_event, thread), stats


def _build_inputs_and_lightconer(cfg):
    import py21cmfast as p21
    from py21cmfast.lightconers import RectilinearLightconer

    source = cfg["source_model"]
    lagrangian = source in ("CHMF-SAMPLER", "L-INTEGRAL", "DEXM-ESF")
    thermal = bool(cfg["thermal"])
    recomb = (
        cfg["recomb_model"]
        if (not lagrangian and thermal)
        else ("inhomogeneous" if thermal else "none")
    )
    if lagrangian:
        updates = dict(
            SOURCE_MODEL=source,
            POWER_SPECTRUM=cfg["power_spectrum"],
            FILTER=cfg["matter_filter"],
            DIM=int(cfg["dim"]),
            HII_DIM=int(cfg["hii_dim"]),
            BOX_LEN=float(cfg["box_len"]),
            N_THREADS=int(cfg["threads"]),
            KEEP_3D_VELOCITIES=True,
            PERTURB_ON_HIGH_RES=bool(cfg["perturb_on_high_res"]),
            SMOOTH_EVOLVED_DENSITY_FIELD=bool(cfg["smooth_density"]),
            USE_TS_FLUCT=thermal,
            RECOMB_MODEL=recomb,
            HII_FILTER="spherical-tophat",
            USE_EXP_FILTER=True,
            CELL_RECOMB=True,
            USE_METALLICITY=True,
            USE_UPPER_STELLAR_TURNOVER=source != "L-INTEGRAL",
        )
    else:
        updates = dict(
            SOURCE_MODEL=source,
            POWER_SPECTRUM=cfg["power_spectrum"],
            FILTER=cfg["matter_filter"],
            DIM=int(cfg["dim"]),
            HII_DIM=int(cfg["hii_dim"]),
            BOX_LEN=float(cfg["box_len"]),
            N_THREADS=int(cfg["threads"]),
            KEEP_3D_VELOCITIES=True,
            PERTURB_ON_HIGH_RES=bool(cfg["perturb_on_high_res"]),
            SMOOTH_EVOLVED_DENSITY_FIELD=bool(cfg["smooth_density"]),
            USE_TS_FLUCT=thermal,
            RECOMB_MODEL=recomb,
            HMF=cfg.get("hmf", "ST"),
            USE_FFTW_WISDOM=False,
            MINIMIZE_MEMORY=False,
            HII_FILTER="spherical-tophat",
            USE_EXP_FILTER=False,
            CELL_RECOMB=False,
            USE_METALLICITY=False,
            USE_UPPER_STELLAR_TURNOVER=False,
        )
    for group in (
        "simulation_options",
        "matter_options",
        "astro_options",
        "astro_params",
        "cosmo_params",
    ):
        if isinstance(cfg.get(group), dict):
            updates.update(cfg[group])

    if "node_redshifts" in cfg and cfg["node_redshifts"] is not None:
        node_redshifts = cfg["node_redshifts"]
    else:
        from py21cmfast.wrapper.inputs import get_logspaced_redshifts

        node_redshifts = get_logspaced_redshifts(
            min_redshift=float(cfg["min_z"]),
            max_redshift=float(cfg.get("z_heat_max", 35.0)),
            z_step_factor=float(cfg.get("z_step_factor", 1.04)),
        )
    inputs = p21.InputParameters(
        random_seed=int(cfg["seed"]),
        node_redshifts=node_redshifts,
    ).evolve_input_structs(**updates)

    quantities = cfg.get("quantities") or (
        list(DEFAULT_QUANTITIES) if thermal else list(NONTHERMAL_QUANTITIES)
    )
    lightconer = RectilinearLightconer.between_redshifts(
        min_redshift=float(cfg["min_z"]),
        max_redshift=float(cfg["max_z"]),
        quantities=tuple(quantities),
        resolution=inputs.simulation_options.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
    )
    return inputs, lightconer


def _run_worker_entry(args):
    import py21cmfast as p21
    from gpu_lightcone import run_lightcone
    from lightcone_io import save_lightcone

    cfg = json.loads(args._task_json.read_text())
    task_out = args._task_out
    task_out.mkdir(parents=True, exist_ok=True)
    scratch = args._worker_scratch
    scratch.mkdir(parents=True, exist_ok=True)

    inputs, lightconer = _build_inputs_and_lightconer(cfg)
    mode = cfg["mode"]
    gpu_compression = cfg["compression"]

    def run_side(side):
        diag_dir = task_out / f"diag_{side}"
        side_scratch = scratch / side
        side_scratch.mkdir(parents=True, exist_ok=True)
        t0 = perf_counter()
        if side == "gpu":
            cone = run_lightcone(
                inputs=inputs,
                lightconer=lightconer,
                diagnostics_dir=diag_dir,
                fallback_to_cpu=False,
                write=False,
                regenerate=True,
            )
            comp = gpu_compression
        else:
            cone = p21.run_lightcone(
                inputs=inputs,
                lightconer=lightconer,
                write=False,
                regenerate=True,
            )
            comp = "none"
        physics_s = perf_counter() - t0
        h5_path = task_out / ("lightcone.h5" if mode != "verify" else f"{side}.h5")
        io_stats = save_lightcone(
            cone,
            h5_path,
            clobber=True,
            compression=comp,
            fsync=True,
        )
        total_s = perf_counter() - t0
        return {
            "physics_seconds": physics_s,
            "write_seconds": io_stats["write_seconds"],
            "fsync_seconds": io_stats["fsync_seconds"],
            "seconds_to_sync": total_s,
            "bytes": io_stats["bytes"],
            "compression": io_stats["compression"],
            "chunks_written": io_stats["chunks_written"],
            "skipped_lzf_chunks": io_stats["skipped_lzf_chunks"],
            "slice_count": len(cone.lightcone_distances),
        }

    if mode == "verify":
        from compare_lightcone_ps import compare_saved_slices

        gpu_stats = run_side("gpu")
        cpu_stats = run_side("cpu")
        cmp_res = compare_saved_slices(task_out / "cpu.h5", task_out / "gpu.h5")
        failed_lc = sum(v["failed"] for v in cmp_res["lightcones"].values())
        failed_gq = sum(v["failed"] for v in cmp_res["global_quantities"].values())
        failed_ps = cmp_res["brightness_transverse_power"]["failed"]
        passed = failed_lc == 0 and failed_gq == 0 and failed_ps == 0
        timing = {
            **gpu_stats,
            "mode": mode,
            "cpu_seconds_to_sync": cpu_stats["seconds_to_sync"],
            "passed": passed,
            "failed_lightcone": failed_lc,
            "failed_global": failed_gq,
            "failed_transverse_ps": failed_ps,
        }
        (task_out / "comparison.json").write_text(json.dumps(cmp_res, indent=2) + "\n")
    else:
        side_stats = run_side(mode)
        timing = {**side_stats, "mode": mode, "passed": True}

    (task_out / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    if not timing["passed"]:
        raise SystemExit(f"Parity check failed for task {cfg['name']}")


def _launch_worker(
    idx, task, args, base_env, out_dir, work_dir, gpu_ids, gpu_compression
):
    name = str(task["name"])
    task_out = out_dir / name
    worker_dir = work_dir / f"w_{idx}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "name": name,
        "mode": args.mode,
        "source_model": args.source_model,
        "power_spectrum": args.power_spectrum,
        "matter_filter": args.matter_filter,
        "dim": int(task.get("dim", args.dim)),
        "hii_dim": int(task.get("hii_dim", args.hii_dim)),
        "box_len": float(task.get("box_len", args.box_len)),
        "min_z": float(task.get("min_z", args.min_z)),
        "max_z": float(task.get("max_z", args.max_z)),
        "perturb_on_high_res": bool(
            task.get("perturb_on_high_res", args.perturb_on_high_res)
        ),
        "smooth_density": bool(task.get("smooth_density", args.smooth_density)),
        "thermal": bool(task.get("thermal", args.thermal)),
        "recomb_model": task.get("recomb_model", args.recomb_model),
        "threads": int(task.get("threads", args.threads_per_worker)),
        "seed": int(task["seed"]),
        "compression": task.get("compression", gpu_compression),
        "node_redshifts": task.get("node_redshifts"),
        "z_step_factor": task.get("z_step_factor", 1.04),
        "z_heat_max": task.get("z_heat_max", 35.0),
        "quantities": task.get("quantities"),
        "simulation_options": task.get("simulation_options"),
        "matter_options": task.get("matter_options"),
        "astro_options": task.get("astro_options"),
        "astro_params": task.get("astro_params"),
        "cosmo_params": task.get("cosmo_params"),
    }
    task_json = worker_dir / "task.json"
    task_json.write_text(json.dumps(cfg, indent=2))

    worker_env = dict(base_env)
    worker_env["OMP_NUM_THREADS"] = str(cfg["threads"])
    gpu_assigned = None
    if args.mode == "cpu":
        worker_env["CUDA_VISIBLE_DEVICES"] = ""
    elif gpu_ids:
        gpu_assigned = gpu_ids[idx % len(gpu_ids)]
        worker_env["CUDA_VISIBLE_DEVICES"] = gpu_assigned

    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--_worker",
        "--_task-json",
        str(task_json),
        "--_task-out",
        str(task_out),
        "--_worker-scratch",
        str(worker_dir / "scratch"),
    ]
    log_path = worker_dir / "worker.log"
    t0 = perf_counter()
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd, env=worker_env, stdout=log_f, stderr=subprocess.STDOUT, check=False
        )
    wall = perf_counter() - t0
    rec = {
        "worker_index": idx,
        "name": name,
        "seed": cfg["seed"],
        "gpu": gpu_assigned,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "worker_wall_seconds": wall,
        "log": str(log_path),
    }
    timing_file = task_out / "timing.json"
    if timing_file.is_file():
        tdata = json.loads(timing_file.read_text())
        rec.update(tdata)
        if not tdata.get("passed", True):
            rec["ok"] = False
    return rec


def main():
    args = build_arg_parser().parse_args()
    if args._worker:
        _run_worker_entry(args)
        return

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    out_dir = args.out.resolve()
    work_dir = args.work.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    tasks = _load_tasks(args)
    seen = set()
    for task in tasks:
        name = str(task["name"])
        if name in seen:
            raise ValueError(f"Duplicate task name: {name}")
        seen.add(name)
        if (out_dir / name).exists():
            raise FileExistsError(f"Output directory already exists: {out_dir / name}")

    base_env = os.environ.copy()
    shared_cache = work_dir / "shared_cache"
    shared_cache.mkdir(parents=True, exist_ok=True)
    base_env.setdefault("P21_GPU_SHIM_CACHE", str(shared_cache / "native-shims"))
    base_env.setdefault("CUPY_CACHE_DIR", str(shared_cache / "cupy"))
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["NUMEXPR_NUM_THREADS"] = "1"
    base_env["PYTHONUNBUFFERED"] = "1"

    gpu_compression = args.compression or (
        "none" if args.mode == "cpu" else "device-lzf"
    )
    prewarm_seconds = 0.0
    if not args.skip_prewarm and args.mode != "cpu":
        pw_min = float(args.min_z)
        pw_lc_max = min(float(args.max_z), pw_min + 0.25)
        pw_node_hi = round(pw_lc_max + 0.5, 4)
        pw_node_mid = round(0.5 * (pw_node_hi + pw_min), 4)
        pw_node_lo = round(max(0.1, pw_min - 0.05), 4)
        prewarm_task = {
            "name": "_prewarm",
            "seed": tasks[0]["seed"],
            "dim": args.dim,
            "hii_dim": args.hii_dim,
            "box_len": args.box_len,
            "min_z": pw_min,
            "max_z": pw_lc_max,
            "node_redshifts": [pw_node_hi, pw_node_mid, pw_node_lo],
            "simulation_options": {"Z_HEAT_MAX": pw_node_hi},
            "compression": gpu_compression,
        }
        prewarm_dir = work_dir / "prewarm_out"
        prewarm_work = work_dir / "prewarm_work"
        t_pw = perf_counter()
        pw_rec = _launch_worker(
            -1,
            prewarm_task,
            argparse.Namespace(**{**vars(args), "mode": "gpu", "concurrency": 1}),
            base_env,
            prewarm_dir,
            prewarm_work,
            [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else [],
            gpu_compression,
        )
        prewarm_seconds = perf_counter() - t_pw
        if not pw_rec["ok"]:
            tail = ""
            log_p = Path(pw_rec["log"])
            if log_p.is_file():
                tail = "\n" + "\n".join(log_p.read_text().splitlines()[-30:])
            raise RuntimeError(f"Prewarm failed (see {pw_rec['log']}){tail}")
        shutil.rmtree(prewarm_dir, ignore_errors=True)
        shutil.rmtree(prewarm_work, ignore_errors=True)

    gpu_ids = (
        [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else []
    )
    if gpu_ids:
        vram_targets = gpu_ids
    else:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        vram_targets = (
            [
                g.strip()
                for g in cvd.split(",")
                if g.strip() and g.strip() != "NoDevFiles"
            ]
            if cvd
            else []
        )

    monitor_handle, vram_stats = (
        (None, {}) if args.mode == "cpu" else _start_vram_monitor(vram_targets)
    )
    run_work = Path(tempfile.mkdtemp(prefix="batch-", dir=work_dir))
    t_start = perf_counter()
    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [
                pool.submit(
                    _launch_worker,
                    idx,
                    task,
                    args,
                    base_env,
                    out_dir,
                    run_work,
                    gpu_ids,
                    gpu_compression,
                )
                for idx, task in enumerate(tasks)
            ]
            for fut in as_completed(futs):
                rec = fut.result()
                results.append(rec)
                print(
                    f"[batch {len(results)}/{len(tasks)}] name={rec['name']} ok={rec['ok']} "
                    f"wall={rec['worker_wall_seconds']:.2f}s sync={rec.get('seconds_to_sync')}",
                    flush=True,
                )
    finally:
        if monitor_handle is not None:
            stop_event, thread = monitor_handle
            stop_event.set()
            thread.join(timeout=2)

    wall_seconds = perf_counter() - t_start
    results.sort(key=lambda r: r["worker_index"])
    completed = sum(1 for r in results if r["ok"])
    failed = len(results) - completed
    throughput = (completed * 3600.0 / wall_seconds) if wall_seconds > 0 else 0.0

    summary = {
        "mode": args.mode,
        "source_model": args.source_model,
        "power_spectrum": args.power_spectrum,
        "filter": args.matter_filter,
        "concurrency": args.concurrency,
        "threads_per_worker": args.threads_per_worker,
        "dim": args.dim,
        "hii_dim": args.hii_dim,
        "box_len": args.box_len,
        "min_z": args.min_z,
        "max_z": args.max_z,
        "compression": gpu_compression,
        "prewarm_seconds": prewarm_seconds,
        "total_tasks": len(tasks),
        "completed": completed,
        "failed": failed,
        "wall_seconds": wall_seconds,
        "lightcones_per_hour": throughput,
        "peak_vram_mib": vram_stats.get("peak_vram_mib") or None,
        "total_vram_mib": vram_stats.get("total_vram_mib") or None,
        "workers": results,
    }
    (out_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"BATCH_SUMMARY: completed={completed}/{len(tasks)} concurrency={args.concurrency} "
        f"wall={wall_seconds:.2f}s rate={throughput:.1f}/h peak_vram_mib={summary['peak_vram_mib']}",
        flush=True,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
