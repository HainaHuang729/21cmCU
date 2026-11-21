"""One-command batch runner for single- or multi-GPU concurrent lightcone execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from time import perf_counter

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from env_detect import detect_cuda_path, detect_native_deps


def build_arg_parser():
    """Construct the CLI parser for multi-worker lightcone batch runs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/batch_out"), help="Output parent directory")
    p.add_argument("--work", type=Path, default=Path("runs/batch_work"), help="Shared build and per-worker scratch directory")
    p.add_argument("--mode", choices=("gpu-density", "cpu"), default="gpu-density")
    p.add_argument("--tasks-json", type=Path, help="JSON file containing a list of task objects (seed, astro, cosmo, etc.)")
    p.add_argument("--seeds", help="Comma-separated random seeds (e.g. 1001,1002,1003,1004)")
    p.add_argument("--count", type=int, default=4, help="Number of sequential seeds to generate when --seeds/--tasks-json are unset")
    p.add_argument("--seed-start", type=int, default=424241, help="Starting random seed when using --count")
    p.add_argument("--concurrency", type=int, default=2, help="Number of concurrent worker processes")
    p.add_argument("--gpus", help="Optional comma-separated GPU IDs to round-robin via CUDA_VISIBLE_DEVICES")
    p.add_argument("--threads-per-worker", type=int, default=4, help="OpenMP/FFTW threads per worker")
    p.add_argument("--ic-threads", type=int, default=None, help="GSL IC thread count (defaults to --threads-per-worker)")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--hii-dim", type=int, default=96)
    p.add_argument("--box-len", type=float, default=375.0)
    p.add_argument("--max-z", type=float, default=14.0)
    p.add_argument("--window-max-z", default="14.0")
    p.add_argument("--skip-build", action="store_true", help="Reuse existing <work>/shared_build/package if present")
    p.add_argument("--gsl-prefix", help="Explicit GSL installation prefix (sets PS_GSL_PREFIX)")
    p.add_argument("--fftw-prefix", help="Explicit FFTW3f installation prefix (sets PS_FFTW_PREFIX)")
    p.add_argument("--cuda-path", help="Explicit CUDA toolkit path (sets CUDA_PATH)")
    return p


def _prepend_env_path(env, key, value):
    cur = env.get(key, "")
    parts = [p for p in cur.split(os.pathsep) if p] if cur else []
    val = str(value)
    if val not in parts:
        env[key] = f"{val}{os.pathsep}{cur}" if cur else val


def _load_tasks(args):
    if args.tasks_json is not None:
        raw = json.loads(args.tasks_json.resolve().read_text())
        if not isinstance(raw, list) or not raw:
            raise ValueError("--tasks-json must point to a non-empty JSON list")
        tasks = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"Task entry {idx} must be a JSON object")
            entry = dict(item)
            entry.setdefault("name", f"task_{idx:03d}")
            tasks.append(entry)
        return tasks

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if not seeds:
            raise ValueError("No valid seeds parsed from --seeds")
    else:
        if args.count <= 0:
            raise ValueError("--count must be positive")
        seeds = [args.seed_start + i for i in range(args.count)]

    return [{"name": f"seed_{seed}", "seed": seed} for seed in seeds]


def _query_vram_mib(gpu_targets):
    cmd = ["nvidia-smi"]
    if gpu_targets:
        cmd.extend(["-i", ",".join(gpu_targets)])
    cmd.extend(["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
    except OSError:
        return None, None
    if res.returncode != 0:
        return None, None
    used_list, total_list = [], []
    for line in res.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            used_list.append(int(parts[0]))
            total_list.append(int(parts[1]))
        except ValueError:
            continue
    if not used_list:
        return None, None
    return max(used_list), max(total_list)


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
            stop_event.wait(0.5)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    return (stop_event, t), stats


def _prepare_worker_build_link(worker_dir, shared_build):
    build_link = worker_dir / "build"
    if build_link.is_symlink() or build_link.is_file():
        build_link.unlink()
    elif build_link.is_dir():
        shutil.rmtree(build_link)
    build_link.symlink_to(shared_build)


def _run_one_task(idx, task, args, stage, base_env, out_dir, work_dir, shared_build, gpu_ids, ic_threads):
    name = str(task["name"])
    task_out = out_dir / name
    worker_dir = work_dir / f"w_{idx}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    _prepare_worker_build_link(worker_dir, shared_build)

    cmd = [
        sys.executable,
        str(stage / "run.py"),
        "--mode", args.mode,
        "--skip-build",
        "--out", str(task_out),
        "--work", str(worker_dir),
        "--dim", str(task.get("dim", args.dim)),
        "--hii-dim", str(task.get("hii_dim", args.hii_dim)),
        "--box-len", str(task.get("box_len", args.box_len)),
        "--max-z", str(task.get("max_z", args.max_z)),
        "--window-max-z", str(task.get("window_max_z", args.window_max_z)),
        "--threads", str(args.threads_per_worker),
        "--ic-threads", str(ic_threads),
    ]
    if "seed" in task and task["seed"] is not None:
        cmd.extend(["--seed", str(task["seed"])])
    if "point_index" in task and task["point_index"] is not None:
        cmd.extend(["--point-index", str(task["point_index"])])
    if task.get("astro") is not None:
        astro_arg = task["astro"] if isinstance(task["astro"], str) else json.dumps(task["astro"])
        cmd.extend(["--astro-json", astro_arg])
    if task.get("cosmo") is not None:
        cosmo_arg = task["cosmo"] if isinstance(task["cosmo"], str) else json.dumps(task["cosmo"])
        cmd.extend(["--cosmo-json", cosmo_arg])
    if task.get("control_ic"):
        cmd.extend(["--control-ic", str(task["control_ic"])])
    if args.gsl_prefix:
        cmd.extend(["--gsl-prefix", args.gsl_prefix])
    if args.fftw_prefix:
        cmd.extend(["--fftw-prefix", args.fftw_prefix])
    if args.cuda_path:
        cmd.extend(["--cuda-path", args.cuda_path])

    worker_env = dict(base_env)
    gpu_assigned = None
    if args.mode != "cpu" and gpu_ids:
        gpu_assigned = gpu_ids[idx % len(gpu_ids)]
        worker_env["CUDA_VISIBLE_DEVICES"] = gpu_assigned

    log_path = worker_dir / "worker.log"
    t0 = perf_counter()
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, env=worker_env, stdout=log_f, stderr=subprocess.STDOUT, check=False)
    wall = perf_counter() - t0

    rec = {
        "worker_index": idx,
        "name": name,
        "seed": task.get("seed"),
        "gpu": gpu_assigned,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "worker_wall_seconds": wall,
        "log": str(log_path),
    }
    timing_file = task_out / "timing.json"
    if timing_file.is_file():
        tdata = json.loads(timing_file.read_text())
        rec["seconds_to_sync"] = tdata.get("seconds_to_sync")
        rec["physics_seconds"] = tdata.get("physics_seconds")
        rec["slice_count"] = tdata.get("slice_count")
        rec["bytes"] = tdata.get("bytes")
    return rec


def main():
    args = build_arg_parser().parse_args()
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    stage = Path(__file__).resolve().parent
    root = Path(os.environ.get("PS_GPU_ROOT", stage.parent)).resolve()
    out_dir = args.out.resolve()
    work_dir = args.work.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = _load_tasks(args)
    seen_names = set()
    for task in tasks:
        name = str(task["name"])
        if name in seen_names:
            raise ValueError(f"Duplicate task name: {name}")
        seen_names.add(name)
        if (out_dir / name).exists():
            raise FileExistsError(f"Task output directory already exists: {out_dir / name}")

    base_env = os.environ.copy()
    if args.gsl_prefix:
        base_env["PS_GSL_PREFIX"] = str(Path(args.gsl_prefix).expanduser().resolve())
    if args.fftw_prefix:
        base_env["PS_FFTW_PREFIX"] = str(Path(args.fftw_prefix).expanduser().resolve())

    deps = detect_native_deps(root=root, gsl_prefix=args.gsl_prefix, fftw_prefix=args.fftw_prefix)
    missing = []
    if not deps["gsl_found"]:
        missing.append("GSL (gsl/gsl_integration.h)")
    if not deps["fftw_found"]:
        missing.append("FFTW3f (fftw3.h)")
    if missing:
        raise RuntimeError(
            "Missing native dependencies: "
            + ", ".join(missing)
            + ". Install via `conda env create -f environment.yml` or pass --gsl-prefix / --fftw-prefix."
        )

    sys_lib_dirs = {"/usr/local/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"}
    for lib_dir in reversed([d for d in deps["library_dirs"] if d not in sys_lib_dirs]):
        _prepend_env_path(base_env, "LD_LIBRARY_PATH", lib_dir)

    cuda_path, cuda_inc = detect_cuda_path(args.cuda_path)
    if cuda_path:
        base_env["CUDA_PATH"] = cuda_path
        if cuda_inc:
            _prepend_env_path(base_env, "CPATH", cuda_inc)
    base_env["PS_GPU_ROOT"] = str(root)
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["NUMEXPR_NUM_THREADS"] = "1"

    shared_build = work_dir / "shared_build"
    if not (args.skip_build and (shared_build / "package").is_dir()):
        if shared_build.exists():
            shutil.rmtree(shared_build)
        shared_build.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, str(stage / "build.py"), str(root), str(shared_build)],
            env=base_env,
            check=True,
        )

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else []
    if gpu_ids:
        vram_targets = gpu_ids
    else:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        vram_targets = [g.strip() for g in cvd.split(",") if g.strip() and g.strip() != "NoDevFiles"] if cvd else []

    ic_threads = args.ic_threads if args.ic_threads is not None else args.threads_per_worker
    monitor_handle, vram_stats = (
        (None, {}) if args.mode == "cpu" else _start_vram_monitor(vram_targets)
    )

    t_start = perf_counter()
    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = [
                pool.submit(
                    _run_one_task,
                    idx,
                    task,
                    args,
                    stage,
                    base_env,
                    out_dir,
                    work_dir,
                    shared_build,
                    gpu_ids,
                    ic_threads,
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
        "concurrency": args.concurrency,
        "threads_per_worker": args.threads_per_worker,
        "ic_threads": ic_threads,
        "dim": args.dim,
        "hii_dim": args.hii_dim,
        "box_len": args.box_len,
        "max_z": args.max_z,
        "window_max_z": args.window_max_z,
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
