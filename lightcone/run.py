"""One-command Wave 6 GPU, CPU baseline, and CPU/GPU verify runner for 21cmFAST-GPU."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from env_detect import detect_cuda_path, detect_native_deps

FIELDS = ("brightness_temp", "xH_box", "x_e_box", "Tk_box", "Ts_box", "dNrec_box")

WAVE6_GPU_ENV = {
    "PS_FFTW_PERTURB_DENSITY": "1",
    "PS_FFTW_HEAT_FILTER": "1",
    "PS_FFTW_BRIGHTNESS": "1",
    "PS_GSL_EXACT_POWER": "1",
    "PS_CONDITIONAL_MODE": "3",
    "PS_BATCH_FILTER": "1",
    "PS_XRAY_NATIVE_MODE": "4",
    "PS_NATIVE_SIGMA": "1",
    "PS_NATIVE_MHR": "1",
    "PS_NATIVE_COLLAPSE": "1",
    "PS_NATIVE_XRAY": "0",
    "PS_QAG_MODE": "deferred",
    "PS_GK61_COMPARE": "0",
    "PS_QAG_PROFILE": "0",
    "PS_CONDITIONAL_FULL_HIGH": "0",
    "PS_CONDITIONAL_NAN_SENTINEL": "1",
    "PS_GUARD_EPS_MULT": "32",
    "PS_XEREC_AMP_MULT": "0",
    "PS_LAZY_COEVAL_IC": "1",
    "PS_PARALLEL_LZF": "1",
    "PS_VERBOSE_HEAT": "0",
    "PS_REUSE_PREV_PERTURB": "1",
    "PS_FAST_LIGHTCONE": "1",
    "PS_CIC_PERTILE": "1",
    "PS_OMP_MHR": "1",
    "PS_CACHE_INIT_PS": "1",
    "PS_ASYNC_FFTW_PREFETCH": "1",
    "PS_FFTW_MEASURE": "0",
    "PS_FFTW_MEASURE_HEAT": "0",
}

CPU_BASELINE_ENV = {
    "PS_FFTW_PERTURB_DENSITY": "0",
    "PS_FFTW_HEAT_FILTER": "0",
    "PS_FFTW_BRIGHTNESS": "0",
    "PS_GSL_EXACT_POWER": "0",
    "PS_CONDITIONAL_MODE": "0",
    "PS_BATCH_FILTER": "0",
    "PS_XRAY_NATIVE_MODE": "0",
    "PS_NATIVE_SIGMA": "0",
    "PS_NATIVE_MHR": "0",
    "PS_NATIVE_COLLAPSE": "0",
    "PS_NATIVE_XRAY": "0",
    "PS_QAG_MODE": "deferred",
    "PS_GK61_COMPARE": "0",
    "PS_QAG_PROFILE": "0",
    "PS_LAZY_COEVAL_IC": "0",
    "PS_PARALLEL_LZF": "0",
    "PS_VERBOSE_HEAT": "0",
    "PS_REUSE_PREV_PERTURB": "0",
    "PS_FAST_LIGHTCONE": "0",
    "PS_CIC_PERTILE": "0",
    "PS_OMP_MHR": "0",
    "PS_CACHE_INIT_PS": "0",
    "PS_ASYNC_FFTW_PREFETCH": "0",
    "PS_FFTW_MEASURE": "0",
    "PS_FFTW_MEASURE_HEAT": "0",
}


def build_arg_parser():
    """Construct the CLI parser for one-command lightcone execution."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("gpu-density", "cpu", "verify"),
        default="gpu-density",
        help="Execution mode: GPU Wave 6, pure CPU baseline, or matched CPU+GPU verify",
    )
    p.add_argument("--out", type=Path, default=Path("runs/lc_out"), help="Output directory")
    p.add_argument("--work", type=Path, default=Path("runs/lc_work"), help="Build and scratch working directory")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--hii-dim", type=int, default=96)
    p.add_argument("--box-len", type=float, default=375.0)
    p.add_argument("--max-z", type=float, default=14.0)
    p.add_argument("--window-max-z", default="none")
    p.add_argument("--seed", type=int, help="Random seed override")
    p.add_argument("--point-dir", type=Path, help="Directory of complete numbered point configs (<index>.json)")
    p.add_argument("--point-index", type=int, help="Registered physical point index")
    p.add_argument("--astro-json", help="Astrophysics parameter checks (JSON string or .json file path)")
    p.add_argument("--cosmo-json", help="Cosmology parameter checks (JSON string or .json file path)")
    p.add_argument("--threads", type=int, default=8, help="OpenMP/FFTW thread count for lightcone evolution")
    p.add_argument("--ic-threads", type=int, default=None, help="GSL IC thread count (defaults to --threads)")
    p.add_argument("--control-ic", type=Path, help="Existing InitialConditions HDF5 file; generated if omitted")
    p.add_argument("--skip-build", action="store_true", help="Reuse existing <work>/build/package if present")
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


def _parse_json_or_file(raw):
    if raw is None:
        return None
    p = Path(raw).expanduser()
    text = p.read_text() if p.is_file() else raw
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    return data


def _generate_ic(base_env, ic_dir, cfg, dim, hii_dim, box_len, ic_threads):
    if ic_dir.exists():
        shutil.rmtree(ic_dir)
    ic_dir.mkdir(parents=True, exist_ok=True)
    ic_env = dict(base_env)
    ic_env.update(CPU_BASELINE_ENV)
    ic_env["CUDA_VISIBLE_DEVICES"] = ""
    ic_env["OMP_NUM_THREADS"] = str(ic_threads)
    ic_user = dict(cfg["user"])
    ic_user.update(
        DIM=int(dim),
        HII_DIM=int(hii_dim),
        BOX_LEN=float(box_len),
        N_THREADS=int(ic_threads),
    )
    ic_script = (
        "import sys, py21cmfast\n"
        "py21cmfast.initial_conditions(\n"
        f"    user_params={ic_user!r},\n"
        f"    cosmo_params={cfg['cosmo']!r},\n"
        f"    random_seed={int(cfg['seed'])!r},\n"
        "    direc=sys.argv[1],\n"
        "    regenerate=True,\n"
        "    write=True,\n"
        "    EVOLVE_DENSITY_LINEARLY=False,\n"
        "    Z_HEAT_MAX=35.0,\n"
        ")\n"
    )
    subprocess.run(
        [sys.executable, "-c", ic_script, str(ic_dir)],
        env=ic_env,
        check=True,
    )
    matches = sorted(ic_dir.glob("InitialConditions*.h5"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected 1 IC file in {ic_dir}, found {matches}")
    return matches[0]


def _run_lightcone(stage, base_env, sub_mode, out_dir, scratch_dir, common_env):
    run_env = dict(base_env)
    run_env.update(WAVE6_GPU_ENV if sub_mode != "cpu" else CPU_BASELINE_ENV)
    if sub_mode == "cpu":
        run_env["CUDA_VISIBLE_DEVICES"] = ""
    run_env.update(common_env)
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(stage / "run_large_lightcone.py"),
            sub_mode,
            str(out_dir),
            str(scratch_dir),
        ],
        env=run_env,
        check=True,
    )


def main():
    args = build_arg_parser().parse_args()
    stage = Path(__file__).resolve().parent
    root = Path(os.environ.get("PS_GPU_ROOT", stage.parent)).resolve()
    out_dir = args.out.resolve()
    work = args.work.resolve()

    if args.mode == "verify":
        for sub in ("cpu", "gpu"):
            if (out_dir / sub).exists():
                raise FileExistsError(f"Verify target already exists: {out_dir / sub}")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        if out_dir.exists():
            raise FileExistsError(f"Output directory already exists: {out_dir}")
        out_dir.parent.mkdir(parents=True, exist_ok=True)

    work.mkdir(parents=True, exist_ok=True)

    from point_parameters import resolve_point
    cfg = resolve_point(root, args.point_index, args.point_dir or root / "points")
    if args.seed is not None:
        cfg["seed"] = args.seed
    for group, raw in (("astro", args.astro_json), ("cosmo", args.cosmo_json)):
        for key, value in (_parse_json_or_file(raw) or {}).items():
            if key not in cfg[group] or value != cfg[group][key]:
                raise ValueError(f"{group}.{key} conflicts with registered point {cfg['index']}")

    base_env = os.environ.copy()
    if args.gsl_prefix:
        base_env["PS_GSL_PREFIX"] = str(Path(args.gsl_prefix).expanduser().resolve())
    if args.fftw_prefix:
        base_env["PS_FFTW_PREFIX"] = str(Path(args.fftw_prefix).expanduser().resolve())

    deps = detect_native_deps(
        root=root,
        gsl_prefix=args.gsl_prefix,
        fftw_prefix=args.fftw_prefix,
    )
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
    elif args.mode != "cpu":
        print("[env] Warning: CUDA_PATH not auto-detected; relying on CuPy defaults", file=sys.stderr)

    print(
        f"[env] GSL=ok FFTW3f=ok lib_dirs={deps['library_dirs'][:3]} CUDA_PATH={cuda_path or 'unset'}",
        flush=True,
    )

    base_env["PS_GPU_ROOT"] = str(root)
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["NUMEXPR_NUM_THREADS"] = "1"

    build_dir = work / "build"
    pkg_dir = build_dir / "package"
    if not (args.skip_build and pkg_dir.is_dir()):
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, str(stage / "build.py"), str(root), str(build_dir)],
            env=base_env,
            check=True,
        )

    _prepend_env_path(base_env, "PYTHONPATH", stage)
    _prepend_env_path(base_env, "PYTHONPATH", pkg_dir)

    gk61_path = work / "gk61.json"
    subprocess.run(
        [sys.executable, str(stage / "gk61_rule.py"), str(gk61_path)],
        env=base_env,
        check=True,
    )

    if args.control_ic is not None:
        control_ic = args.control_ic.resolve()
        if not control_ic.is_file():
            raise FileNotFoundError(f"Control IC not found: {control_ic}")
    else:
        ic_threads = args.ic_threads if args.ic_threads is not None else args.threads
        control_ic = _generate_ic(
            base_env,
            work / "ic",
            cfg,
            args.dim,
            args.hii_dim,
            args.box_len,
            ic_threads,
        )

    common_env = {
        "OMP_NUM_THREADS": str(args.threads),
        "PS_N_THREADS": str(args.threads),
        "PS_GK61_RULE": str(gk61_path),
        "PS_CONTROL_IC": str(control_ic),
        "PS_DIM": str(args.dim),
        "PS_HII_DIM": str(args.hii_dim),
        "PS_BOX_LEN": str(args.box_len),
        "PS_MAX_Z": str(args.max_z),
        "PS_WINDOW_MAX_Z": str(args.window_max_z),
        "PS_SEED": str(cfg["seed"]),
        "PS_POINT_INDEX": str(cfg["index"]),
        "PS_POINT_DIR": str((args.point_dir or root / "points").resolve()),
        "PS_ASTRO_JSON": json.dumps(cfg["astro"]),
        "PS_COSMO_JSON": json.dumps(cfg["cosmo"]),
    }

    if args.mode in ("gpu-density", "cpu"):
        _run_lightcone(stage, base_env, args.mode, out_dir, work / f"lc_scratch_{args.mode}", common_env)
        return

    cpu_out = out_dir / "cpu"
    gpu_out = out_dir / "gpu"
    _run_lightcone(stage, base_env, "cpu", cpu_out, work / "lc_scratch_cpu", common_env)
    _run_lightcone(stage, base_env, "gpu-density", gpu_out, work / "lc_scratch_gpu", common_env)

    comp_proc = subprocess.run(
        [sys.executable, str(stage / "attribute_lightcone.py"), str(out_dir), "gpu"],
        env=base_env,
        capture_output=True,
        text=True,
        check=True,
    )
    (out_dir / "comparison.json").write_text(comp_proc.stdout)

    comp = json.loads(comp_proc.stdout)
    cpu_t = json.loads((cpu_out / "timing.json").read_text())
    gpu_t = json.loads((gpu_out / "timing.json").read_text())
    cpu_s = float(cpu_t["seconds_to_sync"])
    gpu_s = float(gpu_t["seconds_to_sync"])
    speedup = cpu_s / gpu_s if gpu_s > 0 else 0.0
    out_of_tol = int(comp["any_field_out_of_tolerance"])
    per_field = {f: int(comp["counts"][f]) for f in FIELDS}
    failed_fields = sum(1 for v in per_field.values() if v > 0)
    print(
        f"VERIFY_SUMMARY: failed_fields={failed_fields}/6 any_field_out_of_tolerance={out_of_tol} "
        f"cpu_s={cpu_s:.2f} gpu_s={gpu_s:.2f} speedup={speedup:.2f}x per_field={json.dumps(per_field)}",
        flush=True,
    )
    if out_of_tol != 0 or failed_fields != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
