"""One-command CLI wrapper for BT 21cmFAST GPU and CPU lightcone runs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from bt_config import resolve_bt_cosmo, split_bt_10d_params

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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="Output directory (created by run_large_lightcone.py)")
    p.add_argument("--work", type=Path, required=True, help="Build and scratch working directory")
    p.add_argument("--mode", choices=("gpu-density", "cpu"), default="gpu-density")
    p.add_argument("--params-json", type=Path, help="Optional JSON file with flat 10D BT or nested astro/cosmo params")
    p.add_argument("--kp-h-mpc", type=float, help="Break scale k_p in h/Mpc (converted to KP = kp_h_mpc * hlittle)")
    p.add_argument("--ms", type=float, help="Small-scale primordial spectral slope MS above k_p")
    p.add_argument("--seed", type=int, help="Random seed override")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--hii-dim", type=int, default=96)
    p.add_argument("--box-len", type=float, default=375.0)
    p.add_argument("--max-z", type=float, default=14.0)
    p.add_argument("--window-max-z", default="none")
    p.add_argument("--ic-threads", type=int, default=8, help="GSL IC thread count (use 16 to match 16-thread production RNG)")
    p.add_argument("--control-ic", type=Path, help="Pre-existing InitialConditions HDF5 path; generated if omitted")
    p.add_argument("--skip-build", action="store_true", help="Reuse existing <work>/build/package if present")
    return p


def _prepend_env_path(env, key, value):
    cur = env.get(key, "")
    env[key] = f"{value}:{cur}" if cur else str(value)


def main():
    args = build_arg_parser().parse_args()
    stage = Path(__file__).resolve().parent
    root = Path(os.environ.get("PS_GPU_ROOT", stage.parent)).resolve()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((root / "point_config.json").read_text())
    if args.params_json is not None:
        raw_params = json.loads(args.params_json.resolve().read_text())
        cfg = split_bt_10d_params(raw_params, cfg)

    cosmo, kp_h_mpc_eff = resolve_bt_cosmo(
        cfg["cosmo"], kp_h_mpc=args.kp_h_mpc, ms=args.ms
    )
    cfg["cosmo"] = cosmo
    cfg["user"]["POWER_SPECTRUM"] = 6
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    build_dir = work / "build"
    pkg_dir = build_dir / "package"
    base_env = os.environ.copy()
    base_env["PS_GPU_ROOT"] = str(root)
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env["OPENBLAS_NUM_THREADS"] = "1"
    base_env["MKL_NUM_THREADS"] = "1"
    base_env["NUMEXPR_NUM_THREADS"] = "1"
    if (root / "lib").is_dir():
        _prepend_env_path(base_env, "LD_LIBRARY_PATH", root / "lib")

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
        ic_dir = work / "ic"
        if ic_dir.exists():
            shutil.rmtree(ic_dir)
        ic_dir.mkdir(parents=True, exist_ok=True)
        ic_env = dict(base_env)
        ic_env.update(CPU_BASELINE_ENV)
        ic_env["CUDA_VISIBLE_DEVICES"] = ""
        ic_env["OMP_NUM_THREADS"] = str(args.ic_threads)
        ic_user = dict(cfg["user"])
        ic_user.update(
            DIM=int(args.dim),
            HII_DIM=int(args.hii_dim),
            BOX_LEN=float(args.box_len),
            POWER_SPECTRUM=6,
            N_THREADS=int(args.ic_threads),
        )
        ic_script = (
            "import json, sys, py21cmfast\n"
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
        ic_matches = sorted(ic_dir.glob("InitialConditions*.h5"))
        if len(ic_matches) != 1:
            raise RuntimeError(f"Expected 1 IC file in {ic_dir}, found {ic_matches}")
        control_ic = ic_matches[0]

    run_env = dict(base_env)
    run_env.update(WAVE6_GPU_ENV if args.mode == "gpu-density" else CPU_BASELINE_ENV)
    if args.mode == "cpu":
        run_env["CUDA_VISIBLE_DEVICES"] = ""
    run_env["OMP_NUM_THREADS"] = str(base_env.get("OMP_NUM_THREADS", "8"))
    run_env["PS_GK61_RULE"] = str(gk61_path)
    run_env["PS_CONTROL_IC"] = str(control_ic)
    run_env["PS_DIM"] = str(args.dim)
    run_env["PS_HII_DIM"] = str(args.hii_dim)
    run_env["PS_BOX_LEN"] = str(args.box_len)
    run_env["PS_MAX_Z"] = str(args.max_z)
    run_env["PS_WINDOW_MAX_Z"] = str(args.window_max_z)
    run_env["PS_SEED"] = str(cfg["seed"])
    run_env["PS_POINT_INDEX"] = str(cfg["index"])
    run_env["PS_ASTRO_JSON"] = json.dumps(cfg["astro"])
    run_env["PS_COSMO_JSON"] = json.dumps(cfg["cosmo"])
    run_env["PS_KP_H_MPC"] = repr(kp_h_mpc_eff)
    run_env["PS_MS"] = repr(cfg["cosmo"]["MS"])
    run_env["PS_IC_THREADS"] = str(args.ic_threads)

    scratch_dir = work / "lc_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(stage / "run_large_lightcone.py"),
            args.mode,
            str(args.out.resolve()),
            str(scratch_dir),
        ],
        env=run_env,
        check=True,
    )


if __name__ == "__main__":
    main()
