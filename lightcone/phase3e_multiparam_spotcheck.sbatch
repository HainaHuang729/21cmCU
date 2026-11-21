#!/bin/bash
#SBATCH --job-name=ps-p3e-spot
#SBATCH --partition=a
#SBATCH --nodelist=a6
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --time=00:45:00
set -euo pipefail
stage=$1
root=$(dirname "$stage")
out="$stage/results-$SLURM_JOB_ID"
scratch=$(mktemp -d /var/tmp/ps-phase3e-spot.XXXXXXXX)
mkdir -p "$out" "$scratch/build"
trap 'rm -rf "$scratch"' EXIT

export PS_GPU_ROOT="$root" LD_LIBRARY_PATH="$root/lib" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_PATH="$root/cuda118_experiment_headers"
export CPATH="$CUDA_PATH/include${CPATH:+:$CPATH}"
export XDG_CONFIG_HOME="$scratch/config"
export CUPY_CACHE_DIR="$scratch/cupy-spot"

python="$root/env310/bin/python"
"$python" "$stage/build.py" "$root" "$scratch/build" > "$out/build.log" 2>&1
export PYTHONPATH="$scratch/build/package:$stage"
export PS_GK61_RULE="$scratch/gk61.json"
"$python" "$stage/gk61_rule.py" "$PS_GK61_RULE"

run_spotcheck() {
    local tag="$1" dim="$2" hii_dim="$3" box_len="$4" max_z="$5" win_z="$6" seed="$7" point_idx="$8" astro_json="$9" cosmo_json="${10}"
    local hdir="$scratch/$tag"
    mkdir -p "$hdir/ic" "$hdir/work-cpu" "$hdir/work-prime" "$hdir/work-warm"

    export PS_DIM="$dim" PS_HII_DIM="$hii_dim" PS_BOX_LEN="$box_len" PS_MAX_Z="$max_z" PS_WINDOW_MAX_Z="$win_z"
    export PS_SEED="$seed" PS_POINT_INDEX="$point_idx" PS_ASTRO_JSON="$astro_json" PS_COSMO_JSON="$cosmo_json"

    # Record full effective spot-check parameters for auditability
    "$python" - "$out/params-${tag}.json" "$tag" <<'PARAM_PY'
import json, os, sys
from pathlib import Path

cfg = json.loads((Path(os.environ["PS_GPU_ROOT"]) / "point_config.json").read_text())
astro_ov = json.loads(os.environ["PS_ASTRO_JSON"]) if os.environ.get("PS_ASTRO_JSON") else {}
cosmo_ov = json.loads(os.environ["PS_COSMO_JSON"]) if os.environ.get("PS_COSMO_JSON") else {}
cosmo_eff = dict(cfg["cosmo"]); cosmo_eff.update(cosmo_ov)
astro_eff = dict(cfg["astro"]); astro_eff.update(astro_ov)
rec = {
    "tag": sys.argv[2],
    "dim": int(os.environ["PS_DIM"]),
    "hii_dim": int(os.environ["PS_HII_DIM"]),
    "box_len": float(os.environ["PS_BOX_LEN"]),
    "max_z": float(os.environ["PS_MAX_Z"]),
    "win_z": os.environ["PS_WINDOW_MAX_Z"],
    "seed": int(os.environ["PS_SEED"]),
    "point_idx": int(os.environ["PS_POINT_INDEX"]),
    "astro_overrides": astro_ov,
    "cosmo_overrides": cosmo_ov,
    "effective_astro": astro_eff,
    "effective_cosmo": cosmo_eff,
}
Path(sys.argv[1]).write_text(json.dumps(rec, indent=2) + "\n")
PARAM_PY

    CUDA_VISIBLE_DEVICES="" PS_OMP_MHR=0 PS_CACHE_INIT_PS=0 PS_FAST_LIGHTCONE=0 PS_REUSE_PREV_PERTURB=0 PS_PARALLEL_LZF=0 PS_ASYNC_FFTW_PREFETCH=0 PS_LAZY_COEVAL_IC=0 \
    "$python" - "$hdir/ic" <<'INNER_PY'
import json, os, sys
from pathlib import Path
import py21cmfast

cfg = json.loads((Path(os.environ["PS_GPU_ROOT"]) / "point_config.json").read_text())
cfg["user"].update(
    DIM=int(os.environ["PS_DIM"]),
    HII_DIM=int(os.environ["PS_HII_DIM"]),
    BOX_LEN=float(os.environ["PS_BOX_LEN"]),
)
if os.environ.get("PS_COSMO_JSON"):
    cfg["cosmo"].update(json.loads(os.environ["PS_COSMO_JSON"]))
seed = int(os.environ["PS_SEED"])
ic = py21cmfast.initial_conditions(
    user_params=cfg["user"],
    cosmo_params=cfg["cosmo"],
    random_seed=seed,
    direc=sys.argv[1],
    regenerate=True,
    write=True,
    EVOLVE_DENSITY_LINEARLY=False,
    Z_HEAT_MAX=35.,
)
INNER_PY
    export PS_CONTROL_IC="$(printf '%s' "$hdir/ic/"InitialConditions*.h5)"
    test -f "$PS_CONTROL_IC"

    # Authentic CPU baseline
    CUDA_VISIBLE_DEVICES="" \
    PS_FFTW_PERTURB_DENSITY=0 PS_FFTW_HEAT_FILTER=0 PS_GSL_EXACT_POWER=0 \
    PS_CONDITIONAL_MODE=0 PS_BATCH_FILTER=0 \
    PS_XRAY_NATIVE_MODE=0 PS_NATIVE_SIGMA=0 PS_NATIVE_MHR=0 PS_NATIVE_COLLAPSE=0 PS_NATIVE_XRAY=0 \
    PS_QAG_MODE=deferred PS_GK61_COMPARE=0 PS_QAG_PROFILE=0 \
    PS_LAZY_COEVAL_IC=0 PS_PARALLEL_LZF=0 PS_VERBOSE_HEAT=0 \
    PS_REUSE_PREV_PERTURB=0 PS_FAST_LIGHTCONE=0 PS_CIC_PERTILE=0 \
    PS_OMP_MHR=0 PS_CACHE_INIT_PS=0 PS_ASYNC_FFTW_PREFETCH=0 \
    PS_FFTW_MEASURE=0 PS_FFTW_MEASURE_HEAT=0 \
    /usr/bin/time -f '%e' -o "$out/cpu-${tag}-wall.txt" \
        "$python" "$stage/run_large_lightcone.py" cpu "$hdir/cpu" "$hdir/work-cpu"
    cp "$hdir/cpu/timing.json" "$out/cpu-${tag}-timing.json"

    # Full Wave 6 GPU
    export PS_FFTW_PERTURB_DENSITY=1 PS_FFTW_HEAT_FILTER=1 PS_GSL_EXACT_POWER=1
    export PS_CONDITIONAL_MODE=3 PS_BATCH_FILTER=1
    export PS_XRAY_NATIVE_MODE=4 PS_NATIVE_SIGMA=1 PS_NATIVE_MHR=1 PS_NATIVE_COLLAPSE=1 PS_NATIVE_XRAY=0
    export PS_QAG_MODE=deferred PS_GK61_COMPARE=0 PS_QAG_PROFILE=0
    export PS_CONDITIONAL_FULL_HIGH=0 PS_CONDITIONAL_NAN_SENTINEL=1 PS_GUARD_EPS_MULT=32
    export PS_LAZY_COEVAL_IC=1 PS_PARALLEL_LZF=1 PS_VERBOSE_HEAT=0
    export PS_REUSE_PREV_PERTURB=1 PS_FAST_LIGHTCONE=1 PS_CIC_PERTILE=1
    export PS_OMP_MHR=1 PS_CACHE_INIT_PS=1 PS_ASYNC_FFTW_PREFETCH=1
    export PS_FFTW_MEASURE=0 PS_FFTW_MEASURE_HEAT=0

    /usr/bin/time -f '%e' -o "$out/prime-${tag}-wall.txt" \
        "$python" "$stage/run_large_lightcone.py" gpu-density "$hdir/prime_${tag}" "$hdir/work-prime"
    cp "$hdir/prime_${tag}/timing.json" "$out/prime-${tag}-timing.json"

    /usr/bin/time -f '%e' -o "$out/warm-${tag}-wall.txt" \
        "$python" "$stage/run_large_lightcone.py" gpu-density "$hdir/warm_${tag}" "$hdir/work-warm"
    cp "$hdir/warm_${tag}/timing.json" "$out/warm-${tag}-timing.json"

    "$python" "$stage/verify_phase3a.py" repeat "$hdir/prime_${tag}/lightcone.h5" "$hdir/warm_${tag}/lightcone.h5" "$out/repeat-${tag}.json"
    "$python" "$stage/attribute_lightcone.py" "$hdir" "warm_${tag}" > "$out/warm-${tag}-comparison.json"
    "$python" "$stage/verify_phase3a.py" dump_failures "$hdir/cpu/lightcone.h5" "$hdir/warm_${tag}/lightcone.h5" "$out/warm-${tag}-failures.json"
    rm -rf "$hdir"
}

run_spotcheck p1_early_eor_strong_xray 256 64 200.0 20.0 none 111111 91001 \
    '{"F_STAR10": -0.60, "ALPHA_STAR": 0.25, "F_ESC10": -0.45, "ALPHA_ESC": -0.50, "M_TURN": 8.30, "t_STAR": 0.15, "L_X": 41.60, "NU_X_THRESH": 200.0}' \
    '{"KP": 6.78, "MS": 1.20}'

run_spotcheck p2_late_eor_weak_hard_xray 256 64 250.0 25.0 none 222222 91002 \
    '{"F_STAR10": -1.65, "ALPHA_STAR": 0.80, "F_ESC10": -1.40, "ALPHA_ESC": 0.20, "M_TURN": 10.20, "t_STAR": 0.80, "L_X": 38.20, "NU_X_THRESH": 1400.0}' \
    '{"KP": 19.662, "MS": 3.85}'

run_spotcheck p3_cosmo_shift_lcdm_low_kp 256 64 180.0 20.0 none 333333 91003 \
    '{"F_STAR10": -1.10, "ALPHA_STAR": -0.35, "F_ESC10": -0.90, "ALPHA_ESC": -0.10, "M_TURN": 8.80, "t_STAR": 0.40, "L_X": 40.50, "NU_X_THRESH": 500.0}' \
    '{"OMm": 0.325, "OMb": 0.0505, "hlittle": 0.665, "SIGMA_8": 0.840, "POWER_INDEX": 0.955, "KP": 1.33, "MS": 2.10}'

run_spotcheck p4_cosmo_shift_low_s8 256 64 220.0 20.0 none 444444 91004 \
    '{"F_STAR10": -0.15, "ALPHA_STAR": -0.30, "F_ESC10": -1.35, "ALPHA_ESC": -0.22, "M_TURN": 10.60, "t_STAR": 0.90, "L_X": 41.70, "NU_X_THRESH": 180.0}' \
    '{"OMm": 0.290, "OMb": 0.0465, "hlittle": 0.700, "SIGMA_8": 0.785, "POWER_INDEX": 0.980, "KP": 18.06, "MS": 3.80}'

run_spotcheck p5_384_96_mid_eor_cosmo 384 96 300.0 16.0 16.0 555555 91005 \
    '{"F_STAR10": -0.95, "ALPHA_STAR": 0.60, "F_ESC10": -0.80, "ALPHA_ESC": 0.35, "M_TURN": 8.65, "t_STAR": 0.12, "L_X": 40.80, "NU_X_THRESH": 350.0}' \
    '{"OMm": 0.315, "OMb": 0.0492, "hlittle": 0.672, "SIGMA_8": 0.828, "POWER_INDEX": 0.962, "KP": 10.50, "MS": 0.90}'

echo "PHASE3E_SPOTCHECK_DONE $out"
