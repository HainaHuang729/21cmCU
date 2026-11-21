#!/bin/bash
#SBATCH --job-name=ps-p3d-hold
#SBATCH --partition=a
#SBATCH --nodelist=a6
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu=1
#SBATCH --mem=36G
#SBATCH --time=00:30:00
set -euo pipefail
stage=$1
root=$(dirname "$stage")
out="$stage/results-$SLURM_JOB_ID"
scratch=$(mktemp -d /var/tmp/ps-phase3d-hold.XXXXXXXX)
mkdir -p "$out" "$scratch/build"
trap 'rm -rf "$scratch"' EXIT

export PS_GPU_ROOT="$root" LD_LIBRARY_PATH="$root/lib" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_PATH="$root/cuda118_experiment_headers"
export CPATH="$CUDA_PATH/include${CPATH:+:$CPATH}"
export XDG_CONFIG_HOME="$scratch/config"
export CUPY_CACHE_DIR="$scratch/cupy-hold"

python="$root/env310/bin/python"
"$python" "$stage/build.py" "$root" "$scratch/build" > "$out/build.log" 2>&1
export PYTHONPATH="$scratch/build/package:$stage"
export PS_GK61_RULE="$scratch/gk61.json"
"$python" "$stage/gk61_rule.py" "$PS_GK61_RULE"

run_holdout() {
    local tag="$1" dim="$2" hii_dim="$3" box_len="$4" max_z="$5" win_z="$6" seed="$7" point_idx="$8" astro_json="$9"
    local hdir="$scratch/$tag"
    mkdir -p "$hdir/ic" "$hdir/work-cpu" "$hdir/work-prime" "$hdir/work-warm"

    export PS_DIM="$dim" PS_HII_DIM="$hii_dim" PS_BOX_LEN="$box_len" PS_MAX_Z="$max_z" PS_WINDOW_MAX_Z="$win_z"
    export PS_SEED="$seed" PS_POINT_INDEX="$point_idx" PS_ASTRO_JSON="$astro_json"

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

    # Full Wave 4 GPU
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

run_holdout h1_256_64 256 64 200.0 20.0 none 424242 90001 '{"F_STAR10": -0.85, "F_ESC10": -0.95, "M_TURN": 8.9, "L_X": 40.2, "NU_X_THRESH": 600.0}'
run_holdout h2_384_96 384 96 300.0 14.0 14.0 987654321 90002 '{"F_STAR10": -1.25, "F_ESC10": -1.20, "M_TURN": 9.1, "L_X": 38.8, "NU_X_THRESH": 800.0}'

echo "PHASE3D_HOLDOUT_DONE $out"
