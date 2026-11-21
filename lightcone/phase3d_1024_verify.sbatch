#!/bin/bash
#SBATCH --job-name=ps-p3d-1024
#SBATCH --partition=a
#SBATCH --nodelist=a6
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu=1
#SBATCH --mem=96G
#SBATCH --time=00:45:00
set -euo pipefail
stage=$1
root=$(dirname "$stage")
out="$stage/results-$SLURM_JOB_ID"
scratch=$(mktemp -d /var/tmp/ps-phase3d-1024.XXXXXXXX)
mkdir -p "$out" "$scratch/build" "$scratch/work-prime" "$scratch/work-warm"
trap 'rm -rf "$scratch"' EXIT

df -h /var/tmp
test -f /var/tmp/ps-lightcone-pair-488043/cpu/lightcone.h5
test -f /var/tmp/ps-perturb-ic-matrix-fp64-20261120/cpu/cpu-initial-conditions.h5

export PS_GPU_ROOT="$root" LD_LIBRARY_PATH="$root/lib" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PS_FFTW_PERTURB_DENSITY=1 PS_FFTW_HEAT_FILTER=1 PS_GSL_EXACT_POWER=1
export PS_CONDITIONAL_MODE=3 PS_BATCH_FILTER=1
export PS_XRAY_NATIVE_MODE=4 PS_NATIVE_SIGMA=1 PS_NATIVE_MHR=1 PS_NATIVE_COLLAPSE=1 PS_NATIVE_XRAY=0
export PS_QAG_MODE=deferred PS_GK61_COMPARE=0 PS_QAG_PROFILE=0
export PS_CONDITIONAL_FULL_HIGH=0 PS_CONDITIONAL_NAN_SENTINEL=1 PS_GUARD_EPS_MULT=32
export PS_LAZY_COEVAL_IC=1 PS_PARALLEL_LZF=1 PS_VERBOSE_HEAT=0
export PS_REUSE_PREV_PERTURB=1 PS_FAST_LIGHTCONE=1 PS_CIC_PERTILE=1
export PS_OMP_MHR=1 PS_CACHE_INIT_PS=1 PS_ASYNC_FFTW_PREFETCH=1
export PS_FFTW_MEASURE=0 PS_FFTW_MEASURE_HEAT=0
export CUDA_PATH="$root/cuda118_experiment_headers"
export CPATH="$CUDA_PATH/include${CPATH:+:$CPATH}"
export XDG_CONFIG_HOME="$scratch/config"
export CUPY_CACHE_DIR="$scratch/cupy-1024"

export PS_DIM=1024 PS_HII_DIM=256 PS_BOX_LEN=250.0 PS_MAX_Z=35.0 PS_WINDOW_MAX_Z=none
export PS_CONTROL_IC=/var/tmp/ps-perturb-ic-matrix-fp64-20261120/cpu/cpu-initial-conditions.h5

python="$root/env310/bin/python"
"$python" "$stage/build.py" "$root" "$scratch/build" > "$out/build.log" 2>&1
export PYTHONPATH="$scratch/build/package:$stage"
export PS_GK61_RULE="$scratch/gk61.json"
"$python" "$stage/gk61_rule.py" "$PS_GK61_RULE"

/usr/bin/time -f '%e' -o "$out/prime-1024-wall.txt" \
    "$python" "$stage/run_large_lightcone.py" gpu-density "$scratch/prime_1024" "$scratch/work-prime"
cp "$scratch/prime_1024/timing.json" "$out/prime-1024-timing.json"

/usr/bin/time -f '%e' -o "$out/warm-1024-wall.txt" \
    "$python" "$stage/run_large_lightcone.py" gpu-density "$scratch/warm_1024" "$scratch/work-warm"
cp "$scratch/warm_1024/timing.json" "$out/warm-1024-timing.json"

"$python" "$stage/verify_phase3a.py" repeat "$scratch/prime_1024/lightcone.h5" "$scratch/warm_1024/lightcone.h5" "$out/repeat_equal.json"
rm -f "$scratch/prime_1024/lightcone.h5"

ln -s /var/tmp/ps-lightcone-pair-488043/cpu "$scratch/cpu"
"$python" "$stage/attribute_lightcone.py" "$scratch" warm_1024 > "$out/warm-1024-comparison.json"
"$python" "$stage/verify_phase3a.py" dump_failures "$scratch/cpu/lightcone.h5" "$scratch/warm_1024/lightcone.h5" "$out/warm-1024-failures.json"

echo "PHASE3D_1024_DONE $out"
