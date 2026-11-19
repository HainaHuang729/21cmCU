#!/bin/bash
#SBATCH --job-name=coeval_parallel_lzf
#SBATCH --partition=a
#SBATCH --nodelist=a6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G
#SBATCH --time=00:25:00
set -euo pipefail
stage=$1
root=$(dirname "$stage")
out="$stage/results-$SLURM_JOB_ID"
mkdir -p "$out"
scratch=$(mktemp -d "/var/tmp/coeval-lzf-$SLURM_JOB_ID.XXXXXXXX")
payload=$(mktemp -d "$stage/.payload-$SLURM_JOB_ID.XXXXXXXX")
trap 'rm -rf "$scratch" "$payload"' EXIT
export PS_GPU_ROOT="$root" LD_LIBRARY_PATH="$root/lib" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_PATH="${CUDA_PATH:-$root/cuda118_experiment_headers}"
export CPATH="$CUDA_PATH/include${CPATH:+:$CPATH}"
python="$root/env310/bin/python"
"$python" "$stage/launch_ps.py" "$scratch/warm" --cache "$scratch/cache" --seed
for mode in baseline paired; do
    opts=(--coeval-lzf-workers 8)
    if [[ $mode == paired ]]; then opts=(--coeval-lzf-workers 4 --paired-archive); fi
    /usr/bin/time -f '%e' -o "$out/$mode-wall-seconds.txt" "$python" "$stage/launch_ps.py" "$payload/$mode" \
        --cache "$scratch/cache" --seed --full-coeval --include-ic --gpu-full-ic --compression lzf "${opts[@]}"
    mkdir "$out/$mode"
    cp "$payload/$mode"/*.json "$out/$mode/"
    PYTHONPATH="$stage" "$python" - "$payload/$mode" > "$out/$mode/file-check.json" <<'PYTHON'
import hashlib, json
from pathlib import Path
import sys
import h5py
from gpu_check import verify
root=Path(sys.argv[1])
result=verify(root)
with h5py.File(root/"initial-conditions.h5","r") as file:
    checksums={}
    for key,dataset in file["InitialConditions"].items():
        digest=hashlib.sha256()
        for i in range(0,dataset.shape[0],8):
            digest.update(dataset[i:i+8].tobytes())
        checksums[key]=digest.hexdigest()
result["ic_sha256"]=checksums
print(json.dumps(result, indent=2))
PYTHON
    rm -r "$payload/$mode"
done
