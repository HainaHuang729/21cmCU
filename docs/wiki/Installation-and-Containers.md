# Installation & Containers

Back to [Home](Home.md) · Next: [Quickstart](Quickstart.md)

## System Prerequisites

- **OS**: Linux x86_64 (GPU runtime requires NVIDIA Driver + CUDA Toolkit `nvcc`; CPU-only build also works on Linux/macOS)
- **Python**: `>= 3.12` **with development headers** (`Python.h` from `python3.12-dev` or Conda)
- **C/C++ Toolchain & Libraries**: `gcc`/`g++`, GNU Scientific Library (`libgsl-dev`), FFTW3 single-precision (`libfftw3-dev` providing `libfftw3f`), `pkg-config`
- **Python runtime packages** (installed automatically via `pyproject.toml`): `numpy>=2.0`, `scipy`, `astropy>=2.0`, `h5py>=2.8.0`, `cffi>=1.0`, `classy>=3.3.4,<3.4`, `hmf>=3.6.2`, `cosmotile>=0.2.5`, `cyclopts`, `attrs`, `click`, `matplotlib`, `tqdm`, `pyyaml`, `tomlkit`

> **Zero external Python GPU dependencies**: `cupy` and `numba` are **not** required.

---

## Option 1: One-Command Install Script (`scripts/install.sh`)

```bash
git clone https://github.com/HainaHuang729/21cmFAST-GPU.git
cd 21cmFAST-GPU
./scripts/install.sh
```

Useful flags:

- `--cuda-archs 89` — target a specific SM compute capability (`auto` by default, querying `nvidia-smi`)
- `--venv .venv` — create and activate a fresh Python 3.12 virtual environment first
- `--cpu` — build CPU-only (`USE_CUDA=0`) even if `nvcc` is installed

The script verifies `Python >= 3.12`, `Python.h`, `gcc`, `libgsl`, `libfftw3f`, and `nvcc`, runs `pip install -e . --no-build-isolation`, and executes `python -m py21cmcu.cli doctor` to validate the installation.

---

## Option 2: Manual Source Build

```bash
export USE_CUDA=1
export CUDA_ARCHS="80,86,89,90"   # or CUDA_ARCHS=auto
pip install -e . --no-build-isolation
```

`setup.py` / `build_cffi.py` automatically discovers the highest CUDA installation under `/usr/local/cuda*`, queries `nvidia-smi` for SM architectures when `CUDA_ARCHS=auto`, and resolves GSL/FFTW headers via environment paths, active `conda`/`venv` prefixes, or `pkg-config`.

---

## Option 3: Docker & Apptainer (HPC Slurm)

Pre-built container recipes under `docker/` avoid local compiler/header setup:

### Docker
```bash
docker build -f docker/Dockerfile --build-arg CUDA_ARCHS="80,86,89,90" -t 21cmcu:0.6.0 .
docker run --rm --gpus all 21cmcu:0.6.0 doctor
```

Defaults baked into the image: `P21_USE_CUDA=1`, `P21_CUDA_MODE=fast`, `OPENBLAS_NUM_THREADS=1`.

### Apptainer / Singularity (Cluster / Slurm)
```bash
apptainer build 21cmcu.sif docker/21cmcu.def
apptainer run --nv 21cmcu.sif doctor --as-json
srun --gres=gpu:2 apptainer run --nv 21cmcu.sif python examples/03_multigpu_large_ic.py --dim 1536 --devices 0,1
```

---

## Verifying Your Installation (`doctor`)

```bash
21cmcu doctor
21cmcu doctor --as-json
```

`21cmcu doctor` prints:
- `py21cmcu`, Python, and NumPy versions + CPU count
- CUDA compile/runtime status, visible device count, and GPU names/VRAM/compute capability
- `nvidia-smi topo -m` interconnect topology (`NODE`, `SYS`, `PHB`, `PIX`, `NV#`) when $\ge 2$ GPUs are present
- `nvcc`, `gcc`, `libgsl`, and `libfftw3f` shared-library discovery
- Recommended 24 GB GPU count and host RAM across `512³–2048³` grids
