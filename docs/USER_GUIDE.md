# 21cmCU User Guide

**21cmCU** provides a native C/CUDA + cuFFT drop-in GPU engine for 21cmFAST v4 (`py21cmcu`), with zero CuPy or Numba runtime dependencies.

## 1. Installation & Container Setup

### Option A: One-Command Source Install (`scripts/install.sh`)

```bash
git clone https://github.com/HainaHuang729/21cmFAST-GPU.git
cd 21cmFAST-GPU
./scripts/install.sh              # auto-detects CUDA SM arches + toolchain + runs 21cmcu doctor
# Or specify GPU architectures / venv:
./scripts/install.sh --cuda-archs 89 --venv .venv
# Or CPU-only fallback:
./scripts/install.sh --cpu
```

Or manually via `pip` (`Python >= 3.12` + `python3.12-dev`, `gcc`, `libgsl-dev`, `libfftw3-dev`, CUDA toolkit):

```bash
USE_CUDA=1 CUDA_ARCHS="80,86,89,90" pip install -e . --no-build-isolation
```

### Option B: Docker / Apptainer (HPC Slurm)

Pre-configured GPU recipes live in [`docker/`](../docker/):

```bash
# Docker
docker build -f docker/Dockerfile --build-arg CUDA_ARCHS="80,86,89" -t 21cmcu:0.6.0 .
docker run --rm --gpus all 21cmcu:0.6.0 doctor

# Apptainer / Singularity (Slurm clusters)
apptainer build 21cmcu.sif docker/21cmcu.def
apptainer run --nv 21cmcu.sif doctor
```

### Runtime Self-Check (`doctor`)

```bash
21cmcu doctor
21cmcu doctor --as-json
```

Or from Python:

```python
import py21cmcu

report = py21cmcu.doctor()
print(py21cmcu.cuda_compiled(), py21cmcu.cuda_available(), py21cmcu.cuda_device_count())
```

### 5-Minute Runnable Examples (`examples/`)

| File | What it demonstrates |
| --- | --- |
| [`examples/01_quickstart_coeval.py`](../examples/01_quickstart_coeval.py) | Minimal single-GPU coeval box (`L-INTEGRAL + TS`) + 1D $P(k)$ |
| [`examples/02_lightcone_power_spectrum.py`](../examples/02_lightcone_power_spectrum.py) | GPU rectilinear lightcone + RSDs + chunk PS + wedge/noise + `DeviceLZF` HDF5 |
| [`examples/03_multigpu_large_ic.py`](../examples/03_multigpu_large_ic.py) | Multi-GPU slab IC + `PerturbField` + 1D $P(k)$ (`--dim 512..2048 --devices 0,1,2`) |
| [`examples/sbi_lightcone_dataset.py`](../examples/sbi_lightcone_dataset.py) | 4D scrambled Sobol sweep + resumable HDF5 checkpoints via `run_batch` |
| [`examples/quickstart.ipynb`](../examples/quickstart.ipynb) | Interactive Jupyter quickstart covering all four workflows |

## 2. Execution Modes: `strict` vs `fast`

21cmCU supports two verified precision/performance profiles controlled via `CUDA_MODE` (`strict` or `fast`), context managers, or CLI flags:

- **`strict`** (default conservative reproducibility): Uses IEEE FP32 math without fast-math trig approximations and strictly matches CPU boundary/interpolation order (`atol`/`rtol` parity against single-thread CPU reference).
- **`fast`** (recommended for SBI & large sweeps): Enables device-resident FP32 approximations (`--use_fast_math` kernels), batched device perturbations, and fused GPU lightcone/RSD pipelines while keeping scientific parity verified across ionization fraction ($\langle x_{\mathrm{HI}} \rangle$), global 21cm brightness ($\langle \delta T_b \rangle$), and 1D/2D power spectra.

```python
import py21cmcu

py21cmcu.set_cuda_mode("fast")
# Or scoped locally:
with py21cmcu.use_mode("fast", lazy_d2h=True):
    ...
```

## 3. GPU Residency (`CUDA_LAZY_D2H`) & `DeviceLZF` HDF5 Compression

21cmCU eliminates unnecessary device-to-host PCIe transfers across coeval and lightcone loops when opt-in `CUDA_LAZY_D2H=True` is enabled (`False` by default):

- Fields produced by `InitialConditions`, `PerturbField`, `IonizedBox`, and `SpinTemperatureBox` remain live in the CUDA device field registry keyed by opaque GPU handles.
- Downstream stages (`IonizedBox`, `SpinTemperatureBox`, `BrightnessTemp`, `Lightconer`) consume device buffers directly, cutting host D2H traffic by **~21.7%** on multi-redshift `512^3 -> 256^3` runs (`7,114 MB -> 5,570 MB`) and accelerating full runs (`4.48 s -> 3.90 s`).
- For saving large 3D cubes or lightcones to disk, `DeviceLZF` compresses float32 arrays on-GPU (`476.3 MB/s` write vs `259.6 MB/s` CPU LZF at `256^2 x 512`) and writes standard HDF5 LZF chunks directly readable by `h5py` without any custom HDF5 filter plugin.

```python
from py21cmfast._cfg import config
import py21cmcu

with config.use(USE_CUDA=True, CUDA_MODE="fast", CUDA_LAZY_D2H=True):
    ...

py21cmcu.save_lightcone(lc, "lightcone.h5", compression="device-lzf")
```

## 4. Multi-GPU Domain Decomposition & Sizing (`CUDA_DEVICES`)

21cmCU supports two multi-GPU parallelism patterns:

1. **Out-of-single-GPU slab domain decomposition** (automatic for large `InitialConditions` grids when `CUDA_DEVICES` lists $\ge 2$ GPUs): partitions `(X, Y, Z)` 3D grids across multiple GPUs with host-staged pinned-bounce sub-tiled parallel transposes and single-work-slab streaming (`P21_CUDA_MGPU_SINGLE_SLAB`).
2. **Multi-GPU process-pool ensemble execution (`run_batch`)**: assigns independent parameter realizations across all GPUs (`devices=(0, 1, 2)`, `workers_per_gpu=2`).

Configure active GPUs in Python or environment:

```python
import py21cmcu

py21cmcu.set_cuda_devices([0, 1, 2])
```

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
```

### Recommended Hardware & Memory Sizing (24 GB Class GPUs, e.g. RTX 4090)

| Workload | Min GPUs | VRAM / GPU | Host RAM | Verified `v0.6.0` Runtime |
| --- | :---: | --- | --- | --- |
| `512^3 -> 256^3` Coeval (`L-INTEGRAL + TS`, 3z, `fast`) | 1 | ~5 GB | 8 GB | `4.46 s` (`12.36x` vs 16-core CPU) |
| `768^3` EoR Coeval (`fast`) | 1 | ~19 GB | 16 GB | `16.44 s` (`6.71x` vs 16-core CPU) |
| `1024^3` (1.07B) Zel'dovich / 2LPT IC (`fast`) | 1 | ~7–18 GB | 24 GB | `4.29 s` / `8.20 s` |
| `1536^3` (3.62B) Multi-GPU Zel'dovich / 2LPT IC | 2 | ~15 GB | 90 GB | `27.80 s` / `75.23 s` (2 GPUs) |
| `1792^3` (5.75B) Multi-GPU Zel'dovich / 2LPT IC | 3 | ~16 GB | 135 GB | `38.70 s` / `103.89 s` (3 GPUs) |
| `2048^3` (8.59B) Single-Slab Zel'dovich IC | 3 | ~11.6 GB | 72 GB | `58.47 s` native in-VRAM (3 GPUs) |

If a single-slab multi-GPU IC exceeds available GPU count/VRAM or host RAM limits, 21cmCU cleanly falls back to CPU/host FFT or skips cleanly without GPU OOM.

## 5. SBI Dataset Generation (`run_batch`, Wedge Filter & Thermal Noise)

21cmCU provides ready-to-use Simulation-Based Inference (SBI) primitives in `py21cmcu`:

- `generate_sobol_parameter_grid(bounds, n_samples, seed=42)`: scrambled low-discrepancy Sobol ensemble generation.
- `horizon_wedge_mask(...)` & `apply_wedge_filter_3d(...)`: Fourier-space horizon foreground wedge filtering preserving the EoR window ($|k_\parallel| \ge m(z) k_\perp + \text{buffer}$).
- `add_thermal_noise_3d(...)`: deterministic radiometer or explicit `sigma_mk` Gaussian thermal noise.
- `run_batch(..., checkpoint_h5=..., resume=True)`: resumable HDF5-checkpointed GPU worker pool saving clean + observed (wedge + noise) 1D/2D power spectra, global histories, and optional `device-lzf` lightcones.

See [examples/sbi_lightcone_dataset.py](../examples/sbi_lightcone_dataset.py) for a full 4D Sobol lightcone sweep example:

```bash
python examples/sbi_lightcone_dataset.py --samples 16 --devices 0,1,2 --workers-per-gpu 2 --out sbi_16.h5
```

## 6. CLI Reference (`21cmcu`)

The `21cmcu` entrypoint extends upstream `21cmfast` CLI commands (`run coeval`, `run lightcone`, etc.) with GPU workflow subcommands:

- `21cmcu run coeval ...` / `21cmcu run lightcone ...` — standard 21cmFAST runs with `--cuda-mode strict|fast` and `--no-use-cuda` support.
- `21cmcu batch --sobol-samples 16 --mode lightcone --redshifts 6.5,8.0,9.5 --devices 0,1 --workers-per-gpu 2 --wedge-filter --noise-sigma-mk 2.0 --resume --out sbi.h5` — run resumable CSV/JSON or Sobol sweeps directly from the terminal.
- `21cmcu doctor [--as-json]` — print runtime, GPU topology, toolchain, and sizing tables.
- `21cmcu bench --suite lightcone,batch --json results.json` — invoke repository benchmark suites (`--resume` supported).
