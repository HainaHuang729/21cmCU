# 21cmFAST-GPU

CUDA-accelerated **21cmFAST v3** (upstream commit [`aefc273`](https://github.com/21cmfast/21cmFAST/commit/aefc273), standard $\Lambda$CDM `POWER_SPECTRUM=0`) with non-invasive build-time CFFI/CUDA hooks and strict **`0 / 6` voxel-level parity** against CPU (`abs(GPU - CPU) <= 1e-8 + 1e-4 * abs(CPU)` across all 6 lightcone fields). Upstream license: [LICENSE.upstream](LICENSE.upstream).

> **Need the blue-tilted / broken-power-law spectrum (`KP_h_Mpc`, `MS`, `POWER_SPECTRUM=6`)?** Switch to the [`bt`](https://github.com/HainaHuang729/21cmFAST-GPU/tree/bt) branch, which adds `KP_h_Mpc -> KP = KP_h_Mpc * hlittle` unit conversion and 16-thread GSL IC parity.

---

## Quickstart

### 1. Clone and Create Environment

```bash
git clone https://github.com/HainaHuang729/21cmFAST-GPU.git
cd 21cmFAST-GPU
conda env create -f environment.yml
conda activate 21cmfast-gpu
pip install cupy-cuda12x   # or cupy-cuda11x for CUDA 11.x hosts
```

GSL, single-precision FFTW3 (`fftw3f`, `fftw3f_omp`), and CUDA are auto-detected across Conda, `pkg-config`, and system paths (or pass `--gsl-prefix` / `--fftw-prefix` / `--cuda-path`).

### 2. Run Lightcone or Verify `0 / 6` Parity

[lightcone/run.py](lightcone/run.py) builds the private extension, generates `gk61.json` and GSL initial conditions, sets all validated flags, and runs:

```bash
# GPU lightcone (384 -> 96, 375 Mpc, z = 5..14)
python lightcone/run.py \
  --dim 384 --hii-dim 96 --box-len 375.0 --max-z 14.0 \
  --out runs/lc_gpu --work runs/work

# End-to-end 256 -> 64 CPU vs. GPU parity self-test (verifies 0 / 6 failed fields)
python lightcone/run.py --mode verify \
  --dim 256 --hii-dim 64 --box-len 200.0 --max-z 20.0 --seed 424242 \
  --out runs/verify_256 --work runs/work
```

Use `--skip-build` to reuse `<work>/build/package`, `--control-ic` to supply existing ICs, or `--astro-json` / `--cosmo-json` for parameter overrides.

---

## Benchmarks (Single RTX A6000 + 8 CPU Cores)

| Grid / Box / Slices | CPU Wall | GPU Warm (`phys`) | Speedup | Parity / Repeat |
|---|---:|---:|:---:|:---:|
| `256 -> 64`, 200 Mpc, 1,227 | `107.93 s` | **`13.98 s`** (`9.59 s`) | **`7.72×`** | `0 / 6`, `0.0` |
| `384 -> 96`, 300 Mpc, 761 | `266.33 s` | **`23.29 s`** (`18.20 s`) | **`11.44×`** | `0 / 6`, `0.0` |
| `384 -> 96`, 375 Mpc, 609 | `278.18 s` | **`21.02 s`** (`16.21 s`) | **`13.23×`** | `0 / 6`, `0.0` |
| **`1024 -> 256`, 250 Mpc, 3,922** | `6,462.85 s` | **`371.37 s`** (`336.33 s`) | **`17.40×`** | `0 / 6`, `0.0` |

All 4 standard runs, 5 additional multi-parameter/multi-cosmology spot-checks, and 9 unit tests pass with zero out-of-tolerance voxels and exact cold/warm determinism. See [docs/GPU_vs_CPU_comparison.md](docs/GPU_vs_CPU_comparison.md) and [reports/wave6-lightcone/summary.json](reports/wave6-lightcone/summary.json) for stage-by-stage architecture and full metrics.

---

## Repository Layout

- [lightcone/](lightcone): 6-field GPU lightcone pipeline, CUDA kernels, C patchers, toolchain detector, and one-command [run.py](lightcone/run.py)
- [gpu/](gpu): Coeval + power-spectrum pipeline (`512³ -> 128³`, 32 redshifts in `85.69 s` via [launch_ps.py](gpu/launch_ps.py))
- [environment.yml](environment.yml): Conda dependencies (`python=3.10`, `gsl`, `fftw`, `c-compiler`, `openmp`, Python stack)
- `resident_filter_20261118/fix/package/py21cmfast`: Unmodified upstream 21cmFAST v3 C/Python source and data tables
- [docs/GPU_vs_CPU_comparison.md](docs/GPU_vs_CPU_comparison.md): Architecture, CUDA kernel breakdown, and numerical parity analysis
- [reports/](reports): Archived JSON validation logs

---

## Roadmap

- [ ] **Port GPU acceleration to 21cmFAST v4 (`v4.3+`)**: Adapt the CFFI/CUDA hooks to the v4 C backend (`InputParameters`, unified Lagrangian/Eulerian framework) and add support for `HALO_STOCHASTICITY`, Lyman-$\alpha$ multiple scattering, and native CLASS transfer functions while preserving strict CPU voxel parity.
