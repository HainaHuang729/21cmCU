# 21cmFAST-GPU (`bt` Branch)

CUDA-accelerated **21cmFAST v3** (upstream commit [`aefc273`](https://github.com/21cmfast/21cmFAST/commit/aefc273)) with the blue-tilted / broken-power-law primordial spectrum (`POWER_SPECTRUM=6`, `KP_h_Mpc` -> `KP = KP_h_Mpc * hlittle`, `MS`), non-invasive build-time CFFI/CUDA hooks, and strict **`0 / 6` voxel-level CPU parity**. Upstream license: [LICENSE.upstream](LICENSE.upstream).

> **Looking for standard $\Lambda$CDM (`POWER_SPECTRUM=0`, without `KP`/`MS` extensions)?** Use the [`main`](https://github.com/HainaHuang729/21cmFAST-GPU/tree/main) branch.

---

## Quickstart

### 1. Clone and Create Environment

```bash
git clone -b bt https://github.com/HainaHuang729/21cmFAST-GPU.git
cd 21cmFAST-GPU
conda env create -f environment.yml
conda activate 21cmfast-gpu
pip install cupy-cuda12x   # or cupy-cuda11x for CUDA 11.x hosts
```

GSL, single-precision FFTW3 (`fftw3f`, `fftw3f_omp`), and CUDA are auto-detected across Conda, `pkg-config`, and system paths (or pass `--gsl-prefix` / `--fftw-prefix` / `--cuda-path`).

### 2. Run BT Lightcone

[lightcone/run_bt_lightcone.py](lightcone/run_bt_lightcone.py) builds the private extension, converts `KP_h_Mpc` to `KP = KP_h_Mpc * hlittle` ([bt_config.py](lightcone/bt_config.py)), generates `gk61.json` and GSL initial conditions (`--ic-threads 16` matches 16-thread production RNG), sets all validated flags, and runs:

```bash
python lightcone/run_bt_lightcone.py \
  --out runs/bt_out --work runs/bt_work \
  --kp-h-mpc 10.0 --ms 2.5 --seed 725213656658 \
  --dim 384 --hii-dim 96 --box-len 375.0 --max-z 14.0 \
  --ic-threads 16
```

Use `--params-json` to load flat 10D BT or nested parameter JSON files, `--mode cpu` for CPU baseline comparison, and `--skip-build` to reuse `<work>/build/package`. Parameter unit tests: [lightcone/test_bt_config.py](lightcone/test_bt_config.py).

---

## Benchmarks (Single RTX A6000 + 8 CPU Cores)

| Grid / Box / Slices | CPU Wall | GPU Warm (`phys`) | Speedup | Parity / Repeat |
|---|---:|---:|:---:|:---:|
| `256 -> 64`, 200 Mpc, 1,227 | `107.93 s` | **`13.98 s`** (`9.59 s`) | **`7.72×`** | `0 / 6`, `0.0` |
| `384 -> 96`, 300 Mpc, 761 | `266.33 s` | **`23.29 s`** (`18.20 s`) | **`11.44×`** | `0 / 6`, `0.0` |
| `384 -> 96`, 375 Mpc, 609 | `278.18 s` | **`21.02 s`** (`16.21 s`) | **`13.23×`** | `0 / 6`, `0.0` |
| **`1024 -> 256`, 250 Mpc, 3,922** | `6,462.85 s` | **`371.37 s`** (`336.33 s`) | **`17.40×`** | `0 / 6`, `0.0` |

See [docs/GPU_vs_CPU_comparison.md](docs/GPU_vs_CPU_comparison.md) and [reports/wave6-lightcone/summary.json](reports/wave6-lightcone/summary.json) for stage-by-stage architecture, 5 multi-parameter spot-checks, and unit test details.

---

## Repository Layout

- [lightcone/](lightcone): 6-field GPU lightcone pipeline, CUDA kernels (`POWER_SPECTRUM=6`), BT parameter tools ([bt_config.py](lightcone/bt_config.py), [run_bt_lightcone.py](lightcone/run_bt_lightcone.py)), and toolchain detector
- [gpu/](gpu): Coeval + power-spectrum pipeline (`512³ -> 128³`, 32 redshifts in `85.69 s` via [launch_ps.py](gpu/launch_ps.py))
- [environment.yml](environment.yml): Conda dependencies (`python=3.10`, `gsl`, `fftw`, `c-compiler`, `openmp`, Python stack)
- `resident_filter_20261118/fix/package/py21cmfast`: Base 21cmFAST v3 C/Python source with `POWER_SPECTRUM=6` and data tables
- [docs/GPU_vs_CPU_comparison.md](docs/GPU_vs_CPU_comparison.md): Architecture, CUDA kernel breakdown, and numerical parity analysis
- [reports/](reports): Archived JSON validation logs

---

## Roadmap

- [ ] **Port GPU acceleration to 21cmFAST v4 (`v4.3+`)**: Adapt the CFFI/CUDA hooks to the v4 C backend (`InputParameters`, unified Lagrangian/Eulerian framework) and add support for `HALO_STOCHASTICITY`, Lyman-$\alpha$ multiple scattering, and native CLASS transfer functions while preserving strict CPU voxel parity.
