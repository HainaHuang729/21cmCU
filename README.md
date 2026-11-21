# 21cmFAST-GPU

CUDA acceleration for 21cmFAST v3 (based on commit [aefc273](https://github.com/21cmfast/21cmFAST/commit/aefc273), standard LCDM `POWER_SPECTRUM=0`). Lightcone outputs across all 6 fields match CPU runs within `1e-8 + 1e-4 * |CPU|`. Upstream license: [LICENSE.upstream](LICENSE.upstream).

For the broken-power-law primordial spectrum (`KP_h_Mpc`, `MS`, `POWER_SPECTRUM=6`), use the [bt](https://github.com/HainaHuang729/21cmFAST-GPU/tree/bt) branch.

## Setup

```bash
git clone https://github.com/HainaHuang729/21cmFAST-GPU.git
cd 21cmFAST-GPU
conda env create -f environment.yml
conda activate 21cmfast-gpu
pip install cupy-cuda12x   # or cupy-cuda11x for CUDA 11.x

python lightcone/run.py --dim 384 --hii-dim 96 --box-len 375.0 --max-z 14.0 --out runs/lc_gpu --work runs/work
```

[lightcone/run.py](lightcone/run.py) automatically locates GSL, `fftw3f`, and CUDA, compiles the private CFFI library, and generates ICs on the first run; pass `--gsl-prefix` / `--fftw-prefix` for custom system installs, `--mode verify` to compare GPU vs. CPU (`256/64` runs in ~15 s GPU / ~105 s CPU), use [lightcone/run_batch.py](lightcone/run_batch.py) for concurrent single/multi-GPU batches, or see [gpu/launch_ps.py](gpu/launch_ps.py) for coeval power-spectrum archives.

## Benchmarks

Host: AMD Ryzen Threadripper PRO 5975WX (32 cores) + 1x RTX A6000 (GPU runs use 8 CPU threads):

| Grid / Box / Slices | CPU 8T | CPU 16T | CPU 32T | GPU (`phys`) | Speedup (8T / 16T / 32T) | Failed |
|---|---:|---:|---:|---:|:---:|:---:|
| `256 -> 64`, 200 Mpc, 1227 | 107.93 s | 72.99 s | 71.18 s | 13.98 s (9.59 s) | 7.72x / 5.22x / 5.09x | 0 / 6 |
| `384 -> 96`, 300 Mpc, 761 | 266.33 s | - | - | 23.29 s (18.20 s) | 11.44x | 0 / 6 |
| `384 -> 96`, 375 Mpc, 609 | 278.18 s | 163.98 s | 160.13 s | 21.02 s (16.21 s) | 13.23x / 7.80x / 7.62x | 0 / 6 |
| `1024 -> 256`, 250 Mpc, 3922 | 6462.85 s | - | - | 371.37 s (336.33 s) | 17.40x | 0 / 6 |

Single-GPU concurrency at `384 -> 96` (375 Mpc, 609 slices, 1x RTX A6000 + 16 CPU cores for 1-4x / 32 CPU cores for 8x, via [lightcone/run_batch.py](lightcone/run_batch.py), including per-worker `384³` IC generation):

| Concurrency (`threads/worker`) | Batch Wall | Throughput | Peak VRAM |
|---|---:|---:|---:|
| 1 (`8` threads) | 43.76 s | 82.3 / h | 2.3 GB |
| 2 (`8` threads) | 46.08 s | 156.2 / h | 3.8 GB |
| 4 (`4` threads) | 59.41 s | 242.4 / h | 6.7 GB |
| 8 (`4` threads) | 74.05 s | 388.9 / h | 12.7 GB |

Implementation notes, unit tests, and 5 additional parameter spot-checks are in [docs/GPU_vs_CPU_comparison.md](docs/GPU_vs_CPU_comparison.md) and [reports/wave6-lightcone/summary.json](reports/wave6-lightcone/summary.json).

## Roadmap

- Port the CFFI/CUDA hooks to 21cmFAST v4 (`v4.3+`) (`InputParameters`, `HALO_STOCHASTICITY`, Lyman-alpha multiple scattering, native CLASS transfer functions).
