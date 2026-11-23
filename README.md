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

### Single v3 lightcone from a seed

```bash
python lightcone/run.py --single-box --threads 8 --dim 384 --hii-dim 96 --box-len 375 --max-z 14 --window-max-z 14 --out runs/single --work runs/single-work
```

This path keeps the production GSL Fourier realization and computes the IC density and displacements in the evolution process. CUDA handles perturbations, heat filtering, ionization and recombination replay, the auxiliary mass and recombination integrals, brightness, rectilinear slice interpolation, and shuffle/LZF compression. The host generates the initial GSL modes, inspects FFTW plans, drives redshift and scale iteration, exchanges host snapshots required by v3, and commits the HDF5 chunks to disk. The supported branch uses cubic low-resolution 2LPT, the standard Eisenstein–Hu spectrum and supported FFTW ESTIMATE plans.

Use `--skip-build` with the same work directory after the first build. Use `--control-ic FILE` to reuse an existing IC. With `--mode verify --single-box` and no supplied IC, both solvers start from the specified seed. The [single-box acceptance report](reports/v3-single-box-acceptance.json) records full launcher timings and comparisons at the original per-voxel tolerances.

### GPU perturbation FFTs on the v3 lightcone

Select `--perturb-fft cuda-plan` on `lightcone/run.py` or `lightcone/run_batch.py` to execute the v3 perturbation density and line-of-sight velocity FFTs on CUDA. The v3 native library supplies its actual in-place FFTW plans. The converter compiles their supported scalar/SSE2 arithmetic for the GPU and reports the number of completed transforms in `timing.json` under `runtime_summary.cuda_plan_transforms`. The current implementation targets the cubic low-resolution 2LPT path with FFTW wisdom disabled. An unsupported plan in this path raises an error. The existing `fftw` default uses native FFTW for these transforms. The [v3 paired acceptance report](reports/v3-perturb-cuda-plan-acceptance.json) includes six-field comparisons and launcher wall times for both backends.

For a matched six-field comparison, run:

```bash
python lightcone/run.py --mode verify --perturb-fft cuda-plan --dim 384 --hii-dim 96 --box-len 375 --max-z 14 --out runs/lc_plan --work runs/plan_work
```

### Numbered points and independent batches

The default point comes from `point_config.json`. Register another point in `points/<index>.json` or select a catalog with `--point-dir`. Each numbered file contains its own `index`, `seed`, and complete `astro`, `cosmo`, `flags`, and `user` mappings in the same units as the default file. Select it with `--point-index` or the batch task field `point_index`. The grid, box size, thread count and optional seed come from the run settings. Explicit astrophysics and cosmology arguments must agree with the selected point. Each HDF5 output records the simulation object's parameter dictionaries in the root attributes `astro_json`, `cosmo_json`, `flags_json` and `user_json`.

Each invocation of `run_batch.py` reserves a unique `batch-*` directory under `--work`. Separate invocations can share that parent and should use distinct output destinations. Use `--mode verify` for paired CPU/GPU outputs and the six-field comparison. The batch summary records the allocated directory.

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
