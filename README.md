# 21cmCU

**21cmCU** is a standalone native C/CUDA + cuFFT 21cm cosmological simulation engine (originating from the `21cmFAST-GPU` fork and built on top of 21cmFAST v4.3.2 semantics with full `py21cmcu` / `py21cmfast` API and CLI compatibility and zero runtime CuPy dependency).

## Key Capabilities (`v0.2.0` — Phases 0, 1/C & 2/B Complete)

- **Native C/CUDA + cuFFT core**: compiled directly into the CFFI shared library (`c_21cmfast`), supporting `E-INTEGRAL`, `CONST-ION-EFF`, `L-INTEGRAL`, `DEXM-ESF`, and `CHMF-SAMPLER` with `USE_MCGS`, `CLASS` (`V_CB_MODEL="FLUCTS"`), and `LYA_MULTIPLE_SCATTERING`.
- **Dual precision modes (`CUDA_MODE`)**:
  - `strict` (default): CPU-matched parity mode (`rel_rms ~ 1e-4` on `384³ / 128³` lightcones, `< 0.02%` max 3D power spectrum deviation over `k ∈ [0.1, 1.0] Mpc⁻¹`).
  - `fast`: statistically equivalent high-speed mode (parallel Philox IC RNG, GPU chunk-tree hashing, adaptive accumulation via `P21_CUDA_MODE=fast` / `CUDA_MODE="fast"` / `--cuda-mode fast` while keeping global histories and Δ²₂₁(k) unbiased).
- **High-resolution single-GPU residency (`512³–1024³`)**: 64-bit indexing, cuFFT plan LRU caching, `P21_CUDA_REGISTRY_MB` budget control, and OOM-safe LRU reclamation.
- **Standalone bug fixes over v4.3.2**:
  - Photon-conservation calibration redshift/neutral-fraction length alignment (`photoncons.py`).
  - Safe null-output guard in Sheth–Tormen `st_taylor_factor` for `SAMPLE_METHOD="PARTITION"` (`hmf.c`).
  - Separate previous-redshift 1D ACG ionizing-efficiency table (`Nion_conditional_table1D_acg_prev`) for `E-INTEGRAL` + `USE_MCGS=True` without photoheating feedback (`interp_tables.c` / `cuda_ionization.cu`).

### Verified RTX 4090 Highlights (24 GB VRAM, `N_THREADS=8`)

- **`512 / 1024` 2LPT IC + Perturb (`fast`)**: `10.331 s` lo-res (**8.30×** vs `85.784 s` CPU) / `14.257 s` hi-res (**6.47×** vs `92.196 s` CPU).
- **`512 / 1024` Full 2LPT Coeval (`fast`)**: `27.630 s` (`15.58 / 23.55 GB` VRAM).
- **`768³` EoR Box (`fast`)**: `29.673 s` (`18.99 / 23.55 GB` VRAM).
- **Full native GPU suite**: `108/108` tests passing on RTX 4090.

## Quick Start

```bash
export USE_CUDA=1 CUDA_ARCHS=86,89   # adjust compute capability for your GPU
pip install -e . --no-build-isolation

# Enable GPU via environment, Python API, or CLI:
P21_USE_CUDA=1 P21_CUDA_MODE=fast 21cmcu run coeval -z 8 --template simple tiny --out ./out
```

```python
import py21cmcu as p21

with p21.use_mode("strict"):
    lc = p21.run_lightcone(...)

with p21.use_mode("fast"):
    cv = p21.run_coeval(...)
```

## Roadmap & Documentation

- Full milestone record, benchmark tables, and Directions D–G plan: [docs/ROADMAP.md](docs/ROADMAP.md)
- Version history: [CHANGELOG.md](CHANGELOG.md)
- **Active focus**: **Direction D** (GPU RSDs, batch rectilinear slicing, GPU 1D/2D power spectra) → **Direction E** (Multi-GPU batch/SBI runner & `strict` vs `fast` calibration) → **Direction F** (build auto-detection & `v0.3.0` polish).

---

## Archived CuPy v4 Adapter & Legacy v3 Reference (`archive-cupy-v4`)

The previous CuPy v4.3.2 adapter (`v4/`) and v3 CUDA implementation (`lightcone/`, `gpu/`) remain preserved in the repository and tagged at `archive-cupy-v4`:

CUDA acceleration for 21cmFAST v3 (based on commit [aefc273](https://github.com/21cmfast/21cmFAST/commit/aefc273), standard LCDM `POWER_SPECTRUM=0`). Lightcone outputs across all 6 fields match CPU runs within `1e-8 + 1e-4 * |CPU|`. Upstream license: [LICENSE.upstream](LICENSE.upstream).

For the broken-power-law primordial spectrum (`KP_h_Mpc`, `MS`, `POWER_SPECTRUM=6`), use the [bt](https://github.com/HainaHuang729/21cmFAST-GPU/tree/bt) branch.

For 21cmFAST 4.3.2, [v4/gpu_lightcone.py](v4/gpu_lightcone.py) supports `run_lightcone`, `generate_lightcone`, `run_coeval`, and `generate_coeval` with native `InputParameters`, `RectilinearLightconer` (`line_of_sight_axis` 0/1/2) or `AngularLightconer`, across `E-INTEGRAL`, `CONST-ION-EFF`, `CHMF-SAMPLER` (`N_THREADS >= 1`, all five GSL PRNGs), `DEXM-ESF`, and `L-INTEGRAL`, including all 7 `HMF` options, heating/recombination/filter/sampling switches, automatic `photon_calibration=None`, non-cubic grids (`NON_CUBIC_FACTOR != 1`), odd `DIM` / `HII_DIM`, and `USE_FFTW_WISDOM=True`. [v4/verify_full_coverage.py](v4/verify_full_coverage.py) passes **102/102** option/switch/geometry parity tests ([reports/v432-full-option-coverage.json](reports/v432-full-option-coverage.json)), and post-port regressions preserve zero-failure `192³ / 64³` parity (**16.15 s** GPU vs **65.78 s** CPU, **4.07×**; 0 failed voxels/globals/1,376 `ps32` bins) and **8/8** `32³` `CHMF-SAMPLER` catalog parity. On node a8 (RTX A6000), end-to-end device residency (`DeviceFieldRegistry`, perturbation caching, shell skip-downloads, resident ionization radius loop, zero-upload slicing/brightness) brings the matched 4.3.2 Eulerian benchmarks to **15.58 s** at `192³ / 64³` (**4.22×** vs 65.78 s CPU; prior GPU 20.93 s) and **37.60 s** at `384³ / 96³` (**9.24×** vs 347.52 s CPU; prior GPU 45.14 s), with zero failures across all 8 lightcone fields, 903 globals, 1,376 32-bin power spectra, and transverse power bins. On `96³ / 32³` thermal `CHMF-SAMPLER`, default hybrid sampling reaches **4.81 s** at 1 thread (**4.94×** vs 23.78 s CPU; **15.4×** faster than the prior 74.15 s CUDA prefix path) and **3.80 s** at 2 threads (**3.78×** vs 14.37 s CPU). [v4/lightcone_io.py](v4/lightcone_io.py) adds bitwise-exact `DeviceLZF` / `ParallelLZF` HDF5 compression, and [v4/run_batch.py](v4/run_batch.py) drives concurrent single-GPU batches up to **286.9 lightcones/h** (`192³`, `conc=2`) and **118.9 lightcones/h** (`384³`, `conc=4`). The v3 benchmarks below use different physical inputs and timing boundaries. [Supported options and usage](v4/README.md) · [Full option coverage](reports/v432-full-option-coverage.json) · [Eulerian acceptance](reports/v432-current-large-lightcones.json) · [Resident residency telemetry](reports/v432-resident-large-lightcones.json) · [CHMF multithread & hybrid speed](reports/v432-chmf-multithread-and-speed.json) · [LZF & batch throughput](reports/v432-lzf-and-batch.json) · [Angular lightcone acceptance](reports/v432-angular-lightcone.json) · [Constant-efficiency acceptance](reports/v432-constant-lightcones.json) · [Recorded v3 and 4.3.2 workloads](v4/README.md#comparing-recorded-run-times-across-versions)

For v4.3.2 installation and entry points, see the [adapter guide](v4/README.md#run-a-432-lightcone). Upstream review and validation references are in the [integration guide](docs/upstream-gpu-integration.md).

## v3 setup

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

## v3 benchmarks

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

- Completed across the 4.3.2 CUDA port: all five source models (`E-INTEGRAL`, `CONST-ION-EFF`, `CHMF-SAMPLER`, `DEXM-ESF`, `L-INTEGRAL`) with `N_THREADS >= 1` across all five GSL PRNGs; `run_lightcone`, `generate_lightcone`, `run_coeval`, and `generate_coeval` drivers; all 7 HMF options, heating/recombination/filter/sampling switches, automatic `photon_calibration=None`, non-cubic (`NON_CUBIC_FACTOR != 1`), odd `DIM`/`HII_DIM`, and `USE_FFTW_WISDOM=True` (`102/102` option/geometry parity coverage); BBKS/analogs/CLASS (`NONE`, `AVG-AUTO`, `FLUCTS`) spectra and Gaussian matter window; molecular cooling and `LYA_MULTIPLE_SCATTERING`; device residency (`DeviceFieldRegistry`, perturbation caching, shell skip-downloads, resident ionization radius loop, zero-upload slicing/brightness); and `DeviceLZF` HDF5 + concurrent batch throughput. Remaining work: large-box coverage for CLASS `CHMF` / `L-INTEGRAL` / `DEXM-ESF` / `CONST-ION-EFF` / angular lightcones, resolving 1–2 OpenMP float32 boundary voxels on multithreaded `CHMF-SAMPLER` and serial contention in `prefix` sampling / `192³ conc=4`, and committing the working tree changes to Git. [4.3.2 status and file index](docs/21cmFAST-4.3.2-GPU-status.md) · [Full option coverage](reports/v432-full-option-coverage.json) · [4.3.2 port and validation order](v4/README.md#porting-and-precision-checks)
