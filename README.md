# 21cmFAST-GPU

CUDA / mixed-precision-faithful GPU acceleration for **21cmFAST v3** (standard $\Lambda$CDM, `POWER_SPECTRUM=0`, Eisenstein & Hu), along with strict voxel-level regression verification against CPU. Built on top of **21cmFAST v3** (upstream commit [aefc273](https://github.com/21cmfast/21cmFAST/commit/aefc273), v3.3.x–v3.4 dev snapshot), this implementation operates via non-invasive build-time patches and C function-pointer hooks. Upstream licensing is documented in [LICENSE.upstream](LICENSE.upstream).

> **Need the blue-tilted / broken-power-law primordial spectrum (`KP_h_Mpc`, `MS`, `POWER_SPECTRUM=6`) specialized version?** Use the dedicated [`bt`](https://github.com/HainaHuang729/21cmFAST-GPU/tree/bt) branch, which preserves `BTPS` / `POWER_SPECTRUM=6` and provides automatic `KP_h_Mpc -> KP = KP_h_Mpc * hlittle` conversion plus 16-thread GSL IC parity. The `main` branch targets standard $\Lambda$CDM (`POWER_SPECTRUM=0`) without `KP`/`MS` extensions.

The repository provides two validated pipelines:
1. **Wave 6 Full 6-Field Lightcone Pipeline ([lightcone/](lightcone))** — Primary release: passes **`0 / 6` out-of-tolerance voxels** (`abs(GPU - CPU) <= 1e-8 + 1e-4 * abs(CPU)`) and cold/warm deterministic repeatability (**`repeat_equal = true`, `max_diff = 0.0`**) across **all 4 standard benchmarks (up to `1024³ -> 256³`, 3,922 slices) plus 5 multi-parameter / multi-cosmology / multi-seed spot-checks**, achieving **`17.40×` end-to-end speedup** at production scale (`6,462.85 s -> 371.37 s`).
2. **Coeval + Power Spectrum Archive Pipeline ([gpu/](gpu))** — 2026-11-19 validated release: full `512³ -> 128³` 32-redshift coeval boxes and power-spectrum archiving with parallel LZF HDF5 writes (`85.69 s`).

---

## Wave 6 GPU Lightcone vs. CPU Reference Summary

> For the complete stage-by-stage architecture, CUDA kernel split, and numerical precision analysis, see [docs/GPU_vs_CPU_comparison.md](docs/GPU_vs_CPU_comparison.md).

### Key Architectural Differences

| Component | Pure CPU Reference | Wave 6 GPU Lightcone ([lightcone/](lightcone)) |
|---|---|---|
| **Source Layout** | `resident_filter_20261118/fix/package/py21cmfast/src` (unmodified C/Python) | Upstream sources left untouched; [lightcone/build.py](lightcone/build.py) builds a private `c_21cmfast` CFFI extension and registers CUDA callback hooks |
| **Perturbation (2LPT + CIC)** | CPU OpenMP + 3D FFTW | CUDA `cic_pertile` + `cic_repair_exact` ([perturb.cu](lightcone/perturb.cu)); `PS_FFTW_PERTURB_DENSITY=1` preserves CPU `FFTW_ESTIMATE` materialization for $\delta$ and $v_z$ |
| **Thermal Evolution ($T_k, x_e, T_s$)** | CPU multi-scale FFTW filtering + GSL tables + OpenMP ODE integration | Background OpenMP prefetch of padded FFTW boxes + CUDA in-place unpack/bounds reduction; `lookup_sum_batch` / `accum_batch` / `update_stabilized.cu` (`--fmad=false` + [glibc_math.cuh](lightcone/glibc_math.cuh)) resident on GPU |
| **Ionization & Recombinations** | Full-box 3D FFTW spherical top-hat filtering across all scales + per-voxel loop and MHR recombinations | CUDA scale screening (`screen_ion_scale_pass1`) + margin guards (`PS_GUARD_EPS_MULT=32`, `PS_XEREC_AMP_MULT=0`) + sharp-$k$ pruned 3D FFTW `c2r` replays ([ion_hook.h](lightcone/ion_hook.h)) + CUDA collapse, bubble assignment, and MHR kernels ([ion.cu](lightcone/ion.cu)) |
| **21 cm Brightness & Lightcone** | CPU 3D FFTW $\partial_z v_z$ + Astropy Python slice interpolation loop | `PS_FFTW_BRIGHTNESS=1` (`ComputeBrightnessFastFFTW` prevents `Ts ≈ T_CMB` zero-crossing drift) + C/OpenMP `PSFillLightconeSlices` + parallel LZF HDF5 I/O |

### End-to-End Lightcone Verification (Single RTX A6000 + 8 CPU Cores, `0 / 6` Failures, `repeat = 0.0`)

#### A. Four Standard Benchmarks ([summary.json](reports/wave6-lightcone/summary.json))

| Benchmark | Grid / Box / Slices | CPU Wall | **Wave 6 GPU Warm (`phys`)** | Speedup | Parity / Repeat | Report |
|---|---|---:|---:|:---:|:---:|---|
| `384/96` | `384 -> 96`, 375 Mpc, 609 | `278.18 s` | **`21.02 s`** (`16.21 s`) | **`13.23×`** | `0 / 6`, `0.0` | [w4_units.json](reports/wave6-lightcone/results-489606/w4_units.json) |
| `h1_256_64` | `256 -> 64`, 200 Mpc, 1,227 | `107.93 s` | **`13.98 s`** (`9.59 s`) | **`7.72×`** | `0 / 6`, `0.0` | [h1 comparison](reports/wave6-lightcone/results-489605/warm-h1_256_64-comparison.json) |
| `h2_384_96` | `384 -> 96`, 300 Mpc, 761 | `266.33 s` | **`23.29 s`** (`18.20 s`) | **`11.44×`** | `0 / 6`, `0.0` | [h2 comparison](reports/wave6-lightcone/results-489605/warm-h2_384_96-comparison.json) |
| **`1024/256`** | `1024 -> 256`, 250 Mpc, 3,922 | `6,462.85 s` | **`371.37 s`** (`336.33 s`) | **`17.40×`** | `0 / 6`, `0.0` | [warm-1024-timing.json](reports/wave6-lightcone/results-489607/warm-1024-timing.json) |

#### B. Five Multi-Parameter / Multi-Cosmology / Multi-Seed Spot-Checks (Job `489621`)

| Point | Physics & Cosmology Regime | Grid / Box / Slices / Seed | CPU Wall | **Wave 6 GPU Warm (`phys`)** | Speedup | Report |
|---|---|---|---:|---:|:---:|---|
| `p1_early_eor_strong_xray` | Early EoR + strong soft X-rays | `256->64`, 200 Mpc, 1,227, `111111` | `111.69 s` | **`13.24 s`** (`8.83 s`) | **`8.44×`** | [p1](reports/wave6-lightcone/results-489621/warm-p1_early_eor_strong_xray-comparison.json) |
| `p2_late_eor_weak_hard_xray` | Late EoR + weak hard X-rays | `256->64`, 250 Mpc, 982, `222222` | `96.61 s` | **`11.82 s`** (`7.94 s`) | **`8.17×`** | [p2](reports/wave6-lightcone/results-489621/warm-p2_late_eor_weak_hard_xray-comparison.json) |
| `p3_cosmo_shift_lcdm_low_kp` | Shifted LCDM cosmology | `256->64`, 180 Mpc, 1,353, `333333` | `114.76 s` | **`13.79 s`** (`9.14 s`) | **`8.32×`** | [p3](reports/wave6-lightcone/results-489621/warm-p3_cosmo_shift_lcdm_low_kp-comparison.json) |
| `p4_cosmo_shift_low_s8` | Low `SIGMA_8` (`0.785`) + high turnover mass | `256->64`, 220 Mpc, 1,113, `444444` | `101.19 s` | **`13.34 s`** (`9.17 s`) | **`7.59×`** | [p4](reports/wave6-lightcone/results-489621/warm-p4_cosmo_shift_low_s8-comparison.json) |
| `p5_384_96_mid_eor_cosmo` | Mid EoR + mixed cosmology shift | `384->96`, 300 Mpc, 839, `555555` | `266.65 s` | **`21.59 s`** (`16.43 s`) | **`12.35×`** | [p5](reports/wave6-lightcone/results-489621/warm-p5_384_96_mid_eor_cosmo-comparison.json) |

> All 9 low-level unit/bitwise tests (including the 7-radius Wave 6 pruned 3D FFTW `c2r` check) passed with `bitwise_equal = true`. Evidence for why `PS_FFTW_BRIGHTNESS=0` causes zero-crossing `brightness_temp` regressions is archived in [489602 failures](reports/wave6-lightcone/results-489602/warm-h1_256_64-failures.json).

---

## Repository Layout

- [lightcone/](lightcone): Complete Wave 6 GPU lightcone source, CUDA kernels, C patchers, and Slurm verification scripts (60 files)
  - [lightcone/build.py](lightcone/build.py): Patches and compiles the private `c_21cmfast` CFFI extension
  - [lightcone/run_large_lightcone.py](lightcone/run_large_lightcone.py): Unified CPU / GPU 6-field lightcone entrypoint (supports `PS_ASTRO_JSON`, `PS_COSMO_JSON`, `PS_SEED`, `PS_DIM`/`PS_HII_DIM`/`PS_BOX_LEN`)
  - [lightcone/phase3d_holdout_verify.sbatch](lightcone/phase3d_holdout_verify.sbatch), [lightcone/phase3e_multiparam_spotcheck.sbatch](lightcone/phase3e_multiparam_spotcheck.sbatch): Self-contained Slurm verification scripts that generate ICs and CPU references on the fly
  - [lightcone/phase3c_w4_verify.sbatch](lightcone/phase3c_w4_verify.sbatch), [lightcone/phase3d_1024_verify.sbatch](lightcone/phase3d_1024_verify.sbatch): Archival Slurm scripts configured for `/var/tmp`-cached `384/96` and `1024/256` CPU references on the benchmark node
- [gpu/](gpu): Validated Coeval + power spectrum pipeline ([gpu/launch_ps.py](gpu/launch_ps.py), [gpu/integrated-lzf.sbatch](gpu/integrated-lzf.sbatch))
- `resident_filter_20261118/fix/package/py21cmfast`: Unmodified base C/Python 21cmFAST tree
- [docs/GPU_vs_CPU_comparison.md](docs/GPU_vs_CPU_comparison.md): Detailed GPU-vs-CPU architectural, algorithmic, and numerical parity comparison
- [reports/](reports): Validation artifacts including [reports/wave6-lightcone/](reports/wave6-lightcone/summary.json) and earlier coeval/batch4 JSON reports
- `point_config.json`, `reference.npz`, `LICENSE.upstream`

---

## Quickstart (Wave 6 GPU Lightcone)

On a Linux CUDA host, prepare Python 3.10+ (NumPy, SciPy, Astropy, h5py with LZF, cffi, CuPy, PyYAML), an OpenMP-capable C compiler, GSL, and single-precision FFTW3 (`libfftw3f`, `libfftw3f_omp`). Ensure `include/` (`gsl/`, `fftw3.h`) and `lib/` (`libgsl`, `libfftw3f`, `libfftw3f_omp`) exist or are symlinked under the repository root.

### 1. Build Private `c_21cmfast` and Generate GK61 Quadrature Rule

```bash
export PS_GPU_ROOT="$PWD"
export LD_LIBRARY_PATH="$PS_GPU_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_PATH=/path/to/cuda
export CPATH="$CUDA_PATH/include${CPATH:+:$CPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p build_work
python lightcone/build.py "$PS_GPU_ROOT" "$PWD/build_work"
export PYTHONPATH="$PWD/build_work/package:$PWD/lightcone"

export PS_GK61_RULE="$PWD/build_work/gk61.json"
python lightcone/gk61_rule.py "$PS_GK61_RULE"
```

### 2. Run `gpu-density` Mode with Validated Wave 6 Flags

Provide a matching CPU-generated GSL initial conditions file via `PS_CONTROL_IC`:

```bash
export PS_FFTW_PERTURB_DENSITY=1 PS_FFTW_HEAT_FILTER=1 PS_GSL_EXACT_POWER=1
export PS_CONDITIONAL_MODE=3 PS_BATCH_FILTER=1
export PS_XRAY_NATIVE_MODE=4 PS_NATIVE_SIGMA=1 PS_NATIVE_MHR=1 PS_NATIVE_COLLAPSE=1 PS_NATIVE_XRAY=0
export PS_QAG_MODE=deferred PS_GK61_COMPARE=0 PS_QAG_PROFILE=0
export PS_CONDITIONAL_FULL_HIGH=0 PS_CONDITIONAL_NAN_SENTINEL=1 PS_GUARD_EPS_MULT=32
export PS_LAZY_COEVAL_IC=1 PS_PARALLEL_LZF=1 PS_VERBOSE_HEAT=0
export PS_REUSE_PREV_PERTURB=1 PS_FAST_LIGHTCONE=1 PS_CIC_PERTILE=1
export PS_OMP_MHR=1 PS_CACHE_INIT_PS=1 PS_ASYNC_FFTW_PREFETCH=1
export PS_FFTW_MEASURE=0 PS_FFTW_MEASURE_HEAT=0

export PS_DIM=384 PS_HII_DIM=96 PS_BOX_LEN=375.0 PS_MAX_Z=14.0 PS_WINDOW_MAX_Z=14.0
export PS_CONTROL_IC=/path/to/InitialConditions.h5
# Optional cosmology / astrophysics overrides:
# export PS_COSMO_JSON='{"OMm": 0.308, "SIGMA_8": 0.815}'
# export PS_ASTRO_JSON='{"F_STAR10": -1.06, "L_X": 39.25}'

python lightcone/run_large_lightcone.py gpu-density out_gpu work_gpu
```

> Note: [phase3d_holdout_verify.sbatch](lightcone/phase3d_holdout_verify.sbatch) and [phase3e_multiparam_spotcheck.sbatch](lightcone/phase3e_multiparam_spotcheck.sbatch) are self-contained Slurm verification templates; adapt `#SBATCH` headers and environment paths (`python`, `CUDA_PATH`) for other clusters.

---

## Coeval + Power Spectrum Pipeline (`gpu/`)

- Fiducial point `80000`, seed `1850877074`, `250 Mpc` box, `512³` IC grid evolved on `128³`.
- Single RTX A6000 GPU + 8 host threads, outputting 14 IC fields, 416 coeval fields across 32 redshifts, global history, and 32-bin / 60-bin power spectra.
- Warm end-to-end wall time to synced NFS: `98.51 s` baseline, `85.69 s` paired parallel LZF (`0/1024` out-of-tolerance bins on the production 32-bin PS): [coeval-retest.json](reports/coeval-retest.json) · [paired-output.json](reports/paired-output.json) · [parallel-lzf.json](reports/parallel-lzf.json).

```sh
python gpu/launch_ps.py runs/point-80000 --cache runs/cache \
  --seed --full-coeval --include-ic --gpu-full-ic \
  --compression lzf --coeval-lzf-workers 4 --paired-archive
```

Earlier 2-field lightcone batch-throughput tests (4 concurrent workers on 1 GPU in `99.83 s`, ~144 lightcones/hour) are archived in [lightcone-batch4.json](reports/lightcone-batch4.json) · [lightcone-comparison.json](reports/lightcone-comparison.json).

---

## Roadmap

- [ ] **Port GPU acceleration to 21cmFAST v4 (`v4.3+`)**: Adapt the build-time patches and CUDA hooks to the v4 C backend and Python API overhaul (`InputParameters` / TOML templates, unified Lagrangian/Eulerian radiation-field framework) and add support for v4 physics features (stochastic halo sampling `HALO_STOCHASTICITY`, Lyman-$\alpha$ multiple scattering, and native CLASS transfer functions) while preserving strict voxel-level CPU regression verification.
