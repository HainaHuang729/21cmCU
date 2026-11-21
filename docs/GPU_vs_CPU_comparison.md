# Wave 6 GPU Lightcone vs. CPU Reference Comparison

This document details the architectural, algorithmic, and numerical differences between the **Wave 6 GPU Lightcone pipeline** ([lightcone/](../lightcone)) and the **CPU 21cmFAST reference implementation** ([resident_filter_20261118/fix/package/py21cmfast/src](../resident_filter_20261118/fix/package/py21cmfast/src), derived from standard $\Lambda$CDM [21cmFAST aefc273](https://github.com/21cmfast/21cmFAST/commit/aefc273), `POWER_SPECTRUM=0`). For the blue-tilted broken-power-law (`POWER_SPECTRUM=6`) specialized variant, see the `bt-ic2` branch.

---

## 1. Non-Invasive Build-Time Patching & Hook Architecture

The reference CPU C/Python source tree under `resident_filter_20261118/fix/package/py21cmfast/` remains completely unmodified. Instead of maintaining an invasive fork, the GPU version uses **build-time source instrumentation + runtime C function-pointer callbacks**:

1. **Build-Time Instrumentation ([build.py](../lightcone/build.py))**:
   - Copies the base `py21cmfast` tree into an isolated build directory, removes prebuilt `.so` extensions, and applies modular patchers:
     - **Python wrappers**: `outputs.py` (`PS_LAZY_COEVAL_IC` lazy IC loading), `wrapper.py` (`PS_REUSE_PREV_PERTURB`, `_ps_set_lightcone_schedule` / `_ps_clear_lightcone_schedule` async loop hooks), and `lightcones.py` (cached scalar distance bounds and C/OpenMP `fill_lightcone_slices`).
     - **C physics core**: `PerturbField.c` (`SetPerturbGPUHook`, `PSParallelMemcpy`, `PSArrayAllFinite`, `PSFillLightconeSlices`, shared FFTW plan lifecycle `SetPSKeepFFTWPlans`), `SpinTemperatureBox.c` (`SetGPUFilterHook`, `SetGPUHeatHook`, `SetHeatObserver`, `HeatReferenceBatchDensityPreplannedPadded`), `IonisationBox.c` ([ion_patch.py](../lightcone/ion_patch.py) + [ion_hook.h](../lightcone/ion_hook.h) registering `SetIonGPUHook` and pruned 3D FFTW `c2r`), `BrightnessTemperatureBox.c` ([brightness_patch.py](../lightcone/brightness_patch.py) adding `SetBrightnessGPUHook` and `ComputeBrightnessFastFFTW`), `GenerateICs.c` ([ic_patch.py](../lightcone/ic_patch.py), [ic_seed_patch.py](../lightcone/ic_seed_patch.py)), `ps.c` ([conditional_patch.py](../lightcone/conditional_patch.py), [sigma_patch.py](../lightcone/sigma_patch.py), [hmf_patch.py](../lightcone/hmf_patch.py)), `recombinations.c` ([mhr_patch.py](../lightcone/mhr_patch.py)), and `heating_helper_progs.c` ([heat_table_cache_patch.py](../lightcone/heat_table_cache_patch.py)).
   - Compiles the private CFFI extension `py21cmfast.c_21cmfast` with strict IEEE floating-point flags:
     ```text
     -fopenmp -O3 -mavx2 -ffp-contract=off -fno-fast-math -fno-finite-math-only
     ```
2. **Runtime Orchestration ([coupled.py](../lightcone/coupled.py) & [run_large_lightcone.py](../lightcone/run_large_lightcone.py))**:
   - Registers CUDA backends (CuPy `RawModule`/`RawKernel` compiled with `--fmad=false`) into the C hooks, keeping density, thermal states, and conditional tables resident in GPU VRAM while selectively retaining or prefetching sensitive 3D `FFTW_ESTIMATE` operations on host CPU threads.
   - **Pure CPU Fallback**: Running [run_large_lightcone.py](../lightcone/run_large_lightcone.py) in `cpu` mode disables all GPU hooks on the exact same build, enabling paired apples-to-apples CPU/GPU comparisons.

---

## 2. Stage-by-Stage Execution Breakdown (GPU vs. CPU)

| Stage | Pure CPU Reference | Wave 6 GPU Lightcone ([lightcone/](../lightcone)) | Design Rationale & Why Certain Paths Stay on CPU |
|---|---|---|---|
| **1. Initial Conditions & Auxiliary Tables** | GSL Fourier modes + 3D FFTW + serial GSL QAG for $\sigma(M)$, HMF, and MHR recombination tables | Benchmarks reuse identical GSL HDF5 ICs (eliminating RNG-stream variance); C optimizations include `PS_OMP_MHR=1` (OpenMP `init_MHR`), `PS_CACHE_INIT_PS=1` (cosmology table cache), and `PS_GSL_EXACT_POWER=1`; CUDA kernels handle conditional collapse tables ([conditional_runtime.py](../lightcone/conditional_runtime.py), `conditional_gk61.cu`) and X-ray frequency integrals ([xray_runtime.py](../lightcone/xray_runtime.py)) with native fallbacks for boundary/nonfinite entries | Preserves identical initial Gaussian Fourier realizations; `init_MHR` OpenMP parallelization gives a `5.19×` table-setup speedup with `0` bit difference; `sigma` / `MHR` / `collapse` native tables remain configurable via `PS_NATIVE_*` flags |
| **2. Perturbation (2LPT + CIC)** | CPU OpenMP displacement scaling & CIC mass deposition + 3D FFTW smoothing and $v_z$ inversion | **CUDA ([gpu_perturb.py](../lightcone/gpu_perturb.py), [perturb.cu](../lightcone/perturb.cu))**: `cic_pertile` (coarse-cell tiled deposition at `DIM == 4 * HII_DIM`, falling back to `cic`) + `cic_repair_exact` for deterministic atomic accumulation; with `PS_FFTW_PERTURB_DENSITY=1` (default), returns status `4` to materialize final smoothed $\delta$ and $v_z$ via cached-plan CPU `MaterializePerturbDensityAndVel` | Offloads heavy particle-to-grid deposition to GPU while keeping the final 3D FFTW rounding identical to CPU `FFTW_ESTIMATE`, preventing subtle downstream threshold drift |
| **3. Thermal & Spin Temperature ($T_k, x_e, T_s$)** | CPU loops over `NUM_FILTER_STEPS_FOR_Ts` (40) scales with 3D FFTW top-hat filtering, log-table $f_{\rm coll}$ lookups, and OpenMP ODE stepping | **Async Padded CPU FFTW Prefetch + GPU Resident Evolution ([unguarded_resident.py](../lightcone/unguarded_resident.py), [gpu_heat_runtime.py](../lightcone/gpu_heat_runtime.py), [update_stabilized.cu](../lightcone/update_stabilized.cu))**:<br>1) **Filtering**: `PS_FFTW_HEAT_FILTER=1` + `PS_ASYNC_FFTW_PREFETCH=1` + `PS_HEAT_GPU_UNPACK=1` computes the next redshift's padded real-space FFTW batch (`HeatReferenceBatchDensityPreplannedPadded`) on a background thread; GPU `unpack_extrapolate_bounds_pass1/2` unpacks in place and reduces min/max.<br>2) **Collapse & X-rays**: `lookup_sum_batch`, `finish_sum_batch`, and `accum_batch` evaluate all 40 filter scales on GPU.<br>3) **ODE Step**: `update_stabilized.cu` (with `--fmad=false` and [glibc_math.cuh](../lightcone/glibc_math.cuh)) advances $(x_e, T_k, T_s)$ in VRAM and serves `SetGPUPreviousHeatMeans` reductions on GPU | Running 40 parallel filter scales through FP32 cuFFT introduces `~1e-7` rounding offsets near density-interpolation transitions; overlapping background CPU FFTW prefetch with resident GPU unpack + ODE kernels hides filter cost while preserving `0 / 6` accuracy |
| **4. Ionization & Recombinations** | Steps from `R_max` down to `R_min`, running full-box 3D FFTW filtering on $(\delta, x_e, \mathrm{d}N_{\rm rec})$, hard photon-budget comparison (`ion_eff * f_coll >= 1 - x_e + dNrec`), and MHR recombination updates | **Screened Replays + Guard Margin + Pruned 3D FFTW `c2r` ([ion_runtime.py](../lightcone/ion_runtime.py), [ion.cu](../lightcone/ion.cu), [ion_hook.h](../lightcone/ion_hook.h))**:<br>1) `gpu-density` (`PS_ION_SCREENED_DENSITY=1`): GPU `screen_ion_scale_pass1` skips scales far below the ionization threshold (`ratio < 0.99`, `last=0` -> `screened_noop`).<br>2) Active non-final scales filter $(x_e, \mathrm{d}N_{\rm rec})$ on GPU via `filter_pair` / `bubble_filter_pair`, while `min_ion_margin_pass1/2` checks voxel distance to the hard ionization step (`PS_GUARD_EPS_MULT=32`, `PS_XEREC_AMP_MULT=0`). Scales within the guard band (or `last=1`) trigger exact CPU `IonReferenceXeRecBoxOpt`.<br>3) **Wave 6 Pruned 3D FFTW `c2r`**: Exploits exact zero-padding outside the `HII_FILTER=1` sharp-$k$ sphere in `IonReferenceXeRecBoxOpt` to transform only the active $(k_x, k_y, k_z)$ sub-volume with fused in-place clamping.<br>4) Collapse fraction `collapse_with_sum`, bubble assignment `assign_ion`, Poisson draws ([ion_poisson.cu](../lightcone/ion_poisson.cu)), and MHR `recombine` run in CUDA | Because the ionization condition `>=` is a discontinuous step function, tiny FP32 rounding differences near threshold cause single-voxel premature/delayed ionization and $O(1)$ `xH` / recombination divergence. Screening skips redundant large scales, guards catch every borderline scale, and pruned `c2r` cuts replay cost with `0` bit difference |
| **5. 21 cm Brightness Temperature ($T_b$)** | CPU 3D FFTW $\partial_z v_z$ + full-box $T_b$ + RSD loop | **Defaults to `PS_FFTW_BRIGHTNESS=1` ([brightness_patch.py](../lightcone/brightness_patch.py) `ComputeBrightnessFastFFTW`)** reusing `ps_perturb_fftw_box` and OpenMP reductions; GPU kernel [brightness.cu](../lightcone/brightness.cu) remains selectable via `PS_FFTW_BRIGHTNESS=0` | **Why `PS_FFTW_BRIGHTNESS=0` was reverted**: in holdout job `489602`, GPU cuFFT $\partial_z v_z$ passed 5/6 fields with `0` failures but produced `17` (`h1_256_64`) and `4` (`h2_384_96`) `brightness_temp` tolerance failures at $T_s \approx T_{\rm CMB}$ zero-crossings where large opposite-sign boxes cancel down to $\sim 10^{-4}\text{ mK}$. Restoring `PS_FFTW_BRIGHTNESS=1` adds only `~0.08 s` net wall time and eliminates all failures |
| **6. Lightcone Slice Construction** | Per-slice Astropy `Quantity` indexing and Python `redshift_interpolation` loop | **Cached Scalar Weights + C/OpenMP `PSFillLightconeSlices` ([build.py](../lightcone/build.py), `PS_FAST_LIGHTCONE=1`)**: pre-extracts pure `float64` comoving weights and executes the exact `float32` arithmetic order across `collapse(2)` threads | Unit test `W5_lightcone_fill` confirms `bitwise_equal = true` (`max_abs = 0.0`) against native Astropy/NumPy while removing Python slice-loop overhead |
| **7. HDF5 Output & Compression** | Single-threaded `h5py` LZF + shuffle writes | **OpenMP `PSArrayAllFinite` + Parallel LZF Writer ([parallel_lzf.py](../lightcone/parallel_lzf.py), `PS_PARALLEL_LZF=1`) + `os.fsync`** | Compresses and fsyncs 3,922 `1024/256` slices (`3.45 GB` uncompressed across 6 fields) in `11.24 s` write + `5.00 s` fsync |

---

## 3. Numerical Fidelity & Unit-Level Bitwise Gates

1. **No FMA Contraction + glibc Single-Precision Parity**: All critical CUDA kernels compile with `--fmad=false` and include [glibc_math.cuh](../lightcone/glibc_math.cuh) to match host x86-64 glibc `log10f`/`expf` and host FMA behavior.
2. **Cold vs. Warm Determinism**: Every benchmark compares cold JIT-compiling `prime` runs against `warm` runs — all 9 end-to-end runs satisfy `repeat_equal = true` (`max_diff = 0.0` across all 6 output fields).
3. **Zero-Bitwise-Diff Unit Suite ([results-489606/w4_units.json](../reports/wave6-lightcone/results-489606/w4_units.json))**:

| Unit Test | Component Verified | Max Abs Diff | `bitwise_equal` | Measured Gain / Notes |
|---|---|---:|:---:|---|
| `W4_cic_pertile_v2` | Coarse-tiled CIC atomic kernel vs. baseline CIC | `0.0` | `true` | `max_ulp = 0`, `3.37×` faster (`23.74 ms -> 7.04 ms`) |
| `W4_init_mhr_omp` | OpenMP `init_MHR` table vs. serial GSL | `0.0` | `true` | `5.19×` faster (`1641.8 ms -> 316.3 ms`) |
| `W4_heat_preplanned` | Preplanned batched FFTW thermal filter + extrema | `0.0` | `true` | `7t`: `59.43 ms`, `8t`: `61.04 ms` |
| `W4_ion_prefetch` | Async prefetched ionization density batch (32 scales) | `0.0` | `true` | Exact match vs. synchronous path |
| `W5_heat_padded_gpu_unpack` | GPU in-place unpack/extrapolate on padded FFTW output | `0.0` | `true` | Eliminates host-side unpack copy |
| `W5_ion_filter_sparse` | Sparse k-space copy for sharp-$k$ filter (`50/10/3/1 Mpc` + unfiltered) | `0` bit diffs | `true` | Zero bit differences across all tested radii |
| `W5_lightcone_fill` | `PSFillLightconeSlices` vs. native Astropy/NumPy loop | `0.0` | `true` | `last_ref == last_fast == 26` |
| `W5_heat_update_opt` | Optimized thermal rate + ODE update kernels | `0.0` | `true` | `bad_u1 = 0` |
| `W6_ion_pruned_c2r` | Pruned 3D FFTW `c2r` (7 radii `50.0` down to `2.0 Mpc`) | `0.0` | `true` | `0` bit diffs across all 7 radii |

---

## 4. End-to-End Performance & Accuracy Benchmarks

All runs below were executed on the same hardware specification (`1× NVIDIA RTX A6000 48 GB`, `OMP_NUM_THREADS=8`). Every voxel of all 6 fields (`brightness_temp`, `xH_box`, `x_e_box`, `Tk_box`, `Ts_box`, `dNrec_box`) must satisfy `abs(GPU - CPU) <= 1e-8 + 1e-4 * abs(CPU)`.

### 4.1 Four Standard Benchmarks ([summary.json](../reports/wave6-lightcone/summary.json))

| Benchmark | Grid (`DIM->HII_DIM`) / Box / Slices | CPU Wall | **Wave 6 GPU Warm (`phys` / `sync`)** | Speedup | Failed Fields | Repeat Max Diff | Report |
|---|---|---:|---:|:---:|:---:|:---:|---|
| `384/96` | `384 -> 96`, 375 Mpc, 609 | `278.18 s` | **`21.02 s`** (`16.21 s` / `18.67 s`) | **`13.23×`** | **`0 / 6`** | `0.0` | [results-489606](../reports/wave6-lightcone/results-489606/w4_units.json) |
| `h1_256_64` | `256 -> 64`, 200 Mpc, 1,227 | `107.93 s` | **`13.98 s`** (`9.59 s` / `12.42 s`) | **`7.72×`** | **`0 / 6`** | `0.0` | [results-489605](../reports/wave6-lightcone/results-489605/warm-h1_256_64-comparison.json) |
| `h2_384_96` | `384 -> 96`, 300 Mpc, 761 | `266.33 s` | **`23.29 s`** (`18.20 s` / `21.30 s`) | **`11.44×`** | **`0 / 6`** | `0.0` | [results-489605](../reports/wave6-lightcone/results-489605/warm-h2_384_96-comparison.json) |
| **`1024/256`** | `1024 -> 256`, 250 Mpc, 3,922 | `6,462.85 s` | **`371.37 s`** (`336.33 s` / `360.42 s`) | **`17.40×`** | **`0 / 6`** | `0.0` | [results-489607](../reports/wave6-lightcone/results-489607/warm-1024-timing.json) |

- **Production `1024/256` Max Absolute Errors ([warm-1024-comparison.json](../reports/wave6-lightcone/results-489607/warm-1024-comparison.json))**: `brightness_temp = 6.10e-5`, `xH_box = 1.79e-7`, `x_e_box = 4.19e-9`, `Tk_box = 1.37e-4`, `Ts_box = 1.45e-4`, `dNrec_box = 0.0`.
- **Production `1024/256` Warm Runtime Breakdown (`336.33 s` physics / `360.42 s` to fsync)**:
  - `perturbation`: `22.79 s` (92 calls)
  - `thermal`: `46.03 s` (`heat_gpu = 22.58 s`, `heat_filter = 18.96 s`, plus `heat_prefetch_wait = 95.23 s` across 90 prefetch hits / 0 misses)
  - `ionization`: `212.68 s` (`batch_density = 52.88 s`, `xerec_replay = 35.50 s`; out of `2,914` scales, `1,261` screened noop, `1,591` active density replays, `741` guarded `xe/rec` replays)
  - `brightness`: `40.28 s` (`PS_FFTW_BRIGHTNESS=1`)
  - HDF5 LZF write + `fsync`: `11.24 s` + `5.00 s` (`3.45 GB` raw lightcone data across 3,922 slices)

### 4.2 Five Multi-Parameter / Multi-Cosmology / Multi-Seed Spot-Checks (Job `489621`)

| Point ID | Regime & Parameter Shifts | Grid / Box / Slices / Seed | CPU Wall | **GPU Warm (`phys`)** | Speedup | Parity / Repeat |
|---|---|---|---:|---:|:---:|:---:|
| `p1_early_eor_strong_xray` | Early EoR + strong soft X-rays (`M_TURN=8.30, L_X=41.60, NU_X_THRESH=200`) | `256->64`, 200 Mpc, 1,227, `111111` | `111.69 s` | **`13.24 s`** (`8.83 s`) | **`8.44×`** | `0 / 6`, `0.0` |
| `p2_late_eor_weak_hard_xray` | Late EoR + weak hard X-rays (`F_STAR10=-1.65, M_TURN=10.20, L_X=38.20, NU_X_THRESH=1400`) | `256->64`, 250 Mpc, 982, `222222` | `96.61 s` | **`11.82 s`** (`7.94 s`) | **`8.17×`** | `0 / 6`, `0.0` |
| `p3_cosmo_shift_lcdm_low_kp` | Shifted LCDM (`OMm=0.325, OMb=0.0505, h=0.665, s8=0.840, ns=0.955`) | `256->64`, 180 Mpc, 1,353, `333333` | `114.76 s` | **`13.79 s`** (`9.14 s`) | **`8.32×`** | `0 / 6`, `0.0` |
| `p4_cosmo_shift_low_s8` | Low `SIGMA_8` + high star-forming threshold (`OMm=0.290, OMb=0.0465, h=0.700, s8=0.785, ns=0.980, F_STAR10=-0.15, M_TURN=10.60`) | `256->64`, 220 Mpc, 1,113, `444444` | `101.19 s` | **`13.34 s`** (`9.17 s`) | **`7.59×`** | `0 / 6`, `0.0` |
| `p5_384_96_mid_eor_cosmo` | High-resolution mixed shift (`OMm=0.315, OMb=0.0492, h=0.672, s8=0.828, ns=0.962`) | `384->96`, 300 Mpc, 839, `555555` | `266.65 s` | **`21.59 s`** (`16.43 s`) | **`12.35×`** | `0 / 6`, `0.0` |

---

## 5. Relationship to the Earlier `gpu/` Coeval Pipeline

- [gpu/](../gpu) preserves the 2026-11-19 **Coeval + Power Spectrum entrypoint** (`512³ -> 128³` 32-redshift coeval archiving with 14 IC fields and 32-bin/60-bin PS in `85.69 s` parallel LZF wall time).
- [lightcone/](../lightcone) is the current **Wave 6 Full 6-Field Lightcone pipeline**: building on the `gpu/` core fixes, it adds async background FFTW prefetch, padded thermal GPU unpack, screened ionization density replays, the `PS_XEREC_AMP_MULT=0` guard, sharp-$k$ pruned 3D FFTW `c2r`, `ComputeBrightnessFastFFTW`, OpenMP `init_MHR` / `init_ps()` caching, and C/OpenMP `PSFillLightconeSlices` — achieving strict `0 / 6` parity and up to `17.40×` end-to-end acceleration at `1024/256`.
