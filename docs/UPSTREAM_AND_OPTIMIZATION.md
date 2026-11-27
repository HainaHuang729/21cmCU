# Upstream 21cmFAST Trajectory, GPU Coverage Matrix & CPU Optimization Guide

This document records the technical relationship between **21cmCU** (`v0.7.0`) and upstream **21cmFAST v4**, audits exact GPU porting coverage vs remaining CPU fallback paths, outlines high-leverage optimizations for the C/OpenMP CPU backend, and summarizes verified precision and throughput baselines.

---

## 1. Upstream `21cmFAST` v4 Trajectory & Complementarity

Upstream [`21cmfast/21cmFAST`](https://github.com/21cmfast/21cmFAST) drives forward astrophysical modeling and software architecture around the **v4** paradigm (*Davies, Mesinger & Murray 2025, A&A*), evolving through `v4.3.x` releases and community pull requests:

| Dimension | Upstream `21cmFAST` v4 Focus | `21cmCU` (`v0.7.0`) Complementarity |
| --- | --- | --- |
| **Source Physics & Multi-Tracer** | Backward stochastic halo sampler (`CHMF-SAMPLER`) with Sheth–Tormen CHMFs, shot-noise, intrinsic galaxy scatter, joint JWST UVLF + LIM + 21cm forward modeling, Ly$\alpha$ multiple scattering. | Full native CUDA + Philox + table caching implementation of `CHMF-SAMPLER`, `L-INTEGRAL`, `E-INTEGRAL`, `DEXM-ESF`, `USE_MCGS`, `CLASS`, and `LYA_MULTIPLE_SCATTERING`. |
| **Ongoing Upstream Extensions** | Active PR tracks: HEALPix spherical angular lightcones, Fixed-and-Paired variance-reduced ICs, Fuzzy Dark Matter, primordial local non-Gaussianity ($f_{\mathrm{NL}}$). | Orthogonal systems layer ready to accelerate stabilized upstream fields once merged; currently supports `RectilinearLightconer` and `AngularLightconer` (`order ∈ {0, 1}`). |
| **Compute & Memory Execution** | OpenMP + FFTW3f CPU backend (`USE_CUDA=0`), selective coeval caching, sample-first angular lightconing. | Native C/CUDA + cuFFT inside `c_21cmfast`, `DeviceFieldRegistry` GPU residency, `CUDA_LAZY_D2H`, `DeviceLZF` HDF5, and multi-GPU Y-slab FFT (`up to 2048³` Zel'dovich on `3× RTX 4090`). |
| **Inference & Observation Stack** | Separate companion packages (`21CMMC`, `21cmemu`, `tuesday`). | Integrated zero-copy `__dlpack__` / `to_torch`, `SBISimulator`, `Likelihood21cm`, GPU 3D interferometer mocks (`SKA1-Low`/`HERA-350`/`MWA-II`/`LOFAR`), `quick_lightcone`, and WebGL2 3D visualization. |

---

## 2. GPU Porting Coverage & CPU Fallback Matrix

All core 21cm forward-model drivers dispatch directly into **82 native CUDA APIs** (`src/py21cmfast/src/cuda/`):
- `InitialConditions.c`, `PerturbedField.c`, `HaloCatalog.c`, `PerturbedHaloCatalog.c`, `Stochasticity.c`, `EmissivityFields.c`, `RadiationFieldsSetup.c`, `RadiationFields.c`, `SpinTemperatureBox.c`, `IonisationBox.c`, `BrightnessTemperatureBox.c`, `recombinations.c`, `interp_tables.c`, plus Python-level GPU drivers (`rsds.py`, `lightconers.py`, `py21cmcu.analysis`, `observables`, `instrument`, `interop`, `io`).

The table below lists every remaining path that stays on CPU or triggers an automatic CPU fallback:

| Component / Flag | Behavior in `21cmCU v0.7.0` | Code Location | Status / Plan |
| --- | --- | --- | --- |
| `USE_INTERPOLATION_TABLES="no-interpolation"` | Direct GSL quadrature per cell falls back to CPU OpenMP (`uses_hmf_interpolation(...) == false`). | `EmissivityFields.c`, `IonisationBox.c`, `Stochasticity.c`, `interp_tables.c` | Intentional: interpolation tables (`USE_INTERPOLATION_TABLES=DEFAULT`) are standard for production & fully GPU-ported. |
| `strict` mode + `CHMF-SAMPLER` | GPU Philox sampler runs in `CUDA_MODE="fast"` (`p21_cuda_get_fast_halo_sampler()`); `strict` falls back to CPU GSL stream for exact RNG sequence matching. | `Stochasticity.c:1135-1137` | Intentional precision contract (`strict` matches CPU RNG draw-for-draw). |
| `AngularLightconer` (`interpolation_order > 1`) | Cubic/spline (`order >= 2`) falls back to CPU `scipy.ndimage.map_coordinates`; `order ∈ {0, 1}` runs natively on GPU. | `src/py21cmfast/lightconers.py:873-875` | Sufficient for standard `order=0/1`; higher-order spline kernel can be added if needed. |
| `HII_DIM == 1` | Single-voxel debug boxes bypass GPU dispatch. | `EmissivityFields.c`, `IonisationBox.c` | Negligible cost on 1 cell. |
| `photoncons.c` & `LuminosityFunction.c` | Auxiliary 1D calibration / UVLF binning runs on CPU. | `photoncons.c`, `LuminosityFunction.c` | Low wall-time fraction; `PhotonCons` is cached per parameter point. |
| `CLASS` Transfer Functions | `classy` Boltzmann solver runs on CPU, accelerated `~6,000×–7,783×` across runs via in-memory LRU + atomic `.npz` disk cache (`py21cmcu.clear_class_cache()`). | `src/py21cmfast/wrapper/inputs.py` | Cached cleanly (`17.19 s → 2.2 ms` mem / `2.8 ms` disk). |
| Multi-GPU `1024³+` full EoR & `<160 GiB` `2048³` 2LPT | `cuda_mgpu.cu` currently covers Y-slab FFT, IC (`ZELDOVICH` & `2LPT`), `power_spectrum_1d/2d`, and streaming `PerturbField`; full multi-GPU `IonisationBox`/`SpinTemperatureBox` and `<160 GiB` host `2048³` 2LPT remain planned. | `cuda_mgpu.cu`, `cuda_ic.cu` | Scheduled for **`v0.8.0` (Direction P)**. |
| Upstream Unreleased PR Physics | HEALPix lightcones, Fixed-and-Paired ICs, Fuzzy Dark Matter, local $f_{\mathrm{NL}}$. | Upstream PRs | Tracked for post-merge adoption. |

---

## 3. CPU Backend (`C / OpenMP / FFTW3f`) Optimization Roadmap

Even without a GPU (e.g., laptops, macOS, or CPU-only CI/login nodes), profiling identifies **5 high-leverage CPU improvements**:

1. **FFTW Plan LRU Cache (`src/py21cmfast/src/dft.c`) — High Impact, Low Risk**
   - Currently, every call to `dft_r2c_cube` and `dft_c2r_cube` invokes `fftwf_plan_dft_*` → `fftwf_execute` → `fftwf_destroy_plan`. Because `IonisationBox.c` filters across ~20–40 excursion-set radii per redshift, hundreds of identical-shape plans are repeatedly constructed and torn down.
   - **Optimization**: Cache `(kind, dim, dim_los, n_threads)` FFTW plans in a bounded LRU and invoke `fftwf_execute_dft_r2c(plan, ...)` / `fftwf_execute_dft_c2r(plan, ...)` (safe on `fftwf_malloc`-aligned 16/32-byte buffers).
2. **Native C/OpenMP `apply_rsds` Kernel (`src/py21cmfast/rsds.py`) — Eliminates Largest Single-Node CPU Bottleneck**
   - At `128 × 128 × 512`, `apply_rsds` dominates CPU lightcone runtime (`37.65 s` out of `248.77 s` due to Python `RegularGridInterpolator` + `cosmotile` CIC).
   - **Optimization**: Port the column-parallel quadratic-velocity + sub-cell CIC kernel from `cuda_analysis.cu` to C/OpenMP in `c_21cmfast`, yielding an estimated `>10×` pure-CPU lightcone RSD speedup.
3. **Architecture-Specific SIMD & OpenMP Region Fusion (`build_cffi.py`, `IonisationBox.c`, `EmissivityFields.c`)**
   - `build_cffi.py` compiles with `-Ofast`; adding opt-in `-march=native` (`P21_CPU_NATIVE=1`) unlocks AVX2 / AVX-512 / Apple NEON vectorization, while fusing back-to-back short `#pragma omp parallel` loops and applying NUMA first-touch initialization improves 16–64 core scaling.
4. **Fast Counter-Based Host RNG & `k^2 R^2` Filter Simplification (`rng.c`, `filtering.c`)**
   - `rng.c` notes per-snapshot `seed_rng_threads` overhead; an optional host Philox/`xoshiro256++` path speeds up IC/stochastic draws, while `filtering.c` can evaluate $k^2 R^2$ instead of repeated `sqrtf(k^2) * R` evaluations inside inner 3D spectral loops.
5. **Context-Level Scratch Pool (`IonisationBox.c`, `PerturbedField.c`)**
   - Reusing persistent `fftwf_malloc` scratch arrays across filter radii avoids repeated `mmap`/`munmap` kernel page faults on `512³–1024³` CPU runs.

---

## 4. Verified Accuracy & Speed Summary (`v0.7.0`)

### Numerical Precision (`strict` vs CPU, & 6-Seed `fast` vs `strict`)

- **`CUDA_MODE="strict"`**:
  - `384³ / 128³` lightcone voxel-wise `rel_rms ~ 1e-4`; 3D power spectrum deviation **`< 0.02%`** over $k \in [0.1, 1.0]\ \mathrm{Mpc}^{-1}$.
  - GPU 1-point histogram PDF matches CPU exactly (`hist_diff = 0`).
  - Verified across **`168/168`** `v0.6.0` native tests plus **`21/21`** `v0.7.0` tests (Phases 7–9) on RTX 4090 (`CUDA 12.6`, `sm_89`).
- **`CUDA_MODE="fast"`** (6-seed ensembles on `RTX 4090`, full details in [`docs/CALIBRATION.md`](CALIBRATION.md)):

| Regime (`HII_DIM / DIM`) | `strict` [s] | `fast` [s] | $\max\|\Delta \bar{x}_{\mathrm{HI}}\|$ | $\max\|\Delta \bar{T}_b\|$ | $\max\|\Delta \bar{T}_s\|/\bar{T}_s$ | 1D $\Delta^2_{21}$ Med $|\Delta|/\text{SEM}$ | 2D $P$ Med $|\Delta|/\text{SEM}$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `E-INTEGRAL` (`96/192`, 192 Mpc) | `1.230` | `1.062` | `0.00147` | `0.0551 mK` | `0.29%` | `0.68` | `0.70` |
| `L-INTEGRAL` (`96/192`, 192 Mpc) | `1.055` | `0.972` | `0.00023` | `0.2339 mK` | `0.61%` | `0.88` | `0.86` |
| `CHMF-SAMPLER + MCG + CLASS` (`64/192`, 150 Mpc) | `23.161` | `18.933` | `0.00064` | `0.3719 mK` | `1.07%` | `0.92` | `1.11` |

### Throughput Highlights (`16-Core CPU` vs `1×–3× RTX 4090`)

| Workload | Grid | CPU 16T | `21cmCU` GPU (`fast`) | Speedup / Savings |
| --- | --- | ---: | ---: | ---: |
| `apply_rsds` | `128×128×512` | `37.646 s` | `0.0764 s` (`1× 4090`) | **`492.74×`** |
| Rectilinear Lightcone (`+ dvdr + RSD`) | `256/128`, `z=6.5–11` | `248.773 s` | `3.032 s` (`1× 4090`) | **`82.05×`** |
| `SKA1-Low / HERA-350` 3D Mock | `256³` | `1.651 s / 1.616 s` | `23.04 ms / 22.48 ms` | **`71.65× / 71.88×`** |
| 1-Pt Moments+PDF / Horizon Wedge | `256³` | — | Single-pass GPU | **`26.43× / 23.86×`** |
| `AngularLightconer` (`+ dvdr + RSD`) | `256/128`, `z=7.8–9.8` | `97.018 s` | `4.698 s` (`1× 4090`) | **`20.65×`** |
| Coeval 3z (`L-INTEGRAL + TS`) | `512/256`, `z=12,10,8` | `55.079 s` | `4.456 s` (`1× 4090`) | **`12.36×`** |
| `768³` EoR Coeval (`E-INTEGRAL`) | `768³`, `z=7.0` | `110.209 s` | `16.435 s` (`1× 4090`) | **`6.71×`** |
| Zero-D2H Lightcone (`lazy_d2h=True`) | `256/128`, 12z `L-INTEGRAL` | `1.496 s` (`lazy_d2h=False`) | **`1.169 s`** (`1× 4090`) | **`-21.9%` time / `-8.5%` D2H** |
| `2048³` Zel'dovich IC | `3× RTX 4090` | `216.665 s` (`v0.5.0` host) | **`58.467 s`** (native VRAM) | **`3.71×`** |
| Concurrent SBI `run_batch` | `16× 192/96`, `2× 4090` | — | `5.888 s` (**`9,782.0 tasks/h`**) | **`1.41×` vs 1 GPU** |
