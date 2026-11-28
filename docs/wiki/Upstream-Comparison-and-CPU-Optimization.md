# Upstream 21cmFAST Comparison & CPU Optimization

This page summarizes how **21cmCU (`v0.7.0`)** relates to upstream **21cmFAST v4**, which code paths currently remain on CPU, and the highest-impact CPU optimization opportunities alongside verified accuracy & speedup baselines. (See full details in [`docs/UPSTREAM_AND_OPTIMIZATION.md`](https://github.com/HainaHuang729/21cmCU/blob/main/docs/UPSTREAM_AND_OPTIMIZATION.md).)

---

## 1. Upstream `21cmFAST` v4 Trajectory

Upstream `21cmFAST` (`v4.3.x`, Davies, Mesinger & Murray 2025, *A&A*) focuses on **CPU astrophysics and modular C/Python design**:
- **Released v4 core**: backward stochastic halo sampler (`CHMF-SAMPLER`), Sheth–Tormen CHMFs, joint JWST UVLF + LIM + 21cm forward modeling, and `Lyα` multiple scattering.
- **Active upstream PRs**: HEALPix angular lightcones, Fixed-and-Paired initial conditions, Fuzzy Dark Matter, and primordial local non-Gaussianity ($f_{\mathrm{NL}}$).
- **21cmCU's role**: full API/physics parity on the released v4 physics stack, accelerated by **82 native CUDA APIs**, `CUDA_LAZY_D2H` residency, `DeviceLZF` HDF5, multi-GPU Y-slab FFT (`2048³`), single-pass interferometer mocks (`SKA1-Low`/`HERA-350`), zero-copy DLPack/`to_torch`, and interactive 3D WebGL2 visualization.

---

## 2. Unported / CPU-Fallback Paths

All primary simulation stages run natively on GPU (`InitialConditions`, `PerturbedField`, `DEXM-ESF`, `CHMF-SAMPLER`, `EmissivityFields`, `RadiationFields`, `SpinTemperatureBox`, `IonisationBox`, `BrightnessTemperatureBox`, `Recombinations`, RSDs, `RectilinearLightconer`, `AngularLightconer` `order <= 1`, `power_spectrum_1d/2d`, `mock_observation_3d`). Remaining CPU paths are:

| Path / Flag | Behavior | Target Release |
| --- | --- | --- |
| `USE_INTERPOLATION_TABLES="no-interpolation"` | Direct GSL quadrature on CPU | Intentional (use default tables for GPU) |
| `CUDA_MODE="strict"` + `CHMF-SAMPLER` | CPU GSL RNG stream for exact seed matching (`fast` runs on GPU Philox) | Intentional (`strict` parity contract) |
| `AngularLightconer` (`interpolation_order > 1`) | Higher-order spline uses CPU `map_coordinates` | Optional future kernel |
| `photoncons.c` / `LuminosityFunction.c` / `CLASS` | CPU auxiliary steps (`CLASS` cached `17.19 s → 2.2 ms` mem / `2.8 ms` disk) | Cached |
| Multi-GPU `>=1024³` full EoR & `<160 GiB` `2048³` 2LPT | Host staging / single-GPU limits beyond 24 GB | **`v0.8.0` (Direction P)** |
| Upstream PRs (HEALPix, Fixed/Paired IC, FDM, $f_{\mathrm{NL}}$) | Not yet merged upstream | Post-merge evaluation |

---

## 3. CPU Backend (`C / OpenMP / FFTW3f`) Optimization Opportunities

1. **FFTW Plan LRU Cache (`src/py21cmfast/src/dft.c`) — Implemented (`max_abs = 0.0`)**: 32-entry thread-safe LRU cache keyed by `(is_c2r, use_wisdom, dim, dim_los, n_threads, align64)` using `fftwf_execute_dft_*` (`P21_FFTW_PLAN_CACHE=1`). Verified **`589` hits / `9` misses** (`max_abs = 0.0` across all 25 arrays) on `256/128` 16T Coeval.
2. **Native C/OpenMP `apply_rsds` (`src/py21cmfast/src/rsds_cpu.c`, `src/py21cmfast/rsds.py`) — Implemented (`395.26×`, `max_abs = 0.0`)**: Column-parallel OpenMP kernel compiled with `#pragma STDC FP_CONTRACT OFF` and `no-fast-math` (`P21_CPU_RSDS_NATIVE=1`). Accelerates `128×128×512` 16-core CPU `apply_rsds` from **`38.2230 s` → `0.0967 s`** (**`395.26×`**) with zero per-pixel error (`max_abs = 0.0`).
3. **Opt-In `-march=native` (`build_cffi.py`) & Multi-Box Filter Fusion (`IonisationBox.c`, `RadiationFields.c`, `filtering.c`) — Implemented (`max_abs = 0.0`)**: Added `P21_CPU_NATIVE=1` (`-march=native`), hoisted `switch (filter_type)` out of the OpenMP 3D loop in `p21_filter_boxes_fused()`, and grouped shared-filter boxes across radii and spherical radiation shells (`P21_CPU_FILTER_FUSED=1`).
4. **FFTW `fftwf_malloc` Scratch Pool (`src/py21cmfast/src/dft.c`, `IonisationBox.c`, `RadiationFields.c`, `PerturbedField.c`) — Implemented (`max_abs = 0.0`)**: 24-slot mutex-guarded scratch pool (`P21_CPU_SCRATCH_POOL=1`, `P21_CPU_SCRATCH_POOL_MB=2048`), eliminating repeated `mmap`/`munmap` page-fault churn and yielding **`1.227×`** (`256/128` `E-INTEGRAL+TS`, `9.0989 s → 7.4172 s`), **`1.205×`** (`256/128` `L-INTEGRAL+TS+MCGS`, `11.6838 s → 9.6985 s`), and **`1.114×`** (`512/256` `E-INTEGRAL+TS`, `63.2424 s → 56.7649 s`) full 16T CPU speedups with **`max_abs = 0.0`**.
5. **Optional Fast Host RNG (`rng.c`)**: Replacing CPU GSL streams with a host Philox RNG remains intentionally opt-in/future since it changes random sequences relative to upstream CPU.

---

## 4. Verified Accuracy & Speed Summary

- **`strict` mode**: `rel_rms ~ 1e-4` on `384³/128³` lightcones, **`< 0.02%`** max 3D $P(k)$ difference over $k \in [0.1, 1.0]\ \mathrm{Mpc}^{-1}$, and exact PDF `hist_diff = 0`.
- **`fast` mode (6-seed ensembles on RTX 4090)**: $|\Delta \bar{x}_{\mathrm{HI}}| \le 0.00147$, $|\Delta \bar{T}_b| \le 0.3719\ \mathrm{mK}$, and 1D/2D power spectrum median $|\Delta|/\text{SEM} \in [0.68, 1.11]$ (pure cosmic variance).
- **Speedups (`1×–3× RTX 4090` vs `16T CPU`, plus native CPU RSDS)**: Native C/OpenMP `apply_rsds` **`395.26×` CPU speedup** (`38.2230 s → 0.0967 s`, `max_abs = 0.0`), Rectilinear Lightcone **`82.05×`**, `SKA1-Low / HERA-350` 3D Mocks **`71.65× / 71.88×`**, 1-Pt Moments/Wedge **`26.43× / 23.86×`**, `AngularLightconer` **`20.65×`**, 3z `L-INTEGRAL+TS` Coeval **`12.36×`**, Zero-D2H Lightcone **`-21.9%` wall time**, `2048³` Zel'dovich IC **`58.467 s`** (**`3.71×`**), `run_batch` up to **`9,782.0 tasks/h`**.
