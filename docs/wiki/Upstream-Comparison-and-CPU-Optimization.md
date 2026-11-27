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

1. **FFTW Plan LRU Cache (`src/py21cmfast/src/dft.c`)**: Reuse `(kind, dim, dim_los, n_threads)` plans via `fftwf_execute_dft_*` instead of creating/destroying plans every radius in `IonisationBox.c`.
2. **Native C/OpenMP `apply_rsds` (`src/py21cmfast/rsds.py`)**: Port the column-parallel CIC displacement kernel out of Python/`RegularGridInterpolator` (`37.65 s` at `128×128×512`) into C/OpenMP.
3. **SIMD + OpenMP Fusion (`build_cffi.py`, `IonisationBox.c`)**: Optional `-march=native`, `#pragma omp simd`, loop fusion, and NUMA first-touch initialization.
4. **Fast Host RNG & `k^2 R^2` Filtering (`rng.c`, `filtering.c`)**: Host Philox/`xoshiro256++` RNG and avoiding inner-loop `sqrtf` calls.
5. **Scratch-Buffer Reuse**: Persist `fftwf_malloc` buffers across filter radii to eliminate repeated kernel page faults.

---

## 4. Verified Accuracy & Speed Summary

- **`strict` mode**: `rel_rms ~ 1e-4` on `384³/128³` lightcones, **`< 0.02%`** max 3D $P(k)$ difference over $k \in [0.1, 1.0]\ \mathrm{Mpc}^{-1}$, and exact PDF `hist_diff = 0`.
- **`fast` mode (6-seed ensembles on RTX 4090)**: $|\Delta \bar{x}_{\mathrm{HI}}| \le 0.00147$, $|\Delta \bar{T}_b| \le 0.3719\ \mathrm{mK}$, and 1D/2D power spectrum median $|\Delta|/\text{SEM} \in [0.68, 1.11]$ (pure cosmic variance).
- **Speedups (`1×–3× RTX 4090` vs `16T CPU`)**: `apply_rsds` **`492.74×`**, Rectilinear Lightcone **`82.05×`**, `SKA1-Low / HERA-350` 3D Mocks **`71.65× / 71.88×`**, 1-Pt Moments/Wedge **`26.43× / 23.86×`**, `AngularLightconer` **`20.65×`**, 3z `L-INTEGRAL+TS` Coeval **`12.36×`**, Zero-D2H Lightcone **`-21.9%` wall time**, `2048³` Zel'dovich IC **`58.467 s`** (**`3.71×`**), `run_batch` up to **`9,782.0 tasks/h`**.
