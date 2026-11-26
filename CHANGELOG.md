# Changelog

All notable milestones in **21cmCU** are documented in this file.

## [0.3.0-dev] — In Progress (Directions D / E / F)
### Planned & Implementing
- **Direction D**: Native CUDA `include_dvdr_in_tau21`, `rsds_shift` / `apply_rsds`, `RectilinearLightconer` batch GPU slicing, and cuFFT-backed 1D spherical & 2D cylindrical power spectra (`power_spectrum_1d`, `power_spectrum_2d`, `lightcone_chunk_power_spectra`).
- **Direction E**: Spawned multi-GPU / concurrent batch runner (`py21cmcu.run_batch`) for parameter sweeps & SBI plus statistical `strict` vs `fast` calibration across source models (`docs/CALIBRATION.md`).
- **Direction F**: `build_cffi.py` toolchain & SM architecture auto-detection (`CUDA_ARCHS=auto`), reproducible `benchmarks/bench_21cmcu.py` CLI, and `v0.3.0` release packaging.

## [0.2.0] — 2026-11-26 (Phase 1 / C & Phase 2 / B)
### Added
- **Phase 1 / Direction C (`924f940`)**:
  - Parallel Philox 4×32-10 initial-conditions RNG (`cuda_philox.cuh`) and parallel 16 KB GPU chunk-tree hashing (`cuda_mempool.cu`).
  - Full GPU acceleration and table caching for `CHMF-SAMPLER`, `DEXM-ESF`, `L-INTEGRAL`, `USE_MCGS`, `V_CB_MODEL="FLUCTS"` (`CLASS`), `LYA_MULTIPLE_SCATTERING`, and conditional/unconditional emissivity tables.
  - Expanded test coverage in `tests/native/test_21cmcu_phase1.py`.
- **Phase 2 / Direction B (`f276a7a`)**:
  - 64-bit grid-index arithmetic across all CUDA kernels supporting `512³–1024³` boxes and beyond.
  - Configurable device registry budget (`P21_CUDA_REGISTRY_MB`) and stale-LRU reclamation on mempool OOM (`reclaim_stale_registry_for_oom`).
  - Low-VRAM 2LPT potential scheduling (`10.331 s` vs `85.784 s` CPU for `512/1024` lo-res 2LPT on RTX 4090 — **8.30×** speedup), `27.630 s` `512/1024` full coeval, and `29.673 s` `768³` EoR box within 24 GB VRAM.
  - Adaptive float32/float64 perturbation accumulation preserving deterministic CPU/GPU parity on smaller grids.
  - Regression & high-res test suite `tests/native/test_21cmcu_phase2.py` (`108/108` passing tests total on RTX 4090).

## [0.1.0] — 2026-11-26 (Phase 0)
### Added
- **`f3285f4`**: Created standalone **21cmCU** engine (`py21cmcu` / `21cmcu`) on top of native C/CUDA + cuFFT inside `c_21cmfast` (zero CuPy runtime dependency).
- Exposed dual execution modes `strict` (CPU-matched parity) and `fast` (statistically equivalent throughput mode).
- Fixed upstream v4.3.2 bugs in `photoncons.py`, `hmf.c` (`st_taylor_factor` null guard), and `interp_tables.c` / `cuda_ionization.cu` (`Nion_conditional_table1D_acg_prev`).
- Archived legacy CuPy v4 adapter and v3 CUDA paths at tag `archive-cupy-v4`.
