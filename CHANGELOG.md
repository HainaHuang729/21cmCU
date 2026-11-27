# Changelog

All notable milestones in **21cmCU** are documented in this file.

## [0.5.0] — 2026-11-27 (Direction G + Full Benchmarks & Figures)
### Added
- **Direction G (`9ec08a4`)**: Native multi-GPU 1D Y-slab domain decomposition (`cuda_mgpu.cu`/`.cuh`) for 3D R2C/C2R FFTs, `compute_initial_conditions` (`ZELDOVICH` + hybrid host-accumulated `2LPT`), and `power_spectrum_1d` / `power_spectrum_2d` with `CUDA_DEVICES` / `P21_CUDA_DEVICES` configuration, `cuda_device_count()`, `get_cuda_devices()`, `set_cuda_devices()`, `use_mode(..., devices=...)`, and `mgpu_fft_runs` / `mgpu_ic_runs` stats (`12/12` tests passing in `tests/native/test_21cmcu_phase5_g.py` across `2×` and `3× RTX 4090`).
- **Full Benchmark Sweep & Visualization (`docs/BENCHMARKS.md`, `benchmarks/plot_benchmarks.py`, `docs/figures/*.png`)**: measured on node `b1` (16 CPU threads vs `1×–3× RTX 4090`) into `benchmarks/results_rtx4090_v050.json`, demonstrating `492.74×` on `apply_rsds`, `82.05×` on `RectilinearLightconer`, `20.65×` on `AngularLightconer`, `12.36×` on 3z `L-INTEGRAL+TS`, `1.83×` HDF5 write speedup with `DeviceLZF`, `-21.7%` D2H with `CUDA_LAZY_D2H`, `1536³` (`46.131 s` Zel'dovich) & `1792³` (`98.095 s` Zel'dovich) native multi-GPU IC on `3× RTX 4090`, and `10,502.0 tasks/h` multi-GPU `run_batch`.
- **Full `v0.5.0` native GPU suite**: **`162/162`** passing tests (`-k "not speed"` across all 11 native test suites on `3× RTX 4090`).

## [0.4.0] — 2026-11-26 (Directions H, I & J)
### Added
- **Direction H (`ec27d60`)**: In-memory + atomic `.npz` disk caching for `CLASS` `CosmoTables` (`17.1857 s` cold → `0.002208 s` memory **`7,783×`** / `0.0028 s` disk **`6,214×`**, `py21cmcu.clear_class_cache()`), `AngularLightconer` `run_classy` caching, and native CUDA `p21_cuda_angular_slice` / `p21_cuda_angular_los_velocity` (`0.0934 s` GPU vs `0.1644 s` CPU on `64 × 128³` `order=1` slices — **`1.76×`**; `6/6` tests passing in `tests/native/test_21cmcu_phase4_h.py`).
- **Direction I (`0360f70`)**: Native CUDA `DeviceLZF` HDF5 compression (`cuda_io.cu`, `py21cmcu.save_lightcone`, **`2.91×`** faster than CPU LZF at `0.2366 s` vs `0.6893 s` on `129.11 MB` `256/128` lightcones) and opt-in `CUDA_LAZY_D2H` GPU-only intermediate residency (`use_mode(..., lazy_d2h=True)`, PEP 688 `_CudaBufferOwner` GC tracking, cutting `256/128` `L-INTEGRAL+TS` lightcone D2H transfers by **`24.3%`** (`1723.60 → 1304.17 MB`) and runtime `1.912 s → 1.322 s` (**`1.45×`**); `9/9` tests passing in `tests/native/test_21cmcu_phase4_i.py`).
- **Direction J**: Removed legacy v3 (`lightcone/`, `gpu/`), CuPy v4 adapter (`v4/`, `reports/`), and obsolete fixtures from `main` (preserved under git tag `archive-cupy-v4`), added GitHub Actions CPU CI (`.github/workflows/ci.yml`), bumped version to `0.4.0`, and verified **`150/150` passing tests** on RTX 4090 (`sm_89`).

## [0.3.0] — 2026-11-26 (Directions D, E & F)
### Added
- **Direction D (`fa22efc`)**: Native CUDA `include_dvdr_in_tau21`, `rsds_shift` / `apply_rsds` (`476.90×` speedup at `128×128×512`), `RectilinearLightconer` batch GPU slicing, `BrightnessTemp`/`tau_21` `DeviceFieldRegistry` retention, and cuFFT-backed `power_spectrum_1d` / `power_spectrum_2d` / `lightcone_chunk_power_spectra` (`19/19` tests passing in `tests/native/test_21cmcu_phase3_d.py`).
- **Direction E (`b8e7950`)**: Spawned multi-GPU / concurrent batch runner (`py21cmcu.run_batch`, `save_batch_h5`, `4,797.8 tasks/h` at `192/96` on RTX 4090 with `workers_per_gpu=2`) and 6-seed `strict` vs `fast` calibration across `E-INTEGRAL`, `L-INTEGRAL`, and `CHMF-SAMPLER+MCG+CLASS` (`docs/CALIBRATION.md`, `tests/native/test_21cmcu_phase3_e.py`).
- **Direction F**: `build_cffi.py` toolchain & `CUDA_ARCHS=auto` SM auto-detection (verified compiling `sm_89` on RTX 4090 without `CUDA_HOME`), unified `benchmarks/bench_21cmcu.py` CLI, and `v0.3.0` release verification (`135/135` passing tests across all native suites).

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
