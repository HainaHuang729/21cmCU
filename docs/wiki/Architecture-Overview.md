# Architecture Overview

Back to [Home](Home.md) · Previous: [CLI & API Reference](CLI-and-API-Reference.md) · Next: [Troubleshooting & FAQ](Troubleshooting-and-FAQ.md)

## Native C/CUDA Engine Inside `c_21cmfast`

Instead of dispatching PyTorch/CuPy arrays across Python bindings, 21cmCU compiles all CUDA C/C++ kernels directly inside `src/py21cmfast/c_21cmfast` via `build_cffi.py`:

- **Memory & Caching**:
  - `cuda_mempool.cu`: `cudaMemPool` arena allocator, `DeviceFieldRegistry` keyed by 16 KB parallel GPU chunk-tree cryptographic hashes (`F_hash`), LRU stale eviction (`reclaim_stale_registry_for_oom`), and cuFFT plan LRU (up to 16 cached plans).
  - 64-bit element indexing across all 3D kernels eliminates 32-bit overflow above $2^{31}$ voxels (`1536³–2048³`).
- **Physical Simulation Stages**:
  - `cuda_ic.cu` & `cuda_philox.cuh`: parallel Philox `4×32-10` RNG, `EH` / `CLASS` (`FLUCTS`) power-spectrum seeding, Zel'dovich & low-VRAM 2LPT potential scheduling.
  - `cuda_perturb.cu`: adaptive float32/float64 displacement/velocity perturbation (`slab-X` streaming).
  - `cuda_tables_build.cu`, `cuda_sampler.cu`, `cuda_halos.cu`, `cuda_dexm.cu`, `cuda_sources.cu`: discrete halo realizations (`CHMF-SAMPLER`, `DEXM-ESF`, `L-INTEGRAL`, `USE_MCGS`).
  - `cuda_radiation.cu`, `cuda_spin_temp.cu`, `cuda_ionization.cu`, `cuda_brightness.cu`: spin temperature (`LYA_MULTIPLE_SCATTERING`), inhomogeneous recombinations, `E-INTEGRAL`/`CONST-ION-EFF`/`L-INTEGRAL` ionization, and brightness temperature.
- **Analysis, Lightcones & Multi-GPU**:
  - `cuda_analysis.cu`: `include_dvdr_in_tau21`, column-parallel `rsds_shift` / `apply_rsds`, `RectilinearLightconer` / `AngularLightconer` kernels, and cuFFT 1D/2D power spectra.
  - `cuda_io.cu`: GPU byte-shuffled `liblzf` HDF5 chunk compressor (`DeviceLZF`).
  - `cuda_mgpu.cu`: Y-slab distributed 3D cuFFT, 64 MiB pinned portable bounce buffers (`MgpuHostBounce`), `std::thread` sub-tiled transposes, and single-work-slab streaming.

---

## Standalone Bug Fixes Over Upstream 21cmFAST v4.3.2

1. **Photon-conservation calibration arrays (`photoncons.py`)**: aligned redshift and neutral-fraction lengths across `compute_fesc_calibrations` / spline construction.
2. **`st_taylor_factor` null guard (`hmf.c`)**: safe guard for optional null outputs when `SAMPLE_METHOD="PARTITION"`.
3. **Previous-redshift ACG table separation (`interp_tables.c` / `cuda_ionization.cu`)**: dedicated `Nion_conditional_table1D_acg_prev` prevents cross-step table contamination under `E-INTEGRAL` + `USE_MCGS=True` without photoheating feedback.
