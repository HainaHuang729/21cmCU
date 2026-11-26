# 21cmCU Roadmap & Milestone Record

**21cmCU** is a standalone native C/CUDA + cuFFT 21cm cosmological simulation engine derived from the `21cmFAST-GPU` fork (21cmFAST v4.3.2 semantics) with full `py21cmcu` / `py21cmfast` API and CLI compatibility and zero runtime CuPy dependency.

## Core Architecture

- **Native C/CUDA + cuFFT inside `c_21cmfast`**: All GPU kernels (`cuda_*.cu`) compile directly into the CFFI shared library, eliminating Python/CuPy launch overhead and external CuPy requirements.
- **Dual precision modes (`CUDA_MODE`)**:
  - `strict`: CPU-matched parity mode using matched FFT conventions, deterministic RNG paths, and strict IEEE floating-point order.
  - `fast`: Statistically equivalent high-speed mode (parallel Philox 4×32-10 IC RNG, float32 accumulation where bounded, fast halo sampler / FFT paths) preserving unbiased global histories and power spectra.
- **End-to-end GPU residency (`DeviceFieldRegistry` + `cudaMemPool`)**: Host field pointers are tracked with parallel 16 KB chunk-tree cryptographic fingerprints (`F_hash`) so unchanged coeval/perturbation/emissivity fields remain GPU-resident across redshifts and radii with minimal or zero H2D transfers.
- **Cached cuFFT Plan LRU**: Up to 16 padded/unpadded R2C/C2R plans cached with external mempool work areas to handle `256³`–`1024³` transforms inside a single-GPU VRAM budget (e.g., 24 GB RTX 4090).

---

## Completed Milestones

### Phase 0 (`f3285f4`) — Engine Unification & Upstream Bug Fixes
- Branded standalone package as **`21cmCU`** (`py21cmcu` / `21cmcu` CLI) alongside full `py21cmfast` namespace compatibility.
- Preserved legacy CuPy v4 adapter and v3 CUDA sources under tag `archive-cupy-v4`.
- Fixed upstream v4.3.2 blockers:
  - Photon-conservation calibration redshift/neutral-fraction length alignment (`photoncons.py`).
  - Safe null-output guard in Sheth–Tormen `st_taylor_factor` for `SAMPLE_METHOD="PARTITION"` (`hmf.c`).
  - Separate previous-redshift 1D ACG ionizing-efficiency table (`Nion_conditional_table1D_acg_prev`) for `E-INTEGRAL` + `USE_MCGS=True` (`interp_tables.c` / `cuda_ionization.cu`).

### Phase 1 / Direction C (`924f940`) — End-to-End GPU Halo, MCG & CLASS Residency
- Parallel **Philox** IC RNG and **GPU chunk-tree hashing** (`cuda_philox.cuh`, `cuda_mempool.cu`) eliminating host RNG and single-thread hashing bottlenecks at large `DIM`.
- Full GPU acceleration for `CHMF-SAMPLER`, `DEXM-ESF`, `L-INTEGRAL`, `USE_MCGS` molecular cooling, `V_CB_MODEL="FLUCTS"` (`CLASS`), `LYA_MULTIPLE_SCATTERING`, and conditional/unconditional emissivity integration tables (`cuda_tables_build.cu`, `cuda_sampler.cu`, `cuda_sources.cu`, `cuda_halos.cu`, `cuda_dexm.cu`).

### Phase 2 / Direction B (`f276a7a`) — Extreme-Scale (`512³–1024³`) Single-GPU Support
- **64-bit indexing audit**: Fixed 32-bit `int` element offsets across IC/2LPT, perturbation, spin temperature, radiation, and ionization kernels for `DIM >= 1290` (`N >= 2^31`).
- **Dynamic VRAM & registry budget (`P21_CUDA_REGISTRY_MB`)**: Capped registry footprint and added automatic stale-LRU eviction under mempool OOM pressure (`reclaim_stale_registry_for_oom`) so `768³` EoR and `512/1024` 2LPT coevals fit cleanly in 24 GB VRAM.
- **Large-box cuFFT / perturbation residency**: Reordered free-memory-aware 2LPT potential reuse, adaptive float/double perturbation accumulation (keeping deterministic double accumulation for `box_bytes < 2 GiB`), and resident `L-INTEGRAL` emissivity/ionization paths.

#### Verified RTX 4090 Benchmarks (`Phase 2 / f276a7a`, `N_THREADS=8`, `--mem=48G`)

| Benchmark | Config (`HII_DIM / DIM`, Box, Regime, Mode) | Wall Time | vs CPU (8T) | H2D Transfer | Peak / Post VRAM |
|---|---|---:|---:|---:|---|
| `B1` | `256 / 512`, `L-INTEGRAL + TS`, 2z (`fast`) | `5.070 s` | — | `0.00 GB` | `4.71 / 23.55 GB` |
| `B2` | `512 / 512`, `L-INTEGRAL + TS`, 2z (`fast`) | `32.087 s` | — | `10.50 GB` | `17.52 / 23.55 GB` |
| `B3` | `512 / 1024` `IC + PF` (2LPT `lo`, `fast`) | `10.331 s` | **`8.30×`** (`85.784 s`) | `1.50 GB` | `6.71 / 23.55 GB` |
| `B3` | `512 / 1024` `IC + PF` (2LPT `lo`, `strict`) | `11.574 s` | `7.41×` | `1.50 GB` | `6.74 / 23.55 GB` |
| `B4` | `512 / 1024` `IC + PF` (2LPT `hi`, `fast`) | `14.257 s` | **`6.47×`** (`92.196 s`) | `0.00 GB` | `18.52 / 23.55 GB` |
| `B4` | `512 / 1024` `IC + PF` (2LPT `hi`, `strict`) | `15.657 s` | `5.89×` | `0.00 GB` | `18.46 / 23.55 GB` |
| `B5` | `512 / 1024` Full Coeval (2LPT, `fast`, `xH=0.5671`) | `27.630 s` | — | `2.50 GB` | `15.58 / 23.55 GB` |
| `B6` | `768³` EoR (`E-INTEGRAL`, `fast`, `xH=0.2518`) | `29.673 s` | — | `13.50 GB` | `18.99 / 23.55 GB` |

*Test suite status at `f276a7a`: `108/108` passing across `test_cuda_phase1.py` (43), `test_cuda_phase2.py -k "not speed"` (27), `test_cuda_phase3.py` (19), `test_21cmcu_phase1.py` (7), and `test_21cmcu_phase2.py` (12).*

---

## Ongoing Directions (D → E → F → G)

### Direction D — Lightcone Pipeline, GPU RSDs & GPU Power Spectra
- **Native CUDA `include_dvdr_in_tau21` & `rsds_shift` (`cuda_analysis.cu`)**: Replace CPU `scipy.fft`, `RegularGridInterpolator`, and `cosmotile.cic.cloud_in_cell_los` with fused column-parallel GPU kernels handling both periodic FFT and non-periodic 2nd-order velocity gradients plus subcell CIC displacements without `4×` host allocations.
- **Native CUDA `RectilinearLightconer` Batch Slicing**: Extract all `[z_lo, z_hi]` lightcone slices in a single GPU kernel directly from `DeviceFieldRegistry`-resident coevals.
- **Native CUDA 1D & 2D Power Spectra (`power_spectrum_1d`, `power_spectrum_2d`, `lightcone_chunk_power_spectra`)**: cuFFT-backed spherical `P(k)` / `Δ²(k)` and cylindrical `P(k_⊥, k_∥)` with exact R2C Hermitian multiplicity directly from GPU-resident fields.

### Direction E — Multi-GPU Batch / SBI Runner & `strict` vs `fast` Calibration
- **Spawned Multi-GPU / Multi-Worker Batch Runner (`py21cmcu.run_batch`)**: Concurrent parameter-sweep and SBI pipeline extracting global histories + GPU 1D/2D power spectra in memory with optional summary HDF5 persistence and per-worker GPU isolation.
- **Statistical Calibration (`docs/CALIBRATION.md`)**: Ensemble benchmarks across `E-INTEGRAL`, `L-INTEGRAL`, and `CHMF-SAMPLER + USE_MCGS + CLASS` quantifying `strict` vs `fast` global and power-spectrum consistency against cosmic variance.

### Direction F — Engineering Polish & Release Readiness
- **`build_cffi.py` Auto-Detection**: Automatic discovery of CUDA (`nvcc` / `CUDA_HOME`), GSL, FFTW, and local GPU compute architectures (`CUDA_ARCHS=auto`).
- **Reproducible Benchmark CLI (`benchmarks/bench_21cmcu.py`)**: Single-command suite covering `B1`–`B6`, lightcone/RSD/PS, and concurrent batch throughput.
- **Release Packaging (`v0.3.0`)**: Clean documentation, changelog, and clean separation from legacy `archive-cupy-v4` references.

### Direction G (Future) — Multi-GPU Domain Decomposition
- Distributed 2D slab cuFFT / MPI across multiple GPUs for `>= 2048³` grids beyond single-card VRAM limits.
