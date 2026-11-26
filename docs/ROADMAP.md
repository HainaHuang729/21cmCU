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

## Completed Directions (D → E → F, `v0.3.0`)

### Direction D (`fa22efc`) — Lightcone Pipeline, GPU RSDs & GPU Power Spectra
- **Native CUDA `include_dvdr_in_tau21` & `rsds_shift` (`cuda_analysis.cu`)**: Replaced CPU `scipy.fft`, `RegularGridInterpolator`, and `cosmotile.cic.cloud_in_cell_los` with fused column-parallel GPU kernels (`double` interpolation/CIC accumulation, zero `4×` host temporary).
  - Verified on RTX 4090 (`128 × 128 × 512`, `USE_TS_FLUCT=True`): `include_dvdr_in_tau21` **`0.0231 s`** vs `0.3223 s` CPU (**`13.94×`**); `apply_rsds` **`0.0794 s`** vs `37.8637 s` CPU (**`476.90×`**).
- **Native CUDA `RectilinearLightconer` Batch Slicing & `BrightnessTemp` Registry Retention (`cuda_brightness.cu`, `lightconers.py`)**: Batch slice interpolation in a single kernel directly from GPU-resident coeval arrays. End-to-end `256/128` `L-INTEGRAL` lightcone (`z=6.5–11`, `apply_rsds=True`, 777 slices) completes in **`4.066 s`** (`1,212` registry hits, `0.56 GB` H2D).
- **Native CUDA 1D & 2D Power Spectra (`power_spectrum_1d`, `power_spectrum_2d`, `lightcone_chunk_power_spectra`)**: cuFFT-backed spherical `P(k)` / `Δ²(k)` and cylindrical `P(k_⊥, k_∥)` matching `powerbox.get_power` (`19.68×` speedup at `128³`: `0.0067 s` vs `0.1311 s` CPU; `6` lightcone chunk PS in `0.1823 s`).
- Verified by `tests/native/test_21cmcu_phase3_d.py` (`19/19 passed`).

### Direction E (`b8e7950`) — Multi-GPU Batch / SBI Runner & `strict` vs `fast` Calibration
- **Spawned Multi-GPU / Multi-Worker Batch Runner (`py21cmcu.run_batch`, `save_batch_h5`)**: Concurrent in-memory sweeps with `CUDA_VISIBLE_DEVICES` isolation, GPU 1D/2D PS extraction, error isolation, and compact summary HDF5 output (`tests/native/test_21cmcu_phase3_e.py` `2/2 passed`).
  - RTX 4090 single-GPU throughput (`4 × 192/96` coevals, `fast`): `3.816 s` (`3,773.5 tasks/h`) with 1 worker; **`3.001 s` (`4,797.8 tasks/h`)** with `workers_per_gpu=2`.
- **Statistical Calibration (`docs/CALIBRATION.md`)**: 6-seed ensembles across `E-INTEGRAL`, `L-INTEGRAL`, and `CHMF-SAMPLER + USE_MCGS + CLASS` establish global agreement (`max |ΔxH| ≤ 0.00147`, `max |ΔTb| ≤ 0.3719 mK`, `max |ΔTs|/Ts ≤ 1.07%`) and unbiased 1D/2D power spectra (`median |Δ|/SEM = 0.68–1.11`).

### Direction F (`v0.3.0`) — Build Auto-Detection & Release Polish
- **`build_cffi.py` Auto-Detection**: Version-aware CUDA toolkit discovery (preferring highest `/usr/local/cuda*`), `CUDA_ARCHS=auto` / `native` (`nvidia-smi` SM cap detection, verified compiling `sm_89` on RTX 4090), and low-priority `conda`/`venv`/`pkg-config` GSL & FFTW discovery.
- **Reproducible Benchmark CLI ([`benchmarks/bench_21cmcu.py`](../benchmarks/bench_21cmcu.py))**: Single-command `--suite core|highres|lightcone|batch|all` harness.
- **Full `v0.3.0` native test suite**: **`135/135` passing tests** on RTX 4090 (`phase0`, `phase1`, `phase2`, `phase3_d`, `phase3_e`, `cuda_phase1`, `cuda_phase2 -k "not speed"`, `cuda_phase3`).

---

## Future Roadmap

### Direction G — Multi-GPU Domain Decomposition
- Distributed 2D slab cuFFT / MPI across multiple GPUs for `>= 2048³` grids beyond single-card VRAM limits.
