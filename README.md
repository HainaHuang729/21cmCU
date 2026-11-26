# 21cmCU

**21cmCU** is a standalone native C/CUDA + cuFFT 21cm cosmological simulation engine (originating from the `21cmFAST-GPU` fork and built on top of 21cmFAST v4.3.2 semantics with full `py21cmcu` / `py21cmfast` API and CLI compatibility and zero runtime CuPy dependency).

## Key Capabilities (`v0.4.0` — Phases 0–4 Complete: D, E, F, H, I & J)

- **Native C/CUDA + cuFFT core**: compiled directly into the CFFI shared library (`c_21cmfast`), supporting `E-INTEGRAL`, `CONST-ION-EFF`, `L-INTEGRAL`, `DEXM-ESF`, and `CHMF-SAMPLER` with `USE_MCGS`, `CLASS` (`V_CB_MODEL="FLUCTS"`), and `LYA_MULTIPLE_SCATTERING`.
- **Dual precision modes (`CUDA_MODE`)** ([calibration report](docs/CALIBRATION.md)):
  - `strict` (default): CPU-matched parity mode (`rel_rms ~ 1e-4` on `384³ / 128³` lightcones, `< 0.02%` max 3D power spectrum deviation over `k ∈ [0.1, 1.0] Mpc⁻¹`).
  - `fast`: statistically equivalent high-speed mode (parallel Philox IC RNG, GPU chunk-tree hashing, adaptive accumulation via `P21_CUDA_MODE=fast` / `CUDA_MODE="fast"` / `--cuda-mode fast`; 6-seed calibration confirms unbiased global histories `|ΔxH| ≤ 0.0015` and 1D/2D power spectra within ~`0.7–1.1 σ` cosmic variance).
- **High-resolution single-GPU residency (`512³–1024³`)**: 64-bit indexing, cuFFT plan LRU caching, `P21_CUDA_REGISTRY_MB` budget control, OOM-safe LRU reclamation, and opt-in `CUDA_LAZY_D2H` (`P21_CUDA_LAZY_D2H=1` / `use_mode(..., lazy_d2h=True)`) intermediate GPU-only residency with automatic eviction/access materialization (`Direction I`).
- **CLASS Transfer-Table Caching & GPU `AngularLightconer` (`Direction H`)**: thread-safe in-memory + atomic `.npz` disk cache for `InputParameters.cosmo_tables` (`py21cmcu.clear_class_cache()`), lightconer `run_classy` caching, and native CUDA `AngularLightconer` slice & LoS velocity interpolation.
- **Native CUDA `DeviceLZF` HDF5 Lightcone I/O (`Direction I`)**: `py21cmcu.save_lightcone(..., compression="device-lzf")` compresses 3D `float32` lightcone chunks on the GPU using HDF5-compatible byte-shuffle + `liblzf` with automatic non-finite detection and incompressible chunk fallback.
- **GPU RSD, Rectilinear Slicing & 1D/2D Power Spectra (`Direction D`)**: native CUDA `include_dvdr_in_tau21`, `rsds_shift` / `apply_rsds`, `RectilinearLightconer` batch slicing, and cuFFT-backed `power_spectrum_1d`, `power_spectrum_2d`, and `lightcone_chunk_power_spectra`.
- **Spawned Multi-GPU / Concurrent Batch Runner (`Direction E`)**: `py21cmcu.run_batch` extracts in-memory coeval/lightcone observables + GPU power spectra without 3D disk bottlenecks, with optional compact HDF5 output.
- **Toolchain Auto-Detection & CPU CI (`Directions F & J`)**: `build_cffi.py` supports `CUDA_ARCHS=auto` and automatic discovery of CUDA, GSL, and FFTW; GitHub Actions CPU CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) verifies `USE_CUDA=0` builds, `ruff`, and CPU-safe unit tests on every push.
- **Standalone bug fixes over v4.3.2**:
  - Photon-conservation calibration redshift/neutral-fraction length alignment (`photoncons.py`).
  - Safe null-output guard in Sheth–Tormen `st_taylor_factor` for `SAMPLE_METHOD="PARTITION"` (`hmf.c`).
  - Separate previous-redshift 1D ACG ionizing-efficiency table (`Nion_conditional_table1D_acg_prev`) for `E-INTEGRAL` + `USE_MCGS=True` without photoheating feedback (`interp_tables.c` / `cuda_ionization.cu`).

### Verified RTX 4090 Highlights (24 GB VRAM, `N_THREADS=8`, `v0.4.0`)

- **`CLASS CosmoTables` Caching (`192/64` `CHMF+MCG+CLASS`)**: cold `17.1857 s` → **`0.002208 s`** in-memory (**`7,783×`**) / **`0.0028 s`** on-disk (**`6,214×`**); end-to-end `192/64` `fast` coeval completes in **`7.713 s`** (`mem`) / **`7.795 s`** (`disk`).
- **`CUDA_LAZY_D2H` & `DeviceLZF` HDF5 (`256/128` `L-INTEGRAL+TS`, `z=7–11`)**: `CUDA_LAZY_D2H=True` cuts host D2H transfers by **`24.3%`** (`1723.60 MB → 1304.17 MB`) and improves wall time from `1.912 s` to **`1.322 s`** (**`1.45×`**); `save_lightcone(..., compression="device-lzf")` saves all 3 lightcone fields (`129.11 MB` raw) in **`0.2366 s`** vs `0.6893 s` CPU LZF (**`2.91×`**).
- **`AngularLightconer` GPU Slicing (`64 × 128³`, `order=1`)**: **`0.0934 s`** GPU vs `0.1644 s` CPU (**`1.76×`**) with cached `run_classy`.
- **`512 / 1024` 2LPT IC + Perturb (`fast`)**: `10.331 s` lo-res (**8.30×** vs `85.784 s` CPU) / `14.257 s` hi-res (**6.47×** vs `92.196 s` CPU); full `512 / 1024` 2LPT coeval **`27.630 s`** (`15.58 / 23.55 GB` VRAM); `768³` EoR box **`29.673 s`** (`18.99 / 23.55 GB` VRAM).
- **GPU RSDs & Power Spectra (`128 × 128 × 512` lightcone / `128³` cube)**: `apply_rsds` **`0.0794 s`** vs `37.8637 s` CPU (**`476.90×`**); `include_dvdr_in_tau21` **`0.0231 s`** vs `0.3223 s` (**`13.94×`**); 1D+2D PS **`0.0067 s`** vs `0.1311 s` (**`19.68×`**); end-to-end `256/128` `L-INTEGRAL` lightcone (`z=6.5–11`, `apply_rsds=True`) **`4.066 s`** (`4.297 s` via `bench_21cmcu.py`, 6 chunk PS in `0.1823 s`).
- **Concurrent Single-GPU Batch (`4 × 192/96` coevals, `fast`)**: `3.001 s` with `workers_per_gpu=2` (**`4,797.8 tasks/h`**).
- **Full `v0.4.0` native GPU suite**: **`150/150`** tests passing on RTX 4090 (`cuda_arch=sm_89`).

## Quick Start

```bash
export USE_CUDA=1 CUDA_ARCHS=auto   # auto-detects local GPU SM architecture (or set e.g. 86,89)
pip install -e . --no-build-isolation

# Enable GPU via environment, Python API, or CLI:
P21_USE_CUDA=1 P21_CUDA_MODE=fast 21cmcu run coeval -z 8 --template simple tiny --out ./out
```

```python
import py21cmcu as p21

with p21.use_mode("strict"):
    lc = p21.run_lightcone(...)

with p21.use_mode("fast"):
    cv = p21.run_coeval(...)
    ps1 = p21.power_spectrum_1d(cv[0].brightness_temp, box_length=200.0)
```

## Roadmap & Documentation

- Full milestone record, benchmark tables, and future multi-GPU roadmap (Direction G): [docs/ROADMAP.md](docs/ROADMAP.md)
- Statistical `strict` vs `fast` calibration report: [docs/CALIBRATION.md](docs/CALIBRATION.md)
- Reproducible benchmark CLI: `python benchmarks/bench_21cmcu.py --suite lightcone,batch`
- Version history: [CHANGELOG.md](CHANGELOG.md)

---

## Archived CuPy v4 Adapter & Legacy v3 Reference (`archive-cupy-v4`)

The previous CuPy v4.3.2 adapter (`v4/`), legacy v3 CUDA trees (`lightcone/`, `gpu/`), and archived JSON reports have been removed from `main` in `v0.4.0` and remain preserved intact under git tag `archive-cupy-v4`.

Upstream 21cmFAST license: [LICENSE.upstream](LICENSE.upstream).
