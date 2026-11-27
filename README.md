# 21cmCU

**21cmCU** is a standalone native C/CUDA + cuFFT 21cm cosmological simulation engine (originating from the `21cmFAST-GPU` fork and built on top of 21cmFAST v4.3.2 semantics with full `py21cmcu` / `py21cmfast` API and CLI compatibility and zero runtime CuPy dependency).

## Key Capabilities (`v0.5.0` — Phases 0–5 Complete: D, E, F, G, H, I & J)

- **Native C/CUDA + cuFFT core**: compiled directly into the CFFI shared library (`c_21cmfast`), supporting `E-INTEGRAL`, `CONST-ION-EFF`, `L-INTEGRAL`, `DEXM-ESF`, and `CHMF-SAMPLER` with `USE_MCGS`, `CLASS` (`V_CB_MODEL="FLUCTS"`), and `LYA_MULTIPLE_SCATTERING`.
- **Dual precision modes (`CUDA_MODE`)** ([calibration report](docs/CALIBRATION.md)):
  - `strict` (default): CPU-matched parity mode (`rel_rms ~ 1e-4` on `384³ / 128³` lightcones, `< 0.02%` max 3D power spectrum deviation over `k ∈ [0.1, 1.0] Mpc⁻¹`).
  - `fast`: statistically equivalent high-speed mode (parallel Philox IC RNG, GPU chunk-tree hashing, adaptive accumulation via `P21_CUDA_MODE=fast` / `CUDA_MODE="fast"` / `--cuda-mode fast`; 6-seed calibration confirms unbiased global histories `|ΔxH| ≤ 0.0015` and 1D/2D power spectra within ~`0.7–1.1 σ` cosmic variance).
- **Native Multi-GPU 1D Slab Domain Decomposition (`Direction G`)**: distributed Y-slab R2C/C2R/C2C 3D FFTs (`cuda_mgpu.cu`), multi-GPU `compute_initial_conditions` (`ZELDOVICH` + hybrid host-accumulated `2LPT`), and multi-GPU `power_spectrum_1d` / `power_spectrum_2d` controlled via `CUDA_DEVICES` (`P21_CUDA_DEVICES`, `py21cmcu.get_cuda_devices`, `py21cmcu.set_cuda_devices`, `use_mode(..., devices=...)`), scaling natively to `1536³` (`3.62 B` voxels) and `1792³` (`5.75 B` voxels) beyond single-card 24 GB VRAM.
- **High-resolution single-GPU residency (`512³–1024³`)**: 64-bit indexing, cuFFT plan LRU caching, `P21_CUDA_REGISTRY_MB` budget control, OOM-safe LRU reclamation, and opt-in `CUDA_LAZY_D2H` (`P21_CUDA_LAZY_D2H=1` / `use_mode(..., lazy_d2h=True)`) intermediate GPU-only residency with automatic eviction/access materialization (`Direction I`).
- **CLASS Transfer-Table Caching & GPU `AngularLightconer` (`Direction H`)**: thread-safe in-memory + atomic `.npz` disk cache for `InputParameters.cosmo_tables` (`py21cmcu.clear_class_cache()`), lightconer `run_classy` caching, and native CUDA `AngularLightconer` slice & LoS velocity interpolation.
- **Native CUDA `DeviceLZF` HDF5 Lightcone I/O (`Direction I`)**: `py21cmcu.save_lightcone(..., compression="device-lzf")` compresses 3D `float32` lightcone chunks on the GPU using HDF5-compatible byte-shuffle + `liblzf` with automatic non-finite detection and incompressible chunk fallback.
- **GPU RSD, Rectilinear Slicing & 1D/2D Power Spectra (`Direction D`)**: native CUDA `include_dvdr_in_tau21`, `rsds_shift` / `apply_rsds`, `RectilinearLightconer` batch slicing, and cuFFT-backed `power_spectrum_1d`, `power_spectrum_2d`, and `lightcone_chunk_power_spectra`.
- **Spawned Multi-GPU / Concurrent Batch Runner (`Direction E`)**: `py21cmcu.run_batch` extracts in-memory coeval/lightcone observables + GPU power spectra without 3D disk bottlenecks, scaling to **`10,502.0 tasks/h`** (`16 × 192/96` coevals across `3× RTX 4090`).
- **Toolchain Auto-Detection & CPU CI (`Directions F & J`)**: `build_cffi.py` supports `CUDA_ARCHS=auto` and automatic discovery of CUDA, GSL, and FFTW; GitHub Actions CPU CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) verifies `USE_CUDA=0` builds, `ruff`, and CPU-safe native tests on every push.
- **Standalone bug fixes over v4.3.2**:
  - Photon-conservation calibration redshift/neutral-fraction length alignment (`photoncons.py`).
  - Safe null-output guard in Sheth–Tormen `st_taylor_factor` for `SAMPLE_METHOD="PARTITION"` (`hmf.c`).
  - Separate previous-redshift 1D ACG ionizing-efficiency table (`Nion_conditional_table1D_acg_prev`) for `E-INTEGRAL` + `USE_MCGS=True` without photoheating feedback (`interp_tables.c` / `cuda_ionization.cu`).

### Verified RTX 4090 Highlights (`v0.5.0`, Node `b1`, 16 CPU Threads vs 1×–3× RTX 4090)

See the full benchmark report and figures in [**docs/BENCHMARKS.md**](docs/BENCHMARKS.md).

- **Multi-GPU Slab IC (`Direction G`, `3× RTX 4090`)**: native multi-GPU `1536³` (`3.62 B` voxels: **`46.131 s`** Zel'dovich / **`201.302 s`** 2LPT) and `1792³` (`5.75 B` voxels: **`98.095 s`** Zel'dovich / **`365.663 s`** 2LPT) beyond single-GPU 24 GB VRAM (`+ PF + 1D P(k)` in `48.087 s` / `102.545 s` Zel'dovich), with graceful host fallback at `2048³` (`3×` cards) and `10,502.0 tasks/h` multi-GPU `run_batch` (`1.65×` vs 1 GPU).
- **Lightcones & RSDs (`128 × 128 × 512` / `256/128`)**: `apply_rsds` **`0.0764 s`** vs `37.6455 s` CPU (**`492.74×`**); `RectilinearLightconer` (`z=6.5–11`, `dvdr+RSD`) **`3.032 s`** vs `248.773 s` CPU (**`82.05×`**); `AngularLightconer` (`z=7.8–9.8`) **`4.698 s`** vs `97.018 s` CPU (**`20.65×`**); `include_dvdr_in_tau21` **`0.0238 s`** vs `0.305 s` (**`12.81×`**).
- **Core Stages & Coevals (`512/256`–`768³`)**: `L-INTEGRAL + TS` 3z coeval **`4.456 s`** vs `55.079 s` CPU (**`12.36×`**, `0.00 GB` H2D, `408` hits); `768³` EoR **`16.435 s`** vs `110.209 s` CPU (**`6.71×`**); `1024/512` lo-res 2LPT `IC+PF` **`10.495 s`** vs `54.245 s` CPU (**`5.17×`**); `CLASS+FLUCTS` (`256/128`) IC **`0.295 s`** vs `2.827 s` CPU (**`9.57×`**).
- **`CUDA_LAZY_D2H` & `DeviceLZF` HDF5**: `lazy_d2h=True` cuts `B1` 3z D2H transfers by **`21.7%`** (`7,113.54 → 5,570.04 MB`) and runtime by **`12.8%`** (`4.477 s → 3.904 s`); `save_lightcone(..., compression="device-lzf")` writes `256 MB` (`256×256×512`) at **`476.3 MB/s`** vs `259.6 MB/s` CPU LZF (**`1.83×`**).
- **Full `v0.5.0` native GPU suite**: **`162/162`** tests passing across all 11 native suites on `3× RTX 4090` (`cuda_arch=sm_89`).

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

- Full benchmark suite, scaling analysis, and publication figures: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- Milestone record and architecture notes: [docs/ROADMAP.md](docs/ROADMAP.md)
- Statistical `strict` vs `fast` calibration report: [docs/CALIBRATION.md](docs/CALIBRATION.md)
- Reproducible benchmark & figure CLI: `python benchmarks/bench_21cmcu.py --suite all,mgpu` / `python benchmarks/plot_benchmarks.py`
- Version history: [CHANGELOG.md](CHANGELOG.md)

---

## Archived CuPy v4 Adapter & Legacy v3 Reference (`archive-cupy-v4`)

The previous CuPy v4.3.2 adapter (`v4/`), legacy v3 CUDA trees (`lightcone/`, `gpu/`), and archived JSON reports have been removed from `main` in `v0.4.0` and remain preserved intact under git tag `archive-cupy-v4`.

Upstream 21cmFAST license: [LICENSE.upstream](LICENSE.upstream).
