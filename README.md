# 21cmCU

**21cmCU** is a standalone native C/CUDA + cuFFT 21cm cosmological simulation engine (originating from the `21cmFAST-GPU` fork and built on top of 21cmFAST v4.3.2 semantics with full `py21cmcu` / `py21cmfast` API and CLI compatibility and zero runtime CuPy dependency).

## Key Capabilities (`v0.6.0` — Directions D → N Complete)

- **Native C/CUDA + cuFFT core**: compiled directly into the CFFI shared library (`c_21cmfast`), supporting `E-INTEGRAL`, `CONST-ION-EFF`, `L-INTEGRAL`, `DEXM-ESF`, and `CHMF-SAMPLER` with `USE_MCGS`, `CLASS` (`V_CB_MODEL="FLUCTS"`), and `LYA_MULTIPLE_SCATTERING`.
- **Dual precision modes (`CUDA_MODE`)** ([calibration report](docs/CALIBRATION.md)):
  - `strict` (default): CPU-matched parity mode (`rel_rms ~ 1e-4` on `384³ / 128³` lightcones, `< 0.02%` max 3D power spectrum deviation over `k ∈ [0.1, 1.0] Mpc⁻¹`).
  - `fast`: statistically equivalent high-speed mode (parallel Philox IC RNG, GPU chunk-tree hashing, adaptive accumulation via `P21_CUDA_MODE=fast` / `CUDA_MODE="fast"` / `--cuda-mode fast`; 6-seed calibration confirms unbiased global histories `|ΔxH| ≤ 0.0015` and 1D/2D power spectra within ~`0.7–1.1 σ` cosmic variance).
- **Native Multi-GPU Slab Domain Decomposition & Single-Slab `2048³` (`Directions G & K`)**: distributed Y-slab R2C/C2R/C2C 3D FFTs (`cuda_mgpu.cu`) with 64 MiB pinned portable host-bounce buffers, sub-tiled `std::thread` parallel transposes, zero-`h_tmp_ab` 2LPT streaming, and single-work-slab streaming (`P21_CUDA_MGPU_SINGLE_SLAB`). Scales natively in-VRAM up to **`2048³` (`8.590 B` voxels)** Zel'dovich IC on `3× RTX 4090` (`58.467 s`, `3.71×` vs `v0.5.0` host fallback) and delivers `2.20×–3.52×` speedups on `1536³`/`1792³` 2LPT.
- **High-resolution single-GPU residency (`512³–1024³`)**: 64-bit indexing, cuFFT plan LRU caching, `P21_CUDA_REGISTRY_MB` budget control, OOM-safe LRU reclamation, and opt-in `CUDA_LAZY_D2H` (`P21_CUDA_LAZY_D2H=1` / `use_mode(..., lazy_d2h=True)`) intermediate GPU-only residency with automatic eviction/access materialization (`Direction I`).
- **CLASS Transfer-Table Caching & GPU `AngularLightconer` (`Direction H`)**: thread-safe in-memory + atomic `.npz` disk cache for `InputParameters.cosmo_tables` (`py21cmcu.clear_class_cache()`), lightconer `run_classy` caching, and native CUDA `AngularLightconer` slice & LoS velocity interpolation.
- **Native CUDA `DeviceLZF` HDF5 Lightcone I/O (`Direction I`)**: `py21cmcu.save_lightcone(..., compression="device-lzf")` compresses 3D `float32` lightcone chunks on the GPU using HDF5-compatible byte-shuffle + `liblzf` with automatic non-finite detection and incompressible chunk fallback.
- **GPU RSD, Rectilinear Slicing & 1D/2D Power Spectra (`Direction D`)**: native CUDA `include_dvdr_in_tau21`, `rsds_shift` / `apply_rsds`, `RectilinearLightconer` batch slicing, and cuFFT-backed `power_spectrum_1d`, `power_spectrum_2d`, and `lightcone_chunk_power_spectra`.
- **End-to-End SBI Data Pipeline (`Directions E & L`)**: scrambled Sobol parameter sweeps (`generate_sobol_parameter_grid`), Fourier horizon wedge filtering (`horizon_wedge_mask`, `apply_wedge_filter_3d`), deterministic radiometer thermal noise (`add_thermal_noise_3d`), and HDF5-checkpointed resumable worker pools (`py21cmcu.run_batch(..., checkpoint_h5=..., resume=True)`, up to **`9,782.0 tasks/h`** on `2× RTX 4090`, see [`examples/sbi_lightcone_dataset.py`](examples/sbi_lightcone_dataset.py)).
- **Diagnostics, CLI & User Guide (`Directions F, J & M`)**: `py21cmcu.doctor()`, `21cmcu doctor|batch|bench` CLI commands, `docs/USER_GUIDE.md`, and GitHub Actions CPU CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).
- **Standalone bug fixes over v4.3.2**:
  - Photon-conservation calibration redshift/neutral-fraction length alignment (`photoncons.py`).
  - Safe null-output guard in Sheth–Tormen `st_taylor_factor` for `SAMPLE_METHOD="PARTITION"` (`hmf.c`).
  - Separate previous-redshift 1D ACG ionizing-efficiency table (`Nion_conditional_table1D_acg_prev`) for `E-INTEGRAL` + `USE_MCGS=True` without photoheating feedback (`interp_tables.c` / `cuda_ionization.cu`).

### Verified RTX 4090 Highlights (`v0.6.0`, Node `b1`, 16 CPU Threads vs 1×–3× RTX 4090)

See the full benchmark report and figures in [**docs/BENCHMARKS.md**](docs/BENCHMARKS.md) and the standalone methods draft in [**docs/paper/21cmcu_paper.md**](docs/paper/21cmcu_paper.md).

- **Multi-GPU Slab IC (`Directions G & K`, `2×`/`3× RTX 4090`)**: native in-VRAM `2048³` (`8.590 B` voxels) Zel'dovich IC **`58.467 s`** (`3.71×` vs `v0.5.0` host fallback, `64.282 s` w/ `PF + 1D P(k)`); `1792³` (`5.755 B`) **`38.699 s`** Zel'dovich / **`103.892 s`** 2LPT (`2.53×` / `3.52×` vs `v0.5.0`); `1536³` (`3.624 B`) **`27.799 s`** Zel'dovich / **`75.231 s`** 2LPT on `2× RTX 4090` (`1.94×` / `2.80×` vs `v0.5.0`); concurrent `16 × 192/96` `run_batch` reaches **`9,782.0 tasks/h`** (`2 GPUs`) / **`9,544.3 tasks/h`** (`3 GPUs`).
- **Lightcones & RSDs (`128 × 128 × 512` / `256/128`)**: `apply_rsds` **`0.0764 s`** vs `37.6455 s` CPU (**`492.74×`**); `RectilinearLightconer` (`z=6.5–11`, `dvdr+RSD`) **`3.032 s`** vs `248.773 s` CPU (**`82.05×`**); `AngularLightconer` (`z=7.8–9.8`) **`4.698 s`** vs `97.018 s` CPU (**`20.65×`**); `include_dvdr_in_tau21` **`0.0238 s`** vs `0.305 s` (**`12.81×`**).
- **Core Stages & Coevals (`512/256`–`768³`)**: `L-INTEGRAL + TS` 3z coeval **`4.456 s`** vs `55.079 s` CPU (**`12.36×`**, `0.00 GB` H2D, `408` hits); `768³` EoR **`16.435 s`** vs `110.209 s` CPU (**`6.71×`**); `1024/512` lo-res 2LPT `IC+PF` **`10.495 s`** vs `54.245 s` CPU (**`5.17×`**); `CLASS+FLUCTS` (`256/128`) IC **`0.295 s`** vs `2.827 s` CPU (**`9.57×`**).
- **`CUDA_LAZY_D2H` & `DeviceLZF` HDF5**: `lazy_d2h=True` cuts `B1` 3z D2H transfers by **`21.7%`** (`7,113.54 → 5,570.04 MB`) and runtime by **`12.8%`** (`4.477 s → 3.904 s`); `save_lightcone(..., compression="device-lzf")` writes `256 MB` (`256×256×512`) at **`476.3 MB/s`** vs `259.6 MB/s` CPU LZF (**`1.83×`**).
- **Full `v0.6.0` native GPU suite**: **`168/168`** tests passing across all 12 native GPU test suites on `3× RTX 4090` (`cuda_arch=sm_89`).

## Quick Start

```bash
# Option 1: One-command installer + diagnostic verification
./scripts/install.sh
21cmcu doctor

# Option 2: Containerized GPU runtime (Docker / Apptainer)
docker build -f docker/Dockerfile -t 21cmcu:0.6.0 . && docker run --rm --gpus all 21cmcu:0.6.0

# Run minimal GPU examples or CLI:
python examples/01_quickstart_coeval.py
python examples/02_lightcone_power_spectrum.py
python examples/03_multigpu_large_ic.py --dim 512 --devices 0,1
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

- **Wiki** (installation, quickstart, multi-GPU, SBI, API/CLI, FAQ): [docs/wiki/Home.md](docs/wiki/Home.md)
- User & sizing guide: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)
- Standalone technical methods draft: [docs/paper/21cmcu_paper.md](docs/paper/21cmcu_paper.md) (LaTeX: [docs/paper/21cmcu_paper.tex](docs/paper/21cmcu_paper.tex))
- Full benchmark suite, scaling analysis, and publication figures: [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- Milestone record and architecture notes: [docs/ROADMAP.md](docs/ROADMAP.md)
- Statistical `strict` vs `fast` calibration report: [docs/CALIBRATION.md](docs/CALIBRATION.md)
- Reproducible benchmark & figure CLI: `21cmcu bench --suite all,mgpu --json benchmarks/results_rtx4090_v060.json` / `python benchmarks/plot_benchmarks.py`
- Version history: [CHANGELOG.md](CHANGELOG.md)

---

## Archived CuPy v4 Adapter & Legacy v3 Reference (`archive-cupy-v4`)

The previous CuPy v4.3.2 adapter (`v4/`), legacy v3 CUDA trees (`lightcone/`, `gpu/`), and archived JSON reports have been removed from `main` in `v0.4.0` and remain preserved intact under git tag `archive-cupy-v4`.

Upstream 21cmFAST license: [LICENSE.upstream](LICENSE.upstream).
