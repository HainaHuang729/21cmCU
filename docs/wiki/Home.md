# Welcome to the 21cmCU Wiki

**21cmCU** (`py21cmcu` `v0.6.0`) is a native **C/CUDA + cuFFT** cosmological 21cm simulation engine with full **21cmFAST v4** API/CLI compatibility and **zero CuPy or Numba runtime dependency**.

## Headline Performance (`v0.6.0` on NVIDIA RTX 4090 vs 16-Core CPU)

| Workload | CPU 16T | `21cmCU v0.6.0` GPU (`fast`) | Speedup |
| --- | ---: | ---: | ---: |
| `apply_rsds` (`128 × 128 × 512`) | `37.6455 s` | `0.0764 s` (`1× 4090`) | **`492.74×`** |
| Rectilinear Lightcone (`256/128`, `z=6.5–11`, RSD+dvdr) | `248.773 s` | `3.032 s` (`1× 4090`) | **`82.05×`** |
| `AngularLightconer` (`256/128`, `z=7.8–9.8`, RSD+dvdr) | `97.018 s` | `4.698 s` (`1× 4090`) | **`20.65×`** |
| Coeval 3z (`512/256`, `L-INTEGRAL + TS`) | `55.079 s` | `4.456 s` (`1× 4090`) | **`12.36×`** |
| Concurrent Batch (`4 × 192/96`, `w=2`) | `14.184 s` | `3.097 s` (`1× 4090`) | **`4.58×`** |
| `2048³` (`8.590 B` voxels) Zel'dovich IC | `216.665 s` (`v0.5.0` host) | `58.467 s` (`3× 4090` native VRAM) | **`3.71×`** |
| SBI Batch Sweep (`16 × 192/96`, `2× 4090`) | — | `5.888 s` (`9,782.0 tasks/h`) | **`1.41×` vs 1 GPU** |

All **`168/168`** native GPU tests (`-k "not speed"` across 12 native suites) pass on `3× RTX 4090` (CUDA 12.6, `sm_89`).

---

## Wiki Navigation

| Guide | Topic |
| --- | --- |
| [Installation & Containers](Installation-and-Containers.md) | One-command `install.sh`, `USE_CUDA=1` build, Docker & Apptainer (Slurm), `21cmcu doctor` |
| [Quickstart](Quickstart.md) | 5-minute Python & CLI walkthrough + runnable scripts & Jupyter Notebook |
| [Precision Modes & Calibration](Precision-Modes-and-Calibration.md) | `strict` vs `fast` mode parity & 6-seed statistical calibration |
| [Multi-GPU & Large Grids](Multi-GPU-and-Large-Grids.md) | `CUDA_DEVICES`, Y-slab FFT, single-slab `2048³`, and 24 GB sizing table |
| [Residency & I/O](Residency-and-IO.md) | `CUDA_LAZY_D2H`, `DeviceFieldRegistry`, `DeviceLZF` HDF5, and `CLASS` table cache |
| [SBI & Batch Pipeline](SBI-and-Batch-Pipeline.md) | Horizon wedge, thermal noise, moments/PDF, Sobol sampling, `SBISimulator`, `Likelihood21cm` |
| [Interferometer Mock & Forecast](Interferometer-Mock-and-Forecast.md) | `SKA1-Low`/`HERA-350`/`MWA-II`/`LOFAR`, single-pass GPU mock observations, 1D/2D SNR forecasts |
| [Lightcone Optimization & 3D Viz](Lightcone-Optimization-and-3D-Viz.md) | One-line `quick_lightcone`, zero-D2H lightcones, `to_torch` DLPack, and offline WebGL2 3D viewer |
| [CLI & API Reference](CLI-and-API-Reference.md) | `21cmcu` subcommands, `P21_*` environment variables, and `py21cmcu` exports |
| [Architecture Overview](Architecture-Overview.md) | Native CFFI CUDA engine layout, Philox RNG, 64-bit indexing, and engine fixes |
| [Troubleshooting & FAQ](Troubleshooting-and-FAQ.md) | Common HPC build issues, v4 `_ACG` naming, `Z_HEAT_MAX`, and OOM tuning |

---

## Syncing This Wiki to GitHub (`*.wiki.git`)

Repository maintainers can publish updates from `docs/wiki/` to the GitHub Wiki remote directly:

```bash
git clone git@github.com:HainaHuang729/21cmCU.wiki.git /tmp/21cmCU.wiki
cp docs/wiki/*.md /tmp/21cmCU.wiki/
cd /tmp/21cmCU.wiki && git add -A && git commit -m "Sync v0.6.0 wiki" && git push
```
