# 21cmCU Benchmark Suite (`v0.6.0` — NVIDIA RTX 4090)

All measurements in this report were captured directly on cluster node **`b1`** using [`benchmarks/bench_21cmcu.py`](../benchmarks/bench_21cmcu.py) and merged verbatim into [`benchmarks/results_rtx4090_v060.json`](../benchmarks/results_rtx4090_v060.json) (`v0.5.0` single-GPU baselines + `v0.6.0` Direction K multi-GPU sweep [`benchmarks/results_mgpu_v060.json`](../benchmarks/results_mgpu_v060.json)).

## Hardware & Environment

| Component | Configuration |
|---|---|
| Host CPU & RAM | Node `b1`, 64-core x86_64 host, **16 OpenMP threads** (`N_THREADS=16`) for CPU baselines, 196 GiB Slurm memory limit (`129.77 GiB` peak RSS) |
| GPUs | **1×, 2×, and 3× NVIDIA GeForce RTX 4090** (`24,564 MiB` VRAM per GPU, PCIe interconnect, `CUDA_ARCHS=89`) |
| Software | CUDA 12.6, native `c_21cmfast` CFFI extension, NumPy 2.5.5, `21cmCU v0.6.0` |
| Native GPU Verification | **`168/168` tests passing** (`-k "not speed"` across all 12 native GPU test suites on `3× RTX 4090`) |

Reproduce the suites and figures with:

```bash
python benchmarks/bench_21cmcu.py --suite all,mgpu --json benchmarks/results_rtx4090_v060.json
python benchmarks/plot_benchmarks.py --json benchmarks/results_rtx4090_v060.json --out-dir docs/figures
```

---

## 1. Single-GPU vs 16-Core CPU Overview

![21cmCU Single-GPU Speedup Overview](figures/speedup_overview.png)

| Workload | Grid (`DIM / HII_DIM`) | CPU 16T [s] | 1× RTX 4090 (`fast`) [s] | Speedup | Notes |
|---|---|---:|---:|---:|---|
| `apply_rsds` | `128 × 128 × 512` | `37.6455` | `0.0764` | **`492.74×`** | Standalone column-parallel CIC LOS displacement |
| Rectilinear Lightcone (`+ dvdr + RSD`) | `256 / 128`, `z=6.5–11` | `248.773` | `3.032` | **`82.05×`** | `0.562 GB` H2D, `1,212` registry hits, 6 chunk PS `0.0987 s` (vs `0.402 s` CPU) |
| Angular Lightcone (`+ dvdr + RSD`) | `256 / 128`, `z=7.8–9.8` | `97.018` | `4.698` | **`20.65×`** | Isolated slice + LOS vel kernel: `0.0034 s` vs `0.0183 s` (**`5.37×`**) |
| `include_dvdr_in_tau21` | `128 × 128 × 512` | `0.305` | `0.0238` | **`12.81×`** | Periodic gradient + τ₂₁ dv/dr scaling |
| Perturb Field (2LPT) | `256³` | `0.915` | `0.073` | **`12.58×`** | Isolated `perturb_field` at `z=8.0` |
| Coeval 3z (`L-INTEGRAL + TS`) | `512 / 256`, `z=12,10,8` | `55.079` | `4.456` | **`12.36×`** | `0.00 GB` H2D, `408` registry hits (`xH=0.6569`) |
| Ionize Box (`E-INTEGRAL`) | `256³`, `z=8.0` | `2.851` | `0.264` | **`10.81×`** | Isolated `compute_ionization_field` |
| `CLASS + V_CB_MODEL=FLUCTS` IC | `256 / 128` | `2.827` | `0.295` | **`9.57×`** | Warm `CosmoTables`: `0.634 s` cold → `0.000311 s` mem / `0.0027 s` disk |
| EoR Coeval (`E-INTEGRAL`) | `768³`, `z=7.0` | `110.209` | `16.435` | **`6.71×`** | `0.00 GB` H2D, `xH=0.3779` |
| Initial Conditions (2LPT) | `512 / 256` | `6.244` | `1.160` | **`5.38×`** | Isolated `compute_initial_conditions` |
| High-Res `IC + PF` (2LPT `lo`) | `1024 / 512`, `z=8.0` | `54.245` | `10.495` | **`5.17×`** | `1.611 GB` H2D |
| Concurrent Batch (`4 × 192/96`, `w=2`) | `4 × 192 / 96` | `14.184` | `3.097` | **`4.58×`** | **`4,649.0 tasks/h`** (`2,968.9 tasks/h` at `workers_per_gpu=1` vs `1,015.3/h` CPU) |

---

## 2. Direction G + K Multi-GPU Slab Domain Decomposition (`1×`, `2×`, `3× RTX 4090`)

![Direction G+K Multi-GPU Scaling](figures/mgpu_scaling.png)

Direction G partitions 3D complex FFT volumes along Y-slabs across GPUs (`cuda_mgpu.cu`), and **Direction K (`v0.6.0`)** accelerates multi-GPU execution with:
- **64 MiB pinned portable host-bounce buffers (`MgpuHostBounce`)** with sub-tiled `std::thread` parallel transposes and parallel multi-GPU strip transfers;
- **Single-work-slab mode (`P21_CUDA_MGPU_SINGLE_SLAB=1` / automatic for Zel'dovich)** running **`2048³` (`8.590 B` voxels)** Zel'dovich IC **natively in-VRAM on `3× RTX 4090`** (`mgpu_ic_runs=1`) in **`58.467 s`** (**`3.71×`** faster than `v0.5.0` host fallback);
- **Zero-`h_tmp_ab` 2LPT pinned-bounce streaming** eliminating `32 GiB` of host staging at `2048³` and speeding up `1536³`/`1792³` 2LPT by **`2.20×–3.52×`** over `v0.5.0`.

| Benchmark | GPUs | Algo | `v0.5.0` `ic_s` | `v0.6.0` `ic_s` | Speedup | `perturb_s` | `ps1d_s` | Total [s] | Peak RSS [GiB] | Path (`v0.6.0`) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `1024³` (`1.074 B`) | 1 | `ZELDOVICH` | `4.122` | `4.288` | `0.96×` | `0.580` | `0.0094` | `4.878` | `4.66` | 1-GPU native |
| `1024³` (`1.074 B`) | 2 | `ZELDOVICH` | `4.695` | `5.007` | `0.94×` | `0.634` | `0.0094` | `5.650` | `4.74` | Multi-GPU slab |
| `1024³` (`1.074 B`) | 3 | `ZELDOVICH` | `4.604` | `4.397` | `1.05×` | `0.584` | `0.0093` | `4.990` | `4.88` | Multi-GPU slab |
| `1024³` (`1.074 B`) | 1 | `2LPT` | `8.134` | `8.200` | `0.99×` | `0.567` | `0.0093` | `8.777` | `17.01` | 1-GPU native |
| `1024³` (`1.074 B`) | 2 | `2LPT` | `10.307` | `9.999` | `1.03×` | `0.616` | `0.0094` | `10.624` | `17.01` | Multi-GPU slab |
| `1024³` (`1.074 B`) | 3 | `2LPT` | `10.539` | `8.686` | `1.21×` | `0.617` | `0.0090` | `9.312` | `17.10` | Multi-GPU slab |
| `1536³` (`3.624 B`) | 2 | `ZELDOVICH` | `53.903` | **`27.799`** | **`1.94×`** | `1.883` | `0.0093` | **`29.691`** | `27.93` | Multi-GPU slab |
| `1536³` (`3.624 B`) | 3 | `ZELDOVICH` | `46.131` | **`29.005`** | **`1.59×`** | `2.010` | `0.0094` | **`31.024`** | `28.06` | Multi-GPU slab |
| `1536³` (`3.624 B`) | 2 | `2LPT` | `210.680` | **`75.231`** | **`2.80×`** | `3.109` | `0.0101` | **`78.350`** | `82.00` | Multi-GPU pinned stream |
| `1536³` (`3.624 B`) | 3 | `2LPT` | `201.302` | **`91.616`** | **`2.20×`** | `3.591` | `0.0099` | **`95.217`** | `82.13` | Multi-GPU pinned stream |
| `1792³` (`5.755 B`) | 3 | `ZELDOVICH` | `98.095` | **`38.699`** | **`2.53×`** | `3.046` | `0.0094` | **`41.754`** | `82.13` | Multi-GPU slab |
| `1792³` (`5.755 B`) | 3 | `2LPT` | `365.663` | **`103.892`** | **`3.52×`** | `4.373` | `0.0100` | **`108.275`** | `129.77` | Multi-GPU pinned stream |
| `2048³` (`8.590 B`) | 3 | `ZELDOVICH` | `216.665` (host) | **`58.467`** | **`3.71×`** | `5.805` | `0.0094` | **`64.282`** | `129.77` | **Native in-VRAM (`single_slab`)** |
| `2048³` (`8.590 B`) | 3 | `2LPT` | — | *skipped* | — | — | — | — | — | Requires `235.0 GiB` > `196.0 GiB` host budget |

### Concurrent `run_batch` Ensemble (`16 × 192/96` `L-INTEGRAL + TS`, `workers_per_gpu=2`)

| Active GPUs | Concurrent Workers | Wall Time [s] | Throughput [tasks / hour] | Scaling vs 1 GPU |
|---:|---:|---:|---:|---:|
| 1 | 2 | `8.324` | `6,920.1` | `1.00×` |
| 2 | 4 | **`5.888`** | **`9,782.0`** | **`1.41×`** |
| 3 | 6 | `6.035` | `9,544.3` | `1.38×` |

---

## 3. GPU Residency (`CUDA_LAZY_D2H`) & `DeviceLZF` HDF5 I/O

![Residency and DeviceLZF HDF5 I/O](figures/residency_and_io.png)

### `CUDA_LAZY_D2H` Residency (`512/256`, 3z `L-INTEGRAL + TS` Coeval)

| Mode | Run Time [s] | Time w/ `Tb` Read [s] | H2D [MB] | D2H [MB] | Registry Hits |
|---|---:|---:|---:|---:|---:|
| Eager D2H (`lazy_d2h=False`) | `4.477` | `4.484` | `0.00` | `7,113.54` | `408` |
| Lazy D2H (`lazy_d2h=True`) | **`3.904` (-12.8%)** | **`3.910`** | `0.00` | **`5,570.04` (-21.7%)** | `408` |

### HDF5 Lightcone Storage (`256 × 256 × 512`, 2 fields = `256.0 MB` raw, `chunk_z=32`)

| Backend | Compressed [MB] | Ratio | Write Time [s] | Write [MB/s] | Read Time [s] | Read [MB/s] |
|---|---:|---:|---:|---:|---:|---:|
| Uncompressed (`none`) | `256.00` | `1.00×` | `0.1662` | `1,540.3` | `0.1723` | `1,485.6` |
| CPU `h5py` LZF (`lzf`) | `77.48` | `3.30×` | `0.9860` | `259.6` | `0.8207` | `311.9` |
| GPU `DeviceLZF` (`device-lzf`) | `93.15` | `2.75×` | **`0.5375`** | **`476.3` (`1.83×` vs CPU LZF)** | `0.8653` | `295.8` |
