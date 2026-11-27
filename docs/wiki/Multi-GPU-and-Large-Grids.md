# Multi-GPU & Large Grids

Back to [Home](Home.md) · Previous: [Precision Modes & Calibration](Precision-Modes-and-Calibration.md) · Next: [Residency & I/O](Residency-and-IO.md)

21cmCU scales beyond single-card 24 GB VRAM through **1D Y-slab distributed 3D cuFFT** and **out-of-core single-work-slab streaming** (`cuda_mgpu.cu`, `cuda_ic.cu`), without external MPI dependencies.

## Configuring Active GPUs

```python
import py21cmcu as p21

print("Visible GPUs:", p21.cuda_device_count())
p21.set_cuda_devices([0, 1, 2])   # or "all" / "0,1,2"

with p21.use_mode("fast", devices=(0, 1, 2)):
    ...
```

Or via environment variables:

```bash
export P21_CUDA_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0,1,2
```

For large boxes where 2 complex slabs exceed multi-GPU VRAM (`2048³` on 3× 24 GB GPUs), **single-work-slab mode** (`P21_CUDA_MGPU_SINGLE_SLAB=1`, automatically activated for Zel'dovich when needed) halves complex device slab memory.

---

## How Multi-GPU 3D FFTs & Initial Conditions Work

1. Each GPU owns a contiguous range of `Y` slabs of `(X, Y_slab, Z/2 + 1)` complex data.
2. Batched 2D `(X, Z)` R2C/C2R cuFFT plans execute locally per GPU (`cuda_mgpu.cu`).
3. Parallel `std::thread` sub-tiled transposes (`MgpuHostBounce` 64 MiB pinned `cudaHostAllocPortable` buffers or P2P NVLink when enabled) exchange `X↔Y` strips across GPUs.
4. Batched 1D `Y` C2C cuFFT transforms complete the 3D transform.
5. In `v0.6.0` (Direction K), 2LPT streaming eliminates the `32 GiB` `h_tmp_ab` host temporary buffer at `2048³` via pinned-bounce streaming subtraction and parallelizes diagonal component combination with `std::thread`.

---

## Verified RTX 4090 Capacity & Scaling (`v0.5.0` → `v0.6.0`)

Captured on cluster node `b1` (`1×–3× RTX 4090`, 24 GB VRAM/GPU, 196 GiB Slurm RAM limit):

| Grid (`Voxels`) | GPUs | Algo | `v0.5.0` IC | `v0.6.0` IC | Speedup | Total (`+PF+PS`) | Path (`v0.6.0`) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `1024³` (`1.074 B`) | 1 | `ZELDOVICH` | `4.122 s` | `4.288 s` | `0.96×` | `4.878 s` | 1-GPU native |
| `1024³` (`1.074 B`) | 3 | `ZELDOVICH` | `4.604 s` | `4.397 s` | `1.05×` | `4.990 s` | Multi-GPU slab |
| `1024³` (`1.074 B`) | 1 | `2LPT` | `8.134 s` | `8.200 s` | `0.99×` | `8.777 s` | 1-GPU native |
| `1536³` (`3.624 B`) | 2 | `ZELDOVICH` | `53.903 s` | **`27.799 s`** | **`1.94×`** | `29.691 s` | Multi-GPU slab |
| `1536³` (`3.624 B`) | 3 | `ZELDOVICH` | `46.131 s` | **`29.005 s`** | **`1.59×`** | `31.024 s` | Multi-GPU slab |
| `1536³` (`3.624 B`) | 2 | `2LPT` | `210.680 s` | **`75.231 s`** | **`2.80×`** | `78.350 s` | Pinned stream |
| `1536³` (`3.624 B`) | 3 | `2LPT` | `201.302 s` | **`91.616 s`** | **`2.20×`** | `95.217 s` | Pinned stream |
| `1792³` (`5.755 B`) | 3 | `ZELDOVICH` | `98.095 s` | **`38.699 s`** | **`2.53×`** | `41.754 s` | Multi-GPU slab |
| `1792³` (`5.755 B`) | 3 | `2LPT` | `365.663 s` | **`103.892 s`** | **`3.52×`** | `108.275 s` | Pinned stream |
| `2048³` (`8.590 B`) | 3 | `ZELDOVICH` | `216.665 s` (host) | **`58.467 s`** | **`3.71×`** | `64.282 s` | **Native in-VRAM (`single_slab`)** |
| `2048³` (`8.590 B`) | 3 | `2LPT` | — | *skipped* | — | — | Needs `235 GiB` > `196 GiB` host RAM |

---

## Hardware Sizing Recommendations (24 GB Class GPUs)

| Workload | Min GPUs | VRAM / GPU | Host RAM |
| --- | :---: | --- | --- |
| `512³ -> 256³` Coeval (`L-INTEGRAL + TS`, 3z) | 1 | ~5 GB | 8 GB |
| `768³` EoR Coeval | 1 | ~19 GB | 16 GB |
| `1024³` (`1.07B`) 2LPT IC + PF | 1 | ~7–18 GB | 24 GB |
| `1536³` (`3.62B`) Multi-GPU 2LPT IC | 2 | ~15 GB | 90 GB |
| `1792³` (`5.75B`) Multi-GPU 2LPT IC | 3 | ~16 GB | 135 GB |
| `2048³` (`8.59B`) Single-Slab Zel'dovich / 2LPT | 3 | ~11.6 GB | 72 GB (Zel) / 235 GB (2LPT) |

Check live counters via `py21cmcu.cuda_stats()` (`mgpu_ic_runs`, `mgpu_fft_runs`).
