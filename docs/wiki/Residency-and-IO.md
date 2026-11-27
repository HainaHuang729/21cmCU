# Residency & I/O

Back to [Home](Home.md) · Previous: [Multi-GPU & Large Grids](Multi-GPU-and-Large-Grids.md) · Next: [SBI & Batch Pipeline](SBI-and-Batch-Pipeline.md)

21cmCU minimizes PCIe host-to-device (H2D) and device-to-host (D2H) copies as well as disk I/O bottlenecks across multi-redshift simulations.

## 1. `DeviceFieldRegistry` & Opt-In `CUDA_LAZY_D2H`

In default (`eager`) mode, fields produced by GPU stages are copied back to host NumPy arrays while retaining a synchronized GPU copy in `DeviceFieldRegistry` keyed by parallel 16 KB chunk-tree cryptographic fingerprints.

When opt-in **`CUDA_LAZY_D2H=True`** is enabled (`P21_CUDA_LAZY_D2H=1` or `use_mode(..., lazy_d2h=True)`):
- Intermediate fields (`EmissivityFields`, `RadiationFields`, and coeval stages) remain **GPU-resident only** (`_CudaBufferOwner`) until explicitly read back by Python (`array[...]`), evicted by LRU under VRAM pressure, or needed by a CPU fallback.
- Verified on `512/256` 3z `L-INTEGRAL + TS` coeval (`1× RTX 4090`):
  - **D2H traffic**: `7,113.54 MB` → **`5,570.04 MB`** (**`-21.7%`**, `0.00 GB` H2D, `408` registry hits)
  - **Runtime**: `4.477 s` → **`3.904 s`** (**`-12.8%`**)
- Verified on `256/128` `L-INTEGRAL + TS` lightcone (`z=7–11`): D2H drops **`24.3%`** (`1,723.60 MB → 1,304.17 MB`) and runtime improves **`1.45×`** (`1.912 s → 1.322 s`).

### Memory Budget & Cleanup

```bash
export P21_CUDA_REGISTRY_MB=4096          # Cap device registry budget in MiB
export P21_CUDA_REGISTRY_MAX_ENTRY_MB=2048
```

```python
import py21cmcu as p21

print(p21.cuda_stats())   # inspect registry_hits, registry_d2h_bytes, etc.
p21.cleanup_cuda()        # release pooled VRAM, cuFFT plans, and registry buffers
```

---

## 2. `DeviceLZF` GPU-Accelerated HDF5 Compression

Large float32 lightcone cubes bottleneck on CPU HDF5 compression (`h5py` LZF). `21cmCU` implements **HDF5-native byte-shuffled `liblzf` compression directly on the GPU** (`cuda_io.cu`):

```python
import py21cmcu as p21

p21.save_lightcone(lc, "lightcone.h5", compression="device-lzf")
```

- Output files use standard HDF5 LZF chunks readable by any Python/`h5py` client with zero custom HDF5 plugins required.
- Automatically flags non-finite values, falls back per chunk on incompressible noise (`filter_mask`), and falls back to CPU if `USE_CUDA=False`.
- Verified on `256 × 256 × 512` (`256.0 MB` raw, `chunk_z=32`): **`476.3 MB/s`** write (`0.5375 s`, `2.75×` ratio) vs `259.6 MB/s` (`0.9860 s`, `3.30×`) CPU `h5py` LZF (**`1.83×`** faster).

---

## 3. `CLASS` `CosmoTables` Memory & Disk Caching

When running `V_CB_MODEL="FLUCTS"` (`CLASS` transfer tables) or `AngularLightconer`, cold `CosmoTables` construction is cached across Python calls (in-memory LRU) and across processes (atomic `.npz` cache under `config["direc"]/class_tables_cache`):

- Cold `CosmoTables` build: `0.634 s` → **`0.000311 s`** memory hit / **`0.0027 s`** disk hit (`results_rtx4090_v060.json`); full `192/64` `CHMF-SAMPLER + CLASS` table build `17.1857 s` → `0.002208 s` memory (**`7,783×`**) / `0.0028 s` disk (**`6,214×`**).
- Control via `P21_CLASS_CACHE`, `P21_CLASS_DISK_CACHE`, and clear in-memory caches via `py21cmcu.clear_class_cache()`.
