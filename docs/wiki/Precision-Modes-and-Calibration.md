# Precision Modes & Calibration

Back to [Home](Home.md) · Previous: [Quickstart](Quickstart.md) · Next: [Multi-GPU & Large Grids](Multi-GPU-and-Large-Grids.md)

21cmCU separates **bitwise/tolerance regression parity** (`strict`) from **high-throughput scientific production** (`fast`) via `CUDA_MODE`.

## Mode Comparison

| Aspect | `strict` (default) | `fast` |
| --- | --- | --- |
| Mathematical Precision | IEEE FP32 without `--use_fast_math` trig shortcuts | `--use_fast_math` intrinsics & single-precision fused kernels |
| IC RNG & Chunk Hashing | CPU-matched RNG / parity accumulation order | Parallel Philox `4×32-10` IC RNG & parallel 16 KB GPU chunk-tree hashing |
| Target Use Case | CPU-vs-GPU verification, deterministic debugging | Production MCMC, Simulation-Based Inference (SBI), large grids |
| How to Enable | Default, or `use_mode("strict")` | `use_mode("fast")`, `P21_CUDA_MODE=fast`, `--cuda-mode fast` |

### Switching Modes in Python or CLI

```python
import py21cmcu as p21

p21.set_cuda_mode("fast")
print(p21.get_cuda_mode())

with p21.use_mode("strict"):
    # Scoped strict block; previous mode automatically restored on exit
    ...
```

CLI:
```bash
21cmcu run coeval -z 8 --cuda-mode fast --template simple tiny --out ./out
```

---

## Numerical Verification Summary

### 1. `strict` Mode Deterministic Parity
- Rectilinear lightcones (`384³ / 128³`): relative RMS difference vs 16-thread CPU is **`~1e-4`**, with **`< 0.02%`** max 3D power spectrum deviation across $k \in [0.1, 1.0]\ \mathrm{Mpc}^{-1}$.
- All 1D/2D power spectra (`power_spectrum_1d`, `power_spectrum_2d`), RSD shifts (`include_dvdr_in_tau21`, `rsds_shift`, `apply_rsds`), `AngularLightconer`, and `DeviceLZF` HDF5 compression are verified against CPU references across `tests/native/`.

### 2. `fast` Mode 6-Seed Statistical Calibration

From [CALIBRATION.md](../CALIBRATION.md) across `E-INTEGRAL`, `L-INTEGRAL`, and `CHMF-SAMPLER + USE_MCGS + CLASS`:
- **Global histories** (`strict` vs `fast`, 6 random seeds):
  - Max neutral-fraction drift: $|\Delta \langle x_{\mathrm{HI}} \rangle| \le 0.00147$
  - Max brightness-temperature drift: $|\Delta \langle \delta T_b \rangle| \le 0.3719\ \mathrm{mK}$
  - Max spin-temperature relative drift: $|\Delta \langle T_S \rangle| / \langle T_S \rangle \le 1.07\%$
- **1D & 2D Power Spectra**: unbiased relative to cosmic variance across seeds ($\text{median } |\Delta P| / \text{SEM} = 0.68\text{–}1.11$).

## Recommendation

- Use **`fast`** (plus optional `lazy_d2h=True`) for all training-set generation, Sobol sweeps, MCMC chains, and high-resolution runs.
- Use **`strict`** only when performing exact unit-test parity comparisons against upstream CPU runs.
