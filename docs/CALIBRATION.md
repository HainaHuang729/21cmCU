# `strict` vs `fast` Mode Statistical Calibration (`RTX 4090`, 6-Seed Ensembles)

**21cmCU** supports two execution modes (`CUDA_MODE`):
- `strict`: Matches CPU numerical paths (host GSL Fourier ICs, matched halo sampler / FFT paths, IEEE rounding order).
- `fast`: Activates parallel GPU Philox 4×32-10 IC realizations, single-precision-bounded perturbation accumulation (on boxes $\ge 2\text{ GiB}$), and fast GPU halo sampling while keeping cosmological statistics unbiased.

Because `fast` draws an independent Gaussian IC field (Philox vs GSL) when using fast ICs, matched-seed voxel comparisons are not meaningful; instead, ensembles of matched parameter sets must agree within cosmic-variance standard error ($\mathrm{SEM} = \sigma / \sqrt{N_{\text{seeds}}}$).

## Ensemble Calibration Results (`b1`, NVIDIA RTX 4090, `N_THREADS=8`, `N_seeds=6`)

Benchmark script: [`benchmarks/calibrate_strict_vs_fast.py`](../benchmarks/calibrate_strict_vs_fast.py).
Each regime compares 6 seeds (`fast` vs `strict`) across 3 sampled redshifts ($z_{\text{hi}}$, $z_{\text{mid}}$, $z_{\text{lo}}$), evaluating 1D spherical $\Delta^2_{21}(k)$ over $k \in [0.08, 1.0]\text{ Mpc}^{-1}$ (12 log bins) and 2D cylindrical $P(k_\perp, k_\parallel)$ ($6 \times 6$ bins).

| Regime (`HII_DIM / DIM`, Box, Redshifts) | `strict` Wall | `fast` Wall | Speedup | $\max \|\Delta \bar{x}_{\text{HI}}\|$ | $\max \|\Delta \bar{T}_b\|$ | $\max \|\Delta \bar{T}_s\| / \bar{T}_s$ | 1D $\Delta^2_{21}$ Med $\|\Delta\|/\text{SEM}$ | 2D $P$ Med $\|\Delta\|/\text{SEM}$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `E-INTEGRAL` (`96 / 192`, `192 Mpc`, $z = 16.35 \to 6.5$) | `1.230 s` | `1.062 s` | `1.16×` | `0.00147` | `0.0551 mK` | `0.29%` | `0.68` | `0.70` |
| `L-INTEGRAL` (`96 / 192`, `192 Mpc`, $z = 16.35 \to 6.5$) | `1.055 s` | `0.972 s` | `1.08×` | `0.00023` | `0.2339 mK` | `0.61%` | `0.88` | `0.86` |
| `CHMF-SAMPLER + MCG + CLASS` (`64 / 192`, `150 Mpc`, $z = 16.10 \to 7.5$) | `23.161 s` | `18.933 s` | `1.22×` | `0.00064` | `0.3719 mK` | `1.07%` | `0.92` | `1.11` |

*Notes:*
1. Across all three physical regimes, ensemble-mean global histories agree within $\le 0.0015$ in neutral fraction, $\le 0.37\text{ mK}$ in $\bar{T}_b$, and $\le 1.07\%$ in mean spin temperature $\bar{T}_s$.
2. Across both 1D spherical and 2D cylindrical power spectra, median normalized deviations $|\bar{P}_{\text{fast}} - \bar{P}_{\text{strict}}| / \mathrm{SEM}_{\text{strict}}$ lie in `[0.68, 1.11]` — fully consistent with random cosmic variance between two independent 6-seed Gaussian realizations (`O(1) sigma` expected for zero bias).
3. On larger grids (`512 / 1024` 2LPT in [`docs/ROADMAP.md`](ROADMAP.md)), parallel Philox IC generation (`fast`) widens throughput gains further (`10.331 s` `fast` vs `11.574 s` `strict` at `512 / 1024` lo-res 2LPT; `8.30×` over `85.784 s` CPU).

## Mode Selection Guidance

- Use **`CUDA_MODE="strict"`** when validating regression parity against CPU reference runs or reproducing a specific GSL random-seed voxel realization.
- Use **`CUDA_MODE="fast"`** for SBI training-set generation, MCMC/Fisher sweeps, and high-resolution (`512³–1024³`) single-GPU simulations where statistically unbiased ensembles and maximum throughput are desired.
