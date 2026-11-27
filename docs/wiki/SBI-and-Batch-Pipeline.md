# SBI & Batch Pipeline

Back to [Home](Home.md) · Previous: [Residency & I/O](Residency-and-IO.md) · Next: [CLI & API Reference](CLI-and-API-Reference.md)

21cmCU provides an end-to-end **Simulation-Based Inference (SBI)** dataset generation pipeline combining scrambled Sobol sampling, GPU simulations, foreground/thermal noise realism, 1D/2D power spectra, and fault-tolerant HDF5 checkpoints.

## Core Building Blocks (`py21cmcu.observables` & `py21cmcu.batch`)

1. **Scrambled Sobol Parameter Grid**:
   ```python
   import py21cmcu as p21

   bounds = {
       "F_STAR10_ACG": (-1.8, -0.7),
       "F_ESC10_ACG": (-1.8, -0.5),
       "ALPHA_STAR_ACG": (0.2, 0.8),
       "M_TURN": (8.0, 9.5),
   }
   rows = p21.generate_sobol_parameter_grid(bounds, n_samples=16, seed=42)
   ```
   *(Tip: pass powers of 2 for `n_samples` to preserve optimal Sobol balance properties.)*

2. **Horizon Foreground Wedge & Thermal Noise**:
   - `horizon_slope(redshift, cosmo=None)`: horizon slope $m(z) = \frac{H(z)\,\chi(z)}{c\,(1+z)}$.
   - `horizon_wedge_mask(k_perp, k_para, redshift, buffer_h_mpc=0.0, ...)`: boolean mask selecting modes inside the EoR window ($|k_\parallel| \ge m(z)\,k_\perp + b$).
   - `apply_wedge_filter_3d(field, box_length, redshift, los_axis=2, buffer_h_mpc=0.0)`: GPU-accelerated (`23.86×` at `256³`) 3D Fourier wedge filter.
   - `add_thermal_noise_3d(field, box_length, sigma_mk=None, redshift=8.0, t_int_hours=1000.0, n_ant=350, seed=0)`: GPU Philox (`10.71×` at `256³`) deterministic Gaussian thermal noise.
   - `field_moments_and_pdf(field, n_bins=32)`: GPU-accelerated (`26.43×` at `256³`) mean, std, variance, skewness, excess kurtosis, and 1-point PDF.
   - `SBISimulator` & `Likelihood21cm`: high-level wrappers for `sbi` posterior estimation (`"stats"`, `"ps1d"`, `"field"`) and `21CMMC` Gaussian likelihoods.

3. **Resumable Multi-GPU Worker Pool (`run_batch`)**:
   ```python
   tasks = [
       p21.BatchTask(
           task_id=f"sobol_{i:04d}",
           inputs=inp,
           mode="lightcone",
           min_redshift=6.5,
           max_redshift=9.5,
           cuda_mode="fast",
           apply_rsds=True,
           ps_n_bins=12,
           compute_ps_2d=True,
           wedge_filter=True,
           wedge_buffer=0.05,
           noise_sigma_mk=2.0,
           lightcone_out_path=f"lightcones/lc_{i:04d}.h5",
           lightcone_compression="device-lzf",
       )
       for i, inp in enumerate(...)
   ]

   results = p21.run_batch(
       tasks,
       devices=(0, 1, 2),
       workers_per_gpu=2,
       threads_per_worker=4,
       output_h5="sbi_dataset.h5",
       checkpoint_h5="sbi_dataset.h5",
       resume=True,
       use_cuda=True,
   )
   ```
   - `resume=True` reads completed `task_id` entries from `checkpoint_h5` and skips already finished realizations if an HPC job restarts.
   - Each task's failure is isolated in `BatchResult(ok=False, error=...)` without aborting the sweep.

---

## Multi-GPU Ensemble Throughput (`results_rtx4090_v060.json`)

For `16 × 192/96` `L-INTEGRAL + TS` coeval realizations (`workers_per_gpu=2`):

| GPUs | Concurrent Workers | Wall Time [s] | Throughput [tasks / hour] | Scaling vs 1 GPU |
|---:|---:|---:|---:|---:|
| 1 | 2 | `8.324` | `6,920.1` | `1.00×` |
| 2 | 4 | **`5.888`** | **`9,782.0`** | **`1.41×`** |
| 3 | 6 | `6.035` | `9,544.3` | `1.38×` |

See the complete runnable example in [`examples/sbi_lightcone_dataset.py`](../../examples/sbi_lightcone_dataset.py) or run directly from the CLI with `21cmcu batch --sobol-samples 16 ...`.
