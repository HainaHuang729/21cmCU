# Quickstart

Back to [Home](Home.md) · Previous: [Installation & Containers](Installation-and-Containers.md) · Next: [Precision Modes & Calibration](Precision-Modes-and-Calibration.md)

## How Activation Works

By default, 21cmCU mirrors upstream 21cmFAST (`USE_CUDA=False`, `CUDA_MODE="strict"`, `CUDA_LAZY_D2H=False`) so existing scripts remain conservative until you opt in via **Python context manager** or **environment variables**.

### Python Context Manager (Recommended)

```python
import py21cmcu as p21

with p21.use_mode("fast", lazy_d2h=True):
    coeval = p21.run_coeval(inputs=inputs, out_redshifts=[8.0], write=False)[0]
    ps1 = p21.power_spectrum_1d(coeval.brightness_temp, box_length=128.0, n_bins=12, remove_mean=True)
```

### Environment Variables (Zero Code Changes)

```bash
export P21_USE_CUDA=1
export P21_CUDA_MODE=fast
export P21_CUDA_LAZY_D2H=1   # optional GPU intermediate residency

python your_existing_21cmfast_script.py
```

---

## Five-Minute Runnable Examples (`examples/`)

All scripts below have been verified end-to-end on cluster node `b1` (`RTX 4090`):

1. **Single-GPU Coeval + 1D Power Spectrum** (`examples/01_quickstart_coeval.py`):
   ```bash
   python examples/01_quickstart_coeval.py --dim 128 --hii-dim 64 --box-len 128 --redshift 8.0
   ```
   - Verified on RTX 4090: completes `L-INTEGRAL + TS` coeval + 1D `P(k)` in **`0.556 s`** (`584` registry hits, `<x_HI> = 0.6120`, `12/12` finite `k` bins).

2. **GPU Lightcone + RSDs + Wedge + Thermal Noise + `DeviceLZF`** (`examples/02_lightcone_power_spectrum.py`):
   ```bash
   python examples/02_lightcone_power_spectrum.py --dim 128 --hii-dim 64 --out lightcone_quickstart.h5
   ```
   - Verified on RTX 4090: runs rectilinear lightcone (`z=7–10`, `apply_rsds=True`), 4-chunk 1D `P(k)`, foreground wedge filtering, `2 mK` Gaussian thermal noise, and writes `DeviceLZF` HDF5 in **`2.53 s`**.

3. **Multi-GPU Slab Domain-Decomposed IC** (`examples/03_multigpu_large_ic.py`):
   ```bash
   python examples/03_multigpu_large_ic.py --dim 256 --hii-dim 128 --devices 0,1
   python examples/03_multigpu_large_ic.py --dim 1536 --hii-dim 256 --algo 2LPT --devices 0,1
   ```
   - Verified on `b1`: `256³ -> 128³` Zel'dovich on `(0, 1)` finishes IC in **`0.474 s`** (`mgpu_ic_runs=1`); `2LPT` on `(0, 1, 2)` finishes IC in **`0.569 s`**. Scales up to `2048³` Zel'dovich (`58.467 s` on 3× RTX 4090).

4. **4D Sobol SBI Lightcone Sweep** (`examples/sbi_lightcone_dataset.py`):
   ```bash
   python examples/sbi_lightcone_dataset.py --samples 16 --devices 0,1 --workers-per-gpu 2 --out sbi.h5
   ```
   - Runs scrambled Sobol sampling over `(F_STAR10_ACG, F_ESC10_ACG, ALPHA_STAR_ACG, M_TURN)` with resumable HDF5 checkpoints, wedge filter, and thermal noise.

5. **DLPack Zero-Copy & SBI / `21CMMC` Likelihood** (`examples/04_dlpack_and_sbi_inference.py`):
   - Demonstrates `SBISimulator`, `CudaFieldView` / `to_torch()`, `field_moments_and_pdf()`, and `Likelihood21cm`.

6. **Interferometer Mock Observations & Sensitivity Forecasts** (`examples/05_interferometer_mock_and_forecast.py`):
   - Simulates `SKA1-Low` & `HERA-350` 3D mocks on GPU (`~23 ms` at `256³`) and computes 1D/2D EoR-window SNR forecasts.

7. **One-Line Lightcone + 3D WebGL Viewer** (`examples/06_quick_lightcone_and_3d_viz.py`):
   - Generates a lightcone, injects `SKA1-Low` mock effects, and exports a standalone interactive 3D HTML viewer in 1 call (`quick_lightcone`).

8. **Interactive Notebook** (`examples/quickstart.ipynb`):
   - Self-contained 5-cell Jupyter quickstart covering `doctor()`, coeval, lightcone, and `run_batch`.

---

## Important 21cmFAST v4.3 Tips for New Scripts

- For `USE_TS_FLUCT=True` or inhomogeneous recombinations, pass `node_redshifts` extending **above `Z_HEAT_MAX`** (e.g., `node_redshifts=[8.0, 9.5, 11.0, 12.5]` when `Z_HEAT_MAX=12.0`).
- Use updated v4.3 `AstroParams` names (`F_STAR10_ACG`, `F_ESC10_ACG`, `ALPHA_STAR_ACG`, `LX_OVER_SFR_ACG`), or see [Troubleshooting & FAQ](Troubleshooting-and-FAQ.md).
- Always call `py21cmcu.cleanup_cuda()` when looping over many independent grid sizes in the same Python process to free pooled device buffers cleanly.
