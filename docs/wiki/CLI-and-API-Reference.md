# CLI & API Reference

Back to [Home](Home.md) · Previous: [SBI & Batch Pipeline](SBI-and-Batch-Pipeline.md) · Next: [Architecture Overview](Architecture-Overview.md)

## `21cmcu` CLI Commands

`21cmcu` (`py21cmcu.cli:app`) inherits upstream `21cmfast` commands (`run`, `template`, `dev`, `predict`, `plot`, `migrate`) and adds three workflow subcommands:

- **`21cmcu run coeval|lightcone ...`**
  Upstream simulation runner with `--cuda-mode strict|fast` and `--no-use-cuda`.
- **`21cmcu batch`**
  Run CSV/JSON parameter sweeps or Sobol grids directly into checkpointed HDF5 summaries:
  ```bash
  21cmcu batch --sobol-samples 16 --mode lightcone --redshifts 6.5,8.0,9.5 \
    --devices 0,1 --workers-per-gpu 2 --cuda-mode fast \
    --wedge-filter --wedge-buffer 0.05 --noise-sigma-mk 2.0 --resume --out sbi.h5
  ```
  Supports both v4 names (`F_STAR10_ACG`, `F_ESC10_ACG`, `ALPHA_STAR_ACG`, `LX_OVER_SFR_ACG`) and legacy aliases (`F_STAR10`, `F_ESC10`, `ALPHA_STAR`, `L_X`, `F_STAR7_MINI`, `F_ESC7_MINI`, `ALPHA_STAR_MINI`, `L_X_MINI`).
- **`21cmcu doctor [--as-json]`**
  Inspect runtime versions, CUDA compilation/availability, GPU properties, `nvidia-smi topo -m`, compiler/library detection, and 24 GB grid sizing limits.
- **`21cmcu bench [--suite lightcone,batch] [--json out.json] [--resume]`**
  Run `benchmarks/bench_21cmcu.py` suites directly from the CLI.

---

## Environment Variables (`P21_*`)

| Variable | Effect | Default |
| --- | --- | --- |
| `P21_USE_CUDA` | Enable CUDA acceleration globally (`1`/`true`) | `False` |
| `P21_CUDA_MODE` | Precision mode: `strict` or `fast` | `strict` |
| `P21_CUDA_LAZY_D2H` | Opt-in GPU-only intermediate residency (`1`/`true`) | `False` |
| `P21_CUDA_DEVICES` | Active GPU IDs (e.g. `0,1,2` or `all`) | `None` (single GPU 0) |
| `P21_CUDA_FAST_HALO_SAMPLER` | Control GPU halo sampler override | Config-linked |
| `P21_CUDA_REGISTRY_MB` | Cap `DeviceFieldRegistry` VRAM budget (MiB) | Auto |
| `P21_CUDA_REGISTRY_MAX_ENTRY_MB` | Max single cached array size in registry (MiB) | Auto |
| `P21_CUDA_MGPU_SINGLE_SLAB` | Force single-work-slab multi-GPU FFT/IC mode (`1`) | Auto (`2048³` Zel'dovich) |
| `P21_CUDA_FORCE_MGPU` | Force multi-GPU slab path on smaller test grids (`1`) | `0` |
| `P21_CUDA_PERTURB_SLAB_X` | Override perturbation X-slab width | Auto |
| `P21_CUDA_ION_SPILL_KSPACE` | Force host spill for ionization k-space arrays | Auto |
| `P21_CLASS_CACHE` / `P21_CLASS_DISK_CACHE` | Toggle memory / `.npz` disk cache for `CLASS` tables | Enabled |
| `P21_FFTW_PLAN_CACHE` | Toggle 32-entry bit-exact FFTW plan LRU cache in `dft.c` (`0` to disable) | Enabled (`1`) |
| `P21_CPU_SCRATCH_POOL` / `P21_CPU_SCRATCH_POOL_MB` | Toggle 24-slot `fftwf_malloc` scratch pool & max MiB budget (`0` disables) | Enabled (`1`, `2048` MiB) |
| `P21_CPU_FILTER_FUSED` | Toggle hoisted-switch multi-box `k`-space filtering in `filtering.c` | Enabled (`1`) |
| `P21_CPU_RSDS_NATIVE` | Toggle bit-exact C/OpenMP CPU `apply_rsds` / `rsds_shift` (`0` to use Python/SciPy fallback) | Enabled (`1`) |
| `P21_CPU_NATIVE` | Opt-in `-march=native` host compiler flag in `build_cffi.py` | Off |

---

## `py21cmcu` Python API Exports

Besides re-exporting `py21cmfast.*` (`InputParameters`, `RectilinearLightconer`, `ObliqueRectilinearLightconer`, `AngularLightconer`, `run_coeval`, `run_lightcone`, `compute_initial_conditions`, `perturb_field`, `compute_ionization_field`, `spin_temperature`, `brightness_temperature`, `config`, etc.), `py21cmcu` exports:

- **Runtime & Device Control**:
  `cuda_compiled()`, `cuda_available()`, `cuda_device_count()`, `get_cuda_devices()`, `set_cuda_devices()`, `get_cuda_mode()`, `set_cuda_mode()`, `use_mode(...)`, `cuda_stats()`, `reset_cuda_stats()`, `cleanup_cuda()`, `clear_class_cache()`, `doctor()`
- **Analysis, Power Spectra & Non-Gaussian Topology**:
  `power_spectrum_1d()`, `power_spectrum_2d()`, `lightcone_chunk_power_spectra()`, `lightcone_chunk_power_spectra_2d()`, `field_moments_and_pdf()`, `bispectrum_3d()`, `bubble_size_distribution_mfp()`, `minkowski_functionals_3d()`
- **I/O Compression**:
  `save_lightcone()` (`compression="device-lzf" | "lzf" | "none"`)
- **Observables, Interferometers & Forecasts**:
  `horizon_slope()`, `horizon_wedge_mask()`, `apply_wedge_filter_3d()`, `add_thermal_noise_3d()`, `InterferometerConfig`, `INSTRUMENT_PRESETS`, `get_instrument_preset()`, `baseline_density_uv()`, `thermal_noise_power_spectrum()`, `mock_observation_3d()`, `forecast_power_spectrum_error()`, `forecast_power_spectrum_2d_error()`, `FisherForecaster`, `FisherForecastResult`, `fisher_forecast_21cm()`
- **Multi-Probe Cross-Correlations & Topology**:
  `cic_deposit_3d()`, `load_gravitycu_halos()`, `cross_power_21cm_galaxy()`, `cross_power_21cm_galaxy_error()`, `ksz_momentum_map()`, `bispectrum_3d()`, `bubble_size_distribution_mfp()`, `minkowski_functionals_3d()`
- **Ergonomic Workflows & Visualization**:
  `quick_coeval()`, `quick_lightcone()`, `export_volume_3d_html()`, `export_lightcone_3d_html()`, `plot_lightcone_summary()`, `plot_coeval_slices()`, `plot_mock_comparison()`, `plot_fisher_contours()`
- **Zero-Copy Interop & SBI / Batch**:
  `CudaFieldView`, `to_cuda_view()`, `lightcone_to_cuda_views()`, `to_torch()`, `SBISimulator`, `OnlineSBIDataset`, `Likelihood21cm`, `BatchPool`, `BatchTask`, `BatchResult`, `run_batch()`, `save_batch_h5()`, `generate_sobol_parameter_grid()`
