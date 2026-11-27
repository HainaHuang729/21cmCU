# Troubleshooting & FAQ

Back to [Home](Home.md) · Previous: [Architecture Overview](Architecture-Overview.md)

### 1. Build fails with `fatal error: Python.h: No such file or directory`
Your Python environment lacks development headers (common on minimal Slurm/compute nodes).
- **Ubuntu/Debian**: `sudo apt-get install python3.12-dev`
- **Conda**: `conda install python=3.12` inside the active environment includes `Python.h`.
- **Container alternative**: run `docker/Dockerfile` or `docker/21cmcu.def` where headers, `libgsl-dev`, and `libfftw3-dev` are pre-installed.

### 2. Build cannot find `-lgsl` or `-lfftw3f`
Install single-precision FFTW3 (`libfftw3-dev` on Ubuntu, providing `libfftw3f.so.3`) and GSL (`libgsl-dev`), or point `LD_LIBRARY_PATH` / `PKG_CONFIG_PATH` / `CPATH` to your HPC module or Conda environment, then re-run `./scripts/install.sh`.

### 3. My script still runs on CPU instead of GPU!
By default, `USE_CUDA=False` (`strict` mode) to match upstream 21cmFAST behavior. Either wrap your code in `with py21cmcu.use_mode("fast"):` or export `P21_USE_CUDA=1 P21_CUDA_MODE=fast`. Verify with `21cmcu doctor` that `compiled=True` and `available=True`.

### 4. `ValueError: F_STAR10_ACG is set to ... but F_STAR10 is ...!` (or `ALPHA_STAR` / `F_ESC10`)
In 21cmFAST v4.3, astrophysical parameters split into ACG (atomic cooling galaxies) and MCG (molecular cooling minihaloes):
- Use **`F_STAR10_ACG`**, **`F_ESC10_ACG`**, **`ALPHA_STAR_ACG`**, and **`LX_OVER_SFR_ACG`** when calling `InputParameters.evolve_input_structs(...)`.
- Note: `21cmcu batch` CLI automatically maps legacy names in CSV/JSON parameter tables to their `_ACG`/`_MCG` counterparts.

### 5. `ValueError: maximum passed node_redshifts must be above Z_HEAT_MAX`
When `USE_TS_FLUCT=True` or inhomogeneous recombinations are enabled, `node_redshifts` must span from above `Z_HEAT_MAX` down to your target output redshift, e.g.:
```python
inputs = p21.InputParameters(
    random_seed=42,
    node_redshifts=[8.0, 9.5, 11.0, 12.5],
).evolve_input_structs(USE_TS_FLUCT=True, Z_HEAT_MAX=12.0)
coeval = p21.run_coeval(inputs=inputs, out_redshifts=[8.0])[0]
```

### 6. `TypeError: USE_2LPT is not a valid keyword input`
In v4.3, pass **`PERTURB_ALGORITHM="2LPT"`** or **`PERTURB_ALGORITHM="ZELDOVICH"`** to `evolve_input_structs(...)`.

### 7. Out-of-VRAM on single GPU for large boxes (`768³–1024³`)
- Reduce device cache budget: `export P21_CUDA_REGISTRY_MB=2048`.
- Call `py21cmcu.cleanup_cuda()` between unrelated simulation dimensions.
- Use `PERTURB_ON_HIGH_RES=False` (`lo` perturbation on `HII_DIM³` rather than `DIM³`).
- For `1536³–2048³`, assign multiple GPUs (`set_cuda_devices([0, 1, 2])`).

### 8. Why was `2048³` `2LPT` skipped on `196 GiB` host RAM while `2048³` `ZELDOVICH` succeeded?
At `2048³` (`8.590 B` voxels), Zel'dovich fits natively inside `3× 24 GB` VRAM via single-work-slab streaming (`58.467 s`, ~`72 GB` host RAM budget). Full 2LPT requires additional host-staged potential accumulation arrays totaling ~`235 GiB` host RAM, so systems under `235 GiB` skip `2048³ 2LPT` rather than crashing with host OOM.
