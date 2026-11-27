# Lightcone Optimization, Zero-Copy PyTorch Interop & 3D Visualization

Back to [Home](Home.md) · Previous: [Interferometer Mock & Forecast](Interferometer-Mock-and-Forecast.md) · Next: [CLI & API Reference](CLI-and-API-Reference.md)

---

## 1. One-Line Ergonomic API (`quick_coeval` & `quick_lightcone`)

Instead of assembling `InputParameters`, `RectilinearLightconer`, and `use_mode(...)` manually, call `quick_coeval` or `quick_lightcone`:

```python
import py21cmcu as p21

# One-line coeval box
cv = p21.quick_coeval(8.0, box_len=200.0, dim=128)

# One-line optimized GPU lightcone + SKA1-Low mock + 3D HTML viewer
lc = p21.quick_lightcone(
    z_min=6.5,
    z_max=10.5,
    box_len=200.0,
    dim=128,
    instrument="SKA1-Low",
    html_3d_path="out/lightcone_3d.html",
)
```

### Under-the-Hood Lightcone Pipeline Optimizations
Why must lightcones step through redshifts sequentially? When **`USE_TS_FLUCT=True`** or **`RECOMB_MODEL != "none"`**, each redshift integrates the causal spin-temperature / recombination history from $z_{i-1}$. Within that physical constraint, `21cmCU` eliminates three host/memory bottlenecks:
1. **Zero-D2H Global History (`p21_cuda_field_mean_f32`)**: Computes `global_quantities` means directly from GPU-resident lazy arrays via shared-memory reductions without materializing full 3D cubes on the host.
2. **Zero-D2H Batched Lightcone Slicing**: `RectilinearLightconer.make_lightcone_slices` fetches GPU-resident slabs (`_get_coeval_array_no_sync`) directly inside `p21_cuda_rectilinear_slices`.
3. **On-Demand Streaming `PerturbedField` (`stream_perturbed_fields=True`)**: Generates each node's `PerturbedField` on demand inside the redshift loop and releases the previous node immediately ($O(1)$ peak `PerturbedField` memory instead of storing all $N_z$ boxes simultaneously).
4. **Causal-Free Redshift-Parallel Lightcones (`parallel_redshifts=True`)**: For history-independent analytic EoR configurations (`SOURCE_MODEL="L-INTEGRAL"` or `"E-INTEGRAL"`, `use_ts_fluct=False`, `recomb_model="none"`, `USE_MCGS=False`, `PHOTON_CONS_TYPE="no-photoncons"`), shares initial conditions across multiple GPUs/workers to evolve overlapping redshift segments in parallel and stitches them with identical `dvdr`/`RSDs`. When causal history or discrete halo merger trees are enabled, it automatically preserves exact serial ordering.

---

## 2. Zero-Copy PyTorch / ML DLPack Interop (`py21cmcu.interop`)

Under `lazy_d2h=True`, simulated boxes remain on the GPU (`DeviceFieldRegistry`) and export directly to PyTorch via DLPack or `__cuda_array_interface__` with **zero host-to-device transfers** (`registry_h2d_bytes == 0`):

```python
import py21cmcu as p21

with p21.use_mode("fast", lazy_d2h=True):
    cv = p21.quick_coeval(8.0, box_len=128.0, dim=128, lazy_d2h=True)
    tensor = p21.to_torch(cv.brightness_temp)   # zero-copy PyTorch CUDA tensor

    # Or explicit lifetime management with CudaFieldView
    with p21.to_cuda_view(cv.neutral_fraction) as view:
        print(view.__cuda_array_interface__, view.__dlpack_device__())
```

Pair with `p21.Likelihood21cm` (for `21CMMC`-style MCMC) and `p21.SBISimulator` (batched `"stats"`, `"ps1d"`, `"field"` summaries) in [`examples/04_dlpack_and_sbi_inference.py`](../../examples/04_dlpack_and_sbi_inference.py).

---

## 3. Offline Interactive 3D WebGL Viewer & 2D Summary Plots (`py21cmcu.viz`)

No ParaView, PyVista, or internet CDN is required. `export_lightcone_3d_html(...)` and `export_volume_3d_html(...)` embed quantized, zlib-compressed 3D arrays directly into a single self-contained **WebGL2 raymarching HTML5** file you can open in any desktop browser (including locally synced files from an HPC cluster):

- Interactive mouse orbit, scroll zoom, and PNG snapshot download
- Live `X`, `Y`, and `Z / Redshift` orthogonal slice sliders with Mpc / `z` readouts
- Adjustable threshold & volume opacity for inspecting ionized HII bubbles vs neutral ridges
- 5 colormaps (`eor_21cm`, `viridis`, `inferno`, `plasma`, `coolwarm`)

For static paper/notebook figures (requires `matplotlib`):
- `plot_lightcone_summary(lc, out_path="lc_summary.png")`
- `plot_coeval_slices(coeval, out_path="coeval_slices.png")`
- `plot_mock_comparison(clean_tb, mock_tb, box_length=200.0, redshift=8.0, out_path="mock_cmp.png")`
