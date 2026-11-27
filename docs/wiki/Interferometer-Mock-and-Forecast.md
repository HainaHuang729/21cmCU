# Interferometer Mock Observations & Sensitivity Forecasts

Back to [Home](Home.md) · Previous: [SBI & Batch Pipeline](SBI-and-Batch-Pipeline.md) · Next: [Lightcone Optimization & 3D Viz](Lightcone-Optimization-and-3D-Viz.md)

`21cmCU` (`py21cmcu.instrument` & `py21cmcu.observables`) turns raw 3D simulated brightness-temperature boxes into realistic radio interferometer mock observations and 1D/2D sensitivity forecasts in a single GPU pass (`p21_cuda_mock_observation_3d`).

---

## 1. Built-In Telescope Presets (`INSTRUMENT_PRESETS`)

| Preset | Antennas $N_{\mathrm{ant}}$ | Dish $D$ [m] | Baselines $[B_{\min}, B_{\max}]$ [m] | Channel $\Delta\nu$ | Default Wedge |
|---|---:|---:|---:|---:|---|
| **`SKA1-Low`** (`"ska"`) | `512` | `38.0` | `35 – 65,000` | `80.0 kHz` | `horizon` |
| **`HERA-350`** (`"hera"`) | `350` | `14.0` | `14.6 – 876` | `97.66 kHz` | `horizon` |
| **`MWA-II`** (`"mwa"`) | `128` | `4.0` | `7.7 – 5,300` | `80.0 kHz` | `fov` |
| **`LOFAR`** (`"lofar"`) | `48` | `30.75` | `68 – 80,000` | `195.3 kHz` | `fov` |

Customize any preset or supply real 2D antenna ENU coordinates (`antenna_xy_m`):

```python
import py21cmcu as p21

ska_deep = p21.get_instrument_preset("SKA1-Low", t_int_hours=2000.0, default_buffer_h_mpc=0.08)
```

---

## 2. Single-Pass GPU 3D Mock Observation (`mock_observation_3d`)

`mock_observation_3d(...)` executes a fused cuFFT R2C → C2R pipeline on GPU (**`>71×`** speedup over CPU at `256³` on RTX 4090, `~22.5 ms`):
1. Radial baseline UV cut $[k_{\perp,\min}, k_{\perp,\max}]$ from physical baseline lengths $[B_{\min}, B_{\max}]$.
2. Synthesized beam (PSF) Gaussian transverse smoothing ($\sigma_\perp \propto \lambda / B_{\max}$).
3. Frequency channel LOS smoothing (`channel_response="boxcar"` $\mathrm{sinc}$ or `"gaussian"`).
4. Foreground horizon/FoV wedge exclusion ($|k_\parallel| < m(z)\, k_\perp + b$).
5. Hermitian-symmetric Philox thermal noise sampled from $P_N(k_\perp, z) = X^2 Y \frac{\Omega_p \Omega_{\mathrm{eff}}}{N_{\mathrm{pol}}\, t_{\mathrm{int}}\, n_{\mathrm{uv}}(u)} T_{\mathrm{sys}}^2(z)$.

```python
mock_tb, meta = p21.mock_observation_3d(
    tb_box,
    box_length=192.0,
    redshift=8.0,
    instrument="SKA1-Low",
    seed=2026,
    return_metadata=True,
)
print(meta["t_sys_k"], meta["psf_fwhm_arcmin"], meta["backend"])
```

---

## 3. 1D & 2D Sensitivity / SNR Forecasts

```python
fc1d = p21.forecast_power_spectrum_error(
    tb_box.shape, box_length=192.0, redshift=8.0, instrument="SKA1-Low", signal_field=tb_box, n_bins=12
)
fc2d = p21.forecast_power_spectrum_2d_error(
    tb_box.shape, box_length=192.0, redshift=8.0, instrument="HERA-350", signal_field=tb_box, n_kperp=8, n_kpara=8
)
print("SKA1-Low 1D SNR:", fc1d["total_snr"], "HERA-350 2D SNR:", fc2d["total_snr"])
```

See the full runnable script in [`examples/05_interferometer_mock_and_forecast.py`](../../examples/05_interferometer_mock_and_forecast.py).
