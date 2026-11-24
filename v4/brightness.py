"""Coeval brightness and optical depth for 21cmFAST v4.3.2."""

import cupy as cp
import numpy as np
from py21cmfast import BrightnessTemp


_spin_brightness = cp.ElementwiseKernel(
    "float32 prefactor, float32 neutral, float32 density, float32 spin, "
    "float32 radiation, float64 optical_scale, float64 brightness_scale",
    "float32 tau, float32 brightness",
    """
    const float thin = (prefactor * neutral) * (1.f + density);
    tau = (float)((optical_scale / (double)spin) * (double)thin);
    brightness = (float)(((double)(spin - radiation) * (1. - exp(-(double)tau)))
                         * brightness_scale);
    """,
    "v432_spin_brightness",
    options=("--fmad=false",),
)


def brightness_temperature_gpu(
    *, ionized_box, perturbed_field, spin_temp=None, registry=None
):
    inputs = ionized_box.inputs
    use_spin = inputs.astro_options.USE_TS_FLUCT
    if use_spin and spin_temp is None:
        raise ValueError("Spin-temperature fluctuations require a spin temperature box")
    cosmo = inputs.cosmo_params
    z = float(np.float32(ionized_box.redshift))
    h = np.float32(cosmo.hlittle)
    baryons = np.float32(cosmo.OMb) * h * h
    matter = float(np.float32(cosmo.OMm)) * float(h) * float(h)
    scale = 0.015 / matter
    prefactor = np.float32(float(baryons) * (27.0 / 0.023) * np.sqrt(scale * z + scale))
    read = cp.asarray if registry is None else registry.lookup_host
    neutral = read(ionized_box.get("neutral_fraction"))
    density = read(perturbed_field.get("density"))
    box = BrightnessTemp.new(inputs=inputs, redshift=ionized_box.redshift)
    if use_spin:
        # v4.3.2 Constants.c defines the present-day CMB temperature in K.
        radiation = np.float32(2.7255 * float(np.float32(1.0) + np.float32(z)))
        tau, brightness = _spin_brightness(
            prefactor,
            neutral,
            density,
            read(spin_temp.get("spin_temperature")),
            radiation,
            (1.0 + z) * 0.001,
            1000.0 / (1.0 + z),
        )
        box.set("tau_21", tau.get())
        if registry is not None:
            registry.record(box.get("tau_21"), tau)
    else:
        brightness = (prefactor * neutral) * (np.float32(1.0) + density)
    if not bool(cp.isfinite(brightness).all()):
        raise FloatingPointError("Nonfinite v4 brightness temperature")
    box.set("brightness_temp", brightness.get())
    if registry is not None:
        registry.record(box.get("brightness_temp"), brightness)
    return box
