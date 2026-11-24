"""Build the upstream inhomogeneous-recombination table with the v3 CUDA kernels."""

import cupy as cp
import numpy as np


def build(module, rule, density, helium_mass_fraction, alpha):
    slopes_kernel = module.get_function("natural_derivatives")

    def slopes(knots, values):
        result = cp.empty_like(values)
        rows = values.size // knots.size
        slopes_kernel(
            ((rows + 31) // 32,),
            (32,),
            (knots, values, np.int32(rows), np.int32(knots.size), result),
        )
        return result

    def interpolate(z, samples):
        values = cp.asarray(samples, dtype="d")
        x = cp.arange(values.size, dtype="d") + 2
        m = slopes(x, values)
        offset = cp.clip(z - 2, 0, values.size - 1)
        i = cp.minimum(offset.astype("i"), values.size - 2)
        t = offset - i
        u = 1 - t
        return u * u * ((1 + 2 * t) * values[i] + t * m[i]) + t * t * (
            (3 - 2 * t) * values[i + 1] - u * m[i + 1]
        )

    def parameters(z):
        c = interpolate(
            z,
            [
                0.558,
                0.599,
                0.611,
                0.769,
                0.868,
                0.930,
                0.964,
                0.983,
                0.993,
                0.998,
                0.999,
                1.0,
            ],
        )
        beta = interpolate(z, [-2.23, -2.35, -2.48, -2.49, -2.50])
        return c, beta

    az = cp.arange(2, 62, dtype="d")
    pdf = cp.stack((az, *parameters(az)), axis=1)
    norm = cp.empty_like(az)
    module.get_function("mhr_pdf_normalization")(
        (60,), (32,), (pdf, rule, np.int32(60), norm)
    )
    z = (cp.arange(300, dtype="f") * np.float32(0.2)).astype("d")
    rows = cp.stack(
        (z, cp.zeros_like(z), *parameters(z), density * (1 + z) ** 3), axis=1
    )
    module.get_function("mhr_pdf_interpolate")(
        (5,), (64,), (rows, az, norm, slopes(az, norm), np.int32(300), np.int32(60))
    )
    knots = (cp.arange(250, dtype="f") * np.float32(0.1)).astype("d") - 10
    rates = cp.empty((300, 250), dtype="d")
    helium = 1 + 1 / (4 / helium_mass_fraction - 3)
    module.get_function("mhr_rates")(
        (rates.size,),
        (32,),
        (
            rows,
            knots,
            rule,
            np.int32(300),
            np.int32(250),
            np.float64(alpha),
            np.float64(helium),
            rates,
        ),
    )
    deriv = slopes(knots, rates)
    if not bool(cp.all(cp.isfinite(deriv)) & cp.all(cp.isfinite(rates))):
        raise FloatingPointError("Invalid GPU recombination table")
    return knots, rates, deriv
