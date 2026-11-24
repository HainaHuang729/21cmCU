"""Rectilinear lightcone interpolation from consecutive device snapshots."""

import cupy as cp
import numpy as np


_weighted_value = """
template<typename F, typename W>
__device__ W distance_mix(F a, F b, W u, W v, W denom, int mean_max) {
    if (mean_max && a * b < 0) return a > b ? a : b;
    return (u * a + v * b) / denom;
}
"""


interpolate_values = cp.ElementwiseKernel(
    "F a, F b, W u, W v, W denom, int32 mean_max",
    "W y",
    "y = distance_mix(a, b, u, v, denom, mean_max);",
    "interpolate_lightcone_values",
    preamble=_weighted_value,
    options=("--fmad=false",),
)


# F preserves the coeval field precision in the mean_max sign test. W determines
# the precision of the distance-weighted sum (v3 float, v4 Astropy result type).
_interpolate = cp.ElementwiseKernel(
    "raw F a, raw F b, raw int32 k, raw W u, raw W v, "
    "int32 depth, int32 count, W denom, int32 mean_max",
    "W y",
    """int s = i % count;
    long j = (i / count) * depth + k[s];
    F x1 = a[j], x2 = b[j];
    y = distance_mix(x1, x2, u[s], v[s], denom, mean_max);""",
    "interpolate_device_slice",
    preamble=_weighted_value,
    options=("--fmad=false",),
)


def interpolate_slices(a, b, k, u, v, denom, mean_max):
    """Interpolate matching device grids at periodic LOS indices prepared by the caller."""
    nx, ny, depth = a.shape
    slab = _interpolate(
        cp.ascontiguousarray(a),
        cp.ascontiguousarray(b),
        k,
        u,
        v,
        np.int32(depth),
        np.int32(k.size),
        u.dtype.type(denom),
        np.int32(mean_max),
        size=nx * ny * k.size,
    )
    return slab.reshape(nx, ny, k.size)


class DeviceSlices:
    def __init__(self, runtime):
        self.runtime = runtime
        self.previous = {}
        self.calls = 0

    def field(self, name, host):
        addr = host.ctypes.data
        r = self.runtime
        if name == "brightness_temp":
            value = r.brightness.fields.get(addr)
        elif name == "xH_box":
            value = r.ionization.neutral_fields.get(addr)
        elif name == "Ts_box":
            value = r.spin_fields.get(addr)
        elif name == "x_e_box":
            value = r.electron_fields.get(addr)
        elif name == "Tk_box":
            value = r.temperature_fields.get(addr)
        elif name == "dNrec_box":
            value = r.ionization.rec_fields.get(addr)
        else:
            value = None
        return cp.asarray(host) if value is None else value

    def fill(self, names, arrays, indices, k, u, v, denom):
        k, u, v = (
            cp.asarray(k),
            cp.asarray(u, dtype=cp.float32),
            cp.asarray(v, dtype=cp.float32),
        )
        for name, (a, b, output, mean_max) in zip(names, arrays):
            current = self.field(name, a)
            entry = self.previous.get(name)
            prior = (
                entry[1] if entry is not None and entry[0] is b else self.field(name, b)
            )
            output[..., indices] = interpolate_slices(
                current.reshape(a.shape),
                prior.reshape(b.shape),
                k,
                u,
                v,
                denom,
                mean_max,
            ).get()
            self.previous[name] = (a, current)
        self.calls += 1
