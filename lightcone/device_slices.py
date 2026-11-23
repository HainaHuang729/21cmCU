"""Rectilinear lightcone interpolation from consecutive device snapshots."""

import cupy as cp
import numpy as np


class DeviceSlices:
    def __init__(self, runtime):
        self.runtime = runtime
        self.previous = {}
        self.calls = 0
        self.kernel = cp.ElementwiseKernel(
            "raw float32 a, raw float32 b, raw int32 k, raw float32 u, raw float32 v, "
            "int32 side, int32 count, float32 denom, int32 mean_max",
            "float32 y",
            """int s=i%count; long j=(i/count)*side+k[s];
            float x1=a[j], x2=b[j];
            y = mean_max && x1*x2<0 ? fmaxf(x1,x2) : (u[s]*x1+v[s]*x2)/denom;""",
            "interpolate_device_slice",
            options=("--fmad=false",),
        )

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
            side = a.shape[0]
            current = self.field(name, a)
            entry = self.previous.get(name)
            prior = (
                entry[1] if entry is not None and entry[0] is b else self.field(name, b)
            )
            slab = self.kernel(
                current,
                prior,
                k,
                u,
                v,
                np.int32(side),
                np.int32(len(k)),
                np.float32(denom),
                np.int32(mean_max),
                size=side * side * len(k),
            )
            output[..., indices] = slab.reshape(side, side, -1).get()
            self.previous[name] = (a, current)
        self.calls += 1
