"""Supply the existing thermal backend with C-compatible resident density."""
import ctypes as C
import time
import traceback
from pathlib import Path

import cupy as cp
import numpy as np

from gpu_heat_runtime import F, Runtime, arr

FilterHook = C.CFUNCTYPE(
    C.c_int, C.c_int, C.c_int, C.c_double, C.c_float, C.c_float,
    F, F, F, F, C.POINTER(F),
)

class ResidentRuntime(Runtime):
    def __init__(self):
        super().__init__()
        module = cp.RawModule(
            code=Path(__file__).with_name("resident_filter.cu").read_text(),
            options=("--fmad=false",),
        )
        self.filter_spectrum = module.get_function("filter_spectrum")
        self.extrapolate_density = module.get_function("extrapolate_density")
        self.filter_steps = []
        self.fallback_downloads = 0

    def install(self, native):
        super().install(native)
        self.filter_callback = FilterHook(self.filter_density)
        self.lib.SetGPUFilterHook.argtypes = [C.c_void_p]
        self.lib.SetGPUFilterHook.restype = None
        self.lib.SetGPUFilterHook(C.cast(self.filter_callback, C.c_void_p))

    def filter_density(self, side, count, dk, inverse_growth, low, density,
                       radii, minima, maxima, materialize):
        try:
            if materialize:
                for j in range(count):
                    self.density[j].get(out=arr(materialize[j], side**3))
                self.fallback_downloads += 1
                self.download_bytes += count * side**3 * 4
            else:
                self.compute_filter(side, count, dk, inverse_growth, low,
                                    density, radii, minima, maxima)
            return 0
        except Exception:
            self.error = traceback.format_exc()
            print(self.error, flush=True)
            return 1

    def compute_filter(self, side, count, dk, inverse_growth, low,
                       density, radii, minima, maxima):
        start = time.perf_counter()
        shape = (side,) * 3
        size = side**3
        spectrum = cp.fft.rfftn(self.filter_input(density, size).reshape(shape))
        scratch = cp.empty_like(spectrum)
        if self.density is None or self.density.shape != (count, size):
            self.density = cp.empty((count, size), dtype=cp.float32)
        for j, radius in enumerate(arr(radii, count)):
            self.filter_spectrum(
                ((spectrum.size + 255) // 256,), (256,),
                (spectrum, scratch, np.int32(side), np.float64(dk),
                 radius, np.int32(j != 0)),
            )
            smooth = cp.fft.irfftn(scratch, s=shape, norm="forward")
            self.extrapolate_density(
                ((size + 255) // 256,), (256,),
                (smooth, self.density[j], np.int64(size),
                 np.float32(inverse_growth), np.float32(low)),
            )
        bounds = cp.stack((self.density.min(axis=1), self.density.max(axis=1))).get()
        if not np.isfinite(bounds).all():
            raise ValueError("Nonfinite density bounds")
        arr(minima, count)[:] = np.minimum(bounds[0], 0)
        arr(maxima, count)[:] = np.maximum(bounds[1], 0)
        self.download_bytes += bounds.nbytes
        self.filter_steps.append(time.perf_counter() - start)

    def report(self):
        report = super().report()
        report.update(filter_step_seconds=self.filter_steps,
                      no_light_materializations=self.fallback_downloads)
        report["note"] = "GPU resident density, thermal state, and conditional collapse tables; host stages scalar coefficients"
        return report
