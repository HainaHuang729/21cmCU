"""cuFFT density and 1LPT/2LPT, retaining the production Fourier RNG."""
import ctypes as C
import os
from pathlib import Path
import time
import traceback

import cupy as cp
import numpy as np

FloatPtr = C.POINTER(C.c_float)
Callback = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_int, C.c_int, C.c_double,
                      C.c_double, C.c_float, C.POINTER(FloatPtr))


class ICRuntime:
    def __init__(self, native):
        source = Path(__file__).with_name("ic.cu").read_text()
        module = cp.RawModule(code=source, options=("--fmad=false",))
        self.transform = module.get_function("ic_modes")
        self.combine = module.get_function("ic_source")
        self.fft64 = os.environ.get("PS_IC_FFT64") == "1"
        self.device_ic = os.environ.get("PS_DEVICE_IC") == "1"
        self.archive_complete = os.environ.get("PS_ARCHIVE_GPU_IC") == "1"
        self.fields = {}
        self.seconds = []
        self.lib = C.CDLL(native.__file__)
        self.callback = Callback(self.run)
        setter = self.lib.SetICGPUHook
        setter.argtypes = [C.c_void_p]
        setter.restype = None
        setter(self.callback)

    @staticmethod
    def launch(kernel, size, args):
        kernel(((size + 255)//256,), (256,), args)

    def run(self, pointer, side, coarse, dk, volume, radius, outputs):
        try:
            self.compute(pointer, side, coarse, dk, volume, radius, outputs)
            return 0
        except Exception:
            traceback.print_exc()
            return 1

    def compute(self, pointer, side, coarse, dk, volume, radius, outputs):
        start = time.perf_counter()
        spectral_shape = (side, side, side//2 + 1)
        spectral_size = np.prod(spectral_shape)
        packed_host = np.ctypeslib.as_array(C.cast(pointer, FloatPtr), shape=(2*spectral_size,))
        modes = cp.asarray(packed_host.view(np.complex64).reshape(spectral_shape))
        self.compute_modes(side,coarse,dk,volume,radius,modes,outputs,start)

    def compute_modes(self,side,coarse,dk,volume,radius,modes,outputs,start):
        shape = (side,)*3
        spectral_size = modes.size
        buffer = cp.empty_like(modes)
        ratio = side//coarse

        if self.device_ic:
            from plan_cuda.fft_plan_cuda import transform
        def irfft(source):
            if self.device_ic:
                return cp.ascontiguousarray(transform(source, side, True, threads=1))
            if self.fft64:
                precise = cp.fft.irfftn(source.astype(cp.complex128), s=shape, norm="forward")
                return precise.astype(cp.float32)
            return cp.fft.irfftn(source, s=shape, norm="forward")

        capture = getattr(self, "capture", lambda *unused: None)

        def inverse(source, mode, volume, radius, a=0, b=0, stage=0, index=0):
            self.launch(self.transform, spectral_size, (source, buffer, np.int32(side),
                np.float64(dk), np.float64(volume), np.float32(radius), np.int32(mode),
                np.int32(a), np.int32(b), np.int32(self.device_ic)))
            capture({2:1,4:3,5:10,9:8}[stage],index,buffer)
            result=irfft(buffer)
            capture(stage,index,result)
            return result

        def sample(field):
            return cp.ascontiguousarray(field[::ratio, ::ratio, ::ratio])

        if ratio == 1:
            radius = 0.0
        high = irfft(modes)
        capture(0,0,high)
        high /= np.float32(volume)
        smooth = sample(inverse(modes, 0, volume, radius, stage=2))
        smooth /= np.float32(volume)
        displacement = cp.empty((6, coarse**3), dtype=cp.float32)
        for axis in range(3):
            displacement[axis] = sample(inverse(modes, 1, volume, radius, axis, stage=4, index=axis)).ravel()

        diagonal = [inverse(modes, 2, volume, 0, axis, axis, stage=5, index=axis) for axis in range(3)]
        if self.archive_complete:
            # In low-resolution mode the native hires 2LPT slots retain the
            # diagonal Hessian intermediates. The hires 1LPT slots stay zero.
            for index, field in enumerate(diagonal, start=8):
                if not outputs[index]:
                    raise ValueError("Native IC Hessian buffer is missing")
                field.ravel().get(out=np.ctypeslib.as_array(outputs[index], shape=(field.size,)))
        source = cp.zeros_like(high)
        for index, (a,b) in enumerate(((0, 1), (0, 2), (1, 2)),3):
            cross = inverse(modes, 2, volume, 0, a, b, stage=5, index=index)
            self.launch(self.combine, source.size, (source, diagonal[a], diagonal[b],
                                                    cross, np.int64(source.size)))
            del cross
        del diagonal
        source /= np.float32(source.size)
        capture(6,0,source)
        if self.device_ic:
            second_modes = cp.ascontiguousarray(transform(source, side, threads=1))
        elif self.fft64:
            second_modes = cp.fft.rfftn(source.astype(cp.float64)).astype(cp.complex64)
        else:
            second_modes = cp.fft.rfftn(source)
        capture(7,0,second_modes)
        del source
        for axis in range(3):
            displacement[axis+3] = sample(inverse(second_modes, 1, 1.0, radius, axis, stage=9, index=axis)).ravel()

        # The Python data objects still require their public host fields.
        for index, field in enumerate((high, smooth, *displacement)):
            field.ravel().get(out=np.ctypeslib.as_array(outputs[index], shape=(field.size,)))
        key = C.cast(outputs[0], C.c_void_p).value
        self.fields[key] = (high.ravel(), displacement)
        self.seconds.append(time.perf_counter()-start)
