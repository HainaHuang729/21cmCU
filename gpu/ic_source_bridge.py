"""Compute the quadratic 2LPT source on CUDA between reference FFTW transforms."""
import ctypes as C
from pathlib import Path
import time
import traceback

import cupy as cp
import numpy as np

FloatPtr = C.POINTER(C.c_float)
Callback = C.CFUNCTYPE(C.c_int, C.c_int, C.c_int, FloatPtr, FloatPtr, FloatPtr, FloatPtr)
ModesObserver = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_int, C.c_int, C.c_double,
                           C.c_double, C.c_float, C.POINTER(FloatPtr))

class ReferenceFFTIC:
    def __init__(self, native):
        self.kernel = cp.RawKernel(Path(__file__).with_name("ic_source_padded.cu").read_text(),
                                   "ic_source_padded", options=("--fmad=false",))
        self.fields = {}
        self.seconds = []
        self.phase_seconds = []
        self.lib = C.CDLL(native.__file__)
        self.callback = Callback(self.apply)
        self.start_callback = ModesObserver(self.start)
        for name, hook in (("SetICSourceHook", self.callback), ("SetICGPUHook", self.start_callback)):
            setter = getattr(self.lib, name)
            setter.argtypes = [C.c_void_p]
            setter.restype = None
            setter(C.cast(hook, C.c_void_p))

    def start(self, *_):
        # Status 2 leaves the original FFTW path in charge of the transforms.
        return 2

    def apply(self, side, pair, target, left, right, mixed):
        try:
            begin = time.perf_counter()
            stride = 2*(side//2+1)
            n, padded = side**3, side**2*stride
            if pair == 0:
                self.source = cp.zeros(padded, dtype=cp.float32)
                self.diagonal = {}
                self.begin = begin
            axes = ((0, 1), (0, 2), (1, 2))[pair]
            fields = []
            for axis, pointer in zip(axes, (left, right)):
                if axis not in self.diagonal:
                    self.diagonal[axis] = cp.asarray(np.ctypeslib.as_array(pointer, shape=(n,)))
                fields.append(self.diagonal[axis])
            cross = cp.asarray(np.ctypeslib.as_array(mixed, shape=(padded,)))
            self.kernel(((n+255)//256,), (256,),
                        (self.source, *fields, cross, np.int64(n), np.int32(side), np.int32(stride)))
            if pair == 2:
                self.source.get(out=np.ctypeslib.as_array(target, shape=(padded,)))
                self.diagonal.clear()
                self.source = None
            else:
                cp.cuda.get_current_stream().synchronize()
            self.phase_seconds.append(time.perf_counter()-begin)
            if pair == 2:
                self.seconds.append(sum(self.phase_seconds[-3:]))
            return 0
        except Exception:
            traceback.print_exc()
            return 1
