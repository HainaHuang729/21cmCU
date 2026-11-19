"""GSL-compatible Poisson draws for the supported mean <= 10."""
import ctypes as C
import math
from pathlib import Path
import time

import cupy as cp
import numpy as np


class IonPoisson:
    def __init__(self):
        code = Path(__file__).with_name("ion_poisson.cu").read_text()
        self.kernel = cp.RawKernel(code, "ion_poisson", options=("--fmad=false",))
        self.last_seconds = 0.

    def draw(self, count, threads, mean):
        if not 0 <= mean <= 10:
            raise ValueError(f"GPU Poisson mean outside [0,10]: {mean}")
        start = time.perf_counter()
        draws = cp.empty(count, dtype=cp.int32)
        state = cp.empty((threads,624), dtype=cp.uint32)
        self.kernel(((threads+31)//32,), (32,),
                    (draws,state,np.int64(count),np.int32(threads),np.float64(math.exp(-mean))))
        cp.cuda.get_current_stream().synchronize()
        self.last_seconds = time.perf_counter()-start
        return draws


def compare(native, count, threads, mean):
    library = C.CDLL(native.__file__)
    ref = library.IonPoissonDraws
    ref.argtypes = [C.c_int,C.c_double,C.c_ulonglong,C.POINTER(C.c_int)]
    ref.restype = None
    expected = np.empty(count,dtype=np.int32)
    start = time.perf_counter()
    ref(threads,mean,count,expected.ctypes.data_as(C.POINTER(C.c_int)))
    cpu_seconds = time.perf_counter()-start
    gpu = IonPoisson()
    got = gpu.draw(count,threads,mean).get()
    bad = np.flatnonzero(got != expected)
    return dict(count=count,threads=threads,mean=mean,failed=int(bad.size),
                first_failed=int(bad[0]) if bad.size else None,
                cpu_seconds=cpu_seconds,gpu_seconds=gpu.last_seconds)
