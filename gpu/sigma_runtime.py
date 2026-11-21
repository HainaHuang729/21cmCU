"""Compute power spectrum moments in batched CUDA quadrature."""
import ctypes as C
from pathlib import Path
import time
import traceback

import cupy as cp
import numpy as np

D=C.POINTER(C.c_double)
Hook=C.CFUNCTYPE(C.c_int,C.c_int,D,D,D,D,D)


class SigmaMoments:
    def __init__(self,native):
        self.lib=C.CDLL(native.__file__)
        self.kernel=cp.RawKernel(Path(__file__).with_name("sigma.cu").read_text(),
                                 "sigma_moments",options=("--fmad=false",))
        self.nodes=tuple(cp.asarray(v) for v in np.polynomial.legendre.leggauss(16))
        self.calls=0
        self.launches=0
        self.seconds=0.
        self.cache={}
        self.cb=Hook(self.compute)
        setter=self.lib.SetGPUSigmaMoments
        setter.argtypes=[C.c_void_p]
        setter.restype=None
        setter(C.cast(self.cb,C.c_void_p))

    def compute(self,n,p,r,m,out,derivative):
        try:
            start=time.perf_counter()
            params=np.ctypeslib.as_array(p,shape=(9,))
            radius=np.ctypeslib.as_array(r,shape=(n,))
            mass=np.ctypeslib.as_array(m,shape=(n,))
            key=(params.tobytes(),radius.tobytes(),mass.tobytes())
            v=self.cache.get(key)
            if v is None:
                g,d=(cp.empty(n,dtype=cp.float64) for _ in range(2))
                self.kernel((n,),(128,),(cp.asarray(params),cp.asarray(radius),cp.asarray(mass),
                                         *self.nodes,np.int32(n),g,d))
                v=(g.get(),d.get())
                if not all(np.isfinite(x).all() for x in v):
                    raise FloatingPointError("Nonfinite sigma moments")
                self.cache[key]=v
                self.launches+=1
            np.ctypeslib.as_array(out,shape=(n,))[:]=v[0]
            np.ctypeslib.as_array(derivative,shape=(n,))[:]=v[1]
            self.calls+=1
            self.seconds+=time.perf_counter()-start
            return 0
        except Exception:
            traceback.print_exc()
            return 1
