"""GPU quadrature for global Sheth-Tormen collapsed fractions."""
import ctypes as C
from table_rule import table_rule, table_source
import time
import traceback

import cupy as cp
import numpy as np

D=C.POINTER(C.c_double)
F=C.POINTER(C.c_float)
Hook=C.CFUNCTYPE(C.c_int,C.c_int,C.c_int,D,D,F,F,D)


class GlobalCollapse:
    def __init__(self,native):
        self.kernel=cp.RawKernel(table_source("hmf.cu"),
                                 "global_collapse",options=("--fmad=false","--std=c++11"))
        self.rule=table_rule()
        self.lib=C.CDLL(native.__file__)
        self.cb=Hook(self.compute)
        setter=self.lib.SetGPUGlobalCollapse
        setter.argtypes=[C.c_void_p]
        setter.restype=None
        setter(C.cast(self.cb,C.c_void_p))
        self.seconds=0.
        self.calls=0
        self.evaluations=0

    def compute(self,n,ns,p,g,s,d,out):
        try:
            start=time.perf_counter()
            arrays=[cp.asarray(np.ctypeslib.as_array(pointer,shape=(count,)))
                    for pointer,count in ((p,12),(g,n),(s,ns),(d,ns))]
            await_values=cp.empty(n,dtype=cp.float64)
            self.kernel((n,),(32,),(*arrays,self.rule,np.int32(ns),np.int32(n),await_values))
            host=await_values.get()
            if not np.isfinite(host).all():
                raise FloatingPointError("Nonfinite global collapse")
            np.ctypeslib.as_array(out,shape=(n,))[:]=host
            self.seconds+=time.perf_counter()-start
            self.calls+=1
            self.evaluations+=n
            return 0
        except Exception:
            traceback.print_exc()
            return 1
