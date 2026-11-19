"""Keep the recombination rate and natural spline tables resident on CUDA."""
import ctypes as C
from pathlib import Path
import time
import traceback

import cupy as cp
import numpy as np

D=C.POINTER(C.c_double)
Callback=C.CFUNCTYPE(C.c_int,C.c_int,C.c_int,D,D,C.c_double,C.c_double,C.c_int,D)


class MHRTable:
    def __init__(self,native):
        code=Path(__file__).with_name("mhr.cu").read_text()
        module=cp.RawModule(code=code,options=("--fmad=false",))
        self.rates=module.get_function("mhr_rates")
        self.deriv=module.get_function("natural_derivatives")
        self.normalize=module.get_function("mhr_pdf_normalization")
        self.apply_norm=module.get_function("mhr_pdf_interpolate")
        self.pdf_rows=0
        self.pdf_quad=tuple(cp.asarray(x) for x in np.polynomial.legendre.leggauss(32))
        self.gl=tuple(cp.asarray(x) for x in np.polynomial.legendre.leggauss(128))
        self.knots=None
        self.calls=0
        self.seconds=0.
        self.lib=C.CDLL(native.__file__)
        self.callback=Callback(self.compute)
        set_hook=self.lib.SetGPUMHRTable
        set_hook.argtypes=[C.c_void_p]
        set_hook.restype=None
        set_hook(C.cast(self.callback,C.c_void_p))

    def compute(self,nz,ng,params,gamma,alpha,helium,na,pdf):
        try:
            start=time.perf_counter()
            x=cp.asarray(np.ctypeslib.as_array(gamma,shape=(ng,)))
            r=cp.asarray(np.ctypeslib.as_array(params,shape=(nz,5)))
            pdf=cp.asarray(np.ctypeslib.as_array(pdf,shape=(na,3)))
            az=cp.ascontiguousarray(pdf[:,0])
            norm=cp.empty(na,dtype=cp.float64)
            slopes=cp.empty_like(norm)
            self.normalize((na,),(128,),(pdf,*self.pdf_quad,norm))
            self.deriv((1,),(32,),(az,norm,np.int32(1),np.int32(na),slopes))
            self.apply_norm(((nz+63)//64,),(64,),(r,az,norm,slopes,np.int32(nz),np.int32(na)))
            self.pdf_rows+=na
            vals=cp.empty((nz,ng),dtype=cp.float64)
            deriv=cp.empty_like(vals)
            self.rates((nz*ng,),(64,),(r,x,*self.gl,np.int32(nz),np.int32(ng),np.int32(128),
                                      np.float64(alpha),np.float64(helium),vals))
            self.deriv(((nz+31)//32,),(32,),(x,vals,np.int32(nz),np.int32(ng),deriv))
            if not bool(cp.isfinite(vals).all() & cp.isfinite(deriv).all()):
                raise FloatingPointError("MHR GPU table is nonfinite")
            self.knots=[x,vals,deriv]
            self.calls+=1
            self.seconds+=time.perf_counter()-start
            return 0
        except Exception:
            traceback.print_exc()
            return 1
