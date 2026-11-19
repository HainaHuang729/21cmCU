"""Resident 21-cm brightness and redshift-space distortion for the fixed branch."""
import ctypes as C
from pathlib import Path
import time
import traceback

import cupy as cp
import numpy as np
from gpu_heat_runtime import F, arr
from ion_runtime import address

BrightHook = C.CFUNCTYPE(C.c_int,C.c_int,C.c_int,C.c_float,C.c_double,C.c_float,
                         C.c_float,C.c_float,C.c_double,F,F,F,F,F)

class BrightnessRuntime:
    def __init__(self, thermal):
        self.thermal = thermal
        module = cp.RawModule(code=Path(__file__).with_name("brightness.cu").read_text(),
                              options=("--fmad=false",))
        self.derivative = module.get_function("velocity_gradient")
        self.cell = module.get_function("brightness")
        self.fields = {}
        self.seconds = []
        self.seed_uploads = 0
        self.hook = BrightHook(self.compute)
        register = thermal.lib.SetBrightnessGPUHook
        register.argtypes = [C.c_void_p]
        register.restype = None
        register(C.cast(self.hook,C.c_void_p))

    def compute(self,side,rsd,z,dk,factor,trad,hubble,limit,density,velocity,neutral,ts,out):
        try:
            start=time.perf_counter()
            n=side**3
            shape=(side,)*3
            d=self.thermal.perturbation.fields[address(density)].ravel()
            v=self.thermal.perturbation.velocity_fields[address(velocity)]
            x=self.thermal.ionization.neutral_fields[address(neutral)]
            spin=self.thermal.spin_fields.get(address(ts))
            if spin is None:
                # The reference heat solver initializes pre-heating snapshots.
                spin=cp.asarray(arr(ts,n))
                self.seed_uploads+=1
            g=d
            if rsd:
                spec=cp.fft.rfftn(v.reshape(shape))
                deriv=cp.empty_like(spec)
                self.derivative(((spec.size+255)//256,),(256,),
                                (spec,deriv,np.int32(side),np.float64(dk)))
                g=cp.fft.irfftn(deriv,s=shape,norm="forward").ravel()
            result=cp.empty(n,dtype=cp.float32)
            self.cell(((n+255)//256,),(256,),
                      (d,x,spin,g,result,np.int32(n),np.float32(factor),
                       np.float32(trad),np.float32(hubble),np.float64(limit),np.int32(rsd)))
            if not bool(cp.isfinite(result).all()):
                raise ValueError(f"Nonfinite brightness at z={z}")
            self.fields[address(out)]=result.reshape(shape)
            result.get(out=arr(out,n))
            self.seconds.append(time.perf_counter()-start)
            return 0
        except Exception:
            self.thermal.error=traceback.format_exc()
            print(self.thermal.error,flush=True)
            return 1
