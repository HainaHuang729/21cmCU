"""Generate the fixed Eisenstein-Hu IC realization entirely on the device."""
import ctypes as C
from pathlib import Path
import time
import traceback

import cupy as cp
import numpy as np

from ic_runtime import ICRuntime, FloatPtr

DoublePtr=C.POINTER(C.c_double)
SeedHook=C.CFUNCTYPE(C.c_int,C.c_ulonglong,C.c_int,C.c_int,C.c_int,C.c_double,
                       C.c_double,C.c_float,DoublePtr,C.POINTER(FloatPtr))


class SeededIC(ICRuntime):
    def __init__(self,native):
        super().__init__(native)
        module=cp.RawModule(code=Path(__file__).with_name("ic_seed.cu").read_text(),
                                options=("--fmad=false",))
        self.scale_modes=module.get_function("scale_initial_modes")
        self.project_modes=module.get_function("enforce_real_planes")
        self.seed_callback=SeedHook(self.generate)
        setter=self.lib.SetICSeedGPUHook
        setter.argtypes=[C.c_void_p]
        setter.restype=None
        setter(C.cast(self.seed_callback,C.c_void_p))
        self.seed_seconds=[]

    def generate(self,seed,side,coarse,no_rng,dk,volume,radius,params,outputs):
        try:
            start=time.perf_counter()
            shape=(side,side,side//2+1,2)
            if no_rng:
                noise=cp.empty(shape,dtype=cp.float32)
                noise[...,0]=1.
                noise[...,1]=-1.
            else:
                noise=cp.random.RandomState(seed).standard_normal(shape,dtype=cp.float32)
            modes=noise.view(cp.complex64).reshape(shape[:-1])
            device_params=cp.asarray(np.ctypeslib.as_array(params,shape=(11,)))
            self.launch(self.scale_modes,modes.size,(modes,np.int32(side),np.float64(dk),
                                                       np.float64(volume),device_params))
            self.launch(self.project_modes,2*side**2,(modes,np.int32(side)))
            cp.cuda.get_current_stream().synchronize()
            self.seed_seconds.append(time.perf_counter()-start)
            self.compute_modes(side,coarse,dk,volume,radius,modes,outputs,start)
            return 0
        except Exception:
            traceback.print_exc()
            return 1
