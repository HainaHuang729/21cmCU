"""Keep the initial RECFAST fields on the GPU for later evolution."""
import ctypes as C
from pathlib import Path
import traceback

import cupy as cp
import numpy as np

from gpu_heat_runtime import F, arr
from ion_runtime import address

SeedHook = C.CFUNCTYPE(C.c_int, C.c_int, C.c_float, C.c_float, C.c_float,
                       C.c_float, C.c_float, C.c_double, C.c_double, F,F,F,F)


class HeatSeed:
    def __init__(self, parent):
        self.parent = parent
        self.kernel = cp.RawKernel(Path(__file__).with_name("heat_seed.cu").read_text(),
                                   "heat_seed", options=("--fmad=false",))
        register = parent.lib.SetGPUHeatSeedHook
        register.argtypes = [C.c_void_p]
        register.restype = None
        self.callback = SeedHook(self.run)
        self.calls = 0
        register(C.cast(self.callback,C.c_void_p))

    def run(self, side, z, growth, adiabatic, kinetic, electrons, trad, coupling,
            density, *outputs):
        try:
            n = side**3
            source = self.parent.perturbation.fields[address(density)].ravel()
            states = [cp.empty(n,dtype=cp.float32) for _ in range(3)]
            self.kernel(((n+255)//256,), (256,),
                        (source,*states,np.int64(n),np.float32(growth),np.float32(adiabatic),
                         np.float32(kinetic),np.float32(electrons),np.float64(trad),np.float64(coupling)))
            for pointer, device in zip(outputs,states):
                device.get(out=arr(pointer,n))
            self.parent.states[(address(outputs[0]),float(z))] = states
            self.parent.spin_fields[address(outputs[2])] = states[2]
            self.calls += 1
            return 0
        except Exception:
            self.parent.error = traceback.format_exc()
            print(self.parent.error,flush=True)
            return 1
