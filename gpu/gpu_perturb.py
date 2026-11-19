"""CUDA CIC perturbation for the fixed cubic, low-resolution 2LPT path."""
import ctypes as C
from pathlib import Path
import time
import traceback

import cupy as cp
import numpy as np

F = C.POINTER(C.c_float)
Hook = C.CFUNCTYPE(C.c_int,C.c_float,C.c_int,C.c_int,C.c_double,
                  C.c_float,C.c_float,C.c_float,C.c_float,C.POINTER(F),F,F)

class PerturbRuntime:
    def __init__(self, native, shadow=True):
        self.module = cp.RawModule(
            code=Path(__file__).with_name("perturb.cu").read_text(), options=("--fmad=false",))
        self.scaler = self.module.get_function("scale_displacements")
        self.depositer = self.module.get_function("cic")
        self.normalizer = self.module.get_function("density_contrast")
        self.velocity_op = self.module.get_function("velocity_spectrum")
        self.shadow = shadow
        self.previous = None
        self.first_upload_seconds = None
        self.cached_ic_shape = None
        self.host_ic_uploads = 0
        self.host_ic_upload_bytes = 0
        self.current = None
        self.error = None
        self.reference_mass = None
        self.lib = C.CDLL(native.__file__)
        self.set_hook = self.lib.SetPerturbGPUHook
        self.set_hook.argtypes = [C.c_void_p]
        self.set_hook.restype = None
        self.callback = Hook(self.run)

    def load_cpu_ic(self, box):
        """Bind one reloaded CPU initial-condition box to this run."""
        if self.ic is not None or self.cached_ic_shape is not None:
            raise RuntimeError("This runtime already has an IC source")
        fine,coarse=box.user_params.DIM,box.user_params.HII_DIM
        begin=time.perf_counter()
        density=box.hires_density
        components=[getattr(box,"lowres_v"+axis+suffix)
                    for suffix in ("","_2LPT") for axis in "xyz"]
        if density.shape!=(fine,)*3 or any(a.shape!=(coarse,)*3 for a in components):
            raise ValueError("Unexpected CPU IC box shape")
        self.high=cp.asarray(density.reshape(-1))
        self.displacements=cp.asarray(np.stack([a.reshape(-1) for a in components]))
        cp.cuda.get_current_stream().synchronize()
        self.first_upload_seconds=time.perf_counter()-begin
        self.host_ic_uploads=1
        self.host_ic_upload_bytes=self.high.nbytes+self.displacements.nbytes
        self.cached_ic_shape=(fine,coarse)

    def install(self):
        self.set_hook(C.cast(self.callback,C.c_void_p))

    def disable(self):
        self.set_hook(None)

    @staticmethod
    def launch(kernel,size,args):
        kernel(((size+255)//256,),(256,),args)

    def run(self,z,fine,coarse,dk,first,second,initial_growth,dlog_growth_dt,inputs,out_density,out_velocity):
        try:
            self.compute(z,fine,coarse,dk,first,second,initial_growth,dlog_growth_dt,inputs,out_density,out_velocity)
            return 2 if self.shadow else 0
        except Exception:
            self.error = traceback.format_exc()
            print(self.error,flush=True)
            return 1

    def compute(self,z,fine,coarse,dk,first,second,initial_growth,dlog_growth_dt,inputs,out_density,out_velocity):
        n = coarse**3
        shape = (coarse,)*3
        stream = cp.cuda.Stream.null
        start = time.perf_counter()
        key = (fine,coarse,*(C.cast(inputs[i],C.c_void_p).value for i in range(7)))
        workspace_key=(fine,coarse) if self.ic is None else key
        if self.previous != workspace_key:
            high = np.ctypeslib.as_array(inputs[0],shape=(fine**3,))
            if self.ic is not None:
                self.high, self.displacements = self.ic.fields[key[2]]
            elif self.cached_ic_shape != (fine,coarse):
                raise ValueError("CPU IC must be uploaded before perturbation")
            self.mass = cp.empty(n,dtype=cp.float64)
            self.contrast = cp.empty(shape,dtype=cp.float32)
            self.velocity_buffer = cp.empty((coarse,coarse,coarse//2+1),dtype=cp.complex64)
            stream.synchronize()
            self.previous = workspace_key
            if self.shadow:
                # The multiplication is FP32 in the reference before adding one.
                weighted = 0.
                for chunk in np.array_split(high,32):
                    weighted += np.sum(np.float32(initial_growth)*chunk,dtype=np.float64)
                self.reference_mass = fine**3 + weighted

        start = time.perf_counter()
        first,second = np.float32(first),np.float32(second)
        if first == 0 or second == 0:
            raise ValueError("Degenerate growth factor in fixed 2LPT workflow")
        scale_args = (self.displacements,np.int64(n),first,second)
        self.launch(self.scaler,6*n,(*scale_args,np.int32(0)))
        self.mass.fill(0)
        self.launch(self.depositer,fine**3,(self.high,self.displacements,self.mass,
                                           np.int32(fine),np.int32(coarse),np.float32(initial_growth)))
        self.launch(self.scaler,6*n,(*scale_args,np.int32(1)))
        self.launch(self.normalizer,n,(self.mass,self.contrast,np.int64(n),np.float32((fine//coarse)**3)))
        stream.synchronize()
        cic_seconds = time.perf_counter()-start
        start = time.perf_counter()
        spectrum = cp.fft.rfftn(self.contrast)
        self.launch(self.velocity_op,spectrum.size,(spectrum,self.velocity_buffer,
                                                  np.int32(coarse),np.float64(dk),np.float32(dlog_growth_dt)))
        self.density = cp.fft.irfftn(spectrum,s=shape,norm="forward")
        self.density *= np.float32(1./n)
        self.density = cp.where(self.density < -1.0, np.float32(-1.+1e-7), self.density)
        self.velocity_z = cp.fft.irfftn(self.velocity_buffer,s=shape,norm="forward")
        stream.synchronize()
        fft_seconds = time.perf_counter()-start
        start = time.perf_counter()
        if self.shadow:
            density = self.density.get()
            velocity = self.velocity_z.get()
        else:
            density = np.ctypeslib.as_array(out_density,shape=(n,)).reshape(shape)
            velocity = np.ctypeslib.as_array(out_velocity,shape=(n,)).reshape(shape)
            self.density.get(out=density)
            self.velocity_z.get(out=velocity)
        download_seconds = time.perf_counter()-start
        self.current = {
            "z": z, "density": density, "velocity_z": velocity,
            "cic_seconds": cic_seconds, "fft_seconds": fft_seconds,
            "download_seconds": download_seconds,
            "mass_relative_error": (abs(float(self.mass.sum(dtype=cp.float64).get())-self.reference_mass)
                                    / abs(self.reference_mass)) if self.shadow else None,
        }
