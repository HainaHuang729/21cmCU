"""Build all X-ray frequency tables for a redshift in one CUDA callback."""
import ctypes as C
import os
from pathlib import Path
import time
import traceback

import cupy as cp
import numpy as np

F=C.POINTER(C.c_float)
D=C.POINTER(C.c_double)
DD=C.POINTER(D)
Hook=C.CFUNCTYPE(C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,D,F,D,F,F,F,F,F,F,F,DD,DD,DD,C.c_int,D)


class XRayTables:
    def __init__(self,native):
        code=Path(__file__).with_name("xray.cu").read_text() + "\n" + Path(__file__).with_name("xray_quadrature.cu").read_text()
        module=cp.RawModule(code=code,options=("--fmad=false",))
        self.tau=module.get_function("find_tau_energy")
        self.rates=module.get_function("xray_rate")
        self.adaptive=module.get_function("xray_frequency_adaptive")
        self.adaptive_coop=module.get_function("xray_frequency_adaptive_coop")
        self.cutoff_cache_key=None
        self.cutoff_cache={}
        self.adaptive_inputs=None
        self.adaptive_interval_max=0
        self.adaptive_entries=0
        self.adaptive_unique_entries=0
        self.adaptive_fallbacks=0
        self.gl=tuple(cp.asarray(x) for x in np.polynomial.legendre.leggauss(24))
        self.lib=C.CDLL(native.__file__)
        self.callback=Hook(self.fill)
        set_hook=self.lib.SetGPUXRayTables
        set_hook.argtypes=[C.c_void_p]
        set_hook.restype=None
        set_hook(C.cast(self.callback,C.c_void_p))
        set_parity=self.lib.SetGPUXRayParity
        set_parity.argtypes=[C.c_int]
        set_parity.restype=None
        set_parity(int(os.environ.get("PS_XRAY_NATIVE_MODE", "0")))
        self.calls=0
        self.seconds=0.
        self.freq_key=None
        self.cutoffs_above_max=0

    @staticmethod
    def arr(ptr,n):
        return np.ctypeslib.as_array(ptr,shape=(n,))

    def fill(self,nr,ni,ne,nz,params,radii,fcoll,energy,ion,heat,lya,hi,hei,heii,heat_out,ion_out,lya_out,parity_mode,cutoffs):
        try:
            start=time.perf_counter()
            p=self.arr(params,19)
            dparam=cp.asarray(p)
            if parity_mode in (1, 4, 5):
                lower=cp.asarray(self.arr(cutoffs,nr))/p[17]
            else:
                lower=cp.empty(nr,dtype=cp.float64)
                self.tau(((nr+63)//64,),(64,),(dparam,cp.asarray(self.arr(radii,nr)),
                          cp.asarray(self.arr(fcoll,nz)),np.int32(nz),*self.gl,np.int32(nr),np.int32(24),lower))
            if parity_mode == 2:
                self.arr(cutoffs,nr)[:]=(lower*p[17]).get()
                self.calls+=1
                self.seconds+=time.perf_counter()-start
                return 0
            if parity_mode in (3, 4, 5):
                lower_hz = (lower * p[17]).get() if parity_mode == 3 else self.arr(cutoffs, nr)
                cache_key = tuple(float(x) for x in p[8:18])
                if cache_key != self.cutoff_cache_key:
                    self.cutoff_cache.clear()
                    self.cutoff_cache_key = cache_key
                unique_hz, indices = np.unique(lower_hz, return_inverse=True)
                unique_keys = [float(hz) for hz in unique_hz]
                missing_idx = [idx for idx, k in enumerate(unique_keys) if k not in self.cutoff_cache]
                if missing_idx:
                    missing_hz = np.array([unique_keys[idx] for idx in missing_idx], dtype=np.float64)
                    miss_n = missing_hz.size
                    device_cutoffs = cp.asarray(missing_hz)
                    if self.adaptive_inputs is None:
                        self.adaptive_inputs = (
                            cp.asarray(self.arr(energy, ne)),
                            cp.asarray(self.arr(ion, ni)),
                            cp.asarray(np.stack([self.arr(ptr, ni * ne) for ptr in (heat, lya, hi, hei, heii)])),
                        )
                    result = cp.empty((3, ni, miss_n), dtype=cp.float64)
                    interval_counts = cp.empty(result.shape, dtype=cp.uint8)
                    self.adaptive_coop((3 * ni * miss_n,), (16,), (
                        dparam, device_cutoffs, *self.adaptive_inputs, np.int32(miss_n), np.int32(ni), np.int32(ne),
                        result, interval_counts,
                    ))
                    miss_host = result.get()
                    miss_counts = interval_counts.get()
                    self.adaptive_interval_max = max(self.adaptive_interval_max, int(miss_counts.max()))
                    self.adaptive_unique_entries += result.size
                    invalid = (miss_counts == 0) | ~np.isfinite(miss_host)
                    if invalid.any():
                        miss_host[invalid] = np.nan
                        self.adaptive_fallbacks += int(invalid.sum())
                    for local_j, idx in enumerate(missing_idx):
                        self.cutoff_cache[unique_keys[idx]] = miss_host[:, :, local_j].copy()
                unique_table = np.stack([self.cutoff_cache[k] for k in unique_keys], axis=2)
                host = unique_table[:, :, indices]
                host[2] *= p[18]
                self.adaptive_entries += host.size
                if parity_mode == 3:
                    self.arr(cutoffs, nr)[:] = lower_hz
                for output, table in zip((heat_out, ion_out, lya_out), host):
                    for i, row in enumerate(table):
                        self.arr(output[i], nr)[:] = row
                self.calls += 1
                self.seconds += time.perf_counter() - start
                return 0
            key=tuple(p[8:18])
            if key!=self.freq_key:
                n=8193
                grid=cp.exp(cp.linspace(np.log(p[14]),np.log(p[15]),n))
                data=cp.asarray(np.stack([self.arr(f,ni*ne) for f in (heat,lya,hi,hei,heii)]))
                value=cp.empty((3,ni,n),dtype=cp.float64)
                self.rates(((ni*n+255)//256,),(256,),(dparam,grid,cp.asarray(self.arr(energy,ne)),
                           cp.asarray(self.arr(ion,ni)),data,np.int32(n),np.int32(ne),np.int32(ni),value))
                segments=0.5*(value[:,:,1:]+value[:,:,:-1])*cp.diff(grid)
                tail=cp.cumsum(segments[:,:,::-1],axis=2)[:,:,::-1]
                self.freq=cp.concatenate((tail,cp.zeros((3,ni,1),dtype=cp.float64)),axis=2)
                self.grid=grid
                self.freq_key=key
            ind=cp.clip(cp.searchsorted(self.grid,lower)-1,0,self.grid.size-2)
            fraction=cp.clip((lower-self.grid[ind])/(self.grid[ind+1]-self.grid[ind]),0.,1.)
            result=self.freq[:,:,ind]*(1.-fraction)+self.freq[:,:,ind+1]*fraction
            result[2]*=p[18]
            host=result.get()
            if not np.isfinite(host).all():
                raise FloatingPointError("Nonfinite X-ray table")
            for output,table in zip((heat_out,ion_out,lya_out),host):
                for i,row in enumerate(table): self.arr(output[i],nr)[:]=row
            self.calls+=1
            self.seconds+=time.perf_counter()-start
            return 0
        except Exception:
            traceback.print_exc()
            return 1
