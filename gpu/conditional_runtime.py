"""Execute conditional-collapse quadrature for the heat and ion lookup tables."""
import ctypes as C
import os
from pathlib import Path
import time
import traceback

import cupy as cp
import numpy as np

F=C.POINTER(C.c_float)
D=C.POINTER(C.c_double)
Hook=C.CFUNCTYPE(C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,D,F,F,F,F,F,F,F,F)

class ConditionalTables:
    def __init__(self,native):
        self.lib=C.CDLL(native.__file__)
        self.kernel=cp.RawKernel(Path(__file__).with_name("conditional.cu").read_text(),
                                 "conditional_table",options=("--fmad=false",))
        self.gl=tuple(cp.asarray(a,dtype=cp.float64) for a in np.polynomial.legendre.leggauss(16))
        self.low_gl=tuple(cp.asarray(a,dtype=cp.float64) for a in np.polynomial.legendre.leggauss(100))
        self.callback=Hook(self.fill)
        setter=self.lib.SetGPUConditionalTable
        setter.argtypes=[C.c_void_p]
        setter.restype=None
        setter(C.cast(self.callback,C.c_void_p))
        self.calls=[0,0]
        mode="0" if os.environ.get("PS_NATIVE_CONDITIONAL") == "1" else os.environ.get("PS_CONDITIONAL_MODE","3")
        self.gpu_modes=int(mode)
        set_mode=self.lib.SetGPUConditionalMode
        set_mode.argtypes=[C.c_int]
        set_mode.restype=None
        set_mode(self.gpu_modes)
        self.device_rows={}
        self.device_reads=[0,0]
        self.seconds=[0.,0.]

    @staticmethod
    def upload(ptr,n):
        return cp.asarray(np.ctypeslib.as_array(ptr,shape=(n,)))

    def row(self,low,high,heat,nlow=None,nhigh=None):
        if not (self.gpu_modes & (1 << heat)):
            return self.upload(low,nlow),self.upload(high,nhigh)
        key=tuple(C.cast(ptr,C.c_void_p).value for ptr in (low,high))
        rows=self.device_rows.pop(key)
        self.device_reads[heat]+=1
        return rows

    def fill(self,heat,nlow,nhigh,ns,nq,p,sigma,dsigma,low,high,x,w,out_low,out_high):
        try:
            start=time.perf_counter()
            shared=tuple(self.upload(ptr,n) for ptr,n in ((p,15),(sigma,ns),(dsigma,ns)))
            if nq != self.low_gl[0].size:
                raise ValueError('Unexpected conditional quadrature order')
            points=self.low_gl
            rows=[]
            for hi,n,grid in ((0,nlow,low),(1,nhigh,high)):
                result=cp.empty(n,dtype=cp.float32)
                self.kernel((n,),(128,),(*shared,self.upload(grid,n),*points,*self.gl,result,
                                         np.int32(ns),np.int32(nq),np.int32(hi),np.int32(heat)))
                rows.append(result)
            key=tuple(C.cast(ptr,C.c_void_p).value for ptr in (out_low,out_high))
            self.device_rows[key]=tuple(rows)
            self.calls[heat]+=1
            self.seconds[heat]+=time.perf_counter()-start
            return 0
        except Exception:
            traceback.print_exc()
            return 1
