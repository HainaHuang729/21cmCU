"""Resolve only branch-sensitive filtered cells and request CPU solver replay when required."""
import ctypes as C
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from gpu_heat_runtime import F, arr
from unguarded_resident import ResidentRuntime as ResidentBase

LongPtr = C.POINTER(C.c_long)
COLLAPSE_LIMIT = np.float64(.99*1.68)
COLLAPSE_GUARD = np.float64(2.5e-6)

class ResidentRuntime(ResidentBase):
    def __init__(self):
        super().__init__()
        self.candidates = cp.RawKernel(
            Path(__file__).with_name("guard_runtime.cu").read_text(), "collapse_candidates",
            options=("--fmad=false",),
        )
        self.candidate_cells = cp.empty(8192,dtype=cp.int64)
        self.candidate_count = cp.empty(1,dtype=cp.uint32)
        self.collapse_checks = []
        self.cpu_replays = []

    def install(self,native):
        super().install(native)
        self.resolve = self.lib.ResolveCollapseDensity
        self.resolve.argtypes = [F,C.c_float,LongPtr,C.c_long,F]
        self.resolve.restype = C.c_int

    def compute_filter(self,side,count,dk,inverse_growth,low,density,radii,minima,maxima):
        super().compute_filter(side,count,dk,inverse_growth,low,density,radii,minima,maxima)
        self.raw_density = arr(density,side**3).copy()
        self.inverse_growth = inverse_growth

    def correct_collapse(self,a):
        if a.density:
            return
        start = time.perf_counter()
        growth = arr(a.growth,a.nr)
        growth_device = self.upload(growth)
        self.candidate_count.fill(0)
        self.candidates(
            ((a.n*a.nr+255)//256,), (256,),
            (self.density,growth_device,np.int64(a.n),np.int32(a.nr),
             COLLAPSE_LIMIT,COLLAPSE_GUARD,self.candidate_count,self.candidate_cells,
             np.uint32(self.candidate_cells.size)),
        )
        count = int(self.candidate_count.get()[0])
        if count == 0:
            return
        if count > self.candidate_cells.size:
            raise RuntimeError(f"Collapse candidate buffer exceeded: {count}")
        cells = np.sort(self.candidate_cells[:count].get())
        precise = np.empty(count,dtype=np.float32)
        code = self.resolve(
            self.raw_density.ctypes.data_as(F), self.inverse_growth,
            cells.ctypes.data_as(LongPtr), count, precise.ctypes.data_as(F),
        )
        if code:
            raise RuntimeError(f"FFTW collapse density replay failed: {code}")
        flat_density = self.density.reshape(-1)
        device_indices = self.upload(cells)
        old = flat_density[device_indices].get()
        flat_density[device_indices] = self.upload(precise)
        scale_growth = growth[cells//a.n]
        changed = (old*scale_growth).astype(np.float64) < COLLAPSE_LIMIT
        changed ^= (precise*scale_growth).astype(np.float64) < COLLAPSE_LIMIT
        if changed.any():
            print("COLLAPSE_REPLAY", a.scalars[0], cells[changed].tolist(),
                  (old*scale_growth)[changed].tolist(),
                  (precise*scale_growth)[changed].tolist(), flush=True)
        self.download_bytes += cells.nbytes + old.nbytes + self.candidate_count.nbytes
        self.collapse_checks.append({
            "z": a.scalars[0], "cells": count, "scales": int(np.unique(cells//a.n).size),
            "branches_changed": int(changed.sum()), "seconds": time.perf_counter()-start,
        })

    def compute(self,a):
        if not a.init & 2:
            self.correct_collapse(a)
        result = super().compute(a)
        if result == 2:
            self.cpu_replays.append(a.scalars[0])
            print("TS_CPU_REPLAY", a.scalars[0], flush=True)
        return result

    def report(self):
        return {**super().report(), "collapse_checks": self.collapse_checks,
                "ts_cpu_replays": self.cpu_replays}
