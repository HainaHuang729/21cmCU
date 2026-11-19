"""GPU excursion-set filtering, cell ionization, and recombination history."""
import ctypes as C
import os
import os
from pathlib import Path
import time
import traceback

import cupy as cp
import numpy as np
from ion_poisson import IonPoisson
from gpu_heat_runtime import F, arr

class IonArgs(C.Structure):
    _fields_ = (
        [(k,C.c_int) for k in "phase side threads filter filtered last no_rng nl nh".split()]
        + [(k,C.c_float) for k in "z prev_z radius lo hi ls li hs hi_inv crit pixel mmin eff tre dt dz density_limit".split()]
        + [(k,C.c_double) for k in "dk cut npoisson norm gamma fmin mean dz_table max_log".split()]
        + [(k,F) for k in "density xe tk old_rec old_z xhi nrec zre mfp gamma12 temperature fcoll low high".split()]
    )

Callback = C.CFUNCTYPE(C.c_int,C.POINTER(IonArgs))
DoublePtr = C.POINTER(C.c_double)

def address(ptr):
    return C.cast(ptr,C.c_void_p).value

TRACE_COORDS = [(23,5,19), (2,6,123), (8,59,88), (115,74,46), (24,49,72), (127,35,44), (121,7,45)]

class IonRuntime:
    def __init__(self, thermal):
        self.thermal = thermal
        trace_file = os.environ.get("PS_GPU_SCALE_TRACE_FILE")
        self.trajectory = open(trace_file, "w", buffering=1) if trace_file else None

        self.history = {}
        self.counts = [0]*6
        self.neutral_fields = {}
        self.seconds = [0.]*6
        self.reference_seconds=[0.]*4
        self.reference_passes=[0,0]
        self.host_recomb_density_uploads = 0
        self.module = cp.RawModule(code=Path(__file__).with_name("ion.cu").read_text(),
                                  options=("--fmad=false",))
        self.kernels = [self.module.get_function(k) for k in
                        ("bubble_filter","collapse","assign_ion","recombine","rec_boundary","spline_candidates","ion_candidates","ion_temperature")]
        self.batched = os.environ.get("PS_BATCH_FILTER") == "1"
        self.filter_triplet = self.module.get_function("bubble_filter_triplet")
        native = thermal.lib
        self.poisson_generator = IonPoisson()
        self.poisson_gpu_used = False
        self.clamps = cp.zeros(1,dtype=cp.uint32)
        self._callback = Callback(self.run)
        native.SetIonGPUHook.argtypes = [C.c_void_p]
        native.SetIonGPUHook.restype = None
        native.SetIonGPUHook(C.cast(self._callback,C.c_void_p))

    @staticmethod
    def launch(kernel, n, *args):
        kernel(((n+255)//256,), (256,), args)

    def run(self, ptr):
        try:
            a = ptr.contents
            start = time.perf_counter()
            (self.begin,self.filter,self.lookup,self.apply,self.finish,self.early)[a.phase](a)
            self.counts[a.phase] += 1
            self.seconds[a.phase] += time.perf_counter()-start
            return 0
        except Exception:
            self.thermal.error = traceback.format_exc()
            print(self.thermal.error,flush=True)
            return 1

    def early(self,a):
        n=a.side**3
        matching=[state for (ptr,_),state in self.thermal.states.items() if ptr==address(a.xe)]
        if matching:
            xe,tk,_=matching[-1]
        else:
            xe=cp.asarray(arr(a.xe,n))
            tk=cp.asarray(arr(a.tk,n))
        neutral=cp.float32(1.)-xe
        neutral.get(out=arr(a.xhi,n))
        tk.get(out=arr(a.temperature,n))
        self.neutral_fields[address(a.xhi)]=neutral

    def begin(self,a):
        self.shape = (a.side,)*3
        self.n = a.side**3
        if self.trajectory is not None:
            self.trajectory_indices = cp.asarray([np.ravel_multi_index(x, self.shape) for x in TRACE_COORDS])

        self.density = self.thermal.perturbation.fields[address(a.density)].reshape(-1)
        matching = [state for (ptr,z),state in self.thermal.states.items() if ptr==address(a.xe)]
        if not matching:
            raise RuntimeError(f"Missing GPU thermal field at z={a.z}")
        xe, self.tk, _ = matching[-1]
        prior = self.history.get((address(a.old_rec),a.prev_z))
        if prior is None:
            host = arr(a.old_rec,self.n)
            if np.any(host):
                raise RuntimeError(f"Missing GPU recombination history at z={a.prev_z}")
            prior = (cp.zeros(self.n,dtype=cp.float32),cp.asarray(arr(a.old_z,self.n)))
        self.prev_rec,self.prev_z = prior
        input_fields = tuple(f.reshape(self.shape) for f in (self.density,xe,self.prev_rec))
        if self.batched:
            self.spectra = cp.fft.rfftn(cp.stack(input_fields), axes=(-3,-2,-1))
            self.workspace = cp.empty_like(self.spectra)
        else:
            self.spectra = [cp.fft.rfftn(f) for f in input_fields]
            self.workspace = cp.empty_like(self.spectra[0])
        self.xhi = cp.ones(self.n,dtype=cp.float32)
        self.zre = cp.full(self.n,-1.,dtype=cp.float32)
        self.gamma = cp.zeros_like(self.xhi)
        self.mfp = cp.zeros_like(self.xhi)
        self.temp = cp.zeros_like(self.xhi)
        self.fc = cp.empty_like(self.xhi)
        self.bad = cp.zeros(1,dtype=cp.int32)
        self.flags = cp.empty(self.n,dtype=cp.uint8)
        if not hasattr(self,"knots"):
            self.knots = self.thermal.mhr.knots
            if self.knots is None:
                double_pointer = C.POINTER(C.c_double)
                export_knots = self.thermal.lib.IonRecombinationKnots
                export_knots.argtypes = [double_pointer] * 3
                export_knots.restype = None
                arrays = [np.empty(shape, dtype=np.float64) for shape in (250, (300,250), (300,250))]
                export_knots(*(array.ctypes.data_as(double_pointer) for array in arrays))
                self.knots = [cp.asarray(array) for array in arrays]
            if a.no_rng:
                self.poisson=cp.ones(self.n,dtype=cp.int32)
            else:
                self.poisson=self.poisson_generator.draw(self.n,a.threads,a.npoisson)
                self.poisson_gpu_used=True

    def filter(self,a):
        args = (np.int32(a.side),np.float64(a.dk),np.float32(a.radius),np.int32(a.filtered))
        if self.batched:
            self.launch(self.filter_triplet,self.workspace.size//3,self.spectra,self.workspace,*args)
            fields = cp.fft.irfftn(self.workspace,s=self.shape,axes=(-3,-2,-1),
                                 norm="forward").reshape(3,self.n)
        else:
            fields = []
            for field in self.spectra:
                self.launch(self.kernels[0],field.size,field,self.workspace,*args)
                fields.append(cp.fft.irfftn(self.workspace,s=self.shape,norm="forward").reshape(-1))
        self.delta = cp.maximum(fields[0],np.float32(-1.+1e-7))
        self.xe = cp.clip(fields[1],0.,np.float32(.999))
        self.rec = cp.maximum(fields[2],0.)
        limits=cp.stack((self.delta.min(),self.delta.max())).get()
        a.lo=min(0.,float(limits[0]))
        a.hi=max(0.,float(limits[1]))

    def lookup(self,a):
        low,high=self.thermal.tables.row(a.low,a.high,0,a.nl,a.nh)
        self.bad.fill(0)
        self.launch(self.kernels[1],self.n,self.delta,low,high,self.fc,np.int32(self.n),
                    np.int32(a.nl),np.int32(a.nh),*(np.float32(x) for x in
                    (a.ls,a.li,a.hs,a.hi_inv,a.crit)),np.float64(a.cut),self.clamps)
        self.collapse_sum=cp.sum(self.fc,dtype=cp.float64)

    def apply(self,a):
        if self.trajectory is not None:
            ids = self.trajectory_indices
            vals = cp.stack([f[ids] for f in (self.delta,self.xe,self.rec,self.fc,self.xhi,self.gamma)]).get()
            norm = a.norm / max(float(self.collapse_sum.get()) / self.n, a.fmin)
            for i, (d,xe,rec,fc,xhi,gamma) in enumerate(vals.T):
                row=(a.z,i,a.radius,int(a.last),d,xe,rec,fc,norm,a.eff,a.fmin,xhi,gamma,a.lo,a.hi,a.gamma)
                self.trajectory.write("scale "+" ".join(format(float(x),".17g") for x in row)+"\n")

        self.launch(self.kernels[2],self.n,self.delta,self.xe,self.rec,self.fc,self.prev_z,
                    self.tk,self.poisson,self.xhi,self.zre,self.mfp,self.gamma,self.temp,
                    np.int32(self.n),np.int32(a.last),*(np.float32(x) for x in
                    (a.radius,a.z,a.pixel,a.mmin,a.eff,a.tre)),*(np.float64(x) for x in
                    (a.norm,a.gamma,a.fmin,a.npoisson)),self.collapse_sum)

    def finish(self,a):
        self.launch(self.kernels[7],self.n,self.density,self.xhi,self.zre,self.tk,self.temp,
                    np.int32(self.n),np.float32(a.z),np.float32(a.tre),np.float32(a.density_limit))
        if not bool(cp.isfinite(self.temp).all()):
            raise ValueError(f"Nonfinite ionized gas temperature at z={a.z}")
        out = cp.empty_like(self.xhi)
        self.launch(self.kernels[3],self.n,self.density,self.gamma,self.xhi,
                    self.prev_rec,out,*self.knots,np.int32(self.n),np.float32(a.z),
                    np.float32(a.dt),np.float32(a.dz),np.float64(a.dz_table),np.float64(a.max_log))
        if not bool(cp.isfinite(out).all()):
            raise ValueError(f"Nonfinite recombination at z={a.z}")
        for array, ptr in ((self.xhi,a.xhi),(self.zre,a.zre),(self.mfp,a.mfp),
                           (self.gamma,a.gamma12),(out,a.nrec),(self.temp,a.temperature),(self.fc,a.fcoll)):
            array.get(out=arr(ptr,self.n))
        self.neutral_fields[address(a.xhi)]=self.xhi
        self.history[(address(a.nrec),a.z)] = (out,self.zre)
        if self.counts[4] % 10 == 0:
            print(f"GPU_ION z={a.z} active_steps={self.counts[4]+1}",flush=True)

    def report(self):
        return {"phase_counts":self.counts,"poisson_gpu_used":self.poisson_gpu_used,
                "phase_seconds":self.seconds,"table_endpoint_clamps":int(self.clamps.get()[0]),
                "cpu_fft_replays":0,"cpu_recombination_row_replays":0}
