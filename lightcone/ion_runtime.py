"""GPU excursion-set filtering, cell ionization, and recombination history."""
import ctypes as C
import os
import os
from pathlib import Path
import time
import traceback

import cupy as cp
from host_backed import LatestSnapshots, resident_or_host
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

TRACE_COORDS = [(1, 222, 135), (170, 179, 1), (79, 176, 182), (42, 158, 103), (177, 143, 247), (10, 161, 126), (5, 222, 138)]

class IonRuntime:
    def __init__(self, thermal):
        self.thermal = thermal
        trace_file = os.environ.get("PS_GPU_SCALE_TRACE_FILE")
        self.trajectory = open(trace_file, "w", buffering=1) if trace_file else None

        self.history = LatestSnapshots()
        self.counts = [0]*6
        self.neutral_fields = LatestSnapshots()
        self.seconds = [0.]*6
        self.reference_seconds=[0.]*4
        self.reference_passes=[0,0]
        self.host_recomb_density_uploads = 0
        cpu_flags = ""
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("flags"):
                    cpu_flags = f" {line} "
                    break
        except OSError:
            pass
        self.glibc_math_fma = " fma " in cpu_flags
        ion_code = Path(__file__).with_name("glibc_math.cuh").read_text() + "\n" + Path(__file__).with_name("ion.cu").read_text()
        self.module = cp.RawModule(code=ion_code,
                                  options=("--fmad=false", f"-DGLIBC_MATH_FMA={1 if self.glibc_math_fma else 0}"))
        self.kernels = [self.module.get_function(k) for k in
                        ("bubble_filter","collapse","assign_ion","recombine","rec_boundary","spline_candidates","ion_candidates","ion_temperature")]
        self.batched = os.environ.get("PS_BATCH_FILTER") == "1"
        self.filter_triplet = self.module.get_function("bubble_filter_triplet")
        self.filter_pair = self.module.get_function("bubble_filter_pair")
        self.clamp_fields = self.module.get_function("ion_clamp_fields")
        self.skip_density_fft = False
        self.margin_pass1 = self.module.get_function("min_ion_margin_pass1")
        self.margin_pass2 = self.module.get_function("min_ion_margin_pass2")
        self.margin_amp_pass1 = self.module.get_function("min_ion_margin_amp_pass1")
        self.screen_kernel = self.module.get_function("screen_ion_scale_pass1")
        self.init_arrays_kernel = self.module.get_function("ion_init_arrays")
        self.collapse_with_sum = self.module.get_function("collapse_with_sum")
        self.sum_pass2 = self.module.get_function("sum_double_pass2")
        self.margin_partials = cp.empty(256, dtype=cp.float64)
        self.margin_out = cp.empty(1, dtype=cp.float64)
        self._margin_pinned_mem = cp.cuda.alloc_pinned_memory(8)
        self.margin_pinned = np.frombuffer(self._margin_pinned_mem, dtype=np.float64, count=1)
        self.screen_flag = cp.empty(1, dtype=cp.int32)
        self.nonfinite_dev = cp.empty(1, dtype=cp.int32)
        self.max_pair_dev = cp.empty(2, dtype=cp.float32)
        self._scalar_pinned_mem = cp.cuda.alloc_pinned_memory(16)
        sc_i32 = np.frombuffer(self._scalar_pinned_mem, dtype=np.int32, count=2, offset=0)
        self.screen_flag_h = sc_i32[0:1]
        self.nonfinite_h = sc_i32[1:2]
        self.max_pair_h = np.frombuffer(self._scalar_pinned_mem, dtype=np.float32, count=2, offset=8)
        self.xe_max = 1.0
        self.rec_max = 0.0
        self.rec_is_zero = False
        self.input_stack = None
        self.rec_zeros = None
        self.batch_mode = "all3"
        self._finish_pinned_mem = None
        self.finish_pinned = None
        self.scratch_n = 0
        native = thermal.lib
        self.pmemcpy = native.PSParallelMemcpy
        self.pmemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_ulonglong, C.c_int]
        self.pmemcpy.restype = None
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
            status=(self.begin,self.filter,self.lookup,self.apply,self.finish,self.early)[a.phase](a)
            self.counts[a.phase] += 1
            self.seconds[a.phase] += time.perf_counter()-start
            return status or 0
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

        self.density = resident_or_host(self.thermal.perturbation.fields, a.density, self.n).reshape(-1)
        matching = [state for (ptr,z),state in self.thermal.states.items() if ptr==address(a.xe)]
        if matching:
            xe, self.tk, _ = matching[-1]
        else:
            xe, self.tk = (cp.asarray(arr(pointer,self.n)) for pointer in (a.xe,a.tk))
        prior = self.history.get((address(a.old_rec),a.prev_z))
        if prior is None:
            prior = tuple(cp.asarray(arr(pointer,self.n)) for pointer in (a.old_rec,a.old_z))
        self.prev_rec,self.prev_z = prior
        self.max_pair_dev[0] = cp.max(xe)
        self.max_pair_dev[1] = cp.max(self.prev_rec)
        cp.cuda.runtime.memcpyAsync(
            self.max_pair_h.ctypes.data, self.max_pair_dev.data.ptr, 8,
            cp.cuda.runtime.memcpyDeviceToHost, cp.cuda.get_current_stream().ptr,
        )
        if self.scratch_n != self.n:
            self.delta = cp.empty(self.n, dtype=cp.float32)
            self.gamma = cp.empty(self.n, dtype=cp.float32)
            self.mfp = cp.empty(self.n, dtype=cp.float32)
            self.temp = cp.empty(self.n, dtype=cp.float32)
            self.fc = cp.empty(self.n, dtype=cp.float32)
            self.bad = cp.empty(1, dtype=cp.int32)
            self.flags = cp.empty(self.n, dtype=cp.uint8)
            self.sum_partials = cp.empty(256, dtype=cp.float64)
            self.collapse_sum = cp.empty((), dtype=cp.float64)
            self._finish_pinned_mem = cp.cuda.alloc_pinned_memory(7 * self.n * 4)
            self.finish_pinned = np.frombuffer(self._finish_pinned_mem, dtype=np.float32, count=7 * self.n).reshape(7, self.n)
            self.input_stack = None
            self.scratch_n = self.n
        cp.cuda.get_current_stream().synchronize()
        self.xe_max = float(self.max_pair_h[0])
        self.rec_max = float(self.max_pair_h[1])
        self.rec_is_zero = (self.rec_max == 0.0)
        if self.skip_density_fft:
            input_fields = (xe, self.prev_rec)
            self.batch_mode = "xe_rec"
        elif self.rec_is_zero:
            if self.rec_zeros is None or self.rec_zeros.size != self.n:
                self.rec_zeros = cp.zeros(self.n, dtype=cp.float32)
            else:
                self.rec_zeros.fill(0)
            input_fields = (self.density, xe)
            self.batch_mode = "delta_xe"
        else:
            input_fields = (self.density, xe, self.prev_rec)
            self.batch_mode = "all3"
        if self.batched:
            nbuf = len(input_fields)
            if self.input_stack is None or self.input_stack.shape != (nbuf, *self.shape):
                self.input_stack = cp.empty((nbuf, *self.shape), dtype=cp.float32)
                self.workspace = cp.empty((nbuf, a.side, a.side, a.side // 2 + 1), dtype=cp.complex64)
            for idx, f in enumerate(input_fields):
                self.input_stack[idx].ravel()[:] = f
            self.spectra = cp.fft.rfftn(self.input_stack, axes=(-3,-2,-1))
        else:
            self.spectra = [cp.fft.rfftn(f.reshape(self.shape)) for f in input_fields]
            self.workspace = cp.empty_like(self.spectra[0])
        self.xhi = cp.empty(self.n, dtype=cp.float32)
        self.zre = cp.empty(self.n, dtype=cp.float32)
        self.launch(self.init_arrays_kernel, self.n, self.xhi, self.zre, self.gamma, self.mfp, self.temp, np.int32(self.n))
        self.bad.fill(0)
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
        mode = getattr(self, "batch_mode", "all3")
        if self.batched:
            if mode == "xe_rec":
                self.launch(self.filter_pair,self.workspace.size//2,self.spectra,self.workspace,*args)
                fields = cp.fft.irfftn(self.workspace,s=self.shape,axes=(-3,-2,-1),norm="forward").reshape(2,self.n)
                self.launch(self.clamp_fields,self.n,fields,np.int32(self.n),np.int32(0),np.int32(1),np.int32(1))
                self.xe,self.rec = fields[0],fields[1]
                return
            if mode == "delta_xe":
                self.launch(self.filter_pair,self.workspace.size//2,self.spectra,self.workspace,*args)
                fields = cp.fft.irfftn(self.workspace,s=self.shape,axes=(-3,-2,-1),norm="forward").reshape(2,self.n)
                self.launch(self.clamp_fields,self.n,fields,np.int32(self.n),np.int32(1),np.int32(1),np.int32(0))
                self.delta,self.xe,self.rec = fields[0],fields[1],self.rec_zeros
            else:
                self.launch(self.filter_triplet,self.workspace.size//3,self.spectra,self.workspace,*args)
                fields = cp.fft.irfftn(self.workspace,s=self.shape,axes=(-3,-2,-1),norm="forward").reshape(3,self.n)
                self.launch(self.clamp_fields,self.n,fields,np.int32(self.n),np.int32(1),np.int32(1),np.int32(1))
                self.delta,self.xe,self.rec = fields[0],fields[1],fields[2]
        else:
            fields = []
            for field in self.spectra:
                self.launch(self.kernels[0],field.size,field,self.workspace,*args)
                fields.append(cp.fft.irfftn(self.workspace,s=self.shape,norm="forward").reshape(-1))
            if mode == "xe_rec":
                self.xe = cp.clip(fields[0],0.,np.float32(.999))
                self.rec = cp.maximum(fields[1],0.)
                return
            if mode == "delta_xe":
                self.delta = cp.maximum(fields[0],np.float32(-1.+1e-7))
                self.xe = cp.clip(fields[1],0.,np.float32(.999))
                self.rec = self.rec_zeros
            else:
                self.delta = cp.maximum(fields[0],np.float32(-1.+1e-7))
                self.xe = cp.clip(fields[1],0.,np.float32(.999))
                self.rec = cp.maximum(fields[2],0.)
        limits=cp.stack((self.delta.min(),self.delta.max())).get()
        a.lo=min(0.,float(limits[0]))
        a.hi=max(0.,float(limits[1]))

    def filter_xe_rec(self, a):
        args = (np.int32(a.side),np.float64(a.dk),np.float32(a.radius),np.int32(a.filtered))
        if self.batched:
            self.launch(self.filter_pair,self.workspace.size//2,self.spectra,self.workspace,*args)
            fields = cp.fft.irfftn(self.workspace,s=self.shape,axes=(-3,-2,-1),
                                 norm="forward").reshape(2,self.n)
        else:
            fields = []
            for field in self.spectra:
                self.launch(self.kernels[0],field.size,field,self.workspace,*args)
                fields.append(cp.fft.irfftn(self.workspace,s=self.shape,norm="forward").reshape(-1))
        self.xe = cp.clip(fields[0],0.,np.float32(.999))
        self.rec = cp.maximum(fields[1],0.)

    def minimum_margin(self, a):
        self.margin_pass1(
            (256,),
            (256,),
            (self.delta, self.xe, self.rec, self.fc, self.xhi,
             self.collapse_sum, self.margin_partials,
             np.int32(self.n), np.float64(a.norm), np.float64(a.fmin), np.float32(a.eff)),
        )
        self.margin_pass2((1,), (256,), (self.margin_partials, self.margin_out, np.int32(256)))
        self.margin_out.get(out=self.margin_pinned)
        return float(self.margin_pinned[0])

    def minimum_margin_amp(self, a, xe_max, rec_max):
        eps = float(np.finfo(np.float32).eps)
        xe_scale = eps * float(xe_max)
        rec_scale = eps * float(rec_max)
        self.margin_amp_pass1(
            (256,),
            (256,),
            (self.delta, self.xe, self.rec, self.fc, self.xhi,
             self.collapse_sum, self.margin_partials,
             np.int32(self.n), np.float64(a.norm), np.float64(a.fmin),
             np.float32(a.eff), np.float64(xe_scale), np.float64(rec_scale)),
        )
        self.margin_pass2((1,), (256,), (self.margin_partials, self.margin_out, np.int32(256)))
        self.margin_out.get(out=self.margin_pinned)
        return float(self.margin_pinned[0])

    def screen_scale(self, a, ratio_thresh=0.99):
        self.screen_flag.fill(0)
        self.screen_kernel(
            (256,),
            (256,),
            (self.delta, self.xe, self.rec, self.fc, self.xhi,
             self.collapse_sum, self.screen_flag,
             np.int32(self.n), np.float64(a.norm), np.float64(a.fmin),
             np.float32(a.eff), np.float64(a.cut), np.float64(ratio_thresh)),
        )
        cp.cuda.runtime.memcpyAsync(
            self.screen_flag_h.ctypes.data, self.screen_flag.data.ptr, 4,
            cp.cuda.runtime.memcpyDeviceToHost, cp.cuda.get_current_stream().ptr,
        )
        cp.cuda.get_current_stream().synchronize()
        return bool(self.screen_flag_h[0] != 0)

    def lookup(self,a):
        low,high=self.thermal.tables.row(a.low,a.high,0,a.nl,a.nh)
        self.bad.fill(0)
        self.collapse_with_sum((256,),(256,),(self.delta,low,high,self.fc,self.sum_partials,
                    np.int64(self.n),np.int32(a.nl),np.int32(a.nh),*(np.float32(x) for x in
                    (a.ls,a.li,a.hs,a.hi_inv,a.crit)),np.float64(a.cut),self.clamps))
        self.sum_pass2((1,),(256,),(self.sum_partials,self.collapse_sum,np.int32(256)))

    def apply(self,a):
        if self.trajectory is not None:
            ids = self.trajectory_indices
            vals = cp.stack([f[ids] for f in (self.delta,self.xe,self.rec,self.fc,self.xhi,self.gamma)]).get()
            norm = a.norm / max(float(self.collapse_sum.get().item()) / self.n, a.fmin)
            for i, (d,xe,rec,fc,xhi,gamma) in enumerate(vals.T):
                row=(a.z,i,a.radius,int(a.last),d,xe,rec,fc,norm,a.eff,a.fmin,xhi,gamma,a.lo,a.hi,a.gamma)
                self.trajectory.write("scale "+" ".join(format(float(x),".17g") for x in row)+"\n")

        self.launch(self.kernels[2],self.n,self.delta,self.xe,self.rec,self.fc,self.prev_z,
                    self.tk,self.poisson,self.xhi,self.zre,self.mfp,self.gamma,self.temp,
                    np.int32(self.n),np.int32(a.last),*(np.float32(x) for x in
                    (a.radius,a.z,a.pixel,a.mmin,a.eff,a.tre)),*(np.float64(x) for x in
                    (a.norm,a.gamma,a.fmin,a.npoisson)),self.collapse_sum)

    def finish(self,a):
        self.nonfinite_dev.fill(0)
        self.launch(self.kernels[7],self.n,self.density,self.xhi,self.zre,self.tk,self.temp,
                    np.int32(self.n),np.float32(a.z),np.float32(a.tre),np.float32(a.density_limit),self.nonfinite_dev)
        out = cp.empty_like(self.xhi)
        self.launch(self.kernels[3],self.n,self.density,self.gamma,self.xhi,
                    self.prev_rec,out,*self.knots,np.int32(self.n),np.float32(a.z),
                    np.float32(a.dt),np.float32(a.dz),np.float64(a.dz_table),np.float64(a.max_log),self.nonfinite_dev)
        if getattr(self, "same_input_rec", None) is not None:
            self.same_input_rec(a, out)
        copy_nt = min(8, max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
        nbytes = np.uint64(self.n * 4)
        if os.environ.get("PS_ION_SLIM_EXPORT", "0") == "1":
            self.xhi.get(out=self.finish_pinned[0], blocking=False)
            out.get(out=self.finish_pinned[1], blocking=False)
            cp.cuda.runtime.memcpyAsync(
                self.nonfinite_h.ctypes.data, self.nonfinite_dev.data.ptr, 4,
                cp.cuda.runtime.memcpyDeviceToHost, cp.cuda.get_current_stream().ptr,
            )
            cp.cuda.get_current_stream().synchronize()
            nf = int(self.nonfinite_h[0])
            if nf == 2:
                raise ValueError(f"Nonfinite ionized gas temperature at z={a.z}")
            if nf:
                raise ValueError(f"Nonfinite recombination at z={a.z}")
            self.pmemcpy(C.cast(a.xhi, C.c_void_p).value, self.finish_pinned[0].ctypes.data, nbytes, copy_nt)
            self.pmemcpy(C.cast(a.nrec, C.c_void_p).value, self.finish_pinned[1].ctypes.data, nbytes, copy_nt)
        else:
            arrays = (self.xhi, self.zre, self.mfp, self.gamma, out, self.temp, self.fc)
            ptrs = (a.xhi, a.zre, a.mfp, a.gamma12, a.nrec, a.temperature, a.fcoll)
            for idx, array in enumerate(arrays):
                array.get(out=self.finish_pinned[idx], blocking=False)
            cp.cuda.runtime.memcpyAsync(
                self.nonfinite_h.ctypes.data, self.nonfinite_dev.data.ptr, 4,
                cp.cuda.runtime.memcpyDeviceToHost, cp.cuda.get_current_stream().ptr,
            )
            cp.cuda.get_current_stream().synchronize()
            nf = int(self.nonfinite_h[0])
            if nf == 2:
                raise ValueError(f"Nonfinite ionized gas temperature at z={a.z}")
            if nf:
                raise ValueError(f"Nonfinite recombination at z={a.z}")
            for idx, ptr in enumerate(ptrs):
                self.pmemcpy(C.cast(ptr, C.c_void_p).value, self.finish_pinned[idx].ctypes.data, nbytes, copy_nt)
        self.neutral_fields[address(a.xhi)]=self.xhi
        self.history[(address(a.nrec),a.z)] = (out,self.zre)
        if self.counts[4] % 10 == 0:
            print(f"GPU_ION z={a.z} active_steps={self.counts[4]+1}",flush=True)

    def report(self):
        return {"phase_counts":self.counts,"poisson_gpu_used":self.poisson_gpu_used,
                "phase_seconds":self.seconds,"table_endpoint_clamps":int(self.clamps.get()[0]),
                "cpu_fft_replays":0,"cpu_recombination_row_replays":0}
