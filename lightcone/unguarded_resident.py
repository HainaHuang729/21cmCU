"""Supply the existing thermal backend with C-compatible resident density."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import os
import time
import traceback
from pathlib import Path

import cupy as cp
import numpy as np

from gpu_heat_runtime import F, Runtime, arr

FilterHook = C.CFUNCTYPE(
    C.c_int, C.c_int, C.c_int, C.c_double, C.c_float, C.c_float,
    F, F, F, F, C.POINTER(F),
)


class ResidentRuntime(Runtime):
    def __init__(self):
        super().__init__()
        module = cp.RawModule(
            code=Path(__file__).with_name("resident_filter.cu").read_text(),
            options=("--fmad=false",),
        )
        self.filter_spectrum = module.get_function("filter_spectrum")
        self.extrapolate_density = module.get_function("extrapolate_density")
        self.unpack_pass1 = module.get_function("unpack_extrapolate_bounds_pass1")
        self.unpack_pass2 = module.get_function("unpack_extrapolate_bounds_pass2")
        self.filter_steps = []
        self.fallback_downloads = 0
        self.use_fftw_heat_filter = os.environ.get("PS_FFTW_HEAT_FILTER", "1") == "1"
        self.heat_gpu_unpack = self.use_fftw_heat_filter and (os.environ.get("PS_HEAT_GPU_UNPACK", "1") == "1")
        self.prefetch_enabled = os.environ.get("PS_ASYNC_FFTW_PREFETCH", "1") == "1"
        self.nt_heat = max(1, min(8, int(os.environ.get("OMP_NUM_THREADS", "8"))))
        self.pinned_heat_mem = [None, None]
        self.pinned_heat_density = [None, None]
        self.pinned_heat_padded_mem = [None, None]
        self.pinned_heat_padded = [None, None]
        self.density_padded = None
        self.heat_blk_min = None
        self.heat_blk_max = None
        self.heat_min = None
        self.heat_max = None
        self._heat_bounds_mem = None
        self.heat_min_h = None
        self.heat_max_h = None
        self.heat_slot = 0
        self.upload_stream = cp.cuda.Stream(non_blocking=True)
        self.upload_done_event = None
        self.heat_fftw_seconds = 0.0
        self.heat_upload_seconds = 0.0
        self._prefetch_pool = None
        self._heat_future = None
        self._heat_prefetch_rc = None
        self._heat_prefetch_key = None
        self._heat_prefetch_slot = None
        self._heat_prefetch_min = None
        self._heat_prefetch_max = None
        self._heat_prefetch_dens_ref = None
        self._last_heat_params = None
        self._lc_schedule = None
        self._before_heat_filter_cb = None
        self._after_heat_filter_cb = None
        self.heat_prefetch_hits = 0
        self.heat_prefetch_misses = 0
        self.heat_prefetch_wait_seconds = 0.0

    def install(self, native):
        super().install(native)
        self.filter_callback = FilterHook(self.filter_density)
        self.lib.SetGPUFilterHook.argtypes = [C.c_void_p]
        self.lib.SetGPUFilterHook.restype = None
        self.lib.SetGPUFilterHook(C.cast(self.filter_callback, C.c_void_p))
        self.heat_batch_density = self.lib.HeatReferenceBatchDensity
        self.heat_batch_density.argtypes = [C.c_int, C.c_int, C.c_double, C.c_float, F, F, F, F, F]
        self.heat_batch_density.restype = C.c_int
        self.heat_batch_preplanned = self.lib.HeatReferenceBatchDensityPreplanned
        self.heat_batch_preplanned.argtypes = [C.c_int, C.c_int, C.c_double, C.c_float, F, F, F, F, F, C.c_int]
        self.heat_batch_preplanned.restype = C.c_int
        self.heat_batch_padded = self.lib.HeatReferenceBatchDensityPadded
        self.heat_batch_padded.argtypes = [C.c_int, C.c_int, C.c_double, F, F, F, C.c_int]
        self.heat_batch_padded.restype = C.c_int
        self.heat_batch_preplanned_padded = self.lib.HeatReferenceBatchDensityPreplannedPadded
        self.heat_batch_preplanned_padded.argtypes = [C.c_int, C.c_int, C.c_double, F, F, F, C.c_int]
        self.heat_batch_preplanned_padded.restype = C.c_int
        self._dicke = self.lib.dicke
        self._dicke.argtypes = [C.c_double]
        self._dicke.restype = C.c_double

    def _ensure_pinned_slots(self, count, size):
        slots = (0, 1) if self.prefetch_enabled else (0,)
        for s in slots:
            if self.pinned_heat_density[s] is None or self.pinned_heat_density[s].shape != (count, size):
                mem = cp.cuda.alloc_pinned_memory(count * size * 4)
                self.pinned_heat_mem[s] = mem
                self.pinned_heat_density[s] = np.frombuffer(
                    mem, dtype=np.float32, count=count * size
                ).reshape(count, size)

    def _ensure_pinned_padded_slots(self, side, count):
        padded_vol = side * side * (2 * (side // 2 + 1))
        slots = (0, 1) if self.prefetch_enabled else (0,)
        for s in slots:
            if self.pinned_heat_padded[s] is None or self.pinned_heat_padded[s].shape != (count, padded_vol):
                mem = cp.cuda.alloc_pinned_memory(count * padded_vol * 4)
                self.pinned_heat_padded_mem[s] = mem
                self.pinned_heat_padded[s] = np.frombuffer(
                    mem, dtype=np.float32, count=count * padded_vol
                ).reshape(count, padded_vol)
        if self.density_padded is None or self.density_padded.shape != (count, padded_vol):
            self.density_padded = cp.empty((count, padded_vol), dtype=cp.float32)
        if self.heat_blk_min is None or self.heat_blk_min.shape != (count, 256):
            self.heat_blk_min = cp.empty((count, 256), dtype=cp.float32)
            self.heat_blk_max = cp.empty((count, 256), dtype=cp.float32)
            self.heat_min = cp.empty(count, dtype=cp.float32)
            self.heat_max = cp.empty(count, dtype=cp.float32)
            bmem = cp.cuda.alloc_pinned_memory(2 * count * 4)
            self._heat_bounds_mem = bmem
            barr = np.frombuffer(bmem, dtype=np.float32, count=2 * count)
            self.heat_min_h = barr[:count]
            self.heat_max_h = barr[count:]

    def wait_heat_prefetch(self):
        if self._heat_future is not None:
            tw = time.perf_counter()
            self._heat_prefetch_rc = self._heat_future.result()
            self.heat_prefetch_wait_seconds += time.perf_counter() - tw
            self._heat_future = None

    def submit_heat_prefetch_for_index(self, next_iz):
        if not self.prefetch_enabled or self._lc_schedule is None or self._last_heat_params is None:
            return False
        if self._heat_future is not None or self._heat_prefetch_key is not None:
            return False
        scrollz, perturb = self._lc_schedule
        if next_iz < 0 or next_iz >= len(scrollz):
            return False
        side, count, dk, radii_arr = self._last_heat_params
        size = side ** 3
        pf_next = perturb[next_iz]
        dens_arr = pf_next.density
        if not dens_arr.flags.c_contiguous or dens_arr.dtype != np.float32:
            dens_arr = np.ascontiguousarray(dens_arr, dtype=np.float32)
        dens_ptr = dens_arr.ctypes.data
        next_slot = 1 - self.heat_slot
        nt_bg = self.nt_heat
        if self._prefetch_pool is None:
            self._prefetch_pool = ThreadPoolExecutor(max_workers=1)
        if self.heat_gpu_unpack:
            self._ensure_pinned_padded_slots(side, count)
            self._heat_prefetch_rc = None
            self._heat_prefetch_key = (dens_ptr, int(side), int(count), float(dk), radii_arr.tobytes())
            self._heat_prefetch_slot = next_slot
            self._heat_prefetch_min = None
            self._heat_prefetch_max = None
            self._heat_prefetch_dens_ref = dens_arr
            self._heat_future = self._prefetch_pool.submit(
                self.heat_batch_preplanned_padded,
                side, count, dk,
                C.cast(C.c_void_p(dens_ptr), F), radii_arr.ctypes.data_as(F),
                self.pinned_heat_padded[next_slot].ctypes.data_as(F), nt_bg,
            )
            return True
        self._ensure_pinned_slots(count, size)
        gf_z = np.float32(self._dicke(float(np.float32(pf_next.redshift))))
        ig_next = np.float32(1.0 / float(gf_z))
        ig_bytes = ig_next.tobytes()
        min_buf = np.empty(count, dtype=np.float32)
        max_buf = np.empty(count, dtype=np.float32)
        self._heat_prefetch_rc = None
        self._heat_prefetch_key = (
            dens_ptr, int(side), int(count), float(dk), ig_bytes, radii_arr.tobytes()
        )
        self._heat_prefetch_slot = next_slot
        self._heat_prefetch_min = min_buf
        self._heat_prefetch_max = max_buf
        self._heat_prefetch_dens_ref = dens_arr
        self._heat_future = self._prefetch_pool.submit(
            self.heat_batch_preplanned,
            side, count, dk, float(ig_next),
            C.cast(C.c_void_p(dens_ptr), F), radii_arr.ctypes.data_as(F),
            self.pinned_heat_density[next_slot].ctypes.data_as(F),
            min_buf.ctypes.data_as(F), max_buf.ctypes.data_as(F), nt_bg,
        )
        return True

    def join_and_clear_prefetch(self):
        if self._heat_future is not None:
            try:
                self._heat_future.result()
            finally:
                self._heat_future = None
        self._heat_prefetch_rc = None
        self._heat_prefetch_key = None
        self._heat_prefetch_slot = None
        self._heat_prefetch_min = None
        self._heat_prefetch_max = None
        self._heat_prefetch_dens_ref = None
        if self._prefetch_pool is not None:
            self._prefetch_pool.shutdown(wait=True)
            self._prefetch_pool = None
        self._lc_schedule = None
        self._before_heat_filter_cb = None
        self._after_heat_filter_cb = None

    def filter_density(self, side, count, dk, inverse_growth, low, density,
                       radii, minima, maxima, materialize):
        try:
            if materialize:
                ev = self.upload_done_event
                if ev is not None:
                    self.upload_done_event = None
                    ev.synchronize()
                for j in range(count):
                    self.density[j].get(out=arr(materialize[j], side**3))
                self.fallback_downloads += 1
                self.download_bytes += count * side**3 * 4
            else:
                self.compute_filter(side, count, dk, inverse_growth, low,
                                    density, radii, minima, maxima)
            return 0
        except Exception:
            self.error = traceback.format_exc()
            print(self.error, flush=True)
            return 1

    def compute_filter(self, side, count, dk, inverse_growth, low,
                       density, radii, minima, maxima):
        start = time.perf_counter()
        if self.heat_gpu_unpack:
            size = side ** 3
            if self.density is None or self.density.shape != (count, size):
                self.density = cp.empty((count, size), dtype=cp.float32)
            self._ensure_pinned_padded_slots(side, count)
            dens_ptr = C.cast(density, C.c_void_p).value
            radii_arr = arr(radii, count)
            cur_key = (dens_ptr, int(side), int(count), float(dk), radii_arr.tobytes())
            if self._before_heat_filter_cb is not None:
                self._before_heat_filter_cb()
            slot = self.heat_slot
            used_prefetch = False
            if self._heat_future is not None or self._heat_prefetch_key is not None:
                if self._heat_future is not None:
                    self.wait_heat_prefetch()
                code = self._heat_prefetch_rc
                pf_key = self._heat_prefetch_key
                pf_slot = self._heat_prefetch_slot
                self._heat_prefetch_rc = None
                self._heat_prefetch_key = None
                self._heat_prefetch_slot = None
                self._heat_prefetch_min = None
                self._heat_prefetch_max = None
                self._heat_prefetch_dens_ref = None
                if code == 0 and pf_key == cur_key:
                    self.heat_prefetch_hits += 1
                    slot = pf_slot
                    used_prefetch = True
                else:
                    self.heat_prefetch_misses += 1
            if not used_prefetch:
                self.upload_stream.synchronize()
                t_fft = time.perf_counter()
                code = self.heat_batch_padded(
                    side, count, dk, density, radii,
                    self.pinned_heat_padded[slot].ctypes.data_as(F), self.nt_heat,
                )
                if code != 0:
                    raise RuntimeError(f"HeatReferenceBatchDensityPadded failed: {code}")
                self.heat_fftw_seconds += time.perf_counter() - t_fft

            self.heat_slot = slot
            self._last_heat_params = (int(side), int(count), float(dk), radii_arr.copy())
            t_up = time.perf_counter()
            src_h = self.pinned_heat_padded[slot]
            with self.upload_stream:
                cp.cuda.runtime.memcpyAsync(
                    self.density_padded.data.ptr, src_h.ctypes.data, src_h.nbytes,
                    cp.cuda.runtime.memcpyHostToDevice, self.upload_stream.ptr,
                )
                self.unpack_pass1(
                    (256, count), (256,),
                    (self.density_padded, self.density, self.heat_blk_min, self.heat_blk_max,
                     np.int32(side), np.int32(count), np.float32(inverse_growth), np.float64(1.0e-7)),
                )
                self.unpack_pass2(
                    (count,), (256,),
                    (self.heat_blk_min, self.heat_blk_max, self.heat_min, self.heat_max, np.int32(count)),
                )
                cp.cuda.runtime.memcpyAsync(
                    self.heat_min_h.ctypes.data, self.heat_min.data.ptr, count * 4,
                    cp.cuda.runtime.memcpyDeviceToHost, self.upload_stream.ptr,
                )
                cp.cuda.runtime.memcpyAsync(
                    self.heat_max_h.ctypes.data, self.heat_max.data.ptr, count * 4,
                    cp.cuda.runtime.memcpyDeviceToHost, self.upload_stream.ptr,
                )
            self.upload_done_event = None
            if self._after_heat_filter_cb is not None:
                self._after_heat_filter_cb(dens_ptr, int(side), float(dk))
            self.upload_stream.synchronize()
            arr(minima, count)[:] = self.heat_min_h
            arr(maxima, count)[:] = self.heat_max_h
            if not (np.isfinite(arr(minima, count)).all() and np.isfinite(arr(maxima, count)).all()):
                raise ValueError("Nonfinite density bounds")
            self.heat_upload_seconds += time.perf_counter() - t_up
            self.upload_bytes += src_h.nbytes
            self.filter_steps.append(time.perf_counter() - start)
            return

        if self.use_fftw_heat_filter:
            size = side ** 3
            if self.density is None or self.density.shape != (count, size):
                self.density = cp.empty((count, size), dtype=cp.float32)
            self._ensure_pinned_slots(count, size)
            dens_ptr = C.cast(density, C.c_void_p).value
            radii_arr = arr(radii, count)
            ig_bytes = np.float32(inverse_growth).tobytes()
            cur_key = (dens_ptr, int(side), int(count), float(dk), ig_bytes, radii_arr.tobytes())

            if self._before_heat_filter_cb is not None:
                self._before_heat_filter_cb()
            slot = self.heat_slot
            used_prefetch = False
            if self._heat_future is not None or self._heat_prefetch_key is not None:
                if self._heat_future is not None:
                    self.wait_heat_prefetch()
                code = self._heat_prefetch_rc
                pf_key = self._heat_prefetch_key
                pf_slot = self._heat_prefetch_slot
                pf_min = self._heat_prefetch_min
                pf_max = self._heat_prefetch_max
                self._heat_prefetch_rc = None
                self._heat_prefetch_key = None
                self._heat_prefetch_slot = None
                self._heat_prefetch_min = None
                self._heat_prefetch_max = None
                self._heat_prefetch_dens_ref = None
                if code == 0 and pf_key == cur_key:
                    self.heat_prefetch_hits += 1
                    slot = pf_slot
                    arr(minima, count)[:] = pf_min
                    arr(maxima, count)[:] = pf_max
                    used_prefetch = True
                else:
                    self.heat_prefetch_misses += 1
            if not used_prefetch:
                self.upload_stream.synchronize()
                t_fft = time.perf_counter()
                code = self.heat_batch_density(
                    side, count, dk, inverse_growth, density, radii,
                    self.pinned_heat_density[slot].ctypes.data_as(F), minima, maxima,
                )
                if code != 0:
                    raise RuntimeError(f"HeatReferenceBatchDensity failed: {code}")
                self.heat_fftw_seconds += time.perf_counter() - t_fft

            self.heat_slot = slot
            self._last_heat_params = (int(side), int(count), float(dk), radii_arr.copy())
            if not (np.isfinite(arr(minima, count)).all() and np.isfinite(arr(maxima, count)).all()):
                raise ValueError("Nonfinite density bounds")
            t_up = time.perf_counter()
            src_h = self.pinned_heat_density[slot]
            with self.upload_stream:
                cp.cuda.runtime.memcpyAsync(
                    self.density.data.ptr, src_h.ctypes.data, src_h.nbytes,
                    cp.cuda.runtime.memcpyHostToDevice, self.upload_stream.ptr,
                )
                self.upload_done_event = self.upload_stream.record()
            self.heat_upload_seconds += time.perf_counter() - t_up
            self.upload_bytes += count * size * 4
            if self._after_heat_filter_cb is not None:
                self._after_heat_filter_cb(dens_ptr, int(side), float(dk))
            self.filter_steps.append(time.perf_counter() - start)
            return
        shape = (side,) * 3
        size = side**3
        spectrum = cp.fft.rfftn(self.filter_input(density, size).reshape(shape))
        scratch = cp.empty_like(spectrum)
        if self.density is None or self.density.shape != (count, size):
            self.density = cp.empty((count, size), dtype=cp.float32)
        for j, radius in enumerate(arr(radii, count)):
            self.filter_spectrum(
                ((spectrum.size + 255) // 256,), (256,),
                (spectrum, scratch, np.int32(side), np.float64(dk),
                 radius, np.int32(j != 0)),
            )
            smooth = cp.fft.irfftn(scratch, s=shape, norm="forward")
            self.extrapolate_density(
                ((size + 255) // 256,), (256,),
                (smooth, self.density[j], np.int64(size),
                 np.float32(inverse_growth), np.float32(low)),
            )
        bounds = cp.stack((self.density.min(axis=1), self.density.max(axis=1))).get()
        if not np.isfinite(bounds).all():
            raise ValueError("Nonfinite density bounds")
        arr(minima, count)[:] = np.minimum(bounds[0], 0)
        arr(maxima, count)[:] = np.maximum(bounds[1], 0)
        self.download_bytes += bounds.nbytes
        self.filter_steps.append(time.perf_counter() - start)

    def report(self):
        report = super().report()
        report.update(filter_step_seconds=self.filter_steps,
                      no_light_materializations=self.fallback_downloads,
                      fftw_heat_filter=self.use_fftw_heat_filter,
                      heat_fftw_seconds=self.heat_fftw_seconds,
                      heat_upload_seconds=self.heat_upload_seconds,
                      heat_prefetch_hits=self.heat_prefetch_hits,
                      heat_prefetch_misses=self.heat_prefetch_misses,
                      heat_prefetch_wait_seconds=self.heat_prefetch_wait_seconds)
        report["note"] = "GPU resident density, thermal state, and conditional collapse tables; host stages scalar coefficients"
        return report
