"""Execute conditional-collapse quadrature for the heat and ion lookup tables."""
import ctypes as C
import os
import json
from pathlib import Path
import time
import traceback

import cupy as cp
import math
import numpy as np

F = C.POINTER(C.c_float)
D = C.POINTER(C.c_double)
Hook = C.CFUNCTYPE(C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, D, F, F, F, F, F, F, F, F)


class ConditionalTables:
    def __init__(self, native):
        self.lib = C.CDLL(native.__file__)
        self.kernel = cp.RawKernel(
            Path(__file__).with_name("conditional.cu").read_text(),
            "conditional_table",
            options=("--fmad=false",),
        )
        code = Path(__file__).with_name("conditional_gk61.cu").read_text()
        module = cp.RawModule(code=code, options=("--fmad=false",))
        self.qag = module.get_function("conditional_heat_qag")
        self.qag_warp = module.get_function("conditional_heat_qag_warp")
        self.qag_mode = os.environ.get("PS_QAG_MODE", "deferred")
        self.qag_profile_enabled = os.environ.get("PS_QAG_PROFILE") == "1"
        self.full_high = os.environ.get("PS_CONDITIONAL_FULL_HIGH") == "1"
        self.nan_sentinel = os.environ.get("PS_CONDITIONAL_NAN_SENTINEL", "1") == "1"
        self.qag_profiles = []
        self.qag_stats = cp.zeros(2, dtype=cp.uint64)
        self.panel_counts_buf = None
        self.pinned_count = None
        self.rule = cp.asarray(json.loads(Path(os.environ["PS_GK61_RULE"]).read_text()), dtype=cp.float64)
        self.panel_workspace = None
        self.gl = tuple(cp.asarray(a, dtype=cp.float64) for a in np.polynomial.legendre.leggauss(16))
        self.low_gl = self._build_nr_gauleg(100)
        self.callback = Hook(self.fill)
        setter = self.lib.SetGPUConditionalTable
        setter.argtypes = [C.c_void_p]
        setter.restype = None
        setter(C.cast(self.callback, C.c_void_p))
        self.calls = [0, 0]
        mode = "0" if os.environ.get("PS_NATIVE_CONDITIONAL") == "1" else os.environ.get("PS_CONDITIONAL_MODE", "3")
        self.gpu_modes = int(mode)
        set_mode = self.lib.SetGPUConditionalMode
        set_mode.argtypes = [C.c_int]
        set_mode.restype = None
        set_mode(self.gpu_modes)
        self.device_rows = {}
        self.native_first_heat_rows = 0
        self.gpu_adaptive_heat_rows = 0
        self.gpu_adaptive_ion_rows = 0
        self.gpu_adaptive_heat_bins = 0
        self.gpu_adaptive_ion_bins = 0
        self.gpu_high_skipped = [0, 0]
        self.qag_host_seconds = [0.0, 0.0]
        self.gpu_adaptive_panels = 0
        self.compare_native = os.environ.get("PS_GK61_COMPARE") == "1"
        self.device_reads = [0, 0]
        self.seconds = [0.0, 0.0]

    @staticmethod
    def _build_nr_gauleg(n, eps2=3.0e-11):
        # Match Numerical Recipes gauleg in ps.c:1450 on [-1, 1].
        m = (n + 1) // 2
        z_signed = np.empty(n, dtype=np.float64)
        denom = np.empty(n, dtype=np.float64)
        for idx in range(m):
            i = idx + 1
            z = math.cos(3.141592654 * (i - 0.25) / (n + 0.5))
            while True:
                p1 = 1.0
                p2 = 0.0
                for j in range(1, n + 1):
                    p3 = p2
                    p2 = p1
                    p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
                pp = n * (z * p1 - p2) / (z * z - 1.0)
                z1 = z
                z = z1 - p1 / pp
                if abs(z - z1) <= eps2:
                    break
            z_signed[idx] = -z
            z_signed[n - 1 - idx] = z
            d = (1.0 - z * z) * pp * pp
            denom[idx] = d
            denom[n - 1 - idx] = d
        return (cp.asarray(z_signed, dtype=cp.float64), cp.asarray(denom, dtype=cp.float64))

    @staticmethod
    def upload(ptr, n):
        return cp.asarray(np.ctypeslib.as_array(ptr, shape=(n,)))

    def row(self, low, high, heat, nlow=None, nhigh=None):
        if not (self.gpu_modes & (1 << heat)):
            return self.upload(low, nlow), self.upload(high, nhigh)
        key = tuple(C.cast(ptr, C.c_void_p).value for ptr in (low, high))
        rows = self.device_rows.pop(key)
        self.device_reads[heat] += 1
        return rows

    def finish_qag_profile(self):
        before = time.perf_counter()
        stats = self.qag_stats.get()
        total_panels = int(stats[0])
        fail_count = int(stats[1])
        self.gpu_adaptive_panels = total_panels
        if fail_count > 0:
            raise RuntimeError(f"GPU Gauss-Kronrod integration failed on {fail_count} bins")
        final_seconds = time.perf_counter() - before
        values = []
        for u, v, a, b, h, launch, wait, nbytes, s0, s1, g0, d2h in self.qag_profiles:
            b.synchronize()
            values.append(
                dict(
                    kernel_seconds=cp.cuda.get_elapsed_time(a, b) / 1000.0,
                    upload_and_low_row_stream_seconds=cp.cuda.get_elapsed_time(u, v) / 1000.0,
                    host_upload_section_seconds=h,
                    launch_seconds=launch,
                    get_wait_seconds=wait,
                    count_bytes=nbytes,
                    count_copy_stream_seconds=d2h,
                    shared_and_density_upload_stream_seconds=(
                        cp.cuda.get_elapsed_time(s0, s1) + cp.cuda.get_elapsed_time(g0, v)
                    ) / 1000.0,
                )
            )
        return dict(
            mode=self.qag_mode,
            rows=values,
            final_check_seconds=final_seconds,
            total_panels=total_panels,
            fail_count=fail_count,
            ion_qag_rows=self.gpu_adaptive_ion_rows,
            ion_qag_bins=self.gpu_adaptive_ion_bins,
            heat_qag_rows=self.gpu_adaptive_heat_rows,
            heat_qag_bins=self.gpu_adaptive_heat_bins,
            high_skipped=list(self.gpu_high_skipped),
            qag_host_seconds=list(self.qag_host_seconds),
        )

    def fill(self, heat, nlow, nhigh, ns, nq, p, sigma, dsigma, low, high, x, w, out_low, out_high):
        try:
            start = time.perf_counter()
            adaptive_heat = heat == 2
            if adaptive_heat:
                heat = 1
            p_host = np.ctypeslib.as_array(p, shape=(16,))
            deltac = float(p_host[5])
            max_d = float(p_host[15])

            profile_this = self.qag_profile_enabled
            if profile_this:
                transfer_start = time.perf_counter()
                transfer_begin, transfer_end = cp.cuda.Event(), cp.cuda.Event()
                transfer_begin.record()
            shared = (cp.asarray(p_host), self.upload(sigma, ns), self.upload(dsigma, ns))
            if profile_this:
                shared_end = cp.cuda.Event()
                shared_end.record()
            if nq != self.low_gl[0].size:
                raise ValueError("Unexpected conditional quadrature order")
            points = self.low_gl

            # Compute low row
            low_result = cp.empty(nlow, dtype=cp.float32)
            self.kernel(
                (nlow,),
                (128,),
                (*shared, self.upload(low, nlow), *points, *self.gl, low_result,
                 np.int32(ns), np.int32(nq), np.int32(0), np.int32(heat)),
            )

            # Determine active high-table range (always GK61 when active_nr > 0, unless host mode)
            high_host = np.ctypeslib.as_array(high, shape=(nhigh,))
            if self.full_high:
                active_nr = nhigh
            elif max_d < float(high_host[0]) - 2.0e-4:
                active_nr = 0
            else:
                hi_cap = min(max_d, 0.99 * deltac) + 1.0e-4
                idx = int(np.searchsorted(high_host, hi_cap, side="right"))
                active_nr = min(nhigh, max(2, idx + 1))

            if active_nr == 0 and not (adaptive_heat and self.qag_mode == "host"):
                self.gpu_high_skipped[heat] += 1
                high_result = cp.full(nhigh, np.nan, dtype=cp.float32) if self.nan_sentinel else cp.empty(nhigh, dtype=cp.float32)
            else:
                if adaptive_heat and self.qag_mode == "host":
                    high_result = self.upload(out_high, nhigh)
                    self.native_first_heat_rows += 1
                else:
                    qag_t0 = time.perf_counter()
                    if self.panel_workspace is None or self.panel_workspace.shape[0] < nhigh:
                        self.panel_workspace = cp.empty((nhigh, 1000, 4), dtype=cp.float64)
                    if self.panel_counts_buf is None or self.panel_counts_buf.size < nhigh:
                        self.panel_counts_buf = cp.empty(nhigh, dtype=cp.int32)
                    high_result = cp.full(nhigh, np.nan, dtype=cp.float32) if self.nan_sentinel else cp.empty(nhigh, dtype=cp.float32)
                    if profile_this:
                        grid_begin = cp.cuda.Event()
                        grid_begin.record()
                    density = cp.asarray(high_host[:active_nr])
                    if profile_this:
                        transfer_end.record()
                        host_upload_section = time.perf_counter() - transfer_start
                        begin, end = cp.cuda.Event(), cp.cuda.Event()
                        begin.record()
                        mark = time.perf_counter()
                    if self.qag_mode == "serial":
                        self.qag(
                            ((active_nr + 31) // 32,),
                            (32,),
                            (*shared, density, self.rule, high_result, self.panel_workspace,
                             self.panel_counts_buf, self.qag_stats,
                             np.int32(active_nr), np.int32(ns), np.int32(heat)),
                        )
                    else:
                        self.qag_warp(
                            (active_nr,),
                            (32,),
                            (*shared, density, self.rule, high_result, self.panel_workspace,
                             self.panel_counts_buf, self.qag_stats,
                             np.int32(active_nr), np.int32(ns), np.int32(heat)),
                        )
                    if profile_this:
                        launch_seconds = time.perf_counter() - mark
                        end.record()
                        before_wait = time.perf_counter()
                        d2h_seconds = 0.0
                        if self.qag_mode != "deferred":
                            if self.pinned_count is None or self.pinned_count.size != active_nr:
                                self.pinned_count = np.frombuffer(
                                    cp.cuda.alloc_pinned_memory(4 * active_nr), dtype=np.int32, count=active_nr
                                )
                            copy_start, copy_end = cp.cuda.Event(), cp.cuda.Event()
                            copy_start.record()
                            self.panel_counts_buf[:active_nr].get(out=self.pinned_count, blocking=False)
                            copy_end.record()
                            copy_end.synchronize()
                            d2h_seconds = cp.cuda.get_elapsed_time(copy_start, copy_end) / 1000.0
                        self.qag_profiles.append(
                            (transfer_begin, transfer_end, begin, end, host_upload_section,
                             launch_seconds, time.perf_counter() - before_wait, 4 * active_nr,
                             transfer_begin, shared_end, grid_begin, d2h_seconds)
                        )
                    if heat == 0:
                        self.gpu_adaptive_ion_rows += 1
                        self.gpu_adaptive_ion_bins += active_nr
                    else:
                        self.gpu_adaptive_heat_rows += 1
                        self.gpu_adaptive_heat_bins += active_nr
                    if self.compare_native:
                        counts = self.panel_counts_buf[:active_nr].get()
                        native = self.upload(out_high, nhigh)
                        self.native_first_heat_rows += 1
                        prior = cp.empty_like(high_result)
                        self.kernel(
                            (nhigh,),
                            (128,),
                            (*shared, cp.asarray(high_host), *points, *self.gl, prior,
                             np.int32(ns), np.int32(nq), np.int32(1), np.int32(heat)),
                        )
                        arrays = cp.stack((cp.asarray(high_host[:active_nr]), native[:active_nr], prior[:active_nr], high_result[:active_nr])).get()
                        native_values = arrays[1]
                        rel = np.abs(arrays[3] - native_values) / np.maximum(np.abs(native_values), 1e-30)
                        k = int(np.argmax(rel))
                        report = dict(
                            growth=float(p_host[3]),
                            heat=int(heat),
                            active_nr=int(active_nr),
                            max_relative_error=float(rel[k]),
                            max_error_index=k,
                            max_panels=int(counts.max()),
                        )
                        print("GPU_QAG_TABLE", json.dumps(report), flush=True)
                    self.qag_host_seconds[heat] += time.perf_counter() - qag_t0


            key = tuple(C.cast(ptr, C.c_void_p).value for ptr in (out_low, out_high))
            self.device_rows[key] = (low_result, high_result)
            self.calls[heat] += 1
            self.seconds[heat] += time.perf_counter() - start
            return 0
        except Exception:
            traceback.print_exc()
            return 1
