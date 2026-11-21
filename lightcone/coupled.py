"""Pass perturbation density directly to the resident heat filter."""
import ctypes as C
import os
import cupy as cp
import numpy as np

from gpu_perturb import PerturbRuntime
from guarded_resident import ResidentRuntime
from ion_runtime import IonRuntime
from brightness_runtime import BrightnessRuntime
from heat_seed import HeatSeed
from ic_runtime import ICRuntime
from ic_seed_runtime import SeededIC
from conditional_runtime import ConditionalTables
from sigma_runtime import SigmaMoments
from xray_runtime import XRayTables
from mhr_runtime import MHRTable
from hmf_runtime import GlobalCollapse
from host_backed import LatestSnapshots, resident_or_host

class Perturbation(PerturbRuntime):
    def __init__(self, native):
        super().__init__(native, shadow=False)
        self.fields = {}
        self.calls = 0
        self.velocity_fields = {}

    def run(self, z, fine, coarse, dk, first, second, initial_growth,
            dlog_growth_dt, inputs, out_density, out_velocity):
        status = super().run(z, fine, coarse, dk, first, second, initial_growth,
                             dlog_growth_dt, inputs, out_density, out_velocity)
        if status == 0 and os.environ.get("PS_FFTW_PERTURB_DENSITY") == "1":
            return 4 if os.environ.get("PS_FFTW_PERTURB_VEL", "1") == "1" else 3
        return status

    def compute(self, z, fine, coarse, dk, first, second, initial_growth,
                dlog_growth_dt, inputs, out_density, out_velocity):
        super().compute(z, fine, coarse, dk, first, second, initial_growth,
                        dlog_growth_dt, inputs, out_density, out_velocity)
        if self.density is not None:
            self.fields[C.cast(out_density, C.c_void_p).value] = self.density
        if self.velocity_z is not None:
            self.velocity_fields[C.cast(out_velocity,C.c_void_p).value]=self.velocity_z
        self.calls += 1

class GSLDirectIC(ICRuntime):
    """Keep native GSL Fourier weights and enable CUDA moments after IC capture."""
    def __init__(self, native, sigma):
        self.sigma = sigma
        self.sigma_setter = sigma.lib.SetGPUSigmaMoments
        self.sigma_setter(None)
        self.used_native_normalization = False
        super().__init__(native)

    def run(self, *args):
        status = super().run(*args)
        self.used_native_normalization = status == 0
        if os.environ.get('PS_NATIVE_SIGMA') != '1':
            self.sigma_setter(C.cast(self.sigma.cb,C.c_void_p))
        return status

class CoupledRuntime(ResidentRuntime):
    def install(self, native, cached_ic=False, same_gsl=False):
        super().install(native)
        self.perturbation = Perturbation(native)
        self.states = LatestSnapshots()
        self.perturbation.fields = LatestSnapshots()
        self.perturbation.velocity_fields = LatestSnapshots()
        if os.environ.get("PS_FFTW_PERTURB_DENSITY") == "1":
            self.perturbation.install()
            ptr = C.POINTER(C.c_float)
            observer_type = C.CFUNCTYPE(C.c_int, C.c_int, ptr, ptr)
            def capture(side, density, velocity):
                try:
                    self.perturbation.fields.pop(C.cast(density, C.c_void_p).value, None)
                    self.perturbation.velocity_fields.pop(C.cast(velocity, C.c_void_p).value, None)
                    return 0
                except Exception:
                    import traceback
                    traceback.print_exc()
                    return 1
            self._native_perturb_trace = observer_type(capture)
            setter=self.perturbation.lib.SetPerturbTraceObserver
            setter.argtypes=[C.c_void_p]
            setter.restype=None
            setter(C.cast(self._native_perturb_trace,C.c_void_p))
        else:
            self.perturbation.install()
        self.resident_filter_inputs = 0
        self.ionization = IonRuntime(self)
        self.spin_fields=LatestSnapshots()
        self.lib.SetGPUHeatTableCache.argtypes=[C.c_int]
        self.lib.SetGPUHeatTableCache.restype=None
        self.lib.SetGPUHeatTableCache(1)
        self.lib.SetPSKeepFFTWPlans.argtypes=[C.c_int]
        self.lib.SetPSKeepFFTWPlans.restype=None
        self.lib.SetPSKeepFFTWPlans(1)
        self.heat_seed=HeatSeed(self)
        self.tables=ConditionalTables(native)
        self.sigma=SigmaMoments(native)
        self.xray=XRayTables(native)
        self.mhr=MHRTable(native)
        self.hmf=GlobalCollapse(native)
        if cached_ic:
            self.ic=None
        elif same_gsl:
            self.ic=GSLDirectIC(native,self.sigma)
        else:
            self.ic=SeededIC(native)
        self.ic_rng=("CPU-generated HDF5" if cached_ic else
                     "Production CPU GSL Fourier modes" if same_gsl else "CuPy/cuRAND seeded Gaussian")
        for variable, setter in (
            ("PS_NATIVE_CONDITIONAL", "SetGPUConditionalTable"),
            ("PS_NATIVE_SIGMA", "SetGPUSigmaMoments"),
            ("PS_NATIVE_XRAY", "SetGPUXRayTables"),
            ("PS_NATIVE_MHR", "SetGPUMHRTable"),
            ("PS_NATIVE_COLLAPSE", "SetGPUGlobalCollapse"),
        ):
            if os.environ.get(variable) == "1":
                getattr(self.lib, setter)(None)
        self.perturbation.ic=self.ic
        self.brightness=BrightnessRuntime(self)

    def compute(self,a):
        status=super().compute(a)
        if status==0:
            k=(C.cast(a.out_x,C.c_void_p).value,float(a.scalars[0]))
            self.spin_fields[C.cast(a.out_ts,C.c_void_p).value]=self.states[k][2]
        return status

    def filter_input(self, density, size):
        address = C.cast(density, C.c_void_p).value
        field = resident_or_host(self.perturbation.fields, density, size)
        if field is None or field.size != size:
            raise RuntimeError("Heat filter did not receive a GPU perturbation field")
        self.resident_filter_inputs += 1
        return field

    def report(self):
        cache_stats=(C.c_ulonglong*2)()
        self.lib.GetGPUHeatTableCacheStats.argtypes=[C.POINTER(C.c_ulonglong)]
        self.lib.GetGPUHeatTableCacheStats.restype=None
        self.lib.GetGPUHeatTableCacheStats(cache_stats)
        return {**super().report(),
                "evicted_snapshots": {key: cache.evictions for key, cache in {
                    "heat": self.states, "spin": self.spin_fields, "density": self.perturbation.fields,
                    "velocity": self.perturbation.velocity_fields, "ion_history": self.ionization.history,
                    "neutral": self.ionization.neutral_fields, "brightness": self.brightness.fields,
                }.items()},
                "host_heating_tables":{"loads":cache_stats[0],"cache_hits":cache_stats[1]},
                "gpu_perturbation_calls": self.perturbation.calls,
                "resident_filter_inputs": self.resident_filter_inputs,
                "host_filter_uploads": 0, "ionization": self.ionization.report(),
                "gpu_ic_grid_seconds":self.ic.seconds if self.ic is not None else [],
                "gpu_ic_seed_seconds":getattr(self.ic,'seed_seconds',[]),
                "host_ic_uploads":self.perturbation.host_ic_uploads,
                "host_ic_upload_bytes":self.perturbation.host_ic_upload_bytes,
                "host_ic_upload_seconds":self.perturbation.first_upload_seconds,
                "gpu_conditional_tables":{"calls":self.tables.calls,"seconds":self.tables.seconds,"device_rows_consumed":self.tables.device_reads},
                "gpu_sigma":{"calls":self.sigma.calls,"launches":self.sigma.launches,"seconds":self.sigma.seconds},
                "gpu_xray":{"calls":self.xray.calls,"seconds":self.xray.seconds, "adaptive_entries":self.xray.adaptive_entries, "adaptive_unique_entries":self.xray.adaptive_unique_entries,"max_adaptive_intervals":self.xray.adaptive_interval_max, "adaptive_fallbacks":self.xray.adaptive_fallbacks},
                "gpu_mhr":{"calls":self.mhr.calls,"seconds":self.mhr.seconds,"gpu_pdf_rows":self.mhr.pdf_rows},
                "gpu_global_collapse":{"calls":self.hmf.calls,"evaluations":self.hmf.evaluations,"seconds":self.hmf.seconds},
                "gpu_ic_rng":self.ic_rng,
                "ic_native_sigma_normalization":getattr(self.ic,"used_native_normalization",False),
                "initial_gpu_heat_steps":self.heat_seed.calls,
                "gpu_brightness_steps":len(self.brightness.seconds),
                "gpu_brightness_seconds":sum(self.brightness.seconds),
                "brightness_seed_uploads":self.brightness.seed_uploads}
