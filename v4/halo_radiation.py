"""Radiation and spin-temperature hooks consuming 4.3.2 Lagrangian emissivity grids."""

import ctypes as C
import logging
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from py21cmfast.drivers import _global_initialization as initialization
from py21cmfast.wrapper.outputs import RadiationFields, RadiationFieldsSetup, TsBox

from evolution_backend import (
    CB,
    Evolution,
    bind_source_scaling,
    compile_adapter,
    install_entry,
    radiation_setup_source,
    radiation_source,
    shared_source_scaling,
    thermal_source,
)


_STAGES = {
    1: "thermal_update",
    4: "radiation_shell_filter",
    5: "radiation_shell_accumulation",
    14: "radiation_field_normalization",
    15: "radiation_ionization_interpolation",
    17: "radiation_xray_optical_depth_and_frequency",
    18: "radiation_global_nion",
    33: "mcg_xray_optical_depth_and_frequency",
    34: "radiation_lya_scattering_filter",
    21: "radiation_lyman_spectral_factors",
    26: "thermal_initial_conditions",
    27: "thermal_constants",
}


@contextmanager
def halo_radiation_backend(inputs, counts, *, scaling=None, parent_engine=None):
    """Apply the radiation kernels to the native halo emissivity history."""
    options = inputs.astro_options
    if (
        inputs.matter_options.SOURCE_MODEL
        not in ("CHMF-SAMPLER", "L-INTEGRAL", "DEXM-ESF")
        or not options.USE_TS_FLUCT
    ):
        raise ValueError("Unsupported Lagrangian radiation or spin-temperature model")

    heat, cuda, scalars = thermal_source(False)
    engine = Evolution(cuda, scalars, False, scattering_reuse=options.USE_MCGS)
    if parent_engine is not None:
        engine.device_registry = parent_engine.device_registry
        engine._pending_outputs = parent_engine._pending_outputs

    @CB
    def dispatch(stage, size, params, arrays):
        if stage not in _STAGES:
            engine.error = ValueError(f"Unexpected Lagrangian radiation stage {stage}")
            return 1
        if parent_engine is not None:
            engine._record_pending = parent_engine._record_pending
            engine.device_registry = parent_engine.device_registry
            engine._pending_outputs = parent_engine._pending_outputs
        status = engine.dispatch(stage, size, params, arrays)
        if status == 0:
            name = _STAGES[stage]
            counts[name] = counts.get(name, 0) + 1
            if stage == 34 and hasattr(engine, "scattering_window"):
                counts["radiation_lya_scattering_window_reuses"] = (
                    engine.scattering_window.reuses
                )
            if options.USE_MCGS and stage in (1, 5, 14):
                key = {
                    1: "mcg_spin_lw",
                    5: "mcg_lw_shells",
                    14: "mcg_lw_normalizations",
                }[stage]
                counts[key] = counts.get(key, 0) + 1
        return status

    with TemporaryDirectory() as directory, ExitStack() as stack:
        # The homogeneous ACG Nion table uses capped stellar/escape power laws.
        # The X-ray shell instead consumes the actual luminosity density in
        # 10^38 erg/s/Mpc^3, including its metallicity dependence.
        sources = {
            "heat.c": heat,
            "rad.c": radiation_source(),
            "rad_setup.c": radiation_setup_source(),
            "dispatch.c": "int (*gpu_dispatch)(int,int,double*,void**);",
        }
        if options.USE_MCGS:
            import halo_mcg_global

            sources["rad_setup.c"] = halo_mcg_global.source(sources["rad_setup.c"])
        if scaling is not None:
            sources["rad_setup.c"] = shared_source_scaling(sources["rad_setup.c"])
        adapter = compile_adapter(Path(directory), sources)
        if scaling is not None:
            bind_source_scaling(adapter, scaling)
        if options.USE_MCGS:
            _global_callback = halo_mcg_global.bind(
                adapter, counts, lambda exc: setattr(engine, "error", exc)
            )
        C.c_void_p.in_dll(adapter, "gpu_dispatch").value = C.cast(
            dispatch, C.c_void_p
        ).value
        engine.tabulate_xhii_inverse_widths = adapter.tabulate_xhii_inverse_widths
        engine.tabulate_xhii_inverse_widths.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
            C.c_int,
        ]
        engine.tabulate_xhii_inverse_widths.restype = None
        if not options.USE_MCGS:
            previous = initialization.lib

            class HeatLibrary:
                def __getattr__(self, name):
                    return getattr(previous, name)

                def init_heat(self):
                    # Electron energy-deposition arrays are shared with the native
                    # ABI; RECFAST, collisions and Lyman tables are handled by CUDA.
                    engine.initialize_secondaries()
                    return 0

                def destruct_heat(self):
                    pass

            stack.enter_context(patch.object(initialization, "lib", HeatLibrary()))
        install_entry(
            stack, engine, adapter, RadiationFieldsSetup, "SetupRadiationFields"
        )
        for cls, name in (
            (RadiationFields, "UpdateRadiationFields"),
            (TsBox, "ComputeTsBox"),
        ):
            install_entry(stack, engine, adapter, cls, name)
        yield
