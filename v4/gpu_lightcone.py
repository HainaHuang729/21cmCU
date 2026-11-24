"""Run the 21cmFAST 4.3.2 lightcone drivers on CUDA.

Pass the same inputs, lightconer, and driver options as py21cmfast.run_lightcone.
Add the adapter's v4 directory to PYTHONPATH. Pass diagnostics_dir to retain the
stage counters. Use separate processes for simultaneous simulations; the upstream
C backend and its driver hooks are process-wide.
"""

import json
import logging
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import partial
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from unittest.mock import patch

import py21cmfast as p21
from py21cmfast.drivers import (
    coeval as coeval_driver,
)
from py21cmfast.drivers import (
    lightcone as driver,
)
from py21cmfast.drivers import (
    single_field,
)
from py21cmfast.lightconers import AngularLightconer, RectilinearLightconer
from py21cmfast.wrapper.outputs import OutputStruct

_active_backend = Lock()
_logger = logging.getLogger(__name__)


class _UnsupportedCUDAOptionsError(ValueError):
    """Represent valid native options outside the CUDA adapter coverage."""


def _validate(inputs, lightconer, photon_calibration=None):
    if version("21cmfast") != "4.3.2":
        raise _UnsupportedCUDAOptionsError(
            "The CUDA lightcone backend requires 21cmFAST 4.3.2"
        )
    if lightconer is not None:
        if type(lightconer) is RectilinearLightconer:
            if lightconer.line_of_sight_axis not in (-1, 0, 1, 2):
                raise _UnsupportedCUDAOptionsError(
                    "CUDA rectilinear lightcones require line_of_sight_axis in (-1, 0, 1, 2)"
                )
        elif type(lightconer) is not AngularLightconer:
            raise _UnsupportedCUDAOptionsError(
                "CUDA requires a native rectilinear or angular lightconer"
            )
    sim = inputs.simulation_options
    if sim.NON_CUBIC_FACTOR != 1 or sim.DIM % 2 or sim.HII_DIM % 2:
        raise _UnsupportedCUDAOptionsError(
            "CUDA requires even, cubic fine and coarse grids"
        )
    if sim.N_THREADS < 1:
        raise ValueError("CUDA requires N_THREADS >= 1")
    source = inputs.matter_options.SOURCE_MODEL
    cosmo = inputs.cosmo_params
    tiny = 1e-4
    if (
        cosmo.OMl <= 0
        or abs(cosmo.wl + 1) >= tiny
        or abs(cosmo.OMm + cosmo.OMl + cosmo.OMr - 1) >= 0.01
    ):
        raise _UnsupportedCUDAOptionsError(
            "CUDA requires positive Lambda with flat-like expansion (|OMm+OMl+OMr-1| < 0.01) and wl == -1"
        )
    if source not in (
        "CONST-ION-EFF",
        "E-INTEGRAL",
        "CHMF-SAMPLER",
        "DEXM-ESF",
        "L-INTEGRAL",
    ):
        raise _UnsupportedCUDAOptionsError(f"Unsupported CUDA source model: {source}")
    molecular = inputs.astro_options.USE_MCGS
    transfer = inputs.matter_options.POWER_SPECTRUM
    streaming = inputs.matter_options.V_CB_MODEL
    if transfer not in ("EH", "BBKS", "EFSTATHIOU", "PEEBLES", "WHITE", "CLASS"):
        raise _UnsupportedCUDAOptionsError(
            f"Unsupported CUDA power spectrum: {transfer}"
        )
    if molecular and (
        source not in ("E-INTEGRAL", "L-INTEGRAL", "CHMF-SAMPLER")
        or (transfer, streaming)
        not in (("EH", "AVG-DEBUG"), ("CLASS", "AVG-AUTO"), ("CLASS", "FLUCTS"))
        or not inputs.astro_options.USE_TS_FLUCT
    ):
        raise _UnsupportedCUDAOptionsError(
            "Molecular cooling requires a supported thermal source model and "
            "EH with AVG-DEBUG or CLASS with AVG-AUTO or FLUCTS"
        )
    if (
        molecular
        and inputs.matter_options.lagrangian_source_grid
        and inputs.matter_options.PERTURB_ON_HIGH_RES
        and sim.DIM != sim.HII_DIM
    ):
        raise ValueError(
            "Native molecular feedback requires the density and turnover grids to match"
        )
    if molecular and source == "E-INTEGRAL":
        methods = (
            inputs.astro_options.INTEGRATION_METHOD_ACGS,
            inputs.astro_options.INTEGRATION_METHOD_MCGS,
        )
        if any(m not in ("GAUSS-LEGENDRE", "GSL-QAG") for m in methods):
            raise _UnsupportedCUDAOptionsError(
                "Eulerian molecular feedback requires GL or QAG for both source populations"
            )
        if (
            inputs.astro_options.PHOTON_CONS_TYPE != "no-photoncons"
            or inputs.astro_options.RECOMB_MODEL != "inhomogeneous"
        ):
            raise _UnsupportedCUDAOptionsError(
                "Eulerian molecular feedback currently requires inhomogeneous recombinations and no photon calibration"
            )
    correction = inputs.astro_options.PHOTON_CONS_TYPE
    if correction == "no-photoncons":
        if photon_calibration is not None:
            raise ValueError(
                "A photon calibration requires an active photon conservation mode"
            )
    else:
        from photon_conservation import PhotonCalibration

        if not isinstance(photon_calibration, PhotonCalibration):
            raise _UnsupportedCUDAOptionsError(
                "Supply a PhotonCalibration with paired native calibration samples"
            )
        photon_calibration.validate(inputs)
    matter = dict(
        HMF="ST",
        USE_INTERPOLATION_TABLES="hmf-interpolation",
        V_CB_MODEL=streaming if molecular else "NONE",
        USE_FFTW_WISDOM=False,
    )
    if inputs.matter_options.FILTER not in ("spherical-tophat", "gaussian"):
        raise _UnsupportedCUDAOptionsError(
            "CUDA mass variance requires a top-hat or Gaussian filter"
        )
    if inputs.matter_options.PERTURB_ALGORITHM not in ("LINEAR", "ZELDOVICH", "2LPT"):
        raise ValueError("Unknown 4.3.2 perturbation algorithm")
    astro = dict(
        PHOTON_CONS_TYPE=correction,
        USE_MCGS=molecular,
        USE_REIONIZATION_PHOTOHEATING_FEEDBACK=molecular,
        IONISE_ENTIRE_SPHERE=False,
    )
    homogeneous = inputs.astro_options.RECOMB_MODEL == "homogeneous"
    if source == "E-INTEGRAL":
        if inputs.astro_options.INTEGRATION_METHOD_ACGS not in (
            "GAUSS-LEGENDRE",
            "GSL-QAG",
        ):
            raise _UnsupportedCUDAOptionsError(
                "CUDA Eulerian mass-dependent sources require Gauss–Legendre or GSL-QAG quadrature"
            )
        astro.update(
            RECOMB_MODEL="homogeneous"
            if homogeneous
            else ("inhomogeneous" if inputs.astro_options.USE_TS_FLUCT else "none"),
            USE_METALLICITY=False,
            USE_UPPER_STELLAR_TURNOVER=False,
            USE_EXP_FILTER=False,
            CELL_RECOMB=homogeneous,
        )
        if inputs.astro_options.USE_TS_FLUCT:
            astro.update(
                USE_X_RAY_HEATING=True,
                USE_LYA_HEATING=True,
            )
    elif source == "CONST-ION-EFF":
        # The conditional collapse fraction is EPS. Its unconditional normalization
        # follows the requested halo mass function over the native finite mass interval.
        if inputs.matter_options.HMF not in ("PS", "ST"):
            raise _UnsupportedCUDAOptionsError(
                "CUDA constant efficiency supports PS and ST halo mass functions"
            )
        matter["HMF"] = inputs.matter_options.HMF
        astro.update(USE_METALLICITY=False, USE_UPPER_STELLAR_TURNOVER=False)
        if homogeneous:
            astro["CELL_RECOMB"] = True
        elif inputs.astro_options.RECOMB_MODEL not in ("none", "inhomogeneous"):
            raise ValueError("Unknown recombination model for constant efficiency")
        if inputs.astro_options.USE_TS_FLUCT:
            astro.update(
                USE_X_RAY_HEATING=True,
                USE_LYA_HEATING=True,
            )
    else:
        if source == "CHMF-SAMPLER" and inputs.matter_options.SAMPLE_METHOD not in (
            "MASS-LIMITED",
            "NUMBER-LIMITED",
            "BINARY-SPLIT",
            "PARTITION",
        ):
            raise ValueError("Unknown 4.3.2 CHMF progenitor sampling method")
        thermal = inputs.astro_options.USE_TS_FLUCT
        if thermal:
            astro.update(
                USE_X_RAY_HEATING=True,
                USE_LYA_HEATING=True,
            )
        astro.update(
            USE_TS_FLUCT=thermal,
            RECOMB_MODEL="inhomogeneous" if thermal else "none",
            HII_FILTER="spherical-tophat",
            USE_EXP_FILTER=True,
            CELL_RECOMB=True,
            USE_METALLICITY=True,
            USE_UPPER_STELLAR_TURNOVER=source != "L-INTEGRAL",
        )
    wrong = [
        f"{name}={getattr(options, name)!r} (requires {expected!r})"
        for options, required in (
            (inputs.matter_options, matter),
            (inputs.astro_options, astro),
        )
        for name, expected in required.items()
        if getattr(options, name) != expected
    ]
    if p21.config["EXTRA_EMISSIVITY_FIELDS"]:
        wrong.append("EXTRA_EMISSIVITY_FIELDS=True (requires False)")
    if wrong:
        raise _UnsupportedCUDAOptionsError(
            "Unsupported CUDA lightcone options: " + ", ".join(wrong)
        )


@dataclass
class _Timings:
    perturb: list[float] = field(default_factory=list)
    brightness: list[float] = field(default_factory=list)


@contextmanager
def _backend(
    inputs,
    lightconer=None,
    diagnostics_dir=None,
    audit=False,
    photon_calibration=None,
    *,
    fallback_to_cpu=False,
):
    from py21cmfast.drivers._global_initialization import (
        _GlobalInitManagerSingleton as manager,
    )

    try:
        _validate(inputs, lightconer, photon_calibration)
    except _UnsupportedCUDAOptionsError as exc:
        if not fallback_to_cpu or photon_calibration is not None:
            raise
        with manager._lock:
            if _active_backend.locked():
                raise RuntimeError(
                    "Start the CPU fallback outside the active CUDA backend"
                ) from exc
            _logger.warning("Using the native 21cmFAST CPU driver: %s", exc)
            if diagnostics_dir is not None:
                directory = Path(diagnostics_dir)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "backend-routing.json").write_text(
                    json.dumps({"backend": "cpu", "reason": str(exc)}, indent=2) + "\n"
                )
            yield None
        return

    import cupy as cp
    from brightness import brightness_temperature_gpu
    from device_registry import DeviceFieldRegistry
    from evolution_backend import (
        active_emissivity_history,
        evolution_backend,
        lagrangian_backend,
    )
    from lightcone_device import (
        DriverReductions,
        EmissivityHistory,
        LightconeVelocityWorkspace,
        apply_rsds,
        emissivity_history,
        make_lightcone_slices,
        optical_gradient,
    )
    from perturb import perturb_field_gpu

    with manager._lock, ExitStack() as stack:
        if not _active_backend.acquire(blocking=False):
            raise RuntimeError("A CUDA lightcone is already active in this process")
        stack.callback(_active_backend.release)
        if manager._scopes:
            raise RuntimeError(
                "Start the CUDA lightcone outside active 21cmFAST C-state scopes"
            )
        directory = (
            Path(diagnostics_dir)
            if diagnostics_dir is not None
            else Path(stack.enter_context(TemporaryDirectory()))
        )
        directory.mkdir(parents=True, exist_ok=True)
        # A CPU run can leave initialized tables in the upstream singleton. Clear
        # each backend with its own destructors, before restoring the other one.
        manager.free()
        source = inputs.matter_options.SOURCE_MODEL
        lagrangian = inputs.matter_options.lagrangian_source_grid
        report = directory / "gpu-stages.json"
        if lagrangian and audit:
            raise ValueError(
                "The Eulerian thermal audit requires SOURCE_MODEL=E-INTEGRAL"
            )
        setup_start = perf_counter()
        engine = stack.enter_context(
            lagrangian_backend(report, source, inputs=inputs)
            if lagrangian
            else evolution_backend(
                report,
                audit=audit,
                thermal=inputs.astro_options.USE_TS_FLUCT,
                photon_calibration=photon_calibration,
                molecular=inputs.astro_options.USE_MCGS,
            )
        )
        if lagrangian:
            engine.counters["backend_setup_seconds"] = perf_counter() - setup_start
        registry = DeviceFieldRegistry()
        engine.device_registry = registry
        engine._pending_outputs = {}
        engine._record_pending = False
        stack.callback(registry.clear)
        stack.callback(manager.free)

        orig_compute = OutputStruct._compute

        def compute_with_registry(self, allow_already_computed=False, *args, **kwargs):
            previous = engine._record_pending
            engine._record_pending = True
            try:
                out = orig_compute(self, allow_already_computed, *args, **kwargs)
                pending = engine._pending_outputs
                if pending:
                    for array in self.arrays.values():
                        val = array._value
                        if val is not None and array.state.computed_in_mem:
                            dev = pending.pop(int(val.ctypes.data), None)
                            if dev is not None:
                                registry.record(val, dev)
                    pending.clear()
                return out
            finally:
                engine._pending_outputs.clear()
                engine._record_pending = previous

        stack.enter_context(
            patch.object(OutputStruct, "_compute", compute_with_registry)
        )
        timings = _Timings()
        reductions = DriverReductions(registry)
        stack.enter_context(patch.object(driver, "np", reductions))
        stack.enter_context(patch.object(single_field, "np", reductions))
        stack.enter_context(patch.object(coeval_driver, "np", reductions))
        stack.enter_context(
            patch.object(single_field, "interp_emissivity_fields", emissivity_history)
        )
        radiation = single_field.compute_radiation_fields._func
        history_counts = Counter()

        def radiation_with_history(**kwargs):
            options = inputs.astro_options
            workspace = EmissivityHistory(
                history_counts,
                share_scattering_spectra=options.USE_MCGS
                and options.LYA_MULTIPLE_SCATTERING,
                # The native debug summaries read the intermediate host arrays.
                keep_radiation_grids=logging.getLogger(
                    "py21cmfast.wrapper.outputs"
                ).getEffectiveLevel()
                > logging.DEBUG,
                registry=registry,
            )
            token = active_emissivity_history.set(workspace)
            try:
                with patch.object(
                    single_field,
                    "interp_emissivity_fields",
                    partial(emissivity_history, workspace=workspace),
                ):
                    return radiation(**kwargs)
            finally:
                active_emissivity_history.reset(token)

        if inputs.astro_options.USE_TS_FLUCT:
            stack.enter_context(
                patch.object(
                    single_field.compute_radiation_fields,
                    "_func",
                    radiation_with_history,
                )
            )
            stack.callback(
                lambda: (directory / "emissivity-history.json").write_text(
                    json.dumps(history_counts, indent=2)
                )
            )
        if lightconer is not None:

            def rectilinear_slices(lc_self, c1, c2):
                yield from make_lightcone_slices(lc_self, c1, c2, registry=registry)

            stack.enter_context(
                patch.object(
                    RectilinearLightconer,
                    "make_lightcone_slices",
                    rectilinear_slices,
                )
            )
            if type(lightconer) is AngularLightconer:
                from angular_slices import AngularShellGeometry, make_angular_slices

                angular_counts = Counter()
                angular_shells = None

                def angular(geometry, c1, c2):
                    nonlocal angular_shells
                    if (
                        angular_shells is None
                        or angular_shells.lightconer is not geometry
                    ):
                        angular_shells = AngularShellGeometry(geometry)
                    for name, idx, data in make_angular_slices(
                        geometry, c1, c2, angular_shells, registry=registry
                    ):
                        if name is not None:
                            angular_counts[name] += 1
                        yield name, idx, data

                stack.enter_context(
                    patch.object(AngularLightconer, "make_lightcone_slices", angular)
                )
                stack.callback(
                    lambda: (directory / "angular-slices.json").write_text(
                        json.dumps(angular_counts, indent=2)
                    )
                )
            velocity_workspace = LightconeVelocityWorkspace()
            stack.enter_context(
                patch.object(
                    driver,
                    "_apply_rsds",
                    partial(apply_rsds, workspace=velocity_workspace),
                )
            )
            stack.enter_context(
                patch.object(
                    driver,
                    "_apply_dvdr_in_tau21",
                    partial(optical_gradient, workspace=velocity_workspace),
                )
            )

        def perturb(*, redshift, initial_conditions, inputs):
            start = perf_counter()
            if not engine.ic_fields:
                # A supplied or cached IC bypasses ComputeInitialConditions. Load
                # its actual arrays once and retain them across perturbations.
                fields = p21.PerturbedField.new(inputs=inputs, redshift=redshift)
                for name in fields.get_required_input_arrays(initial_conditions):
                    arr = initial_conditions.get(name)
                    dev = cp.asarray(arr)
                    engine.ic_fields[name] = dev
                    registry.record(arr, dev)
            result = perturb_field_gpu(
                redshift=redshift,
                initial_conditions=initial_conditions,
                inputs=inputs,
                strict_fft=True,
                resident=engine,
            )
            timings.perturb.append(perf_counter() - start)
            return result

        def brightness(**kwargs):
            start = perf_counter()
            result = brightness_temperature_gpu(**kwargs, registry=registry)
            timings.brightness.append(perf_counter() - start)
            return result

        # Retain upstream validation, cache loads and writes, and public aliases.
        stack.enter_context(patch.object(single_field.perturb_field, "_func", perturb))
        stack.enter_context(
            patch.object(single_field.brightness_temperature, "_func", brightness)
        )
        try:
            yield timings
        finally:
            if lagrangian:
                engine.counters["gpu_matter_perturbations"] = len(timings.perturb)
                engine.counters["gpu_brightness"] = len(timings.brightness)


def run_lightcone(
    *,
    inputs,
    lightconer,
    diagnostics_dir=None,
    photon_calibration=None,
    fallback_to_cpu: bool = True,
    **kwargs,
):
    """Return a native LightCone using the supplied parameters and upstream run options."""
    with _backend(
        inputs,
        lightconer,
        diagnostics_dir,
        photon_calibration=photon_calibration,
        fallback_to_cpu=fallback_to_cpu,
    ):
        return p21.run_lightcone(inputs=inputs, lightconer=lightconer, **kwargs)


def generate_lightcone(
    *,
    inputs,
    lightconer,
    diagnostics_dir=None,
    photon_calibration=None,
    fallback_to_cpu: bool = True,
    **kwargs,
):
    """Yield upstream lightcone node results. Close the generator when stopping early."""
    with _backend(
        inputs,
        lightconer,
        diagnostics_dir,
        photon_calibration=photon_calibration,
        fallback_to_cpu=fallback_to_cpu,
    ):
        yield from p21.generate_lightcone(
            inputs=inputs, lightconer=lightconer, **kwargs
        )


def run_coeval(
    *,
    inputs,
    out_redshifts=(),
    diagnostics_dir=None,
    photon_calibration=None,
    fallback_to_cpu: bool = True,
    **kwargs,
):
    """Return native Coeval boxes using the supplied parameters and upstream run options."""
    with _backend(
        inputs,
        None,
        diagnostics_dir,
        photon_calibration=photon_calibration,
        fallback_to_cpu=fallback_to_cpu,
    ):
        return coeval_driver.run_coeval(
            inputs=inputs, out_redshifts=out_redshifts, **kwargs
        )


def generate_coeval(
    *,
    inputs,
    out_redshifts=(),
    diagnostics_dir=None,
    photon_calibration=None,
    fallback_to_cpu: bool = True,
    **kwargs,
):
    """Yield native Coeval boxes. Close the generator when stopping early."""
    with _backend(
        inputs,
        None,
        diagnostics_dir,
        photon_calibration=photon_calibration,
        fallback_to_cpu=fallback_to_cpu,
    ):
        yield from coeval_driver.generate_coeval(
            inputs=inputs, out_redshifts=out_redshifts, **kwargs
        )


def save_lightcone(cone, path, **kwargs):
    """Persist a 4.3.2 LightCone using GPU or parallel LZF HDF5 compression."""
    from lightcone_io import save_lightcone as _save

    return _save(cone, path, **kwargs)
