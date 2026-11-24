"""Compare one two-population radiation setup and thermal step using saved native coevals."""

import argparse
import ctypes as C
from contextlib import contextmanager
from tempfile import TemporaryDirectory
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import py21cmfast as p21
from py21cmfast.drivers import single_field as sf
from py21cmfast.drivers._global_initialization import (
    _GlobalInitManagerSingleton as manager,
)
from functools import partial
from verify_chmf_lightcone import read_native_field

from evolution_backend import Evolution, compile_adapter
from halo_radiation import halo_radiation_backend
from lightcone_device import emissivity_history


@contextmanager
def check_cutoffs(diagnostics):
    # These native oracle calls run only in the standalone accuracy checker.
    source = """
    #include <math.h>
    #include "heating_helper_progs.h"
    void reference_cutoffs(int n, double *p, double *shells, double *turns,
                           double *combined, double *atomic) {
        ScalingConstants sc;
        set_scaling_constants(p[0], &sc, false);
        double m = log10(sc.mturn_acg_homogeneous), floor_hz = p[14]*p[17];
        for (int i=0; i<n; i++) {
            combined[i] = fmax(floor_hz, nu_tau_one_mcg(p[0], shells[i], p[1], m, turns[i], &sc));
            atomic[i] = fmax(floor_hz, nu_tau_one_acg(p[0], shells[i], p[1], m, &sc));
        }
    }
    """
    dispatch = Evolution.dispatch
    with TemporaryDirectory() as tmp:
        shim = compile_adapter(Path(tmp), {"cutoffs.c": source})
        oracle = shim.reference_cutoffs
        oracle.argtypes = [
            C.c_int,
            C.POINTER(C.c_double),
            C.c_void_p,
            C.c_void_p,
            np.ctypeslib.ndpointer(np.float64),
            np.ctypeslib.ndpointer(np.float64),
        ]
        oracle.restype = None

        def checked(engine, stage, size, p, a):
            if stage != 33:
                return dispatch(engine, stage, size, p, a)
            n = int(p[5])
            native, atomic = np.empty(n), np.empty(n)
            oracle(n, p, a[0], a[7], native, atomic)
            tau = engine.tau

            def measured(grid, block, args):
                tau(grid, block, args)
                actual = args[-1].get()
                np.testing.assert_allclose(actual, native, rtol=1e-12, atol=0.0)
                diagnostics.update(
                    shells=n,
                    above_threshold=int((native > p[14] * p[17]).sum()),
                    changed_by_molecular_sources=int((native != atomic).sum()),
                    max_relative_error=float(np.max(np.abs(actual / native - 1))),
                    max_molecular_change_Hz=float(np.max(np.abs(native - atomic))),
                    native_range_Hz=[float(native.min()), float(native.max())],
                )

            with patch.object(engine, "tau", measured):
                return dispatch(engine, stage, size, p, a)

        with patch.object(Evolution, "dispatch", checked):
            yield


@contextmanager
def radiation_backend(inputs, counts):
    if inputs.matter_options.lagrangian_source_grid:
        with halo_radiation_backend(inputs, counts):
            yield
    else:
        from evolution_backend import evolution_backend

        with TemporaryDirectory() as tmp:
            report = Path(tmp) / "stages.json"
            with evolution_backend(report, molecular=True):
                yield
            counts.update(
                json.loads(report.with_name("backend-routing.json").read_text())[
                    "counters"
                ]
            )


def replay(directory, z):
    inputs = p21.LightCone.from_file(directory / "cpu.h5").inputs

    read = partial(read_native_field, directory)

    epochs = [e for e in inputs.node_redshifts if e >= z]
    previous_epoch = min(e for e in epochs if e > z)
    previous = read("TsBox", previous_epoch)
    previous_ion = (
        read("IonizedBox", previous_epoch)
        if inputs.astro_options.LYA_MULTIPLE_SCATTERING
        else None
    )
    history = [read("EmissivityFields", e) for e in epochs]
    ic, pf = read("InitialConditions"), read("PerturbedField", z)
    ref_r, ref_t = read("RadiationFields", z), read("TsBox", z)
    opts = dict(write=False, regenerate=True)

    def setup():
        return sf.setup_radiation_fields(
            redshift=z,
            emissivity_fields_list=history,
            previous_spin_temp=previous,
            **opts,
        )

    reference = setup()
    counts, cutoffs = {}, {}
    manager.free()
    with (
        check_cutoffs(cutoffs),
        radiation_backend(inputs, counts),
        patch.object(sf, "interp_emissivity_fields", emissivity_history),
    ):
        try:
            current = setup()
            rad = sf.compute_radiation_fields(
                redshift=z,
                rad_setup=current,
                emissivity_fields_list=history,
                perturbed_field=pf,
                previous_ionize_box=previous_ion,
                previous_spin_temp=previous,
                **opts,
            )
            thermal = sf.compute_spin_temperature(
                initial_conditions=ic,
                perturbed_field=pf,
                inputs=inputs,
                radiation_fields=rad,
                previous_spin_temp=previous,
                **opts,
            )
        finally:
            manager.free()

    metrics = {}

    def compare(name, ref, result, rtol):
        a, b = (np.asarray(field.get(name), dtype="d") for field in (ref, result))
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Invalid {name} grid")
        diff = np.abs(a - b)
        peak = float(np.max(np.abs(a)))
        metrics[name] = dict(
            n=a.size,
            different=int(np.count_nonzero(diff)),
            peak=peak,
            max_abs=float(diff.max()),
            rtol=rtol,
            failed=int((diff > rtol * np.abs(a)).sum()),
        )

    for name in ("inverse_diff", "m_xHII_low_box", "inverse_val_box"):
        compare(name, reference, current, 0.0)
    for name in ("freq_int_heat_tbl", "freq_int_ion_tbl", "freq_int_lya_tbl"):
        compare(name, reference, current, 1e-12)
    for name in (
        "lya_flux_continuum_prefactor_acg",
        "lya_flux_injected_prefactor_acg",
        "lya_flux_continuum_prefactor_mcg",
        "lya_flux_injected_prefactor_mcg",
        "lyw_flux_prefactor_acg",
        "lyw_flux_prefactor_mcg",
    ):
        compare(name, reference, current, 1e-12)
    for ref, got in ((ref_r, rad), (ref_t, thermal)):
        for name, array in ref.arrays.items():
            if array.state.on_disk:
                compare(name, ref, got, 1e-5)
    passed = bool(
        counts.get("mcg_xray_optical_depth_and_frequency")
        and counts.get("radiation_mcg_global_nion")
        and metrics["freq_int_heat_tbl"]["peak"]
        and metrics["J_21_LW"]["peak"]
        and all(x["failed"] == 0 for x in metrics.values())
    )
    return dict(
        job=os.getenv("SLURM_JOB_ID"),
        cpu_job=json.loads((directory / "cpu.json").read_text())["job"],
        redshift=z,
        fields=metrics,
        counters=counts,
        cutoffs=cutoffs,
        passed=passed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("redshift", type=float)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = replay(args.directory, args.redshift)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    raise SystemExit(0 if result["passed"] else 1)
