"""Exercise configured 4.3.2 radiation and heating tables at one redshift."""

import argparse
import ctypes as C
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cupy as cp
import numpy as np
import py21cmfast as p21
from py21cmfast.drivers import single_field
from py21cmfast.drivers._global_initialization import (
    _GlobalInitManagerSingleton as manager,
)
from py21cmfast.drivers._global_initialization import c_state

from evolution_backend import (
    NATIVE,
    SHARED,
    evolution_backend,
    module,
    native_tables,
    view,
)
from halo_radiation import halo_radiation_backend

FIELDS = (
    "lya_flux_continuum_prefactor_acg",
    "lya_flux_injected_prefactor_acg",
    "freq_int_heat_tbl",
    "freq_int_ion_tbl",
    "freq_int_lya_tbl",
)
SECONDARIES = (
    "x_int_fheat",
    "x_int_n_Lya",
    "x_int_nion_HI",
    "x_int_nion_HeI",
    "x_int_nion_HeII",
)


def make_fixture(original, target):
    target.mkdir()
    (target / "x_int_tables").mkdir()
    for name, factors, rows in (
        ("stellar_spectra.dat", (1.0, 1.5, 1.0, 1.0, 1.0), 22),
        ("Lyman_alpha_heating_table.dat", (1.5, 0.75), 101 * 101 * 51),
        ("recfast_LCDM.dat", (1.0, 1.25, 1.0, 0.875), 501),
    ):
        vals = np.loadtxt(original / name, max_rows=rows)
        fmt = (
            ["%d", "%.17g", "%.17g", "%.17g", "%.17g"]
            if name.startswith("stellar")
            else "%.17g"
        )
        np.savetxt(target / name, vals * factors, fmt=fmt)
    for path in (original / "x_int_tables").glob("*.dat"):
        header = "".join(path.read_text().splitlines(keepends=True)[:3])
        vals = np.loadtxt(path, skiprows=3)
        vals[:, [2, 4, 5, 6, 7]] *= [1.125, 0.875, 1.25, 0.75, 1.5]
        with (target / "x_int_tables" / path.name).open("w") as f:
            f.write(header)
            np.savetxt(f, vals, fmt="%.17g")


def inputs_for(model):
    return p21.InputParameters(
        random_seed=31703, node_redshifts=[35.0, 10.22, 10.0]
    ).evolve_input_structs(
        SOURCE_MODEL=model,
        DIM=16,
        HII_DIM=8,
        BOX_LEN=80.0,
        N_THREADS=1,
        USE_TS_FLUCT=True,
        R_MAX_TS=90.0,
        N_STEP_TS=8,
        USE_MCGS=False,
        USE_REIONIZATION_PHOTOHEATING_FEEDBACK=False,
        USE_METALLICITY=model == "L-INTEGRAL",
        USE_UPPER_STELLAR_TURNOVER=False,
        USE_EXP_FILTER=model == "L-INTEGRAL",
        RECOMB_MODEL="inhomogeneous",
    )


def radiation(inputs):
    dims = (8, 8, 8)
    spin = p21.TsBox.new(inputs=inputs, redshift=10.22)
    spin.set(
        "xray_ionised_fraction",
        np.linspace(0.0004, 0.12, np.prod(dims), dtype="f").reshape(dims),
    )
    history = []
    for z in [35.0, 10.0]:
        box = p21.EmissivityFields.new(inputs=inputs, redshift=z)
        box.set("sfrd_acg", np.full(dims, 1.0e-12, dtype="f"))
        history.append(box)
    rad = single_field.setup_radiation_fields(
        redshift=10.0, emissivity_fields_list=history, previous_spin_temp=spin
    )
    return {name: rad.get(name).copy() for name in FIELDS}


def secondaries():
    symbols = native_tables()
    flat = [view(symbols[k], 14 * 258, "f").copy() for k in SECONDARIES]
    return np.concatenate(
        [
            view(symbols["x_int_Energy"], 258, "f"),
            view(symbols["x_int_XHII"], 14, "f"),
            *flat,
        ]
    )


def heating(engine=None):
    # Use physical interior points of the legacy 101 x 101 x 51 interpolation grid.
    tk, ts, tau = 12.7, 10.9, 1.3e5
    z = 23.25
    if engine is None:
        lib = C.CDLL(str(NATIVE))
        lya = lib.Energy_Lya_heating
        lya.argtypes = [C.c_double] * 3 + [C.c_int]
        lya.restype = C.c_double
        rec = []
        for func in (lib.xion_RECFAST, lib.T_RECFAST):
            func.argtypes = [C.c_float, C.c_int]
            func.restype = C.c_double
            rec.append(func(z, 0))
        return np.array([lya(tk, ts, tau, 2), lya(tk, ts, tau, 3), *rec])
    kernel = module(
        [SHARED / "glibc_math.cuh", SHARED / "update_stabilized.cu"],
        """
    extern "C" __global__ void table_sample(double tk,double ts,double tau,
        const double *a,const double *b,double *out) {
        out[0]=Energy_Lya_heating(tk,ts,tau,2,a,b);
        out[1]=Energy_Lya_heating(tk,ts,tau,3,a,b);
    }""",
    ).get_function("table_sample")
    out = cp.empty(2, "d")
    kernel(
        (1,), (1,), (np.float64(tk), np.float64(ts), np.float64(tau), *engine.lya, out)
    )
    return np.concatenate([out.get(), engine.recfast(z).get()])


def compare(reference, actual, rtol):
    np.testing.assert_allclose(actual, reference, rtol=rtol, atol=0)
    valid = reference != 0
    return dict(
        values=reference.size,
        max_relative=float(np.max(np.abs(actual[valid] / reference[valid] - 1)))
        if np.any(valid)
        else 0.0,
    )


def run(out):
    root = Path(p21.config["external_table_path"])
    info = {"version": p21.__version__, "cases": {}}
    with TemporaryDirectory(prefix="v432-tables-") as tmp:
        tmp = Path(tmp)
        custom = tmp / "input"
        make_fixture(root, custom)
        for model in ("E-INTEGRAL", "L-INTEGRAL"):
            inputs = inputs_for(model)
            case = {}
            info["cases"][model] = case
            baseline = None
            for label, folder in (("packaged", root), ("modified", custom)):
                with p21.config.use(external_table_path=folder):
                    manager.free()
                    with c_state(inputs, heat=True):
                        cpu = radiation(inputs)
                        cpu["secondary_arrays"] = secondaries()
                        if model == "E-INTEGRAL":
                            cpu["heating_and_recfast"] = heating()
                    manager.free()
                    counters = {}
                    ctx = (
                        evolution_backend(tmp / "stages.json")
                        if model == "E-INTEGRAL"
                        else halo_radiation_backend(inputs, counters)
                    )
                    with ctx as engine:
                        with c_state(inputs, heat=True):
                            gpu = radiation(inputs)
                            gpu["secondary_arrays"] = secondaries()
                            if model == "E-INTEGRAL":
                                gpu["heating_and_recfast"] = heating(engine)
                        manager.free()
                    details = {
                        k: compare(
                            ref, gpu[k], 0.0 if k == "secondary_arrays" else 1e-11
                        )
                        for k, ref in cpu.items()
                    }
                    if baseline is None:
                        baseline = cpu
                    else:
                        for key in cpu:
                            if np.array_equal(cpu[key], baseline[key]):
                                raise ValueError(
                                    f"Modified fixture did not change {key}"
                                )
                            details[key]["changed_vs_packaged"] = int(
                                np.count_nonzero(cpu[key] != baseline[key])
                            )
                        if model == "E-INTEGRAL":
                            np.testing.assert_allclose(
                                cpu["heating_and_recfast"]
                                / baseline["heating_and_recfast"],
                                [1.5, 0.75, 1.25, 0.875],
                                rtol=1e-6,
                            )
                    case[label] = details
    out.write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
