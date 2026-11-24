"""The paired native redshift and turnover-dependent unconditional Nion histories."""

import ctypes as C
from functools import lru_cache
import json
import re

import cupy as cp
import numpy as np

from evolution_backend import (
    NATIVE,
    ROOT,
    SHARED,
    SRC,
    Table,
    Table2DDouble,
    module,
    native_tables,
    view,
)


def source(native):
    """Insert into the radiation setup shim after the existing ACG row helper."""
    marker = "void setup_z_edges("
    definitions = re.findall(
        r"^#define (?:NMTURN|LOG10_MTURN_MCG_\w+) .*$",
        (SRC / "interp_tables.c").read_text(),
        re.M,
    )
    native = native.replace(
        marker,
        "\n".join(definitions)
        + "\n"
        + (ROOT / "halo_nion_rows.h").read_text()
        + "\n"
        + (ROOT / "halo_mcg_global.h").read_text()
        + "\n"
        + marker,
        1,
    )
    call = "initialize_nion_unconditional_tables(zpp_interp_points_SFR,"
    if native.count(call) != 1 or marker not in native:
        raise ValueError("Unknown molecular radiation Nion history call")
    return native.replace(
        call,
        """if (source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL) &&
            astro_options_global->USE_MCGS) gpu_mcg_nion_history(zpp_interp_points_SFR,
                determine_zpp_min,determine_zpp_max,&sc);
            else """
        + call,
        1,
    )


@cp.memoize(for_each_device=True)
def _resources():
    from halo_ionization import _device_function

    scaling = "\n".join(
        _device_function("scaling_relations.c", sig)
        for sig in ("double log_scaling_single_PL(", "double log_scaling_PL_limit(")
    )
    program = module(
        [
            SHARED / "table_qag.cuh",
            ROOT / "hmf_device.cuh",
            ROOT / "source_integrals.cu",
            ROOT / "halo_mcg_weight.cuh",
        ],
        scaling + (ROOT / "halo_mcg_global.cu").read_text(),
    )
    lib = C.CDLL(str(NATIVE))
    alloc1 = lib.allocate_RGTable1D
    alloc1.argtypes = [C.c_int, C.c_void_p]
    alloc1.restype = None
    alloc2 = lib.allocate_RGTable2D
    alloc2.argtypes = [C.c_int, C.c_int, C.POINTER(Table2DDouble)]
    alloc2.restype = None
    return (
        program.get_function("mcg_nion_history"),
        native_tables(("Nion_z_table_mcg",)),
        cp.asarray(json.loads((ROOT / "gk61.json").read_text())),
        alloc1,
        alloc2,
    )


@lru_cache(maxsize=2)
def _sigma_arrays(device, sigma, slope):
    # The native singleton can free/rebuild its tables at the same address.
    # A small content key retains the device copy only for identical inputs.
    return tuple(cp.asarray(np.frombuffer(data, dtype="f")) for data in (sigma, slope))


def evaluate(rows, turns):
    """Integrate the supplied ACG and MCG rows against the current native sigma tables."""
    kernel, symbols, rule, _, _ = _resources()
    n, nt = len(rows), len(turns)
    sigma, slope = [
        Table.from_address(symbols[name])
        for name in ("Sigma_InterpTable", "dSigmasqdm_InterpTable")
    ]
    if (sigma.n, sigma.low, sigma.width) != (slope.n, slope.low, slope.width):
        raise ValueError("Incompatible native mass-variance tables")
    vals = _sigma_arrays(
        cp.cuda.runtime.getDevice(),
        *(np.ctypeslib.as_array(t.y, (t.n,)).tobytes() for t in (sigma, slope)),
    )
    out = cp.empty((n, nt + 1), dtype="d")
    kernel(
        (out.size,),
        (32,),
        (
            cp.asarray(rows),
            cp.asarray(turns),
            np.int32(nt),
            *vals,
            np.float64(sigma.low),
            np.float64(sigma.width),
            np.int32(sigma.n),
            rule,
            out,
        ),
    )
    data = out.get()
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite molecular Nion history")
    return data


def bind(adapter, counts, set_error):
    ptr = C.POINTER(C.c_double)
    cb = C.CFUNCTYPE(C.c_int, C.c_int, C.c_int, ptr, ptr, ptr)

    @cb
    def run(n, nt, axes, rows, turns):
        try:
            _, symbols, _, alloc1, alloc2 = _resources()
            data = evaluate(
                np.ctypeslib.as_array(rows, (n, 2, 25)),
                np.ctypeslib.as_array(turns, (nt,)),
            )
            acg = Table.from_address(symbols["Nion_z_table_acg"])
            mcg = Table2DDouble.from_address(symbols["Nion_z_table_mcg"])
            if not acg.allocated:
                alloc1(n, C.byref(acg))
            if not mcg.allocated:
                alloc2(n, nt, C.byref(mcg))
            if acg.n != n or (mcg.nx, mcg.ny) != (n, nt):
                raise ValueError("Invalid native history table sizes")
            acg.low, acg.width, mcg.ymin, mcg.ystep = axes[:4]
            mcg.xmin, mcg.xstep = acg.low, acg.width
            view(acg.y, n, "d")[:] = data[:, 0]
            for i in range(n):
                np.ctypeslib.as_array(mcg.values[i], (nt,))[:] = data[i, 1:]
            name = "radiation_mcg_global_nion"
            counts[name] = counts.get(name, 0) + 1
            return 0
        except BaseException as exc:
            set_error(exc)
            return 1

    C.c_void_p.in_dll(adapter, "cuda_mcg_nion_history").value = C.cast(
        run, C.c_void_p
    ).value
    return run
