"""Two-population feedback-dependent conditional Nion, SFRD and X-ray tables."""

import ctypes as C
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
    Table2DFloat as Table2D,
    module,
    native_tables,
    view,
)


def header():
    source = (SRC / "interp_tables.c").read_text()
    dims = re.findall(r"^#define (?:NDELTA|NMTURN|LOG10_MTURN_\w+) .*$", source, re.M)
    return "\n".join(dims) + "\n" + (ROOT / "halo_mcg_tables.h").read_text()


_KINDS = ("Nion", "SFRD", "Xray")
_POPS = ("acg", "mcg")


def table_resources():
    from halo_subgrid import _xray_device_source

    names = [f"{k}_conditional_table2D_{p}" for k in _KINDS for p in _POPS]
    names.extend(f"Nion_conditional_table2D_{p}_prev" for p in _POPS)
    addresses = native_tables(names)
    kernel = module(
        [
            SHARED / "table_qag.cuh",
            ROOT / "hmf_device.cuh",
            ROOT / "source_integrals.cu",
        ],
        _xray_device_source()
        + (ROOT / "halo_mcg_weight.cuh").read_text()
        + (ROOT / "halo_mcg_tables.cu").read_text(),
    ).get_function("population_table")
    allocate = C.CDLL(str(NATIVE)).allocate_RGTable2D_f
    allocate.argtypes = [C.c_int, C.c_int, C.POINTER(Table2D)]
    allocate.restype = None
    return (
        addresses,
        kernel,
        cp.asarray(json.loads((ROOT / "gk61.json").read_text())),
        allocate,
    )


def device_values(resources, p, nx, ny, kind, mini):
    """Evaluate one native table, leaving its completed array on device."""
    addresses, kernel, rule, _ = resources
    rows = np.tile(p, (ny, 1))
    rows[:, 10] = 10.0 ** (
        p[25] + np.arange(ny, dtype="d") / (ny - 1.0) * (p[26] - p[25])
    )
    sigma, slope = [
        Table.from_address(addresses[k])
        for k in ("Sigma_InterpTable", "dSigmasqdm_InterpTable")
    ]
    if (sigma.low, sigma.width, sigma.n) != (slope.low, slope.width, slope.n):
        raise ValueError("Incompatible native mass-variance interpolation grids")
    variance_tables = [
        cp.asarray(np.ctypeslib.as_array(t.y, (t.n,))) for t in (sigma, slope)
    ]
    xi, wi = (
        (np.uint64(0), np.uint64(0))
        if int(p[22]) != 1
        else tuple(cp.asarray(view(addresses[k], 101, "d")) for k in ("xi_GL", "wi_GL"))
    )
    values = cp.empty((nx, ny), dtype="f")
    kernel(
        (nx * ny,),
        (32,),
        (
            cp.asarray(rows),
            np.int32(nx),
            np.int32(ny),
            np.int32(kind),
            np.int32(mini),
            *variance_tables,
            np.float64(sigma.low),
            np.float64(sigma.width),
            np.int32(sigma.n),
            xi,
            wi,
            rule,
            values,
        ),
    )
    return values


def bind(adapter, report, errors, *, symbol="cuda_mcg_table", resident=None):
    callback = C.CFUNCTYPE(C.c_int, *(C.c_int,) * 5, C.POINTER(C.c_double))

    resources = None

    @callback
    def fill(kind, mini, prev, nx, ny, ptr):
        nonlocal resources
        try:
            if resources is None:
                resources = table_resources()
            addresses, _, _, allocate = resources
            name = f"{_KINDS[kind]}_conditional_table2D_{_POPS[mini]}"
            if prev:
                if kind != 0:
                    raise ValueError("Only photon history has a previous-node table")
                name += "_prev"
            table = Table2D.from_address(addresses[name])
            if not table.allocated:
                allocate(nx, ny, C.byref(table))
            if (table.nx, table.ny) != (nx, ny):
                raise ValueError(f"Invalid shape for {name}")
            p = np.ctypeslib.as_array(ptr, (30,))
            table.xmin, table.xstep = p[1], (p[2] - p[1]) / (nx - 1.0)
            table.ymin, table.ystep = p[25], (p[26] - p[25]) / (ny - 1.0)
            values = device_values(resources, p, nx, ny, kind, mini)
            if resident is not None:
                if not bool(cp.isfinite(values).all()):
                    raise ValueError(f"Nonfinite {name} at z={p[0]}")
                resident[name] = (
                    (table.xmin, table.ymin, table.xstep, table.ystep),
                    values,
                )
            else:
                host = values.get()
                if not np.isfinite(host).all():
                    raise ValueError(f"Nonfinite {name} at z={p[0]}")
                for i, row in enumerate(host):
                    np.ctypeslib.as_array(table.values[i], (ny,))[:] = row
            key = f"subgrid_{_KINDS[kind].lower()}_{_POPS[mini]}_2d_tables"
            report[key] = report.get(key, 0) + 1
            return 0
        except BaseException as exc:
            errors.append(exc)
            return 1

    C.c_void_p.in_dll(adapter, symbol).value = C.cast(fill, C.c_void_p).value
    return fill
