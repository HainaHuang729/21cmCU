"""Shared single-turnover ACG conditional integrals for Eulerian and CHMF grids."""

import ctypes as C
import json
import re
from pathlib import Path

import cupy as cp
import numpy as np


def conditional_table(p, kernel, symbols, rule, allocate, ntable, audit_errors=None):
    from evolution_backend import Table, view

    kind = int(p[18])
    name = (
        "Nion_conditional_table1D_acg",
        "SFRD_conditional_table1D_acg",
        "Xray_conditional_table1D_acg",
        "Nion_conditional_table1D_acg",
        "SFRD_conditional_table1D_acg",
    )[kind]
    target = Table.from_address(symbols[name])
    if audit_errors is None:
        if not target.allocated:
            allocate(ntable, C.byref(target))
        target.low = p[1]
        target.width = (p[2] - p[1]) / (target.n - 1)
    host = np.ctypeslib.as_array(target.y, shape=(target.n,))
    values = cp.empty(target.n, dtype="f")
    if kind in (3, 4):
        from constant_efficiency import fill

        fill(p, values, sfr=kind == 4)
    else:
        sigma, slope = [
            Table.from_address(symbols[k])
            for k in ("Sigma_InterpTable", "dSigmasqdm_InterpTable")
        ]
        if sigma.n != slope.n or sigma.low != slope.low or sigma.width != slope.width:
            raise ValueError("Incompatible mass-variance grids")
        xray = kind == 2
        adaptive_only = (int(p[22]) != 1) if xray else (int(p[19]) != 1)
        # Only Gauss-Legendre reads the xi_GL / wi_GL abscissae and weights.
        xi, wi = (
            (np.uint64(0), np.uint64(0))
            if adaptive_only
            else tuple(
                cp.asarray(view(symbols[k], 101, "d")) for k in ("xi_GL", "wi_GL")
            )
        )
        kernel(
            (target.n,),
            (32,),
            (
                cp.asarray(np.ctypeslib.as_array(p, shape=(25 if xray else 22,))),
                np.int32(target.n),
                np.int32(kind),
                cp.asarray(np.ctypeslib.as_array(sigma.y, shape=(sigma.n,))),
                cp.asarray(np.ctypeslib.as_array(slope.y, shape=(slope.n,))),
                np.float64(sigma.low),
                np.float64(sigma.width),
                np.int32(sigma.n),
                xi,
                wi,
                rule,
                values,
            ),
        )
    result = values.get()
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite {name} at z={p[0]}")
    if audit_errors is not None:
        delta = abs(result.astype("d") - host)
        if not np.isfinite(delta).all():
            raise ValueError(f"Nonfinite native {name} at z={p[0]}")
        row = audit_errors.setdefault(name, dict(tested=0, failed=0, max_abs=0.0))
        row["tested"] += target.n
        row["failed"] += int(np.sum(delta > 1e-8 + 1e-4 * abs(host.astype("d"))))
        row["max_abs"] = max(row["max_abs"], float(delta.max()))
    host[:] = result


def source(native):
    # Retain the installed implementation for models outside the CUDA table's
    # input contract, including the Gamma approximation.
    import halo_mcg_tables

    marker = "void set_halo_properties("
    header = (
        Path(__file__).with_name("source_integrals.h").read_text()
        + "\n"
        + Path(__file__).with_name("halo_xray.h").read_text()
    )
    guard = """source_model_uses_lagrangian_grids(matter_options_global->SOURCE_MODEL) &&
               !astro_options_global->USE_MCGS &&
               !astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK &&
               uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)"""
    begin = native.index("int add_integral_contribution(")
    end = native.index("\n}", begin)
    body = native[begin:end]
    for orig, gpu, kind in (
        ("initialize_nion_conditional_tables", "gpu_init_nion", 0),
        ("initialize_sfrd_conditional_tables", "gpu_init_sfr", 1),
        ("initialize_xray_emissivity_conditional_tables", "gpu_init_xray", 2),
    ):
        pattern = rf"{orig}\(([^;]+)\);"

        # Nion alone has a previous-table argument after the scaling constants.
        def call(match):
            args = match[1]
            common, previous = args.rsplit(",", 1) if kind == 0 else (args, "false")
            use_gpu = (
                "use_gpu_subgrid() && astro_options_global->INTEGRATION_METHOD_ACGS != INTEGRATION_METHOD_GAMMA_APPROX"
                if kind == 2
                else "use_gpu_subgrid()"
            )
            return (
                f"if (use_mcg_conditional_gpu()) gpu_pair_tables({common},{kind},{previous});"
                f" else if ({use_gpu}) {gpu}({args}); else {orig}({args});"
            )

        body, count = re.subn(pattern, call, body)
        if count != 1:
            raise ValueError("Unknown ACG conditional table call")
    native = native[:begin] + body + native[end:]
    injected = (
        """
int (*cuda_subgrid_table)(double *);
#define GPU_TABLE_AUDIT 0
#define GPU_CALL(ID, P, A) do { if (cuda_subgrid_table(P)) Throw(ValueError); } while (0)
"""
        + header
        + "\n#undef GPU_CALL\n#undef GPU_TABLE_AUDIT\n"
        + f"static int atomic_hmf_table(void) {{ return {guard}; }}\n"
        + """static int use_gpu_subgrid(void) { return atomic_hmf_table() &&
                    (astro_options_global->INTEGRATION_METHOD_ACGS==INTEGRATION_METHOD_GSL_QAG ||
                     astro_options_global->INTEGRATION_METHOD_ACGS==INTEGRATION_METHOD_GAUSS_LEGENDRE ||
                     astro_options_global->INTEGRATION_METHOD_ACGS==INTEGRATION_METHOD_GAMMA_APPROX); }
"""
    )
    return native.replace(marker, injected + halo_mcg_tables.header() + marker, 1)


def _xray_device_source():
    from halo_ionization import _device_function

    functions = []
    for signature in (
        "double log_scaling_single_PL(",
        "double log_scaling_PL_limit(",
        "double scaling_double_PL(",
        "double lx_on_sfr_doublePL(",
        "void get_halo_metallicity(",
    ):
        body = _device_function("scaling_relations.c", signature)
        if signature.startswith("void "):
            body = body.replace(
                "double *z_out)", "double *z_out, double seconds_per_year)"
            )
            body = body.replace("physconst.s_per_yr", "seconds_per_year").replace(
                "pow(10,", "pow(10.,"
            )
        functions.append(body)
    return "\n".join(functions) + Path(__file__).with_name("halo_xray.cu").read_text()


def bind(adapter, report, errors):
    from evolution_backend import (
        NATIVE,
        ROOT,
        SHARED,
        SRC,
        Table,
        native_tables,
        module,
    )

    callback_type = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_double))
    allocate = C.CDLL(str(NATIVE)).allocate_RGTable1D_f
    allocate.argtypes = [C.c_int, C.POINTER(Table)]
    ntable = int(
        re.search(r"#define NDELTA (\d+)", (SRC / "interp_tables.c").read_text())[1]
    )
    names = ("subgrid_nion_tables", "subgrid_sfrd_tables", "subgrid_xray_tables")
    for name in names:
        report.setdefault(name, 0)
    resources = None
    kernels = {}

    @callback_type
    def fill(p):
        nonlocal resources
        try:
            xray = int(p[18]) == 2
            if resources is None:
                resources = (
                    native_tables(("Xray_conditional_table1D_acg",)),
                    cp.asarray(json.loads((ROOT / "gk61.json").read_text())),
                )
            if xray not in kernels:
                # Keep nonthermal runs on the existing cached Nion/SFRD module.
                program = module(
                    [
                        SHARED / "table_qag.cuh",
                        ROOT / "hmf_device.cuh",
                        ROOT / "source_integrals.cu",
                    ],
                    _xray_device_source() if xray else "",
                )
                kernels[xray] = program.get_function(
                    "xray_conditional_table" if xray else "source_table"
                )
            conditional_table(p, kernels[xray], *resources, allocate, ntable)
            report[names[int(p[18])]] += 1
            return 0
        except Exception as exc:
            errors.append(exc)
            return 1

    C.c_void_p.in_dll(adapter, "cuda_subgrid_table").value = C.cast(
        fill, C.c_void_p
    ).value
    import halo_mcg_tables

    return fill, halo_mcg_tables.bind(adapter, report, errors)
