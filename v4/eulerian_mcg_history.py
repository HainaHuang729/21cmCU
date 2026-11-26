"""Eulerian molecular conditional photon fields for the current excursion radius."""

import ctypes as C
import re

import cupy as cp
import numpy as np

import halo_mcg_tables
from evolution_backend import ROOT, module, view


def source(native):
    """Insert after the Eulerian backend has selected its atomic-only table function."""
    marker = "void setup_integration_tables("
    start = native.index(marker)
    end = native.index("void calculate_conditional_nion(", start)
    block = native[start:end]
    pattern = r"(gpu_init_nion|initialize_nion_conditional_tables)\(([^;]+),\s*(true|false)\);"

    def build(match):
        original, args, prev = match.groups()
        return (
            f"if (use_mcg_conditional_gpu()) gpu_pair_tables({args},0,{prev});"
            f" else if (astro_options_global->USE_MCGS) initialize_nion_conditional_tables({args},{prev});"
            f" else {original}({args},{prev});"
        )

    block, n = re.subn(pattern, build, block)
    if n != 2:
        raise ValueError("Unknown current/previous conditional ionization table setup")
    header = (
        halo_mcg_tables.header()
        .replace("cuda_mcg_table", "cuda_e_ion_table")
        .replace(
            "source_model_uses_lagrangian_grids(matter_options_global->SOURCE_MODEL)",
            "(matter_options_global->SOURCE_MODEL == SOURCE_MODEL_E_INTEGRAL && uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES))",
        )
    )
    native = (
        native[:start]
        + header
        + "\nint (*cuda_e_ion_history)(int,int,int,double*,void**);\n"
        + block
        + native[end:]
    )
    start = native.index("void calculate_conditional_nion(")
    pos = native.index("    double nion_total_acg", start)
    hook = """
    if (use_mcg_conditional_gpu()) {
        int n=HII_TOT_NUM_PIXELS;
        bool need_mcg=consts->need_mcgs_nion;
        bool photo=astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK &&
                   source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL);
        size_t offset=need_mcg ? (size_t)rspec->R_index*n : 0;
        bool prev=need_mcg && (previous_ionize_box->nion_unconditional_mcg + previous_ionize_box->nion_unconditional_acg > 1.e-4);
        double p[]={-1.+FRACT_FLOAT_ERR,consts->scale_consts.atomic_cooling_threshold*1.001,
                    consts->nion_prefactor_acg,consts->nion_prefactor_mcg,
                    photo,need_mcg,consts->scale_consts.mturn_acg_homogeneous};
        void *a[]={fg_struct->deltax_filtered,(need_mcg && prev) ? fg_struct->prev_deltax_filtered : NULL,
                   photo ? fg_struct->log10_mturn_acg_grid_filtered : NULL,
                   need_mcg ? fg_struct->log10_mturn_mcg_grid_filtered : NULL,
                   need_mcg ? previous_ionize_box->nion_conditional_filtered_acg+offset : NULL,
                   need_mcg ? previous_ionize_box->nion_conditional_filtered_mcg+offset : NULL,
                   astro_options_global->USE_TS_FLUCT ? fg_struct->xe_filtered : NULL,
                   consts->filter_recombinations ? fg_struct->N_rec_filtered : NULL,
                   box->nion_conditional_filtered_acg+offset,
                   need_mcg ? box->nion_conditional_filtered_mcg+offset : NULL,
                   &rspec->nion_conditional_ave_acg,&rspec->nion_conditional_ave_mcg};
        if(cuda_e_ion_history(simulation_options_global->HII_DIM,HII_D_PARA,prev,p,a)) Throw(ValueError);
        return;
    }
"""
    return native[:pos] + hook + native[pos:]


def bind(adapter, report, set_error, *, engine=None):
    """Keep per-radius tables on this context's device through the grid lookup."""
    resources = None
    kernel = None
    tables = {}
    axes = {}
    table_cb = C.CFUNCTYPE(C.c_int, *(C.c_int,) * 5, C.POINTER(C.c_double))
    history_cb = C.CFUNCTYPE(
        C.c_int, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_double), C.POINTER(C.c_void_p)
    )

    def count(name):
        report[name] = report.get(name, 0) + 1

    @table_cb
    def build(kind, mini, previous, nx, ny, ptr):
        nonlocal resources
        try:
            if kind != 0:
                raise ValueError(
                    "Only ionizing photon tables enter the excursion barrier"
                )
            if resources is None:
                resources = halo_mcg_tables.table_resources()
            p = np.ctypeslib.as_array(ptr, (30,))
            key = (previous, mini)
            values = halo_mcg_tables.device_values(resources, p, nx, ny, kind, mini)
            if not bool(cp.isfinite(values).all()):
                raise ValueError("Nonfinite conditional ionization table")
            tables[key] = values
            axes[key] = (
                p[1],
                (p[2] - p[1]) / (nx - 1.0),
                p[25] if ny > 1 else 0.0,
                (p[26] - p[25]) / (ny - 1.0) if ny > 1 else 0.0,
                float(nx),
                float(ny),
            )
            count("ion_mcg_previous_tables" if previous else "ion_mcg_current_tables")
            return 0
        except BaseException as exc:
            set_error(exc)
            return 1

    @history_cb
    def update(side, depth, previous, p, a):
        nonlocal kernel
        try:
            if kernel is None:
                kernel = module([ROOT / "eulerian_mcg_history.cu"]).get_function(
                    "ion_history"
                )
            pars = np.ctypeslib.as_array(p, (7,))
            photo, need_mcg = bool(pars[4]), bool(pars[5])
            null = np.uint64(0)
            if need_mcg:
                keys = [(0, 0), (0, 1)] + (
                    [(1, 0), (1, 1)] if previous else []
                )
                rows = [tables[k] for k in keys]
                if not previous:
                    rows.extend((null, null))
                axis_rows = [axes[k] for k in keys]
                if not previous:
                    axis_rows.extend(([0.0] * 6, [0.0] * 6))
            else:
                keys = [(0, 0)]
                rows = [tables[(0, 0)], null, null, null]
                axis_rows = [axes[(0, 0)], *([0.0] * 6 for _ in range(3))]
            dev_axes = cp.asarray(np.asarray(axis_rows, dtype="d"))
            n = side * side * depth
            stride = 2 * (depth // 2 + 1)

            def host(i):
                if i in (4, 5, 8, 9):
                    return view(a[i], n, "f")
                return view(a[i], side * side * stride, "f").reshape(
                    side, side, stride
                )[:, :, :depth]

            ion_filtered = {} if engine is None else engine.ion_filtered
            registry = None if engine is None else engine.device_registry

            def load(i):
                if not a[i]:
                    return null
                ptr = int(a[i])
                res = ion_filtered.get(ptr)
                if res is not None and res.size == n and res.dtype == cp.float32:
                    return res.reshape(-1)
                if i in (4, 5) and registry is not None:
                    cached = registry.lookup_ptr(ptr, n, "f")
                    if cached is not None:
                        return cached
                return cp.asarray(host(i), order="C")

            fields = [load(i) for i in range(8)]
            out_a = cp.empty(n, dtype="f")
            out_m = cp.empty(n, dtype="f") if need_mcg else null
            bad = cp.zeros(1, "i")
            kernel(
                ((n + 255) // 256,),
                (256,),
                (
                    np.int32(n),
                    dev_axes,
                    cp.asarray(pars),
                    *rows,
                    *fields,
                    out_a,
                    out_m,
                    bad,
                ),
            )
            if bad.item():
                raise ValueError(
                    "Filtered density or turnover outside native conditional table"
                )
            residency = engine is not None and engine.ion_residency_enabled
            for j in (0, 6, 7):
                if a[j]:
                    ptr = int(a[j])
                    if ptr in ion_filtered:
                        ion_filtered[ptr] = fields[j]
                    if not (residency and ptr in ion_filtered):
                        host(j)[:] = fields[j].get().reshape(host(j).shape)
            outputs = [out_a] + ([out_m] if need_mcg else [])
            for j, x in enumerate(outputs):
                ptr = int(a[8 + j])
                if residency:
                    ion_filtered[ptr] = x
                # Keep previous_ionize_box->nion_conditional_filtered_* valid for later epochs.
                x.get(out=host(8 + j))
                mean = x.mean(dtype="d").item()
                if not np.isfinite(mean):
                    raise ValueError("Nonfinite conditional ionization grid")
                view(a[10 + j], 1, "d")[0] = mean
            if not need_mcg:
                view(a[11], 1, "d")[0] = 0.0
            count("ion_mcg_radius_history")
            if previous:
                count("ion_mcg_radius_previous_correction")
            tables.clear()
            axes.clear()
            return 0
        except BaseException as exc:
            set_error(exc)
            return 1

    for name, cb in (("cuda_e_ion_table", build), ("cuda_e_ion_history", update)):
        C.c_void_p.in_dll(adapter, name).value = C.cast(cb, C.c_void_p).value
    return build, update
