"""Native CHMF number, collapsed fraction, and inverse number-distribution tables."""

import ctypes as C
import json
import os
import re
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import cupy as cp
import numpy as np

from native_shim import repair_partition_rejection

HERE = Path(__file__).resolve().parent
DPtr = C.POINTER(C.c_double)
FillTables = C.CFUNCTYPE(C.c_int, C.c_int, DPtr, DPtr)
FillInverse = C.CFUNCTYPE(
    C.c_int, C.c_int, C.c_int, DPtr, DPtr, DPtr, C.POINTER(C.c_int), C.POINTER(DPtr)
)


Selector = C.CFUNCTYPE(
    C.c_int,
    C.c_ulonglong,
    C.POINTER(C.c_float),
    C.c_int,
    C.c_double,
    C.c_double,
    C.c_double,
    C.c_int,
    C.POINTER(C.c_int),
)


class NumberTable(C.Structure):
    _fields_ = [
        ("n", C.c_int),
        ("low", C.c_double),
        ("width", C.c_double),
        ("y", DPtr),
        ("allocated", C.c_bool),
    ]


class Tables(NamedTuple):
    allocate: object
    integrate: object
    select_rows: object
    rule: object
    sigma: object
    slope: object
    number: NumberTable
    fraction: NumberTable
    inverse_address: int


def _native_function(source, name):
    start = re.search(r"^(?:int|double|void) " + name + r"\(", source, re.M).start()
    return source[start : source.index("\n}", start) + 2]


def _demand_sampling(src, original):
    """Keep upstream draws and interpolation, extending exactly the rows each draw reads."""
    interp = _native_function(
        (src / "interpolation.c").read_text(), "EvaluateRGTable2D"
    )
    interp = (
        interp.replace("EvaluateRGTable2D", "catalog_interpolate")
        .replace("RGTable2D *table)", "RGTable2D *table, CatalogDemand *demand)")
        .replace(
            "    double x_table =",
            """
    catalog_require(demand, x_idx, y_idx);
    catalog_require(demand, x_idx + 1, y_idx);
    double x_table =""",
        )
    )
    inverse = _native_function(
        (src / "interp_tables.c").read_text(), "evaluate_nhalo_inverse"
    )
    inverse = (
        inverse.replace("evaluate_nhalo_inverse", "catalog_lookup")
        .replace("double prob)", "double prob, CatalogDemand *demand)")
        .replace(
            "EvaluateRGTable2D(condition, lnp, &Nhalo_inv_table)",
            "catalog_interpolate(condition, lnp, catalog_inverse_target, demand)",
        )
    )
    draw = (
        _native_function(original, "sample_dndM_inverse")
        .replace("sample_dndM_inverse", "catalog_draw")
        .replace("gsl_rng *rng)", "gsl_rng *rng, CatalogDemand *demand)")
        .replace(
            "evaluate_nhalo_inverse(condition, p_in)",
            "catalog_lookup(condition, p_in, demand)",
        )
    )
    parts = [interp, inverse, draw]
    # Upstream passes NULL for the unused zeroth-order barrier. Supply its
    # storage in the native fallback so a failed GPU window can be replayed.
    start = original.index("bool partition_rejection(")
    end = original.index("\n}", start) + 2
    rejection = repair_partition_rejection(original[start:end])
    parts.append(
        rejection.replace("partition_rejection(", "catalog_partition_rejection(")
    )
    parts.append(
        _native_function(original, "stoc_partition_sample")
        .replace("stoc_partition_sample(", "catalog_partition_sample(", 1)
        .replace("partition_rejection(", "catalog_partition_rejection(")
    )
    for name in ("stoc_halo_sample", "stoc_mass_sample", "stoc_sample"):
        function = _native_function(original, name).replace(
            name + "(", "catalog_" + name + "(", 1
        )
        function = function.replace(
            "float *M_out) {", "float *M_out, CatalogDemand *demand) {", 1
        )
        function = function.replace(
            "sample_dndM_inverse(tbl_arg, hs_constants, rng)",
            "catalog_draw(tbl_arg, hs_constants, rng, demand)",
        )
        if name == "stoc_sample":
            function = function.replace(
                "    int err;",
                """
    bool partition = hs_constants->from_catalog &&
        matter_options_global->SAMPLE_METHOD == SAMPLE_PARTITION &&
        (matter_options_global->HMF == HMF_ST || matter_options_global->HMF == HMF_PS);
    if (!demand->enabled && !partition) return stoc_sample(hs_constants, rng, n_halo_out, M_out);
    int err;""",
            )
            function = function.replace(
                "err = stoc_partition_sample(hs_constants, rng, n_halo_out, M_out);",
                "err = partition ? catalog_partition_sample(hs_constants, rng, n_halo_out, M_out)"
                " : stoc_partition_sample(hs_constants, rng, n_halo_out, M_out);",
            )
            for routine in ("stoc_halo_sample", "stoc_mass_sample"):
                function = function.replace(
                    routine + "(hs_constants, rng, n_halo_out, M_out)",
                    "catalog_"
                    + routine
                    + "(hs_constants, rng, n_halo_out, M_out, demand)",
                )
        parts.append(function)
    return "\n".join("static " + part for part in parts)


def _inverse_source(src, demand_header):
    raw = (src / "interp_tables.c").read_text()
    start = raw.index("struct rf_inv_params {")
    end = raw.index("double J_integrand(", start)
    impl = raw[start:end]
    impl = impl.replace(
        "struct rf_inv_params {",
        """struct cached_rf_params {
    /* All entries use the same row and normalization. Brent frequently requests
     * the previous quantile's lower endpoint again. */
    double mass[16], log_probability[16];
    unsigned count, cursor;""",
    ).replace("struct rf_inv_params ", "struct cached_rf_params ")
    impl = impl.replace("double dndm_inv_f(", "static double dndm_inv_f(").replace(
        "dndm_inv_f", "cached_dndm_inv_f"
    )
    impl = impl.replace(
        "    double integral =",
        """    for (unsigned i=0; i<p->count; ++i)
        if (p->mass[i] == lnM_min) return p->log_probability[i] - p->rf_target;
    double integral =""",
    )
    impl = impl.replace(
        "    return result - p->rf_target;",
        """    p->mass[p->cursor] = lnM_min;
    p->log_probability[p->cursor] = result;
    p->cursor = (p->cursor+1) % 16;
    if (p->count<16) ++p->count;
    return result - p->rf_target;""",
    )
    impl = impl.replace(
        "params_rf.rf_norm = norm;",
        "params_rf.rf_norm = norm;\n            params_rf.count = 0;\n            params_rf.cursor = 0;",
    )
    impl = impl.replace("#pragma omp for\n", "#pragma omp for schedule(dynamic,1)\n")
    impl = impl.replace(
        "void initialize_dndm_inverse_table(", "static void catalog_cached_inverse("
    )
    impl = impl.replace("Nhalo_inv_table", "(*catalog_inverse_target)")
    impl = impl.replace(
        "double param, bool from_catalog) {",
        "double param, bool from_catalog, const float *samples, size_huge count) {",
    )
    marker = "(*catalog_inverse_target).y_width = pa[1] - pa[0];"
    impl = impl.replace(
        marker,
        marker
        + """
    int needed[nx];
    if (samples) {
        if (catalog_rows_gpu(count, samples, from_catalog, growth_out, xmin,
                             (*catalog_inverse_target).x_width, nx, needed)) Throw(TableGenerationError);
    } else {
        for (int j=0; j<nx; ++j) needed[j]=1;
    }
""",
    )
    at = impl.index("for (i = 0; i < nx; i++) {\n            x = xa[i];")
    impl = impl[:at] + impl[at:].replace(
        "            x = xa[i];",
        "            if (!needed[i]) continue;\n            x = xa[i];",
        1,
    )
    iteration_limit = re.search(r"^#define MAX_ITER_RF \d+$", raw, re.M)[0]
    return (
        f"""
#include <gsl/gsl_roots.h>
#include "interpolation.h"
{iteration_limit}
RGTable2D *catalog_inverse_target;
int (*catalog_rows_gpu)(size_huge, const float *, int, double, double, double, int, int *);
"""
        + impl[: impl.index("static void catalog_cached_inverse(")]
        + demand_header
        + _demand_initializer(impl[impl.index("static void catalog_cached_inverse(") :])
    )


def _demand_initializer(impl):
    impl = impl.replace(
        "const float *samples, size_huge count)",
        "const float *samples, size_huge count, CatalogDemand *demand)",
    )
    impl = impl.replace(
        "    int needed[nx];",
        """
    demand->enabled = true;
    memcpy(demand->targets, pa, np * sizeof(*pa));
    int needed[nx];""",
    )
    impl = impl.replace(
        "if (!needed[i]) continue;",
        """
            CatalogRow *row = &demand->rows[i];
            row->first = needed[i] ? 0 : -1;
            if (!needed[i]) continue;""",
    )
    impl = impl.replace(
        "            M_cond = exp(lnM_cond);",
        """
            if (catalog_inverse_gpu) {
                row->params.lnM_cond = lnM_cond;
                row->params.sigma = sigma_cond;
                row->params.delta = delta;
                row->first = np - 1;
                continue;
            }
            M_cond = exp(lnM_cond);""",
    )
    marker = '    LOG_SUPER_DEBUG("Done.");'
    impl = impl.replace(
        marker,
        """
    if (catalog_inverse_gpu) {
        double bounds[] = {lnM_min, growth_out, physconst.delta_c_sph,
                           RHOcrit * cosmo_params_global->OMm, matter_options_global->HMF,
                           physconst.delta_c_delos};
        double conditions[nx][4];
        int active[nx];
        for (int j = 0; j < nx; ++j) {
            CatalogRow *row = &demand->rows[j];
            active[j] = row->first == np - 1;
            if (!active[j]) continue;
            conditions[j][0] = row->params.lnM_cond;
            conditions[j][1] = row->params.sigma;
            conditions[j][2] = row->params.delta;
            conditions[j][3] = MAX_DELTAC_FRAC * get_delta_crit(
                matter_options_global->HMF, row->params.sigma, growth_out);
        }
        if (catalog_inverse_gpu(nx, np, bounds, conditions[0], pa, active,
                                catalog_inverse_target->z_arr)) Throw(TableGenerationError);
        for (int j = 0; j < nx; ++j)
            if (active[j]) demand->rows[j].first = 0;
    }
"""
        + marker,
    )
    start = impl.index("            lnM_init = lnM_min;")
    end = impl.index("\n        }\n        gsl_root_fsolver_free", start)
    impl = (
        impl[:start]
        + """
            row->params = params_rf;
            row->lower = lnM_min;
            row->first = np - 1;"""
        + impl[end:]
    )
    # No solver is needed while only its native normalization/initial bounds are being saved.
    start = impl.index("        // RF stuff")
    end = impl.index("#pragma omp for", start)
    return (
        impl[:start]
        + "        struct cached_rf_params params_rf = {0};\n"
        + "        params_rf.growthf = growth_out;\n"
        + impl[end:]
    ).replace("        gsl_root_fsolver_free(solver);", "")


def source(src, original):
    demand_header, table_hooks = (
        (HERE / "halo_mass_function.h")
        .read_text()
        .split("/* Existing table hooks */", 1)
    )
    setter = original[
        original.index("void stoc_set_consts_z(") : original.index(
            "// set the constants which are calculated once per condition"
        )
    ]
    setter = setter.replace(
        "void stoc_set_consts_z(", "static void catalog_set_z("
    ).replace(
        "double redshift_desc) {",
        "double redshift_desc, const float *samples, size_huge count, CatalogDemand *demand) {",
    )
    setter = setter.replace("initialize_dndm_tables(", "catalog_dndm(")
    setter = re.sub(
        r"initialize_dndm_inverse_table\(([^;]+)\);",
        r"catalog_inverse(\1, samples, count, demand);",
        setter,
    )
    return (
        _inverse_source(src, demand_header)
        + table_hooks
        + "\n"
        + setter
        + "\n"
        + _demand_sampling(src, original)
    )


def sampling_loop(text):
    # Only the private catalogue invocation knows the complete list of possible
    # conditions. Native single_test_sample and the global setup stay intact.
    marker = "stoc_set_consts_z(&hs_constants, redshift, redshift_desc);"
    if text.count(marker) != 1:
        raise ValueError("Unknown 4.3.2 sampler condition source")
    for name in (
        "sample_halo_grids",
        "sample_halo_progenitors",
        "stochastic_halofield",
    ):
        start = text.index("int " + name + "(")
        end = text.index(") {", start)
        text = text[:end] + ", CatalogDemand *demand" + text[end:]
    text = text.replace("int stochastic_halofield(", "static int catalog_sample_node(")
    text = re.sub(
        r"stoc_sample\(&hs_constants_priv, rng_arr\[threadnum\], ([^;]+?)\);",
        r"catalog_stoc_sample(&hs_constants_priv, rng_arr[threadnum], \1, demand);",
        text,
    )
    text = re.sub(
        r"(sample_halo_(?:grids|progenitors)\(rng_stoc,[^;]*?&hs_constants)\);",
        r"\1, demand);",
        text,
    )
    text = text.replace(
        marker,
        """
       catalog_set_z(&hs_constants, redshift, redshift_desc,
                     redshift_desc > 0 ? halos_desc->halo_masses : dens_field,
                     redshift_desc > 0 ? halos_desc->n_halos : HII_TOT_NUM_PIXELS, demand);""",
    )
    text = text.replace(
        "seed_rng_threads_fast(rng_stoc, seed + seed_fac);",
        "catalog_seed_random(rng_stoc, seed + seed_fac, demand->random);",
    ).replace(
        "free_rng_threads(rng_stoc);",
        "if (!demand->random) free_rng_threads(rng_stoc);",
    )
    text = text.replace(
        "    free_dndm_tables();",
        """
    if (demand->enabled) {
        unsigned long long cells = 0;
        for (int i = 0; i < catalog_inverse_target->nx_bin; ++i) {
            if (demand->rows[i].first >= 0)
                cells += catalog_inverse_target->ny_bin - demand->rows[i].first;
        }
        catalog_inverse_work(cells);
    }
    free_dndm_tables();""",
    )
    return (
        text
        + """
int stochastic_halofield(random_huge seed, float redshift_desc, float redshift, float *dens_field,
                         float *halo_overlap_box, HaloCatalog *halos_desc, HaloCatalog *halos) {
    CatalogDemand state = {0};
    if (catalog_uses_gpu_tables()) {
        state.rows = calloc(simulation_options_global->N_COND_INTERP, sizeof(*state.rows));
        state.targets = malloc(simulation_options_global->N_PROB_INTERP * sizeof(*state.targets));
        if (!state.rows || !state.targets) {
            free(state.rows);
            free(state.targets);
            Throw(MemoryAllocError);
        }
    }
    int threads = simulation_options_global->N_THREADS;
    if (catalog_random_words && threads >= 1 && threads <= 1000) {
        state.random = calloc(threads, sizeof(*state.random));
        if (!state.random) {
            free(state.rows);
            free(state.targets);
            Throw(MemoryAllocError);
        }
    }
    int status;
    Try {
        status = catalog_sample_node(seed, redshift_desc, redshift, dens_field, halo_overlap_box,
                                     halos_desc, halos, &state);
    } Catch(status) {
        catalog_destroy_random(state.random);
        free(state.rows);
        free(state.targets);
        Throw(status);
    }
    catalog_destroy_random(state.random);
    free(state.rows);
    free(state.targets);
    return status;
}
"""
    )


def _resources():
    from evolution_backend import ROOT, SHARED, NATIVE, Table, module, native_tables

    symbols = native_tables(("Nhalo_table", "Fcoll_table", "Nhalo_inv_table"))
    native = C.CDLL(str(NATIVE))
    allocate = native.allocate_RGTable1D
    allocate.argtypes = [C.c_int, C.POINTER(NumberTable)]
    allocate.restype = None
    kernels = module(
        [
            SHARED / "table_qag.cuh",
            HERE / "hmf_device.cuh",
            HERE / "source_integrals.cu",
            HERE / "halo_mass_function.cu",
        ]
    )
    return Tables(
        allocate,
        kernels.get_function("catalog_number_and_fraction"),
        kernels.get_function("select_condition_rows"),
        cp.asarray(json.loads((ROOT / "gk61.json").read_text())),
        Table.from_address(symbols["Sigma_InterpTable"]),
        Table.from_address(symbols["dSigmasqdm_InterpTable"]),
        NumberTable.from_address(symbols["Nhalo_table"]),
        NumberTable.from_address(symbols["Fcoll_table"]),
        symbols["Nhalo_inv_table"],
    )


def fill(resources, nx, bounds, rows):
    sigma, slope = resources.sigma, resources.slope
    number, frac = resources.number, resources.fraction
    for table in (number, frac):
        if not table.allocated:
            resources.allocate(nx, C.byref(table))
        table.low = bounds[0]
        table.width = (bounds[1] - bounds[0]) / (nx - 1)
    result = cp.empty((nx, 2), dtype="d")
    resources.integrate(
        (nx,),
        (32,),
        (
            cp.asarray(bounds),
            cp.asarray(rows),
            cp.asarray(np.ctypeslib.as_array(sigma.y, shape=(sigma.n,))),
            cp.asarray(np.ctypeslib.as_array(slope.y, shape=(slope.n,))),
            np.float64(sigma.low),
            np.float64(sigma.width),
            np.int32(sigma.n),
            resources.rule,
            result,
        ),
    )
    values = result.get()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite conditional halo number or mass fraction")
    for i, table in enumerate((number, frac)):
        np.ctypeslib.as_array(table.y, shape=(nx,))[:] = values[:, i]
    return result


def bind(adapter, report, errors, catalog_tables):
    from catalog_libm import inverse_module, source_for_host
    from catalog_poisson_source import configured_gsl_source

    resources = None
    module = None
    grid_callback = C.c_void_p.in_dll(adapter, "prepare_cuda_grid")
    progenitor_callback = C.c_void_p.in_dll(adapter, "prepare_cuda_progenitors")
    math_path = source_for_host(os.environ.get("P21_CUDA_GLIBC_SOURCE"))
    if math_path is not None:
        module = inverse_module(
            math_path, cp.cuda.runtime.getDevice(), configured_gsl_source()
        )
        inverse = module.get_function("build_inverse_catalog")
    for key in (
        "halo_conditional_number_tables",
        "halo_inverse_cdf_rows",
        "halo_inverse_cdf_possible_rows",
    ):
        report.setdefault(key, 0)

    @FillTables
    def dispatch(nx, bounds, rows):
        nonlocal resources
        try:
            if resources is None:
                resources = _resources()
                C.c_void_p.in_dll(
                    adapter, "catalog_inverse_target"
                ).value = resources.inverse_address
            values = fill(
                resources,
                nx,
                np.ctypeslib.as_array(bounds, shape=(10,)),
                np.ctypeslib.as_array(rows, shape=(nx, 4)),
            )
            if grid_callback.value or progenitor_callback.value:
                catalog_tables["moments"] = (
                    values,
                    resources.number.low,
                    resources.number.width,
                )
            report["halo_conditional_number_tables"] += 1
            return 0
        except Exception as exc:
            errors.append(exc)
            return 1

    @Selector
    def select(count, values, from_catalog, growth, xmin, width, nx, target):
        try:
            used = cp.zeros(nx + 1, dtype=np.int32)
            if count:
                samples = cp.asarray(np.ctypeslib.as_array(values, shape=(count,)))
                if (progenitor_callback if from_catalog else grid_callback).value:
                    catalog_tables["samples"] = (
                        C.cast(values, C.c_void_p).value,
                        samples,
                    )
                if from_catalog and progenitor_callback.value:
                    sigma = resources.sigma
                    catalog_tables["sigma"] = (
                        cp.asarray(np.ctypeslib.as_array(sigma.y, shape=(sigma.n,))),
                        sigma.low,
                        sigma.width,
                    )
                resources.select_rows(
                    ((count + 255) // 256,),
                    (256,),
                    (
                        samples,
                        np.uint64(count),
                        np.int32(from_catalog),
                        np.float64(growth),
                        np.float64(xmin),
                        np.float64(width),
                        np.int32(nx),
                        used,
                        used[nx:],
                    ),
                )
            host = used.get()
            if host[nx]:
                host[:nx] = 1
            np.ctypeslib.as_array(target, shape=(nx,))[:] = host[:nx]
            report["halo_inverse_cdf_rows"] += int(host[:nx].sum())
            report["halo_inverse_cdf_possible_rows"] += nx
            return 0
        except Exception as exc:
            errors.append(exc)
            return 1

    C.c_void_p.in_dll(adapter, "catalog_tables_gpu").value = C.cast(
        dispatch, C.c_void_p
    ).value
    C.c_void_p.in_dll(adapter, "catalog_rows_gpu").value = C.cast(
        select, C.c_void_p
    ).value

    @FillInverse
    def fill_inverse(nx, bins, bounds, conditions, targets, active, out):
        try:
            view = np.ctypeslib.as_array
            selected = np.flatnonzero(view(active, shape=(nx,)))
            if not selected.size:
                return 0

            started = perf_counter()
            sigma, slope = resources.sigma, resources.slope
            saved_sigma = catalog_tables.get("sigma")
            sigma_gpu = (
                saved_sigma[0]
                if saved_sigma is not None
                else cp.asarray(view(sigma.y, shape=(sigma.n,)))
            )
            values = cp.zeros((len(selected), bins), dtype=np.float64)
            inverse(
                (len(selected),),
                (32,),
                (
                    cp.asarray(view(bounds, shape=(6,))),
                    cp.asarray(view(conditions, shape=(nx, 4))[selected]),
                    np.int32(bins),
                    sigma_gpu,
                    cp.asarray(view(slope.y, shape=(slope.n,))),
                    np.int32(sigma.n),
                    np.float64(sigma.low),
                    np.float64(sigma.width),
                    resources.rule,
                    values,
                    cp.asarray(view(targets, shape=(bins,))),
                ),
            )
            host = values.get()
            if not np.isfinite(host).all():
                raise ValueError("CUDA CHMF inverse-CDF root did not converge")
            for row, ratios in zip(selected, host):
                view(out[row], shape=(bins,))[:] = ratios
            report["halo_inverse_cuda_rows"] = report.get(
                "halo_inverse_cuda_rows", 0
            ) + len(selected)
            report["halo_inverse_cuda_seconds"] = (
                report.get("halo_inverse_cuda_seconds", 0.0) + perf_counter() - started
            )
            return 0
        except Exception as exc:
            errors.append(exc)
            return 1

    if math_path is not None:
        C.c_void_p.in_dll(adapter, "catalog_inverse_gpu").value = C.cast(
            fill_inverse, C.c_void_p
        ).value

    @C.CFUNCTYPE(None, C.c_ulonglong)
    def completed(cells):
        key = "halo_inverse_cdf_built_values"
        report[key] = report.get(key, 0) + cells

    C.c_void_p.in_dll(adapter, "catalog_inverse_work").value = C.cast(
        completed, C.c_void_p
    ).value
    return dispatch, select, completed, fill_inverse, module
