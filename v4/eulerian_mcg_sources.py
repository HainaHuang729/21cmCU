"""Eulerian per-cell molecular source integrals and native mean normalization."""

import ctypes as C
import json
import re
from types import SimpleNamespace

import cupy as cp
import numpy as np

import halo_mcg_tables
import halo_mcg_turnover
from evolution_backend import ROOT, SHARED, Table, module, native_tables, view
from halo_advection import _MOLECULAR_TABLES
from halo_mcg_global import _sigma_arrays
from halo_subgrid import _xray_device_source


def source(native, *, audit=False):
    """Extend the composed 4.3.2 Eulerian emitter after its atomic hooks are installed."""
    header = (
        halo_mcg_tables.header()
        .replace(
            "source_model_uses_lagrangian_grids(matter_options_global->SOURCE_MODEL)",
            """matter_options_global->SOURCE_MODEL == SOURCE_MODEL_E_INTEGRAL &&
        uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)""",
        )
        .replace("cuda_mcg_table", "cuda_eulerian_emissivity_table")
    )
    start = native.index("int add_integral_contribution(")
    end = native.index("\n}", start)
    body = native[start:end]

    for func, kind in (
        ("gpu_init_nion", 0),
        ("gpu_init_sfr", 1),
        ("gpu_init_xray", 2),
    ):

        def hook(match):
            args = match[1].strip()
            if kind != 0:
                args += ",false"
            pars, prev = args.rsplit(",", 1)
            originals = (
                "initialize_nion_conditional_tables",
                "initialize_sfrd_conditional_tables",
                "initialize_xray_emissivity_conditional_tables",
            )
            oracle = f"{originals[kind]}({match[1]});" if audit else ""
            return (
                (
                    "if (use_mcg_conditional_gpu() && astro_options_global->INTEGRATION_METHOD_ACGS != INTEGRATION_METHOD_GAMMA_APPROX && (!astro_options_global->USE_MCGS || astro_options_global->INTEGRATION_METHOD_MCGS != INTEGRATION_METHOD_GAMMA_APPROX))"
                    if kind == 2
                    else "if (use_mcg_conditional_gpu())"
                )
                + f" {{ {oracle} gpu_pair_tables({pars},{kind},{prev}); }} else "
                + match[0]
            )

        body, count = re.subn(rf"{func}\(([^;]+)\);", hook, body)
        if count != 1:
            raise ValueError(f"Missing Eulerian conditional source table call {func}")
    pos = body.index("    move_integral_emissivities(")
    body = (
        body[:pos]
        + """
    if (use_mcg_conditional_gpu() && ev_consts->fix_mean) {
        eulerian_molecular_fields(M_min,M_max,dens_pointer,log10_mturn_acg_grid,
                                  log10_mturn_mcg_grid,ev_consts,emissivity_fields);
        free_conditional_tables();
        return 0;
    }
"""
        + body[pos:]
    )
    return halo_mcg_turnover.source(
        native[:start]
        + header
        + "\n"
        + (ROOT / "eulerian_mcg_sources.h").read_text()
        + "\n"
        + body
        + native[end:]
    )


@cp.memoize(for_each_device=True)
def _kernels():
    # Use the same two-population source weights and bilinear interpolation as
    # the Lagrangian emissivity solver, without moving or depositing cells.
    interpolation = (ROOT / "halo_advection.cu").read_text().split('extern "C"')[0]
    extra = (
        _xray_device_source()
        + (ROOT / "halo_mcg_weight.cuh").read_text()
        + (ROOT / "halo_mcg_tables.cu").read_text()
        + interpolation
        + (ROOT / "eulerian_mcg_sources.cu").read_text()
    )
    return module(
        [
            SHARED / "table_qag.cuh",
            ROOT / "hmf_device.cuh",
            ROOT / "source_integrals.cu",
        ],
        extra,
    )


def bind(adapter, counts, set_error):
    errors = SimpleNamespace(append=set_error)
    resident = {}
    tables = halo_mcg_tables.bind(
        adapter,
        counts,
        errors,
        symbol="cuda_eulerian_emissivity_table",
        resident=resident,
    )
    turnover = halo_mcg_turnover.bind(adapter, counts, errors)
    cb = C.CFUNCTYPE(C.c_int, C.c_int, C.POINTER(C.c_double), C.POINTER(C.c_void_p))
    symbols = native_tables((*_MOLECULAR_TABLES, "Nhalo_table", "Fcoll_table"))

    @cb
    def fields(n, ptr, a):
        try:
            pars = np.ctypeslib.as_array(ptr, (13,))
            metal, extra = bool(pars[3]), bool(pars[5])
            thermal, use_mcg, photo = bool(pars[9]), bool(pars[10]), bool(pars[11])
            need_sfr = thermal or extra
            sigma, slope = (
                Table.from_address(symbols[k])
                for k in ("Sigma_InterpTable", "dSigmasqdm_InterpTable")
            )
            sigmas = _sigma_arrays(
                cp.cuda.runtime.getDevice(),
                *(np.ctypeslib.as_array(t.y, (t.n,)).tobytes() for t in (sigma, slope)),
            )

            rule = cp.asarray(json.loads((ROOT / "gk61.json").read_text()))
            rows = cp.asarray(view(a[0], 6 * 30, "d"))
            target = cp.empty(6, dtype="d")
            kernels = _kernels()
            kernels.get_function("eulerian_targets")(
                (target.size,),
                (32,),
                (
                    rows,
                    *sigmas,
                    np.float64(sigma.low),
                    np.float64(sigma.width),
                    np.int32(sigma.n),
                    rule,
                    target,
                ),
            )
            goal_map = {
                0: target[0] + (target[1] if use_mcg else 0.0),
                1: target[2],
                2: target[3],
                3: (target[4] + (target[5] if use_mcg else 0.0))
                if metal
                else (pars[1] * target[2] + (pars[2] * target[3] if use_mcg else 0.0)),
            }
            if extra:
                extra_rows = cp.asarray(view(a[1], 2 * 23, "d"))
                extra_target = cp.empty(2, dtype="d")
                kernels.get_function("global_source_table")(
                    (2,),
                    (32,),
                    (
                        extra_rows,
                        *sigmas,
                        np.float64(sigma.low),
                        np.float64(sigma.width),
                        np.int32(sigma.n),
                        rule,
                        extra_target,
                    ),
                )
                goal_map[4] = extra_target[0]
                goal_map[5] = extra_target[1]
                goal_map[6] = target[2] * pars[8]
                if use_mcg:
                    goal_map[7] = target[3] * pars[8]
                dndm_meta_rows, dndm_vals = [], []
                dndm_offset = 0
                for key in ("Nhalo_table", "Fcoll_table"):
                    tbl = Table.from_address(symbols[key])
                    if not tbl.allocated or not (1 < tbl.n < 1000000) or tbl.width <= 0:
                        raise ValueError(f"Invalid Eulerian dNdM table {key}")
                    dndm_meta_rows.append((tbl.low, tbl.width, dndm_offset, tbl.n))
                    dndm_vals.append(view(C.cast(tbl.y, C.c_void_p).value, tbl.n, "d"))
                    dndm_offset += tbl.n
                dndm_meta = cp.asarray(dndm_meta_rows, dtype="d")
                dndm_values = cp.asarray(np.concatenate(dndm_vals), dtype="d")
            else:
                dndm_meta = np.uint64(0)
                dndm_values = np.uint64(0)
            active_slots = [k for k in range(8) if a[5 + k]]
            goals = cp.stack([goal_map[k] for k in active_slots])
            kinds = (
                ["Nion"]
                + (["SFRD"] if need_sfr else [])
                + (["Xray"] if thermal and metal else [])
            )
            acg_suffix = "conditional_table2D_acg" if photo else "conditional_table1D_acg"
            names = [f"{kind}_{acg_suffix}" for kind in kinds]
            if use_mcg:
                names.extend(f"{kind}_conditional_table2D_mcg" for kind in kinds)
            metadata, values = [], []
            offset = 0
            for name in names:
                axes, table = resident.pop(name)
                ny_tbl = table.shape[1] if table.ndim == 2 else 1
                metadata.append((*axes, offset, table.shape[0], ny_tbl))
                values.append(table.ravel())
                offset += table.size
            meta = cp.asarray(metadata, dtype="d")
            vals = cp.concatenate(values).astype("d")
            density_dev = cp.asarray(view(a[2], n, "f"))
            grids = [
                density_dev,
                cp.asarray(view(a[3], n, "f")) if (photo and a[3]) else np.uint64(0),
                cp.asarray(view(a[4], n, "f")) if (use_mcg and a[4]) else np.uint64(0),
            ]
            raw = cp.empty((8 if (extra or use_mcg) else (4 if thermal else 2), n), dtype="f")
            bad = cp.zeros(1, dtype="i")
            kernels.get_function("eulerian_cells")(
                ((n + 255) // 256,),
                (256,),
                (
                    np.int32(n),
                    meta,
                    vals,
                    dndm_meta,
                    dndm_values,
                    *grids,
                    cp.asarray(pars),
                    raw,
                    bad,
                ),
            )
            if bad.item():
                raise ValueError(
                    "Eulerian query outside native conditional table extent"
                )
            # The native mean loop divides float SFRD and X-ray arrays by a float
            # density factor before its double reduction. Photons use a volume mean.
            factor = cp.where(
                (1.0 + density_dev.astype("d")) > pars[4],
                (1.0 + density_dev.astype("d")).astype("f"),
                np.float32(1.0),
            )
            mean = cp.stack(
                [
                    raw[0].mean(dtype="d")
                    if k == 0
                    else (raw[k] / factor).mean(dtype="d")
                    for k in active_slots
                ]
            )
            ratios = cp.where(mean != 0.0, goals / mean, 1.0)
            packed = cp.stack([raw[k] for k in active_slots])
            result = (packed * ratios[:, None]).astype("f")
            if not bool(cp.isfinite(result).all()):
                raise ValueError("Nonfinite normalized molecular emissivity")
            for pos, k in enumerate(active_slots):
                result[pos].get(out=view(a[5 + k], n, "f"))
            counts["eulerian_mcg_emissivity_cells"] = (
                counts.get("eulerian_mcg_emissivity_cells", 0) + n
            )
            counts["eulerian_mcg_global_integrals"] = (
                counts.get("eulerian_mcg_global_integrals", 0) + target.size + (2 if extra else 0)
            )
            return 0
        except BaseException as exc:
            errors.append(exc)
            return 1

    C.c_void_p.in_dll(adapter, "cuda_eulerian_sources").value = C.cast(
        fields, C.c_void_p
    ).value
    return tables, turnover, fields
