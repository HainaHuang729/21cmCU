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
        astro_options_global->USE_TS_FLUCT && !config_settings.EXTRA_EMISSIVITY_FIELDS &&
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
        ("initialize_xray_emissivity_conditional_tables", 2),
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
                    f"if (use_mcg_conditional_gpu() && astro_options_global->INTEGRATION_METHOD_ACGS != INTEGRATION_METHOD_GAMMA_APPROX && astro_options_global->INTEGRATION_METHOD_MCGS != INTEGRATION_METHOD_GAMMA_APPROX)"
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
    symbols = native_tables(_MOLECULAR_TABLES)

    @cb
    def fields(n, ptr, a):
        try:
            pars = np.ctypeslib.as_array(ptr, (5,))
            metal = bool(pars[3])
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
            target = cp.empty(6 if metal else 4, dtype="d")
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
            goals = cp.stack(
                (
                    target[0] + target[1],
                    target[2],
                    target[3],
                    target[4] + target[5]
                    if metal
                    else pars[1] * target[2] + pars[2] * target[3],
                )
            )
            names = [
                name
                for name in _MOLECULAR_TABLES
                if metal or not name.startswith("Xray")
            ]
            metadata, values = [], []
            offset = 0
            for name in names:
                axes, table = resident.pop(name)
                metadata.append((*axes, offset, *table.shape))
                values.append(table.ravel())
                offset += table.size
            meta = cp.asarray(metadata, dtype="d")
            vals = cp.concatenate(values).astype("d")
            grids = [cp.asarray(view(a[i], n, "f")) for i in (1, 2, 3)]
            raw = cp.empty((4, n), dtype="f")
            bad = cp.zeros(1, dtype="i")
            kernels.get_function("eulerian_cells")(
                ((n + 255) // 256,),
                (256,),
                (
                    np.int32(n),
                    meta,
                    vals,
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
                (1.0 + grids[0].astype("d")) > pars[4],
                (1.0 + grids[0].astype("d")).astype("f"),
                np.float32(1.0),
            )
            mean = cp.concatenate(
                (
                    raw[:1].mean(axis=1, dtype="d"),
                    (raw[1:] / factor).mean(axis=1, dtype="d"),
                )
            )
            result = (raw * (goals / mean)[:, None]).astype("f")
            if not bool(cp.isfinite(result).all()):
                raise ValueError("Nonfinite normalized molecular emissivity")
            for i in range(4):
                result[i].get(out=view(a[4 + i], n, "f"))
            counts["eulerian_mcg_emissivity_cells"] = (
                counts.get("eulerian_mcg_emissivity_cells", 0) + n
            )
            counts["eulerian_mcg_global_integrals"] = (
                counts.get("eulerian_mcg_global_integrals", 0) + target.size
            )
            return 0
        except BaseException as exc:
            errors.append(exc)
            return 1

    C.c_void_p.in_dll(adapter, "cuda_eulerian_sources").value = C.cast(
        fields, C.c_void_p
    ).value
    return tables, turnover, fields
