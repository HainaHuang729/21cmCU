"""Apply 4.3.2 conditional source tables in the Lagrangian grid, then advect and deposit."""

import ctypes as C
from time import perf_counter

import cupy as cp
import numpy as np


def source(native, src_dir):
    # Reuse the upstream displacement factors (Mpc to density-grid coordinates)
    # and cell-volume ratio verbatim. The CHMF caller uses the IC density grid's
    # velocities at precisely the same resolution, so resample_index is identity.
    original = (src_dir / "map_mass.c").read_text()
    start = original.index("void move_integral_emissivities(")
    end = original.index("#pragma omp parallel", start)
    setup = original[start:end].replace(
        "void move_integral_emissivities(", "static void gpu_advect_sources(", 1
    )
    setup += """
    int dims[]={dens_dim[0],dens_dim[1],dens_dim[2],out_dim[0],out_dim[1],out_dim[2]};
    double rho_m=RHOcrit * cosmo_params_global->OMm;
    bool thermal=astro_options_global->USE_TS_FLUCT, metal=astro_options_global->USE_METALLICITY;
    bool extra=config_settings.EXTRA_EMISSIVITY_FIELDS, recomb=uses_recombination(astro_options_global->RECOMB_MODEL);
    // The native OpenMP for distributes contiguous x slabs. Advance the two
    // equally sized slabs together for thermal L-INTEGRAL source deposition.
    int slabs=thermal && matter_options_global->SOURCE_MODEL == SOURCE_MODEL_L_INTEGRAL &&
        simulation_options_global->N_THREADS == 2 && dens_dim[0] % 2 == 0 ? 2 : 1;
    double p[]={growth_factor, dim_ratio_out, velocity_displacement_factor[0],
        velocity_displacement_factor[1], velocity_displacement_factor[2],
        velocity_displacement_factor_2LPT[0],velocity_displacement_factor_2LPT[1],
        velocity_displacement_factor_2LPT[2],matter_options_global->PERTURB_ALGORITHM==PERTURB_ALGORITHM_2LPT,
        thermal, metal, extra, vol_ratio_out,consts->M_cell/rho_m,rho_m,consts->sfr_timescale,
        RHOcrit*cosmo_params_global->OMb/consts->sfr_timescale,consts->lx_over_sfr_acg,slabs,
        astro_options_global->USE_MCGS,consts->atomic_cooling_threshold*1.001,consts->lx_over_sfr_mcg};
    void *a[]={dens_pointer,vel_pointers[0],vel_pointers[1],vel_pointers[2],
        vel_pointers_2LPT[0],vel_pointers_2LPT[1],vel_pointers_2LPT[2],
        emissivity_fields->n_ion,thermal?emissivity_fields->sfrd_acg:NULL,
        thermal?emissivity_fields->xray_emissivity:NULL,recomb?emissivity_fields->fesc_weighted_sfrd:NULL,
        extra?emissivity_fields->halo_number:NULL, extra?emissivity_fields->halo_mass_density:NULL,
        extra?emissivity_fields->stellar_mass_density_acg:NULL,
        (thermal && astro_options_global->USE_MCGS)?emissivity_fields->sfrd_mcg:NULL,
        (extra && astro_options_global->USE_MCGS)?emissivity_fields->stellar_mass_density_mcg:NULL,
        log10_mturn_acg_grid,log10_mturn_mcg_grid};
    if (cuda_advect_sources(dims,p,a)) Throw(ValueError);
}
"""
    call = native.index("    move_integral_emissivities(")
    last = native.index(";", call) + 1
    original_call = native[call:last]
    guarded = (
        """    if (source_model_uses_lagrangian_grids(matter_options_global->SOURCE_MODEL) &&
        uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES) &&
        ((!astro_options_global->USE_MCGS && !astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK) ||
         (astro_options_global->USE_MCGS && astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK &&
          astro_options_global->USE_TS_FLUCT && astro_options_global->USE_METALLICITY &&
          grid_dim[0] == simulation_options_global->HII_DIM))) {
"""
        + original_call.replace("move_integral_emissivities", "gpu_advect_sources", 1)
        + "\n    } else\n"
        + original_call
    )
    native = native[:call] + guarded + native[last:]
    i = native.index("int add_integral_contribution(")
    return (
        native[:i]
        + "int (*cuda_advect_sources)(int *,double *,void **);\n"
        + setup
        + native[i:]
    )


_TABLES = (
    "Nion_conditional_table1D_acg",
    "SFRD_conditional_table1D_acg",
    "Xray_conditional_table1D_acg",
    "Nhalo_table",
    "Fcoll_table",
)


_MOLECULAR_TABLES = tuple(
    f"{quantity}_conditional_table2D_{population}"
    for population in ("acg", "mcg")
    for quantity in ("Nion", "SFRD", "Xray")
)


def _molecular_tables(symbols, needed_2d, extra):
    from evolution_backend import Table, Table2DFloat, view

    nodes = []
    meta = np.zeros((8 if extra else 6, 7), dtype="d")
    offset = 0
    for idx, name in enumerate(_MOLECULAR_TABLES):
        if not needed_2d[idx]:
            continue
        table = Table2DFloat.from_address(symbols[name])
        if (
            not table.allocated
            or min(table.nx, table.ny) < 2
            or min(table.xstep, table.ystep) <= 0
        ):
            raise ValueError(f"Invalid molecular emissivity table {name}")
        meta[idx] = (
            table.xmin,
            table.ymin,
            table.xstep,
            table.ystep,
            offset,
            table.nx,
            table.ny,
        )
        nodes.extend(
            np.ctypeslib.as_array(table.values[i], shape=(table.ny,))
            for i in range(table.nx)
        )
        offset += table.nx * table.ny
    if extra:
        for idx, name in ((6, "Nhalo_table"), (7, "Fcoll_table")):
            table = Table.from_address(symbols[name])
            if not table.allocated or table.n < 2 or table.width <= 0:
                raise ValueError(f"Invalid unresolved emissivity table {name}")
            values = view(table.y, table.n, "d")
            nodes.append(values)
            meta[idx, :4] = table.low, table.width, offset, table.n
            offset += table.n
    return meta, np.concatenate(nodes).astype("d")


def _atomic_tables(symbols, needed):
    from evolution_backend import Table, view

    nodes = []
    meta = np.zeros((len(_TABLES), 4), dtype="d")
    offset = 0
    for j, name in enumerate(_TABLES):
        if needed[j]:
            table = Table.from_address(symbols[name])
            if not table.allocated or table.n < 2 or table.width <= 0:
                raise ValueError(f"Invalid unresolved emissivity table {name}")
            values = view(table.y, table.n, "f" if j < 3 else "d").astype(
                "d", copy=False
            )
            nodes.append(values)
            meta[j] = table.low, table.width, offset, table.n
            offset += table.n
    return meta, np.concatenate(nodes)


def bind(adapter, module, report, errors):
    from evolution_backend import native_tables, view
    from halo_grid import _ordered_deposit

    symbols = native_tables(_TABLES[2:] + _MOLECULAR_TABLES)
    callback_type = C.CFUNCTYPE(
        C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_double), C.POINTER(C.c_void_p)
    )
    kernel = module.get_function("advect_unresolved_cells")

    @callback_type
    def run(dims_ptr, p_ptr, a):
        try:
            start = perf_counter()
            dims = np.ctypeslib.as_array(dims_ptr, (6,))
            p = np.ctypeslib.as_array(p_ptr, (22,))
            second, thermal, metal, extra = map(bool, p[8:12])
            n, ng = (int(np.prod(dims[j : j + 3], dtype=np.int64)) for j in (0, 3))
            if max(n, ng) > np.iinfo(np.int32).max:
                raise ValueError("Unresolved halo grids exceed the signed cell index")
            molecular = bool(p[19])
            stride = (9 if extra else 8) if molecular else 7
            if molecular:
                needed_2d = (True, thermal or extra, thermal and metal) * 2
                meta, tables = _molecular_tables(symbols, needed_2d, extra)
            else:
                needed = (True, thermal or extra, thermal and metal, extra, extra)
                meta, tables = _atomic_tables(symbols, needed)
            fields = [
                j for j in range(stride) if a[j + 7] and j != 3 and (j != 2 or metal)
            ]
            rows = {field: i for i, field in enumerate(fields)}
            source_fields = [view(a[i], n, "f") for i in range(7 if second else 4)]
            if molecular:
                source_fields.extend(view(a[i], n, "f") for i in (16, 17))
            dev_p, dev_dims, dev_meta, dev_tables = map(
                cp.asarray, (p, dims, meta, tables)
            )
            dev_fields = cp.asarray(fields, dtype="i")
            grids = cp.empty((len(fields), ng), dtype="f")
            for j, field in enumerate(fields):
                grids[j].set(view(a[7 + field], ng, "f"))
            bad = cp.zeros(1, dtype="i")
            dummy = cp.empty(1, dtype="f")
            slabs = int(p[18])
            slab_size = n // slabs
            batch_per_slab = 131072 // slabs
            for first in range(0, slab_size, batch_per_slab):
                slab_count = min(slab_size - first, batch_per_slab)
                count = slab_count * slabs
                cells = cp.empty(8 * count, dtype="i")
                weights = cp.empty(8 * count, dtype="d")
                props = cp.empty((count, stride), dtype="d")
                # Each input transfers one contiguous portion of each slab. The
                # CUDA output interleaves these portions in local source order.
                inputs = [
                    cp.asarray(
                        x.reshape(slabs, slab_size)[:, first : first + slab_count],
                        order="C",
                    )
                    for x in source_fields
                ]
                if not second:
                    inputs[4:4] = [dummy] * 3
                if not molecular:
                    inputs.extend([dummy] * 2)
                kernel(
                    ((count + 127) // 128,),
                    (128,),
                    (
                        np.int32(first),
                        np.int32(count),
                        dev_dims,
                        dev_p,
                        dev_meta,
                        dev_tables,
                        *inputs,
                        cells,
                        weights,
                        props,
                        bad,
                    ),
                )
                _ordered_deposit(cells, weights, props, grids, dev_fields, module)
            if bad.item():
                raise ValueError(
                    "Unresolved halo density outside its native source table"
                )
            outputs = {field: grids[j] for j, field in enumerate(fields)}
            if thermal and not metal:
                xray = grids[rows[1]].astype("d") * p[17]
                if molecular:
                    xray = xray + grids[rows[7]].astype("d") * p[21]
                outputs[2] = xray.astype("f")
            if a[10]:
                outputs[3] = (grids[rows[0]].astype("d") * p[16]).astype("f")
            for field, arr in outputs.items():
                arr.get(out=view(a[7 + field], ng, "f"))
            h2d = (len(source_fields) * n + len(fields) * ng + len(fields)) * 4
            h2d += p.nbytes + dims.nbytes + meta.nbytes + tables.nbytes
            updates = {
                "subgrid_advection_calls": 1,
                "subgrid_advection_cells": n,
                "subgrid_advection_input_bytes": h2d,
                "subgrid_advection_output_bytes": 4 * (ng * len(outputs) + 1),
                "subgrid_advection_seconds": perf_counter() - start,
            }
            if molecular:
                updates["subgrid_molecular_advection_cells"] = n
            for key, value in updates.items():
                report[key] = report.get(key, 0) + value
            return 0
        except BaseException as exc:
            errors.append(exc)
            return 1

    C.c_void_p.in_dll(adapter, "cuda_advect_sources").value = C.cast(
        run, C.c_void_p
    ).value
    return run
