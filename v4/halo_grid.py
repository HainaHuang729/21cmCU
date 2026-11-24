"""CUDA gridding of the sampled ACG and MCG catalogue in the 4.3.2 Lagrangian source models."""

from contextlib import contextmanager
import ctypes as C
from importlib.metadata import version
from pathlib import Path
import tempfile
from unittest.mock import patch

from native_shim import load_shim

import cupy as cp
import numpy as np
import py21cmfast as p21
from py21cmfast.c_21cmfast import ffi, lib
from py21cmfast.wrapper.outputs import EmissivityFields


# Native scaling constants include upstream units, overrides and redshift evolution.
_PARAMS = {
    "length_xy": "boxlen",
    "length_z": "boxlen*simulation_options_global->NON_CUBIC_FACTOR",
    "vel_cell_inv": "vel_dim[0]/simulation_options_global->BOX_LEN",
    "displacement": "consts->growth_factor-init_growth",
    "displacement2": "(-3./7.)*(consts->growth_factor*consts->growth_factor-init_growth*init_growth)",
    "second_order": "matter_options_global->PERTURB_ALGORITHM==PERTURB_ALGORITHM_2LPT",
    "volume_inv": "inv_cell*inv_cell*inv_cell",
    "fstar10": "consts->fstar_10_acg",
    "alpha_star": "consts->alpha_star_acg",
    "upper_turnover": "astro_options_global->USE_UPPER_STELLAR_TURNOVER",
    "upper_index": "consts->alpha_upper",
    "upper_pivot": "consts->pivot_upper",
    "upper_ratio": "consts->upper_pivot_ratio",
    "star_sigma": "consts->sigma_star",
    "median": "consts->scaling_median",
    "mturn": "consts->mturn_acg_homogeneous",
    "baryon_ratio": "cosmo_params_global->OMb/cosmo_params_global->OMm",
    "baryon_density": "RHOcrit*cosmo_params_global->OMb",
    "fesc10": "consts->fesc_10_acg",
    "alpha_esc": "consts->alpha_esc",
    "pop2": "consts->pop2_ion",
    "timescale": "consts->sfr_timescale",
    "sfr_sigma": "consts->sigma_sfr_lim",
    "sfr_sigma_index": "consts->sigma_sfr_idx",
    "xray_sigma": "consts->sigma_xray",
    "lx_sfr": "consts->lx_over_sfr_acg",
    "thermal": "astro_options_global->USE_TS_FLUCT",
    "recombinations": "uses_recombination(astro_options_global->RECOMB_MODEL)",
    "metallicity": "astro_options_global->USE_METALLICITY",
    "redshift": "redshift",
    "seconds_year": "physconst.s_per_yr",
    "mcg": "astro_options_global->USE_MCGS",
    "photoheating": "astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK",
    "fstar7": "consts->fstar_7_mcg",
    "alpha_star_mcg": "consts->alpha_star_mcg",
    "fesc7": "consts->fesc_7_mcg",
    "pop3": "consts->pop3_ion",
    "atomic_threshold": "consts->atomic_cooling_threshold",
    "lx_mcg": "consts->lx_over_sfr_mcg",
}

_CALLBACK = C.CFUNCTYPE(
    C.c_int,
    C.c_ulonglong,
    C.POINTER(C.c_int),
    C.POINTER(C.c_double),
    C.POINTER(C.c_void_p),
)


def _source(source):
    start = source.index(
        "    move_halo_emissivities(", source.index("void sum_halos_onto_grid(")
    )
    hook = """
    if (source_model_uses_sampled_halos(matter_options_global->SOURCE_MODEL)) {
        double boxlen=simulation_options_global->BOX_LEN;
        double inv_cell=out_dim[0]/simulation_options_global->BOX_LEN;
        double init_growth=dicke(simulation_options_global->INITIAL_REDSHIFT);
        double p[]={PARAMS};
        int dims[]={vel_dim[0],vel_dim[1],vel_dim[2],out_dim[0],out_dim[1],out_dim[2]};
        int ts=astro_options_global->USE_TS_FLUCT, extra=config_settings.EXTRA_EMISSIVITY_FIELDS;
        int rec=uses_recombination(astro_options_global->RECOMB_MODEL);
        void *a[]={halos->halo_masses,halos->halo_coords,halos->star_rng,halos->sfr_rng,halos->xray_rng,
            vel_pointers[0],vel_pointers[1],vel_pointers[2],vel_pointers_2LPT[0],vel_pointers_2LPT[1],vel_pointers_2LPT[2],
            emissivity_fields->n_ion,ts?emissivity_fields->sfrd_acg:NULL,ts?emissivity_fields->xray_emissivity:NULL,
            rec?emissivity_fields->fesc_weighted_sfrd:NULL,extra?emissivity_fields->halo_number:NULL,
            extra?emissivity_fields->halo_mass_density:NULL,extra?emissivity_fields->stellar_mass_density_acg:NULL,
            ts&&astro_options_global->USE_MCGS?emissivity_fields->sfrd_mcg:NULL,
            extra&&astro_options_global->USE_MCGS?emissivity_fields->stellar_mass_density_mcg:NULL,
            log10_mturn_acg_grid,log10_mturn_mcg_grid};
        if (cuda_halo_grid(halos->n_halos,dims,p,a)) Throw(ValueError);
    } else
    """.replace("PARAMS", ", ".join(_PARAMS.values()))
    declaration = "int (*cuda_halo_grid)(unsigned long long,int*,double*,void**);\n"
    return declaration + source[:start] + hook + source[start:]


def _host_floats(ptr, count):
    return np.ctypeslib.as_array((C.c_float * count).from_address(ptr))


def _ordered_deposit(cells, weights, props, grids, dev_fields, module):
    order = cp.argsort(cells).astype("i")  # CuPy uses a stable sort.
    ordered_cells = cells[order]
    splits = cp.flatnonzero(ordered_cells[1:] != ordered_cells[:-1]) + 1
    starts = cp.concatenate(
        (cp.zeros(1, "i"), splits, cp.asarray([cells.size]))
    ).astype("i")
    ns = len(starts) - 1
    module.get_function("add_segments")(
        ((ns + 127) // 128,),
        (128,),
        (
            np.int32(ns),
            np.int32(grids.shape[1]),
            np.int32(grids.shape[0]),
            np.int32(props.shape[1]),
            cells,
            order,
            starts,
            dev_fields,
            weights,
            props,
            grids,
        ),
    )


def _accumulate(n, dims, par, a, module, *, engine=None):
    if not n:
        return
    vsize, gsize = (int(np.prod(dims[i : i + 3], dtype=np.int64)) for i in (0, 3))
    if gsize > np.iinfo(np.int32).max:
        raise ValueError("Sampled halo grid exceeds the supported signed cell index")
    if any(not a[i] for i in range(8)) or not a[11]:
        raise ValueError("Missing catalogue, RNG, velocity or ionization output")
    registry = None if engine is None else getattr(engine, "device_registry", None)

    def dev_f32(addr, size):
        if registry is not None:
            cached = registry.lookup_ptr(int(addr), size, "f")
            if cached is not None:
                return cached
        return cp.asarray(_host_floats(addr, size))

    second_order = par[list(_PARAMS).index("second_order")]
    empty = cp.empty(1, np.float32)
    vel = [
        dev_f32(a[i], vsize) if i < 8 or second_order else empty for i in range(5, 11)
    ]
    cols = [
        _host_floats(a[i], n * (3 if i == 1 else 1)).reshape(n, -1) for i in range(5)
    ]
    mcg = bool(par[list(_PARAMS).index("mcg")])
    use_acg_grid = bool(par[list(_PARAMS).index("photoheating")])
    if (use_acg_grid and not a[20]) or (mcg and not a[21]):
        raise ValueError("Missing local halo turnover grid")
    turnovers = [
        dev_f32(a[i], gsize) if required else empty
        for i, required in ((20, use_acg_grid), (21, mcg))
    ]
    stride = 9 if mcg else 7
    fields = [i for i in range(stride) if a[i + 11]]
    host = [_host_floats(a[11 + i], gsize) for i in fields]
    grids = cp.zeros((len(fields), gsize), dtype=cp.float32)
    dev_par, dev_dim, dev_fields = (
        cp.asarray(par),
        cp.asarray(dims),
        cp.asarray(fields, dtype="i"),
    )
    # Bound the sort workspace and retain catalogue order across chunks.
    offset = 0
    while offset < n:
        stop = min(n, offset + 131072)
        nh = stop - offset
        cells = cp.empty(nh * 8, "i")
        weights = cp.empty(nh * 8, "d")
        props = cp.empty((nh, stride), "d")
        batch = [cp.asarray(col[offset:stop]) for col in cols]
        module.get_function("sampled_halos")(
            ((nh + 127) // 128,),
            (128,),
            (
                np.int32(nh),
                dev_par,
                dev_dim,
                *batch,
                *vel,
                *turnovers,
                cells,
                weights,
                props,
            ),
        )
        _ordered_deposit(cells, weights, props, grids, dev_fields, module)
        offset = stop
    volume_inv = par[list(_PARAMS).index("volume_inv")]
    module.get_function("normalize_sampled_grid")(
        ((gsize + 255) // 256, len(fields)),
        (256,),
        (np.int32(gsize), np.float64(volume_inv), dev_fields, grids),
    )
    for idx, row, destination in zip(fields, grids, host, strict=True):
        row.get(out=destination)


@contextmanager
def halo_grid_backend(report, *, scaling=None, engine=None):
    """Use CUDA for discrete halo gridding and integrated Lagrangian emissivities."""
    import halo_advection
    import halo_mcg_turnover
    import halo_subgrid

    if version("21cmfast") != "4.3.2":
        raise ValueError("Sampled halo gridding requires 21cmFAST 4.3.2")
    src_dir = Path(p21.__file__).parent / "src"
    native = Path(p21.c_21cmfast.__file__)
    kernel = Path(__file__).with_suffix(".cu").read_text()
    enum = "enum HaloParameters {" + ", ".join(_PARAMS) + "};\n"
    module = cp.RawModule(
        code=enum + kernel + Path(__file__).with_name("halo_advection.cu").read_text(),
        options=("--fmad=false",),
    )
    errors = []
    report.setdefault("halo_grid_calls", 0)
    report.setdefault("halo_grid_halos", 0)
    report.setdefault("halo_grid_empty", 0)

    @_CALLBACK
    def callback(n, dims, par, grids):
        try:
            _accumulate(
                n,
                np.ctypeslib.as_array(dims, (6,)),
                np.ctypeslib.as_array(par, (len(_PARAMS),)),
                grids,
                module,
                engine=engine,
            )
            if n:
                report["halo_grid_calls"] += 1
                report["halo_grid_halos"] += n
                if par[list(_PARAMS).index("mcg")]:
                    report["sampled_mcg_halos"] = report.get("sampled_mcg_halos", 0) + n
            else:
                report["halo_grid_empty"] += 1
            return 0
        except BaseException as exc:
            errors.append(exc)
            return 1

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        source = halo_advection.source(
            halo_subgrid.source(_source((src_dir / "EmissivityFields.c").read_text())),
            src_dir,
        )
        source = halo_mcg_turnover.source(source)
        if scaling is not None:
            from evolution_backend import bind_source_scaling, shared_source_scaling

            source = shared_source_scaling(source)
        adapter = load_shim(
            folder, "halo_grid.so", {"halo_grid.c": source}, src_dir, native
        )
        if scaling is not None:
            bind_source_scaling(adapter, scaling)
        _turnover_callback = halo_mcg_turnover.bind(
            adapter,
            report,
            errors,
            registry=None if engine is None else engine.device_registry,
        )
        _subgrid_callback = halo_subgrid.bind(adapter, report, errors)
        _advection_callback = halo_advection.bind(adapter, module, report, errors)
        C.c_void_p.in_dll(adapter, "cuda_halo_grid").value = C.cast(
            callback, C.c_void_p
        ).value
        compute = ffi.cast(
            ffi.typeof(lib.ComputeEmissivityFields),
            C.cast(adapter.ComputeEmissivityFields, C.c_void_p).value,
        )

        def invoke(*args):
            status = compute(*args)
            if errors:
                raise errors.pop()
            return status

        with patch.object(
            EmissivityFields, "_c_compute_function", staticmethod(invoke)
        ):
            yield
