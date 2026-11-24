"""CUDA adapters for the cubic Lagrangian-source centre-cell ionisation paths."""

import ctypes as C
import json
import logging
from unittest.mock import patch

import cupy as cp
import numpy as np
from py21cmfast.c_21cmfast import ffi, lib
from py21cmfast.wrapper.outputs import IonizedBox

from evolution_backend import CB, ROOT, SHARED, SRC, insert, module, view
from fft_plan_cuda import transform
from mhr import build as build_mhr


def _unconditional_source(src, *, eulerian=False):
    # Keep the native recurrence and validation in one place. Substitute only
    # the six calls that produce the photons-per-baryon integrals.
    start = src.index("void set_unconditional_nion(")
    end = src.index("double set_fully_neutral_box(", start)
    body = src[start:end]
    pair = "    ScalingConstants *sc_ptr = &(c->scale_consts);"
    precompute = """
    bool gpu_integrals = c->lagrangian_source_grids && astro_options_global->USE_MCGS &&
        uses_interpolation_tables(matter_options_global->USE_INTERPOLATION_TABLES);
    double photons[3][2];
    int floor_row=0, prev_row=0;
    if (gpu_integrals) {
        bool prev_acg = !(prev_box->nion_unconditional_acg < 1.e-4);
        bool prev_mcg = !(prev_box->nion_unconditional_mcg < 1.e-4);
        int nz=1;
        double z[3] = {c->redshift};
        if (simulation_options_global->Z_HEAT_MAX != z[0]) {
            floor_row=nz;
            z[nz++]=simulation_options_global->Z_HEAT_MAX;
        }
        if ((prev_acg || prev_mcg) && c->prev_redshift != z[0]) {
            if (c->prev_redshift == z[floor_row]) prev_row=floor_row;
            else {
                prev_row=nz;
                z[nz++]=c->prev_redshift;
            }
        }
        double rows[3][2][25];
        for (int i=0; i<nz; ++i)
            global_population_rows(rows[i], z[i], c->lnMmin, c->lnMmax_gl, mturn_acg, sc_ptr);
        if (prev_row != 0 && prev_row != floor_row) {
            if (!prev_acg) rows[prev_row][0][24] = 1.;
            if (!prev_mcg) rows[prev_row][1][24] = 1.;
        }
        void *grids[] = {rows, &mturn_mcg, photons};
        if (halo_ion_dispatch(7, nz, NULL, grids)) Throw(ValueError);
    }
"""
    if eulerian:
        precompute = precompute.replace(
            "c->lagrangian_source_grids",
            "matter_options_global->SOURCE_MODEL == SOURCE_MODEL_E_INTEGRAL",
        )
        precompute = precompute.replace(
            "halo_ion_dispatch(7,", "cuda_eulerian_mcg_ion(4,"
        )
    body = body.replace(pair, pair + precompute, 1)
    for j, pop in enumerate(("acg", "mcg")):
        for i, z in enumerate(
            ("c->redshift", "simulation_options_global->Z_HEAT_MAX", "c->prev_redshift")
        ):
            call = f"nion_unconditional_{pop}({z},"
            if body.count(call) != 1:
                raise ValueError("Unknown native unconditional ionization integral")
            row = ("0", "floor_row", "prev_row")[i]
            body = body.replace(call, f"gpu_integrals ? photons[{row}][{j}] : " + call)
    common = (ROOT / "global_integrals.h").read_text()
    first = common.index("static void global_acg_row(")
    last = common.index("\n}", first) + 2
    definitions = (ROOT / "halo_nion_rows.h").read_text()
    if not eulerian:
        definitions = common[first:last] + "\n" + definitions
    return src[:start] + definitions + "\n" + body + src[end:]


def ionization_source():
    src = _unconditional_source((SRC / "IonisationBox.c").read_text())
    # Lagrangian photons come from filtered emissivity grids. Only the
    # Eulerian conditional-mass calculation reads a radius's sigma_maxmass,
    # and 4.3.2 never reads the IonBoxConstants.sigma_minmass member.
    for mass in ("consts->M_min", "(*rspec_array)[i].M_max_R"):
        src = src.replace(
            f"= sigma_z0({mass});",
            f"= consts->lagrangian_source_grids ? 0. : sigma_z0({mass});",
        )
    header = """
int (*halo_ion_dispatch)(int,int,double*,void**);
int halo_use_mhr;
#define HALO_ION(ID,P,A) do { if (halo_ion_dispatch(ID,HII_TOT_NUM_PIXELS,P,A)) Throw(ValueError); } while(0)
"""
    active = """consts->lagrangian_source_grids && !consts->need_mcgs_nion
        && astro_options_global->RECOMB_MODEL != RECOMB_MODEL_HOMOGENEOUS
        && !astro_options_global->IONISE_ENTIRE_SPHERE
        && simulation_options_global->NON_CUBIC_FACTOR == 1.0"""
    src = insert(
        src,
        "void prepare_box_for_filtering(",
        "    int i, j, k;",
        """
    if (simulation_options_global->NON_CUBIC_FACTOR == 1.0 && !matter_options_global->USE_FFTW_WISDOM) {
        double p[]={simulation_options_global->HII_DIM, const_factor, limit_min, limit_max,
                    simulation_options_global->N_THREADS};
        void *a[]={input_box,output_c_box};
        HALO_ION(0,p,a); return;
    }
""",
    )
    src = insert(
        src,
        "void copy_filter_transform(",
        "    memcpy(",
        """
    if (ACTIVE && !matter_options_global->USE_FFTW_WISDOM && consts->hii_filter >= 0 && consts->hii_filter <= 2) {
        const float radius = (float)rspec.R, mfp = (float)consts->mfp_meandens;
        const int kind = astro_options_global->USE_EXP_FILTER ? FILTER_EXP_MFP : consts->hii_filter;
        double p[]={box_dim[0],rspec.R_index,2.*M_PI/simulation_options_global->BOX_LEN,
                    radius,mfp,kind == FILTER_EXP_MFP && rspec.R_index > 0 ? exp(-radius/mfp) : 0.,
                    consts->hii_filter,kind,simulation_options_global->N_THREADS,astro_options_global->USE_TS_FLUCT,
                    uses_recombination(astro_options_global->RECOMB_MODEL),consts->filter_recombinations};
        void *a[]={fg_struct->deltax_unfiltered,fg_struct->deltax_filtered,
                   fg_struct->stars_unfiltered,fg_struct->stars_filtered,
                   p[9] ? fg_struct->xe_unfiltered : NULL,p[9] ? fg_struct->xe_filtered : NULL,
                   p[10] ? fg_struct->sfr_unfiltered : NULL,p[10] ? fg_struct->sfr_filtered : NULL,
                   p[11] ? fg_struct->N_rec_unfiltered : NULL,p[11] ? fg_struct->N_rec_filtered : NULL};
        HALO_ION(1,p,a); return;
    }
""".replace("ACTIVE", active),
    )
    src = insert(
        src,
        "void calculate_conditional_nion(",
        "    double nion_total_acg",
        """
    if (ACTIVE) {
        double p[]={simulation_options_global->HII_DIM,-1.+FRACT_FLOAT_ERR,astro_options_global->USE_TS_FLUCT,
                    uses_recombination(astro_options_global->RECOMB_MODEL),consts->filter_recombinations};
        void *a[]={fg_struct->deltax_filtered,fg_struct->stars_filtered,&rspec->nion_conditional_ave_acg,
                   p[2] ? fg_struct->xe_filtered : NULL,p[3] ? fg_struct->sfr_filtered : NULL,
                   p[4] ? fg_struct->N_rec_filtered : NULL};
        HALO_ION(4,p,a); rspec->nion_conditional_ave_mcg=0.; return;
    }
""".replace("ACTIVE", active),
    )
    src = insert(
        src,
        "void find_ionised_regions(",
        "#pragma omp parallel",
        """
    if (ACTIVE) {
        double p[]={box_dim[0],rspec.R_index,consts->redshift,consts->photoncons_adjustment_factor,
                    mean_fix_term_acg,nion_limit_acg,consts->mass_dep_zeta,
                    astro_options_global->USE_TS_FLUCT ? 0. : consts->TK_nofluct,
                    astro_options_global->USE_TS_FLUCT ? 0. : consts->adia_TK_term,consts->T_re,astro_options_global->USE_TS_FLUCT,
                    uses_recombination(astro_options_global->RECOMB_MODEL),consts->filter_recombinations,
                    rspec.R,consts->gamma_prefactor,
                    astro_options_global->USE_MCGS && consts->mass_dep_zeta ? fmax(0.,nion_limit_mcg) : 0.};
        void *a[]={perturbed_field->density,fg_struct->deltax_filtered,fg_struct->stars_filtered,
                   previous_ionize_box->z_reion,box->neutral_fraction,box->z_reion,
                   matter_options_global->MINIMIZE_MEMORY?NULL:box->kinetic_temperature,
                   p[10] ? fg_struct->xe_filtered : NULL,p[10] ? spin_temp->kinetic_temp_neutral : NULL,
                   p[11] ? (p[12] ? (void*)fg_struct->N_rec_filtered : (void*)previous_ionize_box->cumulative_recombinations) : NULL,
                   p[11] ? fg_struct->sfr_filtered : NULL,p[11] ? box->ionisation_rate_G12 : NULL,
                   (p[11] && !matter_options_global->MINIMIZE_MEMORY) ? box->mean_free_path : NULL};
        HALO_ION(2,p,a); return;
    }
""".replace("ACTIVE", active),
    )
    src = insert(
        src,
        "void set_ionized_temperatures(",
        "#pragma omp parallel",
        """
    if (ACTIVE && !matter_options_global->MINIMIZE_MEMORY) {
        double p[]={consts->stored_redshift,consts->T_re,astro_options_global->USE_TS_FLUCT ? 0. : consts->TK_nofluct,
                    astro_options_global->USE_TS_FLUCT ? 0. : consts->adia_TK_term,astro_options_global->USE_TS_FLUCT};
        void *a[]={perturbed_field->density,box->neutral_fraction,box->z_reion,box->kinetic_temperature,
                   p[4] ? spin_temp->kinetic_temp_neutral : NULL};
        HALO_ION(3,p,a); return;
    }
""".replace("ACTIVE", active),
    )
    src = insert(
        src,
        "// Inhomogeneous recombination rate",
        "        bool finite_error",
        """
        if (halo_use_mhr) {
            double p[]={consts->stored_redshift,consts->fabs_dtdz,consts->dz};
            void *a[]={perturbed_field->density,box->ionisation_rate_G12,box->neutral_fraction,
                       previous_ionize_box->cumulative_recombinations,box->cumulative_recombinations};
            HALO_ION(6,p,a); return;
        }
""",
    )
    begin = src.index(
        "#pragma omp parallel shared(box)",
        src.index("            float global_photoionization_rate = 0;"),
    )
    stop = src.index("            if (isfinite(global_xH)", begin)
    src = (
        src[:begin]
        + """if (ACTIVE) {
        void *a[]={box->neutral_fraction,&global_xH}; HALO_ION(8,NULL,a);
    } else {
""".replace("ACTIVE", active.replace("consts->", "ionbox_constants."))
        + src[begin:stop]
        + "}\n"
        + src[stop:]
    )
    return (
        header
        + src
        + """
int halo_init_mhr(void) {
    double p[]={No,cosmo_params_global->Y_He,alpha_B(1.e4)};
    return halo_ion_dispatch(5, 1, p, NULL);
}
"""
    )


def _device_function(filename, signature):
    source = (SRC / filename).read_text()
    start = source.index(signature)
    lines = source[start:].splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("}") or (i == 0 and line.rstrip().endswith("}")):
            body = "".join(lines[: i + 1])
            # C pow has double arguments. NVRTC otherwise selects its integer-exponent overload.
            for exponent in ("2", "3", "-3"):
                body = body.replace(", " + exponent + ")", ", " + exponent + ".)")
            return "__device__ " + body
    raise ValueError(f"Missing function body for {signature}")


@cp.memoize(for_each_device=True)
def halo_window():
    functions = [
        _device_function("filtering.c", "double " + name + "(")
        for name in (
            "real_tophat_filter",
            "sharp_k_filter",
            "gaussian_filter",
            "exp_mfp_filter",
        )
    ]
    return module(
        [], "\n".join(functions) + "\n" + (ROOT / "halo_window.cu").read_text()
    ).get_function("halo_window")


def install_ionization(stack, adapter, counters, *, use_mhr=False, engine=None):
    partial = _device_function(
        "thermochem.c", "float ComputePartiallyIonizedTemperature("
    )
    kernels = module(
        [SHARED / "glibc_math.cuh", SHARED / "ion.cu"],
        partial + "\n" + (ROOT / "halo_ionization.cu").read_text(),
        options=("-DRECOMB_V432",),
    )
    window = halo_window()
    barrier = kernels.get_function("halo_barrier")
    temperature = kernels.get_function("ion_temperature")
    rec_tables = None
    max_log = None
    stage_names = (
        "ionization_forward_fft",
        "ionization_radius_inverse_pair",
        "ionization_halo_barrier",
        "ionization_temperature",
        "ionization_conditional_nion",
        "ionization_mhr_table",
        "ionization_cumulative_recombinations",
        "ionization_global_nion",
        "ionization_global_neutral_fraction",
    )
    errors = []
    spectra = {}
    filtered_cache = {}
    ion_state = {}
    ion_state_meta = {}
    residency = (
        logging.getLogger("py21cmfast.wrapper.outputs").getEffectiveLevel()
        > logging.DEBUG
    )

    @CB
    def callback(stage, n, p, a):
        nonlocal rec_tables, max_log
        try:
            registry = None if engine is None else engine.device_registry

            def grid(i):
                ptr = int(a[i]) if a[i] else 0
                if ptr:
                    arr_dev = ion_state.get(ptr)
                    if arr_dev is not None:
                        return arr_dev
                    arr_dev = filtered_cache.get(ptr)
                    if arr_dev is not None:
                        return arr_dev
                    if registry is not None:
                        cached = registry.lookup_ptr(ptr, n, "f")
                        if cached is not None:
                            return cached
                return cp.asarray(view(a[i], n, "f"))

            def ion_buf(i):
                ptr = int(a[i])
                buf = ion_state.get(ptr)
                if buf is None:
                    buf = grid(i)
                    ion_state[ptr] = buf
                    ion_state_meta[ptr] = (n, "f")
                return buf

            def padded(i):
                return view(a[i], side * side * (side // 2 + 1) * 2, "f").reshape(
                    side, side, -1
                )

            def launch(kernel, size, *args):
                kernel(((size + 255) // 256,), (256,), args)

            if stage == 0:
                side = int(p[0])
                clipped = cp.clip(grid(0).astype("d") * p[1], p[2], p[3]).astype("f")
                spectrum = cp.ascontiguousarray(
                    transform(clipped.reshape((side,) * 3), side, False, int(p[4]))
                )
                # The native -Ofast loop multiplies real and imaginary components by
                # the same rounded reciprocal, including on non-power-of-two grids.
                spectrum.view("f")[:] *= np.float32(1) / np.float32(n)
                if residency:
                    spectra[int(a[1])] = spectrum
                else:
                    spectrum.get(out=padded(1).view("complex64"))
            elif stage == 1:
                side = int(p[0])
                pairs = [(0, int(p[6])), (2, int(p[7]))]
                if p[9]:
                    pairs.append((4, int(p[6])))
                if p[10]:
                    pairs.append((6, int(p[7])))
                if p[11]:
                    pairs.append((8, int(p[6])))
                for field, kind in pairs:
                    ptr_in = int(a[field])
                    spectrum = spectra.get(ptr_in)
                    if spectrum is None:
                        spectrum = cp.asarray(padded(field).view("complex64"))
                    elif p[1] > 0:
                        spectrum = spectrum.copy()
                    if p[1] > 0:
                        launch(
                            window,
                            spectrum.size,
                            spectrum,
                            np.int32(side),
                            np.float64(p[2]),
                            np.float32(p[3]),
                            np.float32(p[4]),
                            np.float64(p[5]),
                            np.int32(kind),
                        )
                    real = cp.ascontiguousarray(
                        transform(spectrum, side, True, int(p[8]))
                    )
                    if residency:
                        filtered_cache[int(a[field + 1])] = real.reshape(-1)
                    else:
                        padded(field + 1)[:, :, :side] = real.get()
            elif stage == 2:
                side = int(p[0])

                def filtered(idx):
                    ptr = int(a[idx])
                    cached = filtered_cache.get(ptr)
                    if cached is not None:
                        return cached
                    return cp.asarray(padded(idx)[:, :, :side], order="C")

                unused = cp.empty(1, dtype="f")
                has_temp = bool(a[6])
                has_mfp = bool(a[12])
                mutable = tuple(
                    i
                    for i in ((4, 5, 6, 11, 12) if p[11] else (4, 5, 6))
                    if (i != 6 or has_temp) and (i != 12 or has_mfp)
                )
                buf_4 = ion_buf(4) if residency else grid(4)
                buf_5 = ion_buf(5) if residency else grid(5)
                buf_6 = (
                    (ion_buf(6) if residency else grid(6)) if has_temp else np.uint64(0)
                )
                arrays = [
                    grid(0),
                    filtered(1),
                    filtered(2),
                    grid(3),
                    buf_4,
                    buf_5,
                    buf_6,
                ]
                arrays.extend((filtered(7), grid(8)) if p[10] else (unused, unused))
                if p[11]:
                    buf_11 = ion_buf(11) if residency else grid(11)
                    buf_12 = (
                        (ion_buf(12) if residency else grid(12))
                        if has_mfp
                        else np.uint64(0)
                    )
                    arrays.extend(
                        (
                            filtered(9) if p[12] else grid(9),
                            filtered(10),
                            buf_11,
                            buf_12,
                        )
                    )
                else:
                    arrays.extend((unused,) * 4)
                launch(
                    barrier,
                    n,
                    *arrays,
                    cp.asarray(np.array(p[:16], dtype="d")),
                    np.int32(n),
                )
                if not residency:
                    for i in mutable:
                        arrays[i].get(out=view(a[i], n, "f"))
                        if (
                            engine is not None
                            and engine._record_pending
                            and engine._pending_outputs is not None
                        ):
                            engine._pending_outputs[int(a[i])] = arrays[i]
            elif stage == 3:
                density, neutral, zre = map(grid, range(3))
                result = ion_buf(3) if residency else grid(3)
                floor = (
                    grid(4)
                    if p[4]
                    else (p[2] * (1.0 + p[3] * density.astype("d"))).astype("f")
                )
                invalid = cp.zeros(1, dtype="i")
                launch(
                    temperature,
                    n,
                    density,
                    neutral,
                    zre,
                    floor,
                    result,
                    np.int32(n),
                    np.float32(p[0]),
                    np.float32(p[1]),
                    np.float32(9e-8),
                    invalid,
                )
                if invalid.item():
                    raise ValueError("Nonfinite CHMF gas temperature")
                if not residency:
                    result.get(out=view(a[3], n, "f"))
                    if (
                        engine is not None
                        and engine._record_pending
                        and engine._pending_outputs is not None
                    ):
                        engine._pending_outputs[int(a[3])] = result
            elif stage == 4:
                side = int(p[0])
                bounds = [(0, p[1]), (1, 0.0)]
                bounds.extend(
                    (idx, 0.0)
                    for idx, flag in ((3, p[2]), (4, p[3]), (5, p[4]))
                    if flag
                )
                for idx, lower in bounds:
                    ptr = int(a[idx])
                    arr_dev = filtered_cache.get(ptr)
                    if arr_dev is not None:
                        clipped = cp.fmax(arr_dev, np.float32(lower))
                        if idx == 3:
                            clipped = cp.fmin(clipped, np.float32(0.999))
                        filtered_cache[ptr] = clipped
                    else:
                        field = padded(idx)[:, :, :side]
                        clipped = cp.fmax(cp.asarray(field), np.float32(lower))
                        if idx == 3:
                            clipped = cp.fmin(clipped, np.float32(0.999))
                        field[:] = clipped.get()
                    if idx == 1:
                        view(a[2], 1, "d")[0] = float(cp.mean(clipped, dtype="d"))
            elif stage == 5:
                rule = cp.asarray(json.loads((ROOT / "gk61.json").read_text()))
                rec_tables = build_mhr(
                    module([SHARED / "table_qag.cuh", SHARED / "mhr.cu"]), rule, *p[:3]
                )
                max_log = np.float64(rec_tables[0][-1].item() - 1.0e-7)
            elif stage == 6:
                if rec_tables is None:
                    raise RuntimeError(
                        "MHR table unavailable for cumulative recombinations"
                    )
                result = cp.empty(n, dtype="f")
                invalid = cp.zeros(1, dtype="i")
                launch(
                    kernels.get_function("recombine"),
                    n,
                    *map(grid, range(4)),
                    result,
                    *rec_tables,
                    np.int32(n),
                    *map(np.float64, p[:3]),
                    np.float64(np.float32(0.2)),
                    max_log,
                    invalid,
                )
                if invalid.item():
                    raise ValueError("Nonfinite cumulative recombinations")
                result.get(out=view(a[4], n, "f"))
                if (
                    engine is not None
                    and engine._record_pending
                    and engine._pending_outputs is not None
                ):
                    engine._pending_outputs[int(a[4])] = result
            elif stage == 7:
                from halo_mcg_global import evaluate

                rows = view(a[0], n * 2 * 25, "d").reshape(n, 2, 25)
                data = evaluate(rows, view(a[1], 1, "d"))
                view(a[2], n * 2, "d")[:] = data.ravel()
                for column, pop in enumerate(("acg", "mcg")):
                    name = f"ionization_global_nion_{pop}_integrals"
                    counters[name] = counters.get(name, 0) + int(
                        np.count_nonzero(rows[:, column, 24] == 0.0)
                    )
            elif stage == 8:
                view(a[1], 1, "d")[0] = grid(0).mean(dtype="d").item()
            else:
                raise ValueError(f"Unexpected ionization stage {stage}")
            name = stage_names[stage]
            counters[name] = counters.get(name, 0) + 1
            return 0
        except BaseException as exc:
            errors.append(exc)
            return 1

    pointer = C.c_void_p.in_dll(adapter, "halo_ion_dispatch")
    pointer.value = C.cast(callback, C.c_void_p).value
    if use_mhr:
        from py21cmfast.drivers import _global_initialization as initialization

        delegate = initialization.lib
        enabled = C.c_int.in_dll(adapter, "halo_use_mhr")
        enabled.value = 1
        stack.callback(setattr, enabled, "value", 0)

        class MHRInit:
            def __getattr__(self, name):
                return getattr(delegate, name)

            def init_MHR(self):
                if adapter.halo_init_mhr():
                    raise errors.pop()

            def free_MHR(self):
                nonlocal rec_tables, max_log
                rec_tables = max_log = None

        stack.enter_context(patch.object(initialization, "lib", MHRInit()))
    native = ffi.cast(
        ffi.typeof(lib.ComputeIonizedBox),
        C.cast(adapter.ComputeIonizedBox, C.c_void_p).value,
    )

    # ctypes only stores the callback address. Keep its owner until the native hook is removed.
    stack.callback(setattr, pointer, "value", None)
    stack.callback(lambda owner=callback: None)

    def compute(*args):
        try:
            status = native(*args)
            if errors:
                raise errors.pop()
            if status == 0 and ion_state:
                for ptr, arr_dev in ion_state.items():
                    size, dtype = ion_state_meta[ptr]
                    arr_dev.get(out=view(ptr, size, dtype).reshape(arr_dev.shape))
                    if (
                        engine is not None
                        and engine._record_pending
                        and engine._pending_outputs is not None
                    ):
                        engine._pending_outputs[ptr] = arr_dev
            return status
        finally:
            spectra.clear()
            filtered_cache.clear()
            ion_state.clear()
            ion_state_meta.clear()

    stack.enter_context(
        patch.object(IonizedBox, "_c_compute_function", staticmethod(compute))
    )
