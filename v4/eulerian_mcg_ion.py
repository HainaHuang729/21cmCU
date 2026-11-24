"""Eulerian molecular ionization uses the current LW grid and two photon populations."""

import ctypes as C

import cupy as cp
import numpy as np

from evolution_backend import CB, insert, view
from fft_plan_cuda import transform
from halo_ionization import _unconditional_source
from halo_mcg_global import evaluate
from halo_mcg_turnover import device_turnovers


def source(native):
    native = _unconditional_source(native, eulerian=True)
    native = "int (*cuda_eulerian_mcg_ion)(int,int,double*,void**);\n" + native
    active = """(matter_options_global->SOURCE_MODEL == SOURCE_MODEL_E_INTEGRAL &&
        astro_options_global->USE_MCGS && simulation_options_global->NON_CUBIC_FACTOR == 1. &&
        !matter_options_global->USE_FFTW_WISDOM &&
        astro_options_global->HII_FILTER >= 0 && astro_options_global->HII_FILTER <= 2)"""
    native = insert(
        native,
        "void calculate_mcrit_boxes(",
        "    double log10_mturn_mcg_avg",
        """
    if (ACTIVE) {
        bool history = consts->redshift < simulation_options_global->Z_HEAT_MAX;
        bool photo = astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK;
        bool streaming = matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS;
        const ScalingConstants *sc = &consts->scale_consts;
        double p[]={consts->redshift,sc->mturn_acg_homogeneous,
            astro_params_global->M_TURN_STELLAR_FEEDBACK,sc->vcb_const,
            cosmo_tables_global->V_CB_AVG * sqrt(3*M_PI/8),
            astro_params_global->A_LW,astro_params_global->BETA_LW,
            astro_params_global->A_VCB,astro_params_global->BETA_VCB,simulation_options_global->HII_DIM};
        void *a[]={history ? spin_temp->J_21_LW : NULL,
            streaming ? ini_boxes->lowres_vcb : NULL,
            photo && history ? prev_ionbox->ionisation_rate_G12 : NULL,
            photo && history ? prev_ionbox->z_reion : NULL,
            photo ? log10_mturn_acg_grid : NULL,log10_mturn_mcg_grid,
            log10_mturn_acg_avg_out,log10_mturn_mcg_avg_out};
        if (cuda_eulerian_mcg_ion(0,HII_TOT_NUM_PIXELS,p,a)) Throw(ValueError);
        return;
    }
""".replace("ACTIVE", active),
    )
    # Store ordinary spectra on device. The dual-population radius hook is their sole reader.
    start = native.index("void prepare_box_for_filtering(")
    body = native.index("{", start) + 1
    native = (
        native[:body]
        + """
    if (ACTIVE) {
        double p[]={simulation_options_global->HII_DIM,const_factor,limit_min,limit_max,
                    simulation_options_global->N_THREADS,1.};
        void *a[]={input_box,output_c_box};
        GPU_CALL(11,p,a); return;
    }
""".replace("ACTIVE", active)
        + native[body:]
    )
    for population in ("acg", "mcg"):
        grid = f"grid_struct->log10_mturn_{population}_grid_unfiltered"
        target = native.index(grid + "[ct] /=")
        start = native.rindex("dft_r2c_cube(", 0, target)
        end = native.index("}", target) + 1
        call = """
        if (ACTIVE) {
            double p[]={simulation_options_global->HII_DIM,simulation_options_global->N_THREADS};
            void *a[]={GRID};
            if (cuda_eulerian_mcg_ion(1,HII_TOT_NUM_PIXELS,p,a)) Throw(ValueError);
        } else {
        """.replace("GRID", grid).replace("ACTIVE", active)
        native = (
            native[:start] + call + native[start:end] + "\n        }" + native[end:]
        )
    native = insert(
        native,
        "void copy_filter_transform(",
        "    memcpy(",
        """
    if (ACTIVE) {
        bool photo = astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK;
        double p[]={box_dim[0],rspec.R_index,(float)rspec.R,consts->hii_filter,
                    simulation_options_global->N_THREADS,simulation_options_global->BOX_LEN};
        void *a[]={fg_struct->deltax_unfiltered,fg_struct->deltax_filtered,
            fg_struct->prev_deltax_unfiltered,fg_struct->prev_deltax_filtered,
            photo ? fg_struct->log10_mturn_acg_grid_unfiltered : NULL,photo ? fg_struct->log10_mturn_acg_grid_filtered : NULL,
            fg_struct->log10_mturn_mcg_grid_unfiltered,fg_struct->log10_mturn_mcg_grid_filtered,
            astro_options_global->USE_TS_FLUCT ? fg_struct->xe_unfiltered : NULL,
            astro_options_global->USE_TS_FLUCT ? fg_struct->xe_filtered : NULL,
            consts->filter_recombinations ? fg_struct->N_rec_unfiltered : NULL,
            consts->filter_recombinations ? fg_struct->N_rec_filtered : NULL};
        if (cuda_eulerian_mcg_ion(2,HII_TOT_NUM_PIXELS,p,a)) Throw(ValueError);
        return;
    }
""".replace("ACTIVE", active),
    )
    return insert(
        native,
        "void find_ionised_regions(",
        "#pragma omp parallel",
        """
    if (ACTIVE && !astro_options_global->IONISE_ENTIRE_SPHERE) {
        bool ts=astro_options_global->USE_TS_FLUCT, rec=uses_recombination(astro_options_global->RECOMB_MODEL);
        index_huge offset=(index_huge)fc_r_idx*HII_TOT_NUM_PIXELS;
        double p[]={box_dim[2],rspec.R_index,rspec.R,consts->redshift,consts->photoncons_adjustment_factor,
            mean_fix_term_acg,nion_limit_acg,consts->mass_dep_zeta,consts->gamma_prefactor,consts->T_re,
            ts,rec,consts->filter_recombinations,ts?0.:consts->TK_nofluct,ts?0.:consts->adia_TK_term,
            astro_options_global->RECOMB_MODEL == RECOMB_MODEL_HOMOGENEOUS,mean_fix_term_mcg,nion_limit_mcg};
        void *a[]={perturbed_field->density,fg_struct->deltax_filtered,box->nion_conditional_filtered_acg+offset,
            ts?(void*)fg_struct->xe_filtered:NULL,rec?(consts->filter_recombinations?
            (void*)fg_struct->N_rec_filtered:(void*)previous_ionize_box->cumulative_recombinations):NULL,
            previous_ionize_box->z_reion,ts?spin_temp->kinetic_temp_neutral:NULL,box->neutral_fraction,box->z_reion,
            (rec && !matter_options_global->MINIMIZE_MEMORY)?box->mean_free_path:NULL,rec?box->ionisation_rate_G12:NULL,
            matter_options_global->MINIMIZE_MEMORY?NULL:box->kinetic_temperature,
            box->nion_conditional_filtered_mcg+offset};
        if (cuda_eulerian_mcg_ion(3,HII_TOT_NUM_PIXELS,p,a)) Throw(ValueError);
        return;
    }
""".replace("ACTIVE", active),
    )


def bind(adapter, engine):
    counts = engine.eulerian_mcg_stages
    names = (
        "turnover_grids",
        "turnover_forward_fft",
        "filter_radius",
        "ion_center",
        "global_nion",
    )

    def count(label, increment):
        name = "eulerian_mcg_ion_" + label
        counts[name] = counts.get(name, 0) + increment

    @CB
    def dispatch(stage, n, p, a):
        try:
            if stage == 0:
                side = int(p[9])
                width = 2 * (side // 2 + 1)

                def store(idx, data):
                    padded = view(a[idx], side * side * width, "f").reshape(
                        side, side, width
                    )
                    padded[..., :side] = data.get().reshape(side, side, side)

                acg, mcg, grid_a, grid_m = device_turnovers(
                    n, p, a, registry=engine.device_registry
                )
                if a[4]:
                    store(4, grid_a)
                    average = acg.mean().item()
                else:
                    average = np.log10(p[1])
                store(5, grid_m)
                view(a[6], 1, "d")[0] = average
                view(a[7], 1, "d")[0] = mcg.mean().item()
                if a[4]:
                    engine.ion_filtered[int(a[4])] = grid_a
                engine.ion_filtered[int(a[5])] = grid_m
            elif stage == 1:
                side, threads = map(int, p[:2])
                width = 2 * (side // 2 + 1)
                resident = engine.ion_filtered.get(int(a[0]))
                if resident is None:
                    real = view(a[0], side * side * width, "f").reshape(
                        side, side, width
                    )[..., :side]
                    source_grid = cp.asarray(real, order="C")
                else:
                    source_grid = resident.reshape(side, side, side)
                spectrum = cp.ascontiguousarray(
                    transform(source_grid, side, False, threads)
                )
                spectrum *= np.float32(1.0 / n)
                engine.ion_spectra[int(a[0])] = spectrum
            elif stage in (2, 3):
                if engine.dispatch(31 if stage == 2 else 2, n, p, a, molecular=True):
                    return 1
            elif stage == 4:
                rows = view(a[0], n * 2 * 25, "d").reshape(n, 2, 25)
                view(a[2], n * 2, "d")[:] = evaluate(rows, view(a[1], 1, "d")).ravel()
                for i, pop in enumerate(("acg", "mcg")):
                    count(
                        "global_nion_" + pop,
                        int(np.count_nonzero(rows[:, i, 24] == 0.0)),
                    )
            else:
                raise ValueError(
                    f"Unexpected Eulerian molecular ionization stage {stage}"
                )
            count(names[stage], 1)
            return 0
        except BaseException as exc:
            engine.error = exc
            return 1

    C.c_void_p.in_dll(adapter, "cuda_eulerian_mcg_ion").value = C.cast(
        dispatch, C.c_void_p
    ).value
    return dispatch
