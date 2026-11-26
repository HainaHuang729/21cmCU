// Populations may select different quadrature rules for the same physical grid.
static bool use_mcg_conditional_gpu(void) {
    int atomic = astro_options_global->INTEGRATION_METHOD_ACGS;
    int mini = astro_options_global->INTEGRATION_METHOD_MCGS;
    return (astro_options_global->USE_MCGS ||
            astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK) &&
           source_model_uses_lagrangian_grids(matter_options_global->SOURCE_MODEL) &&
           uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES) &&
           (atomic == INTEGRATION_METHOD_GAUSS_LEGENDRE || atomic == INTEGRATION_METHOD_GSL_QAG ||
            atomic == INTEGRATION_METHOD_GAMMA_APPROX) &&
           (!astro_options_global->USE_MCGS ||
            mini == INTEGRATION_METHOD_GAUSS_LEGENDRE || mini == INTEGRATION_METHOD_GSL_QAG ||
            mini == INTEGRATION_METHOD_GAMMA_APPROX);
}

int (*cuda_mcg_table)(int, int, int, int, int, double *);

static void gpu_pair_tables(double z, double lo, double hi, double mmin, double mmax, double cell,
                            ScalingConstants *sc, int kind, bool prev) {
    bool photo = astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK &&
                 source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL);
    bool mcg = astro_options_global->USE_MCGS;
    if (kind == 2 &&
        (astro_options_global->INTEGRATION_METHOD_ACGS == INTEGRATION_METHOD_GAMMA_APPROX ||
         (mcg && astro_options_global->INTEGRATION_METHOD_MCGS == INTEGRATION_METHOD_GAMMA_APPROX))) {
        initialize_xray_emissivity_conditional_tables(z, lo, hi, mmin, mmax, cell, sc);
        return;
    }
    double growth = dicke(z), sigma = evaluate_sigma(log(cell));
    int max_mini = mcg ? 1 : 0;
    for (int mini = 0; mini <= max_mini; ++mini) {
        double star = mini ? sc->fstar_7_mcg : sc->fstar_10_acg;
        double escape = mini ? sc->fesc_7_mcg : sc->fesc_10_acg;
        double prefactor = kind == 0
                               ? star * escape * (mini ? sc->pop3_ion : sc->pop2_ion)
                               : RHOcrit * cosmo_params_global->OMb * star / sc->sfr_timescale;
        int method = mini ? astro_options_global->INTEGRATION_METHOD_MCGS
                          : astro_options_global->INTEGRATION_METHOD_ACGS;
        int ny = (mini || photo) ? NMTURN : 1;
        double ymin = mini ? LOG10_MTURN_MCG_MIN
                           : (photo ? LOG10_MTURN_ACG_MIN : log10(sc->mturn_acg_homogeneous));
        double ymax = mini ? LOG10_MTURN_MCG_MAX : (photo ? LOG10_MTURN_ACG_MAX : ymin);
        double p[] = {z,
                      lo,
                      hi,
                      log(mmin),
                      log(mmax),
                      log(cell),
                      growth,
                      sigma,
                      get_delta_crit(matter_options_global->HMF, sigma, growth),
                      physconst.delta_c_sph,
                      (!mini && !photo) ? sc->mturn_acg_homogeneous : 0.,
                      mini ? sc->alpha_star_mcg : sc->alpha_star_acg,
                      kind == 0 ? sc->alpha_esc : 0.,
                      log(star),
                      kind == 0 ? log(escape) : 0.,
                      log(mini ? sc->Mlim_Fstar_mcg : sc->Mlim_Fstar_acg),
                      kind == 0 ? log(mini ? sc->Mlim_Fesc_mcg : sc->Mlim_Fesc_acg) : 0.,
                      prefactor,
                      kind,
                      sc->sfr_timescale,
                      mini ? sc->lx_over_sfr_mcg : sc->lx_over_sfr_acg,
                      physconst.s_per_yr,
                      method,
                      sc->atomic_cooling_threshold,
                      mini && sc->mturn_acg_homogeneous > 1.001 * sc->atomic_cooling_threshold,
                      ymin,
                      ymax,
                      matter_options_global->HMF,
                      physconst.delta_c_delos,
                      0.};
        if (kind == 2) {
            p[14] = cosmo_params_global->OMb;
            p[16] = cosmo_params_global->OMm;
            p[17] = RHOcrit;
        }
        if (cuda_mcg_table(kind, mini, prev, NDELTA, ny, p)) Throw(ValueError);
    }
}
