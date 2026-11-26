// Inserted after the native v4 headers in the two source-table callers.
// The native table is retained as an independent oracle during acceptance.
static void gpu_source_table(double z, double lo, double hi, double mmin, double mmax, double cell,
                             ScalingConstants *sc, int sfr) {
    if (matter_options_global->SOURCE_MODEL == SOURCE_MODEL_CONST_ION_EFF) {
        float sigma_min = sfr ? sc->sigma_min_sfr : sc->sigma_min_ion;
        float sigma_cell = evaluate_sigma(log(cell));
        if (sigma_cell > sigma_min) Throw(ValueError);
        // dfcoll_dz_conditional_eps takes float z, and its helper takes the
        // two perturbed redshifts as floats before evaluating dicke(double).
        float z_sfr = sc->redshift;
        double p[19] = {z,
                        lo,
                        hi,
                        sigma_min,
                        sigma_cell,
                        dicke(z),
                        sfr ? dicke((float)(z_sfr + .001)) : 0.,
                        sfr ? dicke((float)(z_sfr - .001)) : 0.,
                        physconst.delta_c_sph,
                        RHOcrit,
                        cosmo_params_global->OMb,
                        sc->fstar_10_acg,
                        sc->sfr_timescale,
                        astro_params_global->HII_EFF_FACTOR};
        p[18] = sfr ? 4 : 3;
        GPU_CALL(6, p, NULL);
        return;
    }
    if (astro_options_global->USE_MCGS ||
        astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK ||
        !source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL))
        Throw(ValueError);
    double g = dicke(z), sig = evaluate_sigma(log(cell));
    double p[] = {z,
                  lo,
                  hi,
                  log(mmin),
                  log(mmax),
                  log(cell),
                  g,
                  sig,
                  get_delta_crit(matter_options_global->HMF, sig, g),
                  physconst.delta_c_sph,
                  sc->mturn_acg_homogeneous,
                  sc->alpha_star_acg,
                  sfr ? 0. : sc->alpha_esc,
                  log(sc->fstar_10_acg),
                  sfr ? 0. : log(sc->fesc_10_acg),
                  log(sc->Mlim_Fstar_acg),
                  sfr ? 0. : log(sc->Mlim_Fesc_acg),
                  sfr ? RHOcrit * cosmo_params_global->OMb * sc->fstar_10_acg / sc->sfr_timescale
                      : sc->fstar_10_acg * sc->fesc_10_acg * sc->pop2_ion,
                  sfr,
                  astro_options_global->INTEGRATION_METHOD_ACGS,
                  matter_options_global->HMF,
                  physconst.delta_c_delos};
    GPU_CALL(6, p, NULL);
}
static void gpu_init_nion(double z, double lo, double hi, double mmin, double mmax, double cell,
                          ScalingConstants *sc, bool prev) {
    if (GPU_TABLE_AUDIT) initialize_nion_conditional_tables(z, lo, hi, mmin, mmax, cell, sc, prev);
    gpu_source_table(z, lo, hi, mmin, mmax, cell, sc, 0);
}
static void gpu_init_sfr(double z, double lo, double hi, double mmin, double mmax, double cell,
                         ScalingConstants *sc) {
    if (GPU_TABLE_AUDIT) initialize_sfrd_conditional_tables(z, lo, hi, mmin, mmax, cell, sc);
    gpu_source_table(z, lo, hi, mmin, mmax, cell, sc, 1);
}
static void gpu_dndm_tables(double xmin, double xmax, double ymin, double ymax,
                            double growth, double lnM_cond, int unused) {
    if (!uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)) {
        initialize_dndm_tables(xmin, xmax, ymin, ymax, growth, lnM_cond, unused);
        return;
    }
    if (GPU_TABLE_AUDIT) initialize_dndm_tables(xmin, xmax, ymin, ymax, growth, lnM_cond, unused);
    double sigma = evaluate_sigma(lnM_cond);
    double p[] = {xmin, xmax, ymin, ymax, growth, lnM_cond, sigma, physconst.delta_c_sph,
                  RHOcrit * cosmo_params_global->OMm, matter_options_global->HMF,
                  physconst.delta_c_delos,
                  get_delta_crit(matter_options_global->HMF, sigma, growth),
                  simulation_options_global->N_COND_INTERP};
    GPU_CALL(35, p, NULL);
}
