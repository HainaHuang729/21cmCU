// The caller already applied the upstream scatter convention to sc. Store the
// luminosity-density LUT in its native logarithmic units of 10^38 erg/s/Mpc^3.
static void gpu_init_xray(double z, double lo, double hi, double mmin, double mmax, double cell,
                          ScalingConstants *sc) {
    if (GPU_TABLE_AUDIT)
        initialize_xray_emissivity_conditional_tables(z, lo, hi, mmin, mmax, cell, sc);
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
                  0.,
                  log(sc->fstar_10_acg),
                  cosmo_params_global->OMb,
                  log(sc->Mlim_Fstar_acg),
                  cosmo_params_global->OMm,
                  RHOcrit,
                  2.,
                  sc->sfr_timescale,
                  sc->lx_over_sfr_acg,
                  physconst.s_per_yr,
                  astro_options_global->INTEGRATION_METHOD_ACGS,
                  matter_options_global->HMF,
                  physconst.delta_c_delos};
    GPU_CALL(6, p, NULL);
}
