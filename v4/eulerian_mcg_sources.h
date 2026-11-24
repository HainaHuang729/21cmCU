// Inputs for two unconditional populations and three quantities, in the same
// units as get_uhmf_averages. The HMF and source-weight kernels are shared.
int (*cuda_eulerian_sources)(int, double *, void **);
static void eulerian_molecular_fields(double mmin, double mmax, float *density, float *acg,
                                      float *mcg, ScalingConstants *sc, EmissivityFields *out) {
    double at = pow(10., out->log10_mturn_acg_ave), mt = pow(10., out->log10_mturn_mcg_ave);
    double rows[3][2][30] = {0};
    for (int kind = 0; kind < 3; ++kind) {
        for (int mini = 0; mini < 2; ++mini) {
            double *p = rows[kind][mini];
            global_acg_row(p, sc->redshift, log(mmin), log(mmax), at, sc, kind != 0);
            p[0] = sc->redshift;
            if (mini) {
                p[10] = mt;
                p[11] = sc->alpha_star_mcg;
                p[13] = log(sc->fstar_7_mcg);
                p[15] = log(sc->Mlim_Fstar_mcg);
                if (!kind) {
                    p[14] = log(sc->fesc_7_mcg);
                    p[16] = log(sc->Mlim_Fesc_mcg);
                    p[17] = sc->fstar_7_mcg * sc->fesc_7_mcg * sc->pop3_ion;
                } else {
                    p[17] =
                        RHOcrit * cosmo_params_global->OMb * sc->fstar_7_mcg / sc->sfr_timescale;
                }
                p[23] = sc->atomic_cooling_threshold;
                p[24] = at > sc->atomic_cooling_threshold * 1.001;
            }
            if (kind == 2) {
                p[14] = cosmo_params_global->OMb;
                p[16] = cosmo_params_global->OMm;
                p[17] = RHOcrit * cosmo_params_global->OMm;
                p[19] = sc->sfr_timescale;
                p[20] = mini ? sc->lx_over_sfr_mcg : sc->lx_over_sfr_acg;
                p[21] = physconst.s_per_yr;
            }
            p[25] = matter_options_global->HMF;
            p[26] = physconst.delta_c_delos;
            p[27] = cosmo_params_global->OMm;
            p[28] = cosmo_params_global->OMl;
            p[29] = cosmo_params_global->OMr;
        }
    }
    double params[] = {sc->atomic_cooling_threshold * 1.001, sc->lx_over_sfr_acg,
                       sc->lx_over_sfr_mcg, astro_options_global->USE_METALLICITY, FRACT_FLOAT_ERR};
    void *a[] = {rows,       density,       acg,           mcg,
                 out->n_ion, out->sfrd_acg, out->sfrd_mcg, out->xray_emissivity};
    if (cuda_eulerian_sources(HII_TOT_NUM_PIXELS, params, a)) Throw(ValueError);
}
