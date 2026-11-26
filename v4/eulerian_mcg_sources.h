// Inputs for two unconditional populations and three quantities, in the same
// units as get_uhmf_averages. The HMF and source-weight kernels are shared.
int (*cuda_eulerian_sources)(int, double *, void **);
static void eulerian_molecular_fields(double mmin, double mmax, float *density, float *acg,
                                      float *mcg, ScalingConstants *sc, EmissivityFields *out) {
    bool photo = astro_options_global->USE_REIONIZATION_PHOTOHEATING_FEEDBACK &&
                 source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL);
    bool use_mcg = astro_options_global->USE_MCGS;
    bool thermal = astro_options_global->USE_TS_FLUCT;
    bool extra = config_settings.EXTRA_EMISSIVITY_FIELDS;
    double at = photo ? pow(10., out->log10_mturn_acg_ave) : sc->mturn_acg_homogeneous;
    double mt = use_mcg ? pow(10., out->log10_mturn_mcg_ave) : 0.;
    double rows[3][2][30] = {0};
    for (int kind = 0; kind < 3; ++kind) {
        for (int mini = 0; mini < 2; ++mini) {
            double *p = rows[kind][mini];
            if ((mini && !use_mcg) || (kind == 1 && !thermal && !extra) ||
                (kind == 2 && (!thermal || !astro_options_global->USE_METALLICITY))) {
                p[24] = 1.;
                continue;
            }
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
    double dndm_rows[2][23];
    if (extra) {
        global_acg_row(dndm_rows[0], sc->redshift, log(mmin), log(mmax), 0., sc, false);
        dndm_rows[0][11] = 0.; dndm_rows[0][12] = 0.; dndm_rows[0][13] = 0.; dndm_rows[0][14] = 0.;
        dndm_rows[0][15] = -1000.; dndm_rows[0][16] = 0.;
        dndm_rows[0][17] = RHOcrit * cosmo_params_global->OMm * VOLUME / HII_TOT_NUM_PIXELS;
        memcpy(dndm_rows[1], dndm_rows[0], sizeof(dndm_rows[0]));
        dndm_rows[1][15] = 0.;
        dndm_rows[1][17] = RHOcrit * cosmo_params_global->OMm;
    }
    double rho_m = RHOcrit * cosmo_params_global->OMm;
    double params[] = {sc->atomic_cooling_threshold * 1.001, sc->lx_over_sfr_acg,
                       sc->lx_over_sfr_mcg, astro_options_global->USE_METALLICITY, FRACT_FLOAT_ERR,
                       extra, sc->M_cell / rho_m, rho_m, sc->sfr_timescale,
                       thermal, use_mcg, photo, sc->mturn_acg_homogeneous};
    void *a[] = {rows,
                 extra ? dndm_rows : NULL,
                 density,
                 photo ? acg : NULL,
                 use_mcg ? mcg : NULL,
                 out->n_ion,
                 thermal ? out->sfrd_acg : NULL,
                 (thermal && use_mcg) ? out->sfrd_mcg : NULL,
                 thermal ? out->xray_emissivity : NULL,
                 extra ? out->halo_number : NULL,
                 extra ? out->halo_mass_density : NULL,
                 extra ? out->stellar_mass_density_acg : NULL,
                 (extra && use_mcg) ? out->stellar_mass_density_mcg : NULL};
    if (cuda_eulerian_sources(HII_TOT_NUM_PIXELS, params, a)) Throw(ValueError);
}
