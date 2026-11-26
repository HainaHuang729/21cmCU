// Native cosmological scalars for the unconditional mass integrals.
#include <string.h>
#include "hmf.h"
// A row contains the mass bounds, the growth at one or two redshifts,
// the collapse threshold, and the physical normalization. 4.3.2 casts
// z to float before taking dfcoll/dz, while the mass bounds are held fixed.
static void constant_global_row(double row[12], double z, double ln_min, double ln_max,
                                const ScalingConstants *sc, bool sfr, double timescale) {
    double z_sfr = (float)z;
    row[0] = ln_min;
    row[1] = ln_max;
    row[2] = dicke(sfr ? z_sfr + .001 : z);
    row[3] = sfr ? dicke(z_sfr - .001) : 0.;
    row[4] = physconst.delta_c_sph;
    row[5] = sfr ? RHOcrit * cosmo_params_global->OMb * sc->fstar_10_acg / timescale
                 : astro_params_global->HII_EFF_FACTOR;
    row[6] = sfr ? z_sfr : z;
    row[7] = matter_options_global->HMF;
    row[8] = physconst.delta_c_delos;
    row[9] = cosmo_params_global->OMm;
    row[10] = cosmo_params_global->OMl;
    row[11] = cosmo_params_global->OMr;
}

static bool constant_efficiency(void) {
    return matter_options_global->SOURCE_MODEL == SOURCE_MODEL_CONST_ION_EFF;
}

static int global_mass_kernel(void) {
    if (!constant_efficiency()) return 0;
    return matter_options_global->HMF + 1;
}

static void gpu_global_table(int n, float zmin, float zmax, ScalingConstants *sc, int sfr) {
    // The native library divides by the run-time interval count. Keep GCC
    // from folding a known count into a rounded reciprocal under -Ofast.
    volatile double denominator = n - 1.;
    double width = (zmax - zmin) / denominator;
    double p[] = {n, zmin, width, sfr, global_mass_kernel()};
    if (constant_efficiency()) {
        double rows[n][12];
        for (int i = 0; i < n; ++i) {
            double z = zmin + i * width;
            constant_global_row(rows[i], z, log(minimum_source_mass(z, true)), log(M_MAX_INTEGRAL),
                                sc, sfr, sfr ? dtdz(z) : 0.);
        }
        void *a[] = {rows};
        GPU_CALL(18, p, a);
        return;
    }
    double cosm[] = {cosmo_params_global->OMm,
                     cosmo_params_global->OMl,
                     cosmo_params_global->OMr,
                     cosmo_params_global->OMk,
                     cosmo_params_global->hlittle,
                     cosmo_params_global->wl,
                     astro_params_global->M_TURN_STELLAR_FEEDBACK,
                     astro_options_global->M_MIN_in_Mass,
                     astro_params_global->X_RAY_Tvir_MIN,
                     RHOcrit * cosmo_params_global->OMb,
                     astro_params_global->t_STAR,
                     cosmo_params_global->OMtot};
    double base[] = {0,
                     0,
                     0,
                     0,
                     log(M_MAX_INTEGRAL),
                     0,
                     0,
                     0,
                     0,
                     physconst.delta_c_sph,
                     0,
                     sc->alpha_star_acg,
                     sfr ? 0. : sc->alpha_esc,
                     log(sc->fstar_10_acg),
                     sfr ? 0. : log(sc->fesc_10_acg),
                     log(sc->Mlim_Fstar_acg),
                     sfr ? 0. : log(sc->Mlim_Fesc_acg),
                     sfr ? sc->fstar_10_acg : sc->fstar_10_acg * sc->fesc_10_acg * sc->pop2_ion,
                     matter_options_global->HMF,
                     physconst.delta_c_delos,
                     cosmo_params_global->OMm,
                     cosmo_params_global->OMl,
                     cosmo_params_global->OMr};
    void *a[] = {cosm, base};
    GPU_CALL(18, p, a);
}

// Nion counts photons per baryon. SFRD carries the baryon mass density
// (Msun/Mpc^3) divided by the star-formation timescale (s).
static void global_acg_row(double row[23], double z, double ln_min, double ln_max, double mturn,
                           const ScalingConstants *sc, bool sfr) {
    const double data[23] = {
        z,
        0,
        0,
        ln_min,
        ln_max,
        0,
        dicke(z),
        0,
        0,
        physconst.delta_c_sph,
        mturn,
        sc->alpha_star_acg,
        sfr ? 0. : sc->alpha_esc,
        log(sc->fstar_10_acg),
        sfr ? 0. : log(sc->fesc_10_acg),
        log(sc->Mlim_Fstar_acg),
        sfr ? 0. : log(sc->Mlim_Fesc_acg),
        sfr ? RHOcrit * cosmo_params_global->OMb * sc->fstar_10_acg / sc->sfr_timescale
            : sc->fstar_10_acg * sc->fesc_10_acg * sc->pop2_ion,
        matter_options_global->HMF,
        physconst.delta_c_delos,
        cosmo_params_global->OMm,
        cosmo_params_global->OMl,
        cosmo_params_global->OMr};
    memcpy(row, data, sizeof(data));
}

// The ionization barrier asks for Nion at two epochs with identical mass
// bounds. Emissivity mean fixing asks for Nion and optionally SFRD at one epoch.
static void gpu_global_pair(int stage, int count, const double z[2], bool second_is_sfr,
                            double ln_min, double ln_max, double mturn, const ScalingConstants *sc,
                            double result[2]) {
    double p[] = {count, global_mass_kernel()};
    if (constant_efficiency()) {
        double rows[2][12];
        for (int i = 0; i < count; ++i) {
            constant_global_row(rows[i], z[i], ln_min, ln_max, sc, i && second_is_sfr,
                                sc->sfr_timescale);
        }
        void *a[] = {rows, result};
        GPU_CALL(stage, p, a);
    } else {
        double rows[2][23];
        for (int i = 0; i < count; ++i) {
            global_acg_row(rows[i], z[i], ln_min, ln_max, mturn, sc, i && second_is_sfr);
        }
        void *a[] = {rows, result};
        GPU_CALL(stage, p, a);
    }
}

static void gpu_global_dndm(int stage, double ln_min, double ln_max, const ScalingConstants *sc,
                            double cell_volume, double result[2]) {
    double p[] = {2., 0.};
    double rows[2][23];
    global_acg_row(rows[0], sc->redshift, ln_min, ln_max, 0., sc, false);
    rows[0][11] = 0.; rows[0][12] = 0.; rows[0][13] = 0.; rows[0][14] = 0.;
    rows[0][15] = -1000.; rows[0][16] = 0.;
    rows[0][17] = RHOcrit * cosmo_params_global->OMm * cell_volume;
    memcpy(rows[1], rows[0], sizeof(rows[0]));
    rows[1][15] = 0.;
    rows[1][17] = RHOcrit * cosmo_params_global->OMm;
    void *a[] = {rows, result};
    GPU_CALL(stage, p, a);
}
