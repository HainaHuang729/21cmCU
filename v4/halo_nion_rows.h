// Both ionization and radiation supply the relevant ScalingConstants: the
// current node for all ionization epochs, and one evolved set per radiation row.
static void global_population_rows(double rows[2][25], double z, double ln_min, double ln_max,
                                   double turn_acg, const ScalingConstants *sc) {
    double *acg = rows[0], *mcg = rows[1];
    global_acg_row(acg, z, ln_min, ln_max, turn_acg, sc, false);
    acg[23] = sc->atomic_cooling_threshold;
    acg[24] = 0.;
    memcpy(mcg, acg, 25 * sizeof(double));
    mcg[11] = sc->alpha_star_mcg;
    mcg[13] = log(sc->fstar_7_mcg);
    mcg[14] = log(sc->fesc_7_mcg);
    mcg[15] = log(sc->Mlim_Fstar_mcg);
    mcg[16] = log(sc->Mlim_Fesc_mcg);
    mcg[17] = sc->fstar_7_mcg * sc->fesc_7_mcg * sc->pop3_ion;
    mcg[24] = turn_acg > 1.001 * sc->atomic_cooling_threshold;
}
