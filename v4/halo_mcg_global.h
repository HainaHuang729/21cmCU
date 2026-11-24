// Header follows global_integrals.h so both emission modes share global_acg_row.
int (*cuda_mcg_nion_history)(int, int, double *, double *, double *);
static void gpu_mcg_nion_history(int n, float zmin, float zmax, ScalingConstants *sc) {
    // Native redshift interval subtraction is single precision; the division
    // by a runtime interval count and each subsequent redshift are double.
    volatile double intervals = n - 1.;
    double axes[] = {zmin, (zmax - zmin) / intervals, LOG10_MTURN_MCG_MIN,
                     (LOG10_MTURN_MCG_MAX - LOG10_MTURN_MCG_MIN) / (NMTURN - 1.)};
    double turns[NMTURN], rows[n][2][25];
    for (int j = 0; j < NMTURN; ++j) turns[j] = pow(10., axes[2] + j * axes[3]);
    for (int i = 0; i < n; ++i) {
        double z = axes[0] + i * axes[1];
        ScalingConstants sc_z = evolve_scaling_constants_to_redshift(z, sc, false);
        global_population_rows(rows[i], z, log(minimum_source_mass(z, true)), log(M_MAX_INTEGRAL),
                               sc_z.mturn_acg_homogeneous, &sc_z);
    }
    if (cuda_mcg_nion_history(n, NMTURN, axes, &rows[0][0][0], turns)) Throw(ValueError);
}
