extern "C" __global__ void halo_barrier(const float *raw, const float *filtered,
                                        const float *photons, const float *old_z, float *neutral,
                                        float *zre, float *temperature, const float *electron,
                                        const float *neutral_tk,
                                        const float *cumulative_recombinations, const float *sfr,
                                        float *gamma12, float *mean_free_path, const double *p,
                                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool last = p[1] == 0.;
    double density = last ? raw[i] * p[3] : filtered[i];
    double nion = p[4] * photons[i];
    nion *= 1. / (1. + density);
    if (p[6] && nion < p[5]) nion = p[5];
    // The Lagrangian photon grid already includes both populations. The
    // separate MCG contribution here is only the native high-z lower bound.
    const double mcg_floor = p[15];
    double xe = p[10] ? electron[i] : 0.;
    double rec_per_baryon = p[11] ? cumulative_recombinations[i] / (1. + density) : 0.;
    if (exceeds_ionization_barrier(nion + mcg_floor, xe, rec_per_baryon)) {
        if (p[11] && neutral[i] > 1.e-7) {
            gamma12[i] = p[13] * p[14] / (1. + density) * sfr[i];
            if (mean_free_path) mean_free_path[i] = (float)p[13];
        }
        zre[i] = old_z[i] < 0.f ? (float)p[2] : old_z[i];
        neutral[i] = 0.f;
    } else if (last && neutral[i] > 1.e-30) {
        double residual = 1. - nion - mcg_floor;
        if (temperature) {
            float tk = p[10] ? neutral_tk[i] : (float)(p[7] * (1. + p[8] * raw[i]));
            temperature[i] = ComputePartiallyIonizedTemperature(tk, (float)residual, (float)p[9]);
        }
        neutral[i] = fmin(fmax(residual - xe, 0.), 1.);
    }
}
