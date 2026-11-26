__device__ __forceinline__ void eval_halo_barrier_inputs(
    int i, const float *raw, const float *filtered, const float *photons, const float *electron,
    const float *cumulative_recombinations, const double *p, double &density, double &nion,
    double &mcg_floor, double &xe, double &rec_per_baryon) {
    bool last = p[1] == 0.;
    density = last ? raw[i] * p[3] : filtered[i];
    nion = p[4] * photons[i];
    nion *= 1. / (1. + density);
    if (p[6]) nion = fmax(nion, p[5]);
    // The Lagrangian photon grid already includes both populations. The
    // separate MCG contribution here is only the native high-z lower bound.
    mcg_floor = p[15];
    xe = p[10] ? electron[i] : 0.;
    int rec_idx = p[16] ? 0 : i;
    rec_per_baryon = p[11] ? cumulative_recombinations[rec_idx] / (1. + density) : 0.;
}

extern "C" __global__ void halo_barrier(const float *raw, const float *filtered,
                                        const float *photons, const float *old_z, float *neutral,
                                        float *zre, float *temperature, const float *electron,
                                        const float *neutral_tk,
                                        const float *cumulative_recombinations, const float *sfr,
                                        float *gamma12, float *mean_free_path, const double *p,
                                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double density, nion, mcg_floor, xe, rec_per_baryon;
    eval_halo_barrier_inputs(i, raw, filtered, photons, electron, cumulative_recombinations, p,
                             density, nion, mcg_floor, xe, rec_per_baryon);
    bool last = p[1] == 0.;
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

extern "C" __global__ void halo_barrier_sphere_mark(
    const float *raw, const float *filtered, const float *photons, const float *old_z,
    const float *neutral, float *zre, const float *electron, const float *cumulative_recombinations,
    const float *sfr, const double *p, int n, unsigned char *triggered, unsigned char *eligible,
    float *gamma_cand) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double density, nion, mcg_floor, xe, rec_per_baryon;
    eval_halo_barrier_inputs(i, raw, filtered, photons, electron, cumulative_recombinations, p,
                             density, nion, mcg_floor, xe, rec_per_baryon);
    if (exceeds_ionization_barrier(nion + mcg_floor, xe, rec_per_baryon)) {
        triggered[i] = 1;
        if (eligible) {
            bool ok = p[11] && neutral[i] > 1.e-7;
            eligible[i] = ok ? 1 : 0;
            if (ok) gamma_cand[i] = (float)(p[13] * p[14] / (1. + density) * sfr[i]);
        }
        zre[i] = old_z[i] < 0.f ? (float)p[2] : old_z[i];
    } else {
        triggered[i] = 0;
        if (eligible) eligible[i] = 0;
    }
}

extern "C" __global__ void halo_barrier_sphere_partial(
    const float *raw, const float *filtered, const float *photons, float *neutral,
    float *temperature, const float *electron, const float *neutral_tk,
    const unsigned char *triggered, const int *min_painter, const double *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int painter = min_painter ? min_painter[i] : 0x7fffffff;
    if (!triggered[i] && (neutral[i] > 1.e-30) && painter > i) {
        double density = raw[i] * p[3];
        double nion = p[4] * photons[i];
        nion *= 1. / (1. + density);
        if (p[6]) nion = fmax(nion, p[5]);
        double residual = 1. - nion - p[15];
        if (temperature) {
            float tk = p[10] ? neutral_tk[i] : (float)(p[7] * (1. + p[8] * raw[i]));
            temperature[i] = ComputePartiallyIonizedTemperature(tk, (float)residual, (float)p[9]);
        }
        double xe = p[10] ? electron[i] : 0.;
        neutral[i] = fmin(fmax(residual - xe, 0.), 1.);
    }
    if (painter < 0x7fffffff) neutral[i] = 0.f;
}
