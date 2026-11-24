// Emission-frame energy integral across the rest of a stellar spectral band.
__device__ double stellar_lw(double f, int band, const float *nu, const double *c,
                             const double *a) {
    float exponent = (float)a[band] + 2.f;
    double upper = band < 21 ? (double)nu[band + 1] : 4. / 3.;
    double energy =
        ((float)c[band] / exponent) * (pow(upper, (double)exponent) - pow(f, (double)exponent));
    return energy > 0. ? energy : 1e-40;
}

__device__ int stellar_band(double f, const float *nu) {
    for (int b = 0; b < 21; b++)
        if (f >= nu[b] && f < nu[b + 1]) return b;
    return 21;
}

// Each population uses its own spectrum. Shells share the partially illuminated edge.
extern "C" __global__ void lyman_shells(double zp, int nr, double pop2, double pop3, double nu_lya,
                                        double shield, double lw_threshold, const float *stars,
                                        const double *recycle, const double *zshell,
                                        double *output) {
    bool mini = gridDim.x == 2;
    double population = blockIdx.x ? pop3 : pop2;
    int column = blockIdx.x ? 3 : 1;
    double *continuum = output + blockIdx.x * 4 * nr, *injected = continuum + nr;
    double *combined = injected + nr, *lw = combined + nr;
    const int count = 22;
    __shared__ double c[count], a[count];
    __shared__ float nu[count];
    int i = threadIdx.x;
    if (i < count) {
        double n = stars[5 * i];
        nu[i] = (4. / 3.) * (1. - 1. / (n * n));
    }
    __syncthreads();
    if (i < count) {
        a[i] = stars[5 * i + column + 1];
        c[i] = stars[5 * i + column];
        if (i + 1 < count) {
            double exponent = (float)(stars[5 * i + column + 1] + 1.f);
            double delta = pow((double)nu[i + 1], exponent) - pow((double)nu[i], exponent);
            c[i] = (float)(c[i] * (exponent / delta * population));
        }
    }
    __syncthreads();
    if (i) return;
    double prev_c = 0., prev_j = 0.;
    bool edge = false;
    for (int r = 0; r < nr; r++) {
        double z = zshell[r], cc = 0., jj = 0., energy = 0.;
        double ratio = (1. + z) / (1. + zp);
        float limit2 = (1.f + (float)zp) * (1. - 1. / 9.) / .75 - 1.;
        if (mini && z < limit2) {
            double f = lw_threshold * ratio;
            if (f < (1. - 1. / 9.) / .75)
                energy += (1. - shield) * stellar_lw(f, stellar_band(f, nu), nu, c, a);
        }
        for (int n = 23; n >= 2; n--) {
            double line = 1. - 1. / (n * n);
            float limit = (1.f + (float)zp) * (1. - 1. / ((n + 1) * (n + 1))) / line - 1.;
            if (n == 2 ? z >= limit : z > limit) continue;
            double f = (line / .75) * ((1. + z) / (1. + zp));
            int band = stellar_band(f, nu);
            if (mini && n > 2 && f < (1. - 1. / ((n + 1) * (n + 1))) / .75)
                energy += (1. - shield) * stellar_lw(f, band, nu, c, a);
            double j = recycle[n] * (c[band] * pow(f, a[band]) / nu_lya);
            if (n == 2)
                cc = j;
            else
                jj += j;
        }
        if (r > 1 && cc + jj == 0. && prev_c + prev_j > 0. && !edge) {
            float limit = (1.f + (float)zp) * (1. - 1. / 9.) / .75 - 1.;
            float weight = 0;
            // The native optimized loop multiplies by the float32 reciprocal.
            for (int j = 0; j < 1000; j++)
                if (zshell[r - 1] + (z - zshell[r - 1]) * j / 999. > limit) {
                    weight = (float)j * (1.f / 1000.f);
                    break;
                }
            cc = weight * prev_c;
            jj = weight * prev_j;
            edge = true;
        }
        double pref = (1. + zp) * (1. + zp) * (1. + z);
        continuum[r] = pref * cc;
        injected[r] = pref * jj;
        combined[r] = pref * (cc + jj);
        if (mini) lw[r] = (pref * ((1. + zp) * (1. + zp)) / ((1. + z) * (1. + z))) * energy;
        prev_c = cc;
        prev_j = jj;
    }
}
