__device__ double table_value(float x, const float *values, double low, double step, int count) {
    int j = (int)floor(((double)x - low) / step);
    if (j < 0 || j + 1 >= count) return nan("");
    double t = ((double)x - (low + step * (float)j)) / step;
    return exp(values[j] * (1. - t) + values[j + 1] * t);
}
extern "C" __global__ void emission(int n, const float *density, const float *photons, double pl,
                                    double pw, int pn, const float *sfr, double sl, double sw,
                                    int sn, double lx, float *outp, float *outs, float *outx,
                                    double floor, double *partials) {
    double sums[3] = {};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float d = density[i];
        outp[i] = table_value(d, photons, pl, pw, pn);
        if (outs) {
            outs[i] = table_value(d, sfr, sl, sw, sn) * (1. + d);
            outx[i] = lx * outs[i];
        }
        if (partials) {
            sums[0] += outp[i];
            if (outs) {
                // Native tests the double sum, stores a float divisor and
                // rounds the individual quotients before adding to double means.
                float factor = (1. + d > floor) ? (float)(1. + d) : 1.f;
                sums[1] += outs[i] / factor;
                sums[2] += outx[i] / factor;
            }
        }
    }
    if (!partials) return;
    __shared__ double shared[256];
    for (int j = 0; j < (outs ? 3 : 1); ++j) {
        shared[threadIdx.x] = sums[j];
        __syncthreads();
        for (int off = blockDim.x / 2; off; off /= 2) {
            if (threadIdx.x < off) shared[threadIdx.x] += shared[threadIdx.x + off];
            __syncthreads();
        }
        if (threadIdx.x == 0) partials[j * gridDim.x + blockIdx.x] = shared[0];
        __syncthreads();
    }
}

extern "C" __global__ void normalize_emission(int n, float *nion, float *sfr, float *xray,
                                              const double *sums, double expected_nion,
                                              double expected_sfr, double expected_xray) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    nion[i] *= expected_nion / (sums[0] / n);
    if (sfr) {
        sfr[i] *= expected_sfr / (sums[1] / n);
        xray[i] *= expected_xray / (sums[2] / n);
    }
}
extern "C" __global__ void ion_center(int n, const double *p, const float *density,
                                      const float *filtered, const float *photons, const float *xe,
                                      const float *rec, const float *oldz, const float *tk,
                                      float *xh, float *zr, float *mfp, float *gamma, float *temp,
                                      const float *photons_mcg, int contiguous_filtered) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int side = (int)p[0], j = contiguous_filtered ? i : i / side * (2 * (side / 2 + 1)) + i % side;
    double d = p[1] == 0 ? (double)density[i] * p[4] : filtered[j], ion = p[5] * photons[i];
    if (p[7] && ion < p[6]) ion = p[6];
    double mini = 0.;
    if (photons_mcg) {
        mini = p[16] * photons_mcg[i];
        if (p[7] && mini < p[17]) mini = p[17];
    }
    double all_photons = ion + mini;
    double electron = p[10] ? xe[j] : 0.;
    int rec_index = p[15] ? 0 : (p[12] ? j : i);
    double recombinations = p[11] ? rec[rec_index] / (1. + d) : 0.;
    if (exceeds_ionization_barrier(all_photons, electron, recombinations)) {
        if (p[11] && xh[i] > 1.e-7) {
            gamma[i] = p[2] * p[8] * all_photons;
            if (mfp) mfp[i] = p[2];
        }
        zr[i] = oldz[i] < 0.f ? p[3] : oldz[i];
        xh[i] = 0;
    } else if (p[1] == 0 && xh[i] > 1.e-30) {
        double neutral = 1. - ion - mini;
        if (temp) {
            float f = neutral,
                  neutral_tk = p[10] ? tk[i] : (float)(p[13] * (1. + p[14] * density[i]));
            temp[i] = f <= 0.f   ? (float)p[9]
                      : f >= 1.f ? neutral_tk
                                 : neutral_tk * f + (float)p[9] * (1. - f);
        }
        xh[i] = fmin(fmax(neutral - electron, 0.), 1.);
    }
}
extern "C" __global__ void shell_window(float2 *f, int side, double dk, double inner,
                                        double outer) {
    int i = blockDim.x * blockIdx.x + threadIdx.x, h = side / 2 + 1;
    if (i >= side * side * h) return;
    int x = i / (side * h), y = (i / h) % side, z = i % h;
    float kx = (x > side / 2 ? x - side : x) * dk, ky = (y > side / 2 ? y - side : y) * dk,
          kz = z * dk;
    double k = sqrt((double)(kx * kx + ky * ky + kz * kz)), a = k * inner, b = k * outer;
    double w = b < 1.e-4
                   ? 1. - b * b / 10 * (pow(inner / outer, 5.) - 1) / (pow(inner / outer, 3.) - 1)
                   : 3. / (pow(b, 3.) - pow(a, 3.)) * (sin(b) - cos(b) * b - sin(a) + cos(a) * a);
    // The native vector loop rounds the reciprocal once before scaling each mode.
    const float inv_n = 1.f / (float)(side * side * side);
    f[i] = make_float2((f[i].x * inv_n) * w, (f[i].y * inv_n) * w);
}
extern "C" __global__ void accumulate(int n, const double *p, const float *sfr, const float *em,
                                      const int *idx, const float *frac, const double *h,
                                      const double *dh, const double *q, const double *dq,
                                      const double *l, const double *dl, double *heat, double *ion,
                                      double *lya, double *cont, double *inj, const float *mini_sfr,
                                      double *lw, const float *acg_lw, const float *mcg_lw) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int j = idx[i] * (int)p[3] + (int)p[2];
    double t = frac[i], sf = sfr[i] * p[0], xr = em[i] * p[0] * p[1] * 1e38;
    heat[i] += xr * (dh[j] * t + h[j]);
    ion[i] += xr * (dq[j] * t + q[j]);
    lya[i] += xr * (dl[j] * t + l[j]);
    if (mini_sfr) {
        double mini = mini_sfr[i] * p[0];
        cont[i] += sf * p[4] + mini * p[6];
        inj[i] += sf * p[5] + mini * p[7];
        double sf_lw = acg_lw ? acg_lw[i] * p[0] : sf;
        double mini_lw = mcg_lw ? mcg_lw[i] * p[0] : mini;
        lw[i] += sf_lw * p[8] + mini_lw * p[9];
    } else {
        cont[i] += sf * p[4];
        inj[i] += sf * p[5];
    }
}

extern "C" __global__ void spectral_filter(float2 *field, int side, double dk, float radius,
                                           int kind) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = side / 2 + 1;
    if (idx >= side * side * half) return;
    int x = idx / (side * half), y = idx / half % side;
    float kx = (x > side / 2 ? x - side : x) * dk;
    float ky = (y > side / 2 ? y - side : y) * dk;
    float kz = (idx % half) * dk;
    float k2 = kx * kx + ky * ky + kz * kz;
    double kr = (float)(sqrt((double)k2) * radius);
    double weight;
    if (kind == 0)
        weight =
            kr < 1.e-4 ? 1. - .1 * (kr * kr) : (3. / ((kr * kr) * kr)) * (sin(kr) - kr * cos(kr));
    else if (kind == 1)
        weight = kr * .413566994 > 1. ? 0. : 1.;
    else
        weight = exp((-.643 * .643 / 2.) * (k2 * (radius * radius)));
    field[idx] = make_float2(field[idx].x * weight, field[idx].y * weight);
}
