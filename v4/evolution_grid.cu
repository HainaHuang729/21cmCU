__device__ double table_value(float x, const float *values, double low, double step, int count) {
    int j = (int)floor(((double)x - low) / step);
    if (j < 0 || j + 1 >= count) return nan("");
    double t = ((double)x - (low + step * (float)j)) / step;
    return exp(values[j] * (1. - t) + values[j + 1] * t);
}
__device__ double table_linear_d(float x, const double *values, double low, double step, int count) {
    int j = (int)floor(((double)x - low) / step);
    if (j < 0 || j + 1 >= count) return nan("");
    double t = ((double)x - (low + step * (float)j)) / step;
    return values[j] * (1. - t) + values[j + 1] * t;
}
extern "C" __global__ void emission(int n, const float *density, const float *photons, double pl,
                                    double pw, int pn, const float *sfr, double sl, double sw,
                                    int sn, const double *nh_tbl, double nhl, double nhw, int nhn,
                                    const double *fc_tbl, double fcl, double fcw, int fcn,
                                    const float *xray_tbl, double xrl, double xrw, int xrn,
                                    int metal, double lx, double cell_volume, double rho_m,
                                    double sfr_timescale, double const_sm_fac, int mass_dep,
                                    float *outp, float *outs, float *outx, float *outnh,
                                    float *outhm, float *outsm, double floor, double *partials) {
    double sums[6] = {};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const float d = density[i];
        double nion_val = table_value(d, photons, pl, pw, pn);
        outp[i] = nion_val;
        double sfr_val = sfr ? table_value(d, sfr, sl, sw, sn) * (1. + d) : 0.;
        if (outs) {
            outs[i] = sfr_val;
            outx[i] = metal ? table_value(d, xray_tbl, xrl, xrw, xrn) * (1. + d) : lx * outs[i];
        }
        if (outnh) {
            outnh[i] = table_linear_d(d, nh_tbl, nhl, nhw, nhn) * cell_volume * (1. + d);
            outhm[i] = table_linear_d(d, fc_tbl, fcl, fcw, fcn) * rho_m * (1. + d);
            outsm[i] = mass_dep ? sfr_val * sfr_timescale : nion_val * const_sm_fac * (1. + d);
        }
        if (partials) {
            sums[0] += outp[i];
            float factor = (1. + d > floor) ? (float)(1. + d) : 1.f;
            if (outs) {
                // Native tests the double sum, stores a float divisor and
                // rounds the individual quotients before adding to double means.
                sums[1] += outs[i] / factor;
                sums[2] += outx[i] / factor;
            }
            if (outnh) {
                sums[3] += outnh[i] / factor;
                sums[4] += outhm[i] / factor;
                sums[5] += outsm[i] / factor;
            }
        }
    }
    if (!partials) return;
    __shared__ double shared[256];
    for (int j = 0; j < 6; ++j) {
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
                                              float *nh, float *hm, float *sm,
                                              const double *sums, double expected_nion,
                                              double expected_sfr, double expected_xray,
                                              double expected_nh, double expected_hm,
                                              double expected_sm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (sums[0] != 0.) nion[i] *= expected_nion / (sums[0] / n);
    if (sfr) {
        if (sums[1] != 0.) sfr[i] *= expected_sfr / (sums[1] / n);
        if (sums[2] != 0.) xray[i] *= expected_xray / (sums[2] / n);
    }
    if (nh) {
        if (sums[3] != 0.) nh[i] *= expected_nh / (sums[3] / n);
        if (sums[4] != 0.) hm[i] *= expected_hm / (sums[4] / n);
        if (sums[5] != 0.) sm[i] *= expected_sm / (sums[5] / n);
    }
}
__device__ __forceinline__ void eval_ion_inputs(
    int i, const double *p, const float *density, const float *filtered, const float *photons,
    const float *xe, const float *rec, const float *photons_mcg, int contiguous_filtered,
    double &d, double &ion, double &mini, double &all_photons, double &electron,
    double &recombinations) {
    int side = (int)p[0], j = contiguous_filtered ? i : i / side * (2 * (side / 2 + 1)) + i % side;
    d = p[1] == 0 ? (double)density[i] * p[4] : filtered[j];
    ion = p[5] * photons[i];
    if (p[7]) ion = fmax(ion, p[6]);
    mini = 0.;
    if (photons_mcg) {
        mini = p[16] * photons_mcg[i];
        if (p[7]) mini = fmax(mini, p[17]);
    }
    all_photons = ion + mini;
    electron = p[10] ? xe[j] : 0.;
    int rec_index = p[15] ? 0 : (p[12] ? j : i);
    recombinations = (p[11] && rec) ? rec[rec_index] / (1. + d) : 0.;
}

extern "C" __global__ void ion_center(int n, const double *p, const float *density,
                                      const float *filtered, const float *photons, const float *xe,
                                      const float *rec, const float *oldz, const float *tk,
                                      float *xh, float *zr, float *mfp, float *gamma, float *temp,
                                      const float *photons_mcg, int contiguous_filtered) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double d, ion, mini, all_photons, electron, recombinations;
    eval_ion_inputs(i, p, density, filtered, photons, xe, rec, photons_mcg, contiguous_filtered, d,
                    ion, mini, all_photons, electron, recombinations);
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

extern "C" __global__ void ion_center_sphere_mark(
    int n, const double *p, const float *density, const float *filtered, const float *photons,
    const float *xe, const float *rec, const float *oldz, const float *xh, float *zr,
    const float *photons_mcg, int contiguous_filtered, unsigned char *triggered,
    unsigned char *eligible, float *gamma_cand) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double d, ion, mini, all_photons, electron, recombinations;
    eval_ion_inputs(i, p, density, filtered, photons, xe, rec, photons_mcg, contiguous_filtered, d,
                    ion, mini, all_photons, electron, recombinations);
    if (exceeds_ionization_barrier(all_photons, electron, recombinations)) {
        triggered[i] = 1;
        if (eligible) {
            bool ok = p[11] && xh[i] > 1.e-7;
            eligible[i] = ok ? 1 : 0;
            if (ok) gamma_cand[i] = (float)(p[2] * p[8] * all_photons);
        }
        zr[i] = oldz[i] < 0.f ? p[3] : oldz[i];
    } else {
        triggered[i] = 0;
        if (eligible) eligible[i] = 0;
    }
}

extern "C" __global__ void ion_center_sphere_partial(
    int n, const double *p, const float *density, const float *filtered, const float *photons,
    const float *xe, const float *tk, float *xh, float *temp, const float *photons_mcg,
    int contiguous_filtered, const unsigned char *triggered, const int *min_painter) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int painter = min_painter ? min_painter[i] : 0x7fffffff;
    if (!triggered[i] && (xh[i] > 1.e-30) && painter > i) {
        double d, ion, mini, all_photons, electron, recombinations;
        eval_ion_inputs(i, p, density, filtered, photons, xe, nullptr, photons_mcg,
                        contiguous_filtered, d, ion, mini, all_photons, electron, recombinations);
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
    if (painter < 0x7fffffff) xh[i] = 0.f;
}
extern "C" __global__ void shell_window(float2 *f, int nx, int ny, int nz, double dk_xy, double dk_z,
                                        double inner, double outer) {
    int hz = nz / 2 + 1;
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= (long long)nx * ny * hz) return;
    int x = (int)(i / ((long long)ny * hz)), y = (int)((i / hz) % ny), z = (int)(i % hz);
    float kx = (x > nx / 2 ? x - nx : x) * dk_xy, ky = (y > ny / 2 ? y - ny : y) * dk_xy,
          kz = z * dk_z;
    double k = sqrt((double)(kx * kx + ky * ky + kz * kz)), a = k * inner, b = k * outer;
    double w = b < 1.e-4
                   ? 1. - b * b / 10 * (pow(inner / outer, 5.) - 1) / (pow(inner / outer, 3.) - 1)
                   : 3. / (pow(b, 3.) - pow(a, 3.)) * (sin(b) - cos(b) * b - sin(a) + cos(a) * a);
    // The native vector loop rounds the reciprocal once before scaling each mode.
    const float inv_n = 1.f / (float)((long long)nx * ny * nz);
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
    if (heat) heat[i] += xr * (dh[j] * t + h[j]);
    ion[i] += xr * (dq[j] * t + q[j]);
    lya[i] += xr * (dl[j] * t + l[j]);
    if (mini_sfr) {
        double mini = mini_sfr[i] * p[0];
        cont[i] += sf * p[4] + mini * p[6];
        if (inj) inj[i] += sf * p[5] + mini * p[7];
        double sf_lw = acg_lw ? acg_lw[i] * p[0] : sf;
        double mini_lw = mcg_lw ? mcg_lw[i] * p[0] : mini;
        lw[i] += sf_lw * p[8] + mini_lw * p[9];
    } else {
        cont[i] += sf * p[4];
        if (inj) inj[i] += sf * p[5];
    }
}

extern "C" __global__ void spectral_filter(float2 *field, int nx, int ny, int nz, double dk_xy,
                                           double dk_z, float radius, int kind) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int half = nz / 2 + 1;
    if (idx >= (long long)nx * ny * half) return;
    int x = (int)(idx / ((long long)ny * half)), y = (int)(idx / half % ny);
    float kx = (x > nx / 2 ? x - nx : x) * dk_xy;
    float ky = (y > ny / 2 ? y - ny : y) * dk_xy;
    float kz = (idx % half) * dk_z;
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
