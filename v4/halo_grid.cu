// Each sampled halo has ACG and MCG components with the same three catalogue draws.
__device__ double sampled_xray(double stars, double sfr, double norm, const double *p) {
    if (!p[metallicity]) return norm * sfr;
    double mz = 1.;
    if (stars > 0. && sfr > 0.) {
        double m0 = 1.28825e10 * pow(sfr * p[seconds_year], .56);
        mz = pow(1. + pow(stars / m0, -2.1), -.148);
    }
    double metal = 1.23 * mz * pow(10., -.056 * p[redshift] + .064);
    return (norm / (1. + pow(metal / .05, .64))) * sfr;
}

__device__ double sampled_turnover(const float *log_turnover, const int *cell,
                                   const double *weights) {
    double average = 0.;
    for (int k = 0; k < 8; ++k) average += weights[k] * log_turnover[cell[k]];
    return pow(10., average);
}
extern "C" __global__ void sampled_halos(int nh, const double *p, const int *dims,
                                         const float *mass, const float *coords, const float *sr,
                                         const float *fr, const float *xr, const float *vx,
                                         const float *vy, const float *vz, const float *vx2,
                                         const float *vy2, const float *vz2,
                                         const float *log_turn_acg, const float *log_turn_mcg,
                                         int *cell, double *weights, double *props) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nh) return;
    double m = mass[i];
    if (m == 0.) {
        for (int k = 0; k < 8; ++k) cell[8 * i + k] = -1;
        return;
    }

    const float *v[] = {vx, vy, vz}, *v2[] = {vx2, vy2, vz2};
    double pos[3], dist[3];
    int raw[3], left[3], right[3];
    for (int j = 0; j < 3; ++j) {
        pos[j] = coords[3 * i + j];
        int k = (int)(pos[j] * p[vel_cell_inv] + 0.5);
        raw[j] = (k % dims[j] + dims[j]) % dims[j];
    }
    long long vi = ((long long)raw[0] * dims[1] + raw[1]) * dims[2] + raw[2];
    for (int j = 0; j < 3; ++j) {
        pos[j] += v[j][vi] * p[displacement];
        if (p[second_order]) pos[j] -= v2[j][vi] * p[displacement2];
        pos[j] = pos[j] * dims[3 + j] / (j == 2 ? p[length_z] : p[length_xy]);
        double lo = floor(pos[j]);
        dist[j] = pos[j] - lo;
        left[j] = ((int)lo % dims[j + 3] + dims[j + 3]) % dims[j + 3];
        right[j] = (left[j] + 1) % dims[j + 3];
    }
    for (int k = 0; k < 8; ++k) {
        int x = k & 1, y = (k >> 1) & 1, z = k >> 2;
        cell[8 * i + k] =
            ((x ? right[0] : left[0]) * dims[4] + (y ? right[1] : left[1])) * dims[5] +
            (z ? right[2] : left[2]);
        weights[8 * i + k] = (x ? dist[0] : 1. - dist[0]) * (y ? dist[1] : 1. - dist[1]) *
                             (z ? dist[2] : 1. - dist[2]);
    }

    double fstar = p[fstar10] * pow(m / 1e10, p[alpha_star]);
    if (p[upper_turnover] && p[alpha_star] > p[upper_index]) {
        double u = m / p[upper_pivot];
        fstar = p[fstar10] * p[upper_ratio] / (pow(u, -p[alpha_star]) + pow(u, -p[upper_index]));
    }
    double s = p[star_sigma], adjustment = p[median] ? 0. : s * s / 2.;
    double turn_acg =
        p[photoheating] ? sampled_turnover(log_turn_acg, cell + 8 * i, weights + 8 * i) : p[mturn];
    double stars =
        fmin(1., fstar * exp(-turn_acg / m + sr[i] * s - adjustment)) * m * p[baryon_ratio];
    double photons = p[pop2] * fmin(1., p[fesc10] * pow(m / 1e10, p[alpha_esc]));
    double stars_mcg = 0., photons_mcg = 0., sfr_mcg = 0.;
    if (p[mcg]) {
        if (turn_acg <= 1.001 * p[atomic_threshold]) {
            double turn_mcg = sampled_turnover(log_turn_mcg, cell + 8 * i, weights + 8 * i);
            double eff = p[fstar7] * pow(m / 1e7, p[alpha_star_mcg]);
            double duty = exp(-turn_mcg / m - m / p[atomic_threshold] + sr[i] * s - adjustment);
            stars_mcg = fmin(1., eff * duty) * m * p[baryon_ratio];
        }
        photons_mcg = p[pop3] * fmin(1., p[fesc7] * pow(m / 1e7, p[alpha_esc]));
    }
    double sfr = 0., xray = 0.;
    if (p[thermal] || p[recombinations]) {
        s = p[sfr_sigma] > 0.
                ? fmax(p[sfr_sigma],
                       p[sfr_sigma_index] * log10((stars + stars_mcg) / 1e10) + p[sfr_sigma])
                : 0.;
        double scatter = exp(fr[i] * s - (p[median] ? 0. : s * s / 2.));
        sfr = (stars / p[timescale]) * scatter;
        if (p[mcg]) sfr_mcg = (stars_mcg / p[timescale]) * scatter;
    }
    if (p[thermal]) {
        xray = sampled_xray(stars, sfr, p[lx_sfr], p);
        if (p[mcg]) xray += sampled_xray(stars_mcg, sfr_mcg, p[lx_mcg], p);
        s = p[xray_sigma];
        xray *= exp(xr[i] * s - (p[median] ? 0. : s * s / 2.));
    }
    double *out = props + (p[mcg] ? 9 : 7) * i;
    out[0] = (stars * photons + stars_mcg * photons_mcg) / p[baryon_density];
    out[1] = sfr;
    out[2] = xray;
    out[3] = sfr * photons + sfr_mcg * photons_mcg;
    out[4] = 1.;
    out[5] = m;
    out[6] = stars;
    if (p[mcg]) {
        out[7] = sfr_mcg;
        out[8] = stars_mcg;
    }
}

// CuPy compiles CUDA kernels with -ftz=true. Native CIC temporarily widens
// the previous float and rounds back to float after every halo addition.
// Keep subnormal floats in both conversions, including during normalization.
__device__ __forceinline__ double widen_grid_value(float value) {
    double result;
    asm("cvt.f64.f32 %0,%1;" : "=d"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ float round_grid_value(double value) {
    float result;
    asm("cvt.rn.f32.f64 %0,%1;" : "=f"(result) : "d"(value));
    return result;
}

// A stable cell sort preserves halo order, with one writer per non-empty cell.
extern "C" __global__ void add_segments(int ns, int ng, int nf, int stride, const int *cell,
                                        const int *order, const int *start, const int *fields,
                                        const double *weights, const double *props, float *grids) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= ns) return;
    int begin = start[s], end = start[s + 1], pix = cell[order[begin]];
    if (pix < 0) return;
    for (int j = 0; j < nf; ++j) {
        float *value = grids + (long long)j * ng + pix;
        float sum = *value;
        for (int k = begin; k < end; ++k) {
            int entry = order[k];
            sum = round_grid_value(widen_grid_value(sum) +
                                   weights[entry] * props[stride * (entry / 8) + fields[j]]);
        }
        *value = sum;
    }
}

extern "C" __global__ void normalize_sampled_grid(int cells, double inverse_volume,
                                                  const int *fields, float *grids) {
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    int field = blockIdx.y;
    if (cell >= cells || fields[field] == 4) return;  // Halo counts have no volume factor.
    float *value = grids + (long long)field * cells + cell;
    *value = round_grid_value(widen_grid_value(*value) * inverse_volume);
}
