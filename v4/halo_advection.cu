// Regular-grid log emissivities use float nodes promoted to double. Number and
// collapsed-mass tables already use double nodes. Both share the same native interpolant.
__device__ double linear_lookup(double delta, const double *m, const double *values, int *bad) {
    int j = (int)floor((delta - m[0]) / m[1]);
    if (j < 0 || j >= (int)m[3] - 1) {
        atomicExch(bad, 1);
        return 0.;
    }
    double x = m[0] + m[1] * j;
    double w = (delta - x) / m[1];
    const double *y = values + (int)m[2];
    return y[j] * (1. - w) + y[j + 1] * w;
}

__device__ double molecular_emissivity(double delta, double turn, int field, const double *meta,
                                       const double *values, int *bad) {
    const double *m = meta + 7 * field;
    int x = (int)floor((delta - m[0]) / m[2]), y = (int)floor((turn - m[1]) / m[3]);
    int nx = (int)m[5], ny = (int)m[6];
    if (x < 0 || x >= nx - 1 || y < 0 || y >= ny - 1) {
        atomicExch(bad, 1);
        return 0.;
    }
    double wx = (delta - (m[0] + m[2] * x)) / m[2];
    double wy = (turn - (m[1] + m[3] * y)) / m[3];
    const double *row = values + (int)m[4] + x * ny + y;
    double left = row[0] * (1. - wy) + row[1] * wy;
    double right = row[ny] * (1. - wy) + row[ny + 1] * wy;
    return exp(left * (1. - wx) + right * wx);
}

extern "C" __global__ void advect_unresolved_cells(
    int offset, int n, const int *dims, const double *p, const double *meta, const double *values,
    const float *density, const float *vx, const float *vy, const float *vz, const float *vx2,
    const float *vy2, const float *vz2, const float *turn_acg, const float *turn_mcg, int *cells,
    double *weights, double *props, int *bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ix = offset + i, input_index = i;
    if (p[18] == 2.) {
        int slab = i % 2, local = i / 2;
        ix = slab * ((dims[0] / 2) * dims[1] * dims[2]) + offset + local;
        input_index = slab * (n / 2) + local;
    }
    int coord[3] = {ix / (dims[1] * dims[2]), ix / dims[2] % dims[1], ix % dims[2]};
    const float *v[3] = {vx, vy, vz}, *v2[3] = {vx2, vy2, vz2};
    int left[3], right[3];
    double dist[3];
    for (int j = 0; j < 3; ++j) {
        double pos = coord[j] + v[j][input_index] * p[2 + j];
        if (p[8]) pos -= v2[j][input_index] * p[5 + j];
        pos *= p[1];
        int lo = (int)floor(pos), dim = dims[3 + j];
        dist[j] = pos - lo;
        left[j] = (lo % dim + dim) % dim;
        right[j] = (left[j] + 1) % dim;
    }
    for (int k = 0; k < 8; ++k) {
        int x = k & 1, y = (k >> 1) & 1, z = k >> 2;
        cells[8 * i + k] =
            ((x ? right[0] : left[0]) * dims[4] + (y ? right[1] : left[1])) * dims[5] +
            (z ? right[2] : left[2]);
        weights[8 * i + k] = (x ? dist[0] : 1. - dist[0]) * (y ? dist[1] : 1. - dist[1]) *
                             (z ? dist[2] : 1. - dist[2]);
    }
    double delta = density[input_index] * p[0], volume = p[12];
    if (p[19]) {
        double a = turn_acg[input_index], m = turn_mcg[input_index];
        bool active = pow(10., a) <= p[20];
        double *out = props + (p[11] ? 9 : 8) * i;
        double nion = molecular_emissivity(delta, a, 0, meta, values, bad);
        if (active) {
            nion += molecular_emissivity(delta, m, 3, meta, values, bad);
        }
        out[0] = nion * volume;
        double sfr_acg =
            (p[9] || p[11]) ? molecular_emissivity(delta, a, 1, meta, values, bad) : 0.;
        double sfr_mcg =
            (active && (p[9] || p[11])) ? molecular_emissivity(delta, m, 4, meta, values, bad) : 0.;
        out[1] = sfr_acg * volume;
        out[7] = sfr_mcg * volume;
        if (p[9] && p[10]) {
            double xray = molecular_emissivity(delta, a, 2, meta, values, bad);
            if (active) xray += molecular_emissivity(delta, m, 5, meta, values, bad);
            out[2] = xray * volume;
        }
        if (p[11]) {
            out[4] = linear_lookup(delta, meta + 7 * 6, values, bad) * p[13];
            out[5] = linear_lookup(delta, meta + 7 * 7, values, bad) * p[14] * volume;
            out[6] = out[1] * p[15];
            out[8] = out[7] * p[15];
        }
        return;
    }
    double *out = props + 7 * i;
    out[0] = exp(linear_lookup(delta, meta, values, bad)) * volume;
    double sfr = 0.;
    if (p[9] || p[11]) sfr = exp(linear_lookup(delta, meta + 4, values, bad));
    out[1] = sfr * volume;
    if (p[9] && p[10]) out[2] = exp(linear_lookup(delta, meta + 8, values, bad)) * volume;
    if (p[11]) {
        out[4] = linear_lookup(delta, meta + 12, values, bad) * p[13];
        out[5] = linear_lookup(delta, meta + 16, values, bad) * p[14] * volume;
        out[6] = sfr * p[15] * volume;
    }
}
