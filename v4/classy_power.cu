// v4.3.2 transfer_function_CLASS and power_in_k, k in comoving Mpc^-1.
__device__ double eh_density_transfer(double k, const double *p) {
    double q = k * p[6] * p[6] / p[3];
    double a = sqrt(p[1]);
    double qe = q / (a + (1. - a) / (1. + pow(.43 * k * p[0], 4.)));
    double tf = log(2.718281828459045 + 1.84 * p[2] * a * qe);
    tf /= tf + qe * qe * (14.4 + 325. / (1. + 60.5 * pow(qe, 1.11)));
    double qn = 3.92 * q / sqrt(p[4]);
    return tf * (1. + 1.2 * pow(p[4], .64) / (pow(qn, -1.6) + pow(qn, .8)));
}

struct ClassDensity {
    const double *table;  // k, T, and three polynomial coefficients, each n doubles
    int n;
    int suppress_streaming;
    __device__ double spline(double k) const {
        int lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (k < table[mid])
                hi = mid;
            else
                lo = mid;
        }
        double dx = k - table[lo];
        return table[n + lo] +
               dx * (table[2 * n + lo] + dx * (table[3 * n + lo] + dx * table[4 * n + lo]));
    }
    __device__ double transfer(double k, const double *p) const {
        if (k > table[n - 1]) return table[5 * n - 1] * eh_density_transfer(k, p) * k * k;
        return spline(k);
    }
    __device__ double velocity_power(double k, const double *p) const {
        if (k == 0.) return 0.;
        double t;
        if (k > table[n - 1]) {
            // Native velocity extrapolation is linear in log(k), log(T_v).
            double slope = (log(table[2 * n - 1]) - log(table[2 * n - 2])) /
                           (log(table[n - 1]) - log(table[n - 2]));
            t = exp(log(table[2 * n - 1]) + slope * (log(k) - log(table[n - 1])));
        } else {
            t = spline(k);
        }
        return p[7] * (p[8] * pow(k / .05, p[9] - 1.)) * t * t / (k * k * k);
    }
    __device__ double power(double k, const double *p) const {
        if (k == 0.) return 0.;
        double tf = transfer(k, p);
        double power = p[7] * (p[8] * pow(k / .05, p[9] - 1.)) * tf * tf / (k * k * k);
        if (!suppress_streaming) return power;
        // The native streaming models suppress density power on both sides of the last knot.
        double log_ratio = log(k / 300.);
        return power * (1. - .24 * exp(-(log_ratio * log_ratio) / (2. * .9 * .9)));
    }
};

extern "C" __global__ void class_power_grid(int count, const double *k, const double *p,
                                            const double *table, int rows, int suppress_streaming,
                                            int velocity, double *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        ClassDensity spline{table, rows, suppress_streaming};
        out[i] = velocity ? spline.velocity_power(k[i], p) : spline.power(k[i], p);
    }
}
