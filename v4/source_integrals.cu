// v4 conditional Sheth--Tormen mass integral, one warp per table entry.
// p = redshift, density limits, log mass limits, log cell mass, growth, sigma
// at cell mass, collapse threshold, spherical delta_c, turnover mass, stellar
// and escape exponents, log normalizations and limits, and output prefactor.
struct SourceIntegral {
    const double *p;
    const float *sigma, *slope;
    double low, step, density;
    int ns, hmf;
    double delta_c_delos;
    __device__ double interp(const float *v, double x) const {
        int i = (int)floor((x - low) / step);
        if (i < 0 || i + 1 >= ns) return nan("");
        double t = (x - (low + step * (float)i)) / step;
        return v[i] * (1. - t) + v[i + 1] * t;
    }
    __device__ double weighted_mass(double x) const {
        if (p[15] < -900.) return 1.;
        double a = p[11], e = p[12], pivot = 10. * 2.30258509299404568402;
        double star = ((a > 0 && x > p[15]) || (a < 0 && x < p[15])) ? -p[13] : a * (x - pivot);
        double esc = ((e > 0 && x > p[16]) || (e < 0 && x < p[16])) ? -p[14] : e * (x - pivot);
        return exp(star + esc - p[10] / exp(x) + x);
    }
    __device__ double mass_function(double x, double s) const {
        return hmf_conditional(hmf, p[6], density, p[7], s, pow(10., interp(slope, x)), p[9],
                               delta_c_delos);
    }
    __device__ double operator()(double x) const {
        double s = interp(sigma, x);
        if (s < p[7]) return 0.;
        double cmf = mass_function(x, s);
        return weighted_mass(x) * cmf;
    }
};
// One warp owns a density node. The low-density GL sum retains native node order.
template <class Integrand>
__device__ double conditional_quadrature(const Integrand &f, bool adaptive, double lo, double hi,
                                         const double *xi, const double *wi, const double *rule) {
    if (adaptive) return table_integral(lo, hi, 1.e-3, f, rule);
    __shared__ double terms[100];
    for (int j = threadIdx.x; j < 100; j += blockDim.x) terms[j] = wi[j + 1] * f(xi[j + 1]);
    __syncwarp();
    double integral = 0.;
    if (!threadIdx.x)
        for (int j = 0; j < 100; ++j) integral += terms[j];
    return integral;
}

extern "C" __global__ void source_table(const double *p, int n, int sfr, const float *sigma,
                                        const float *slope, double low, double step, int ns,
                                        const double *xi, const double *wi, const double *rule,
                                        float *out) {
    int i = blockIdx.x;
    double d = p[1] + (float)i / ((float)n - 1.) * (p[2] - p[1]);
    int method = (int)p[19];
    SourceIntegral f = {p, sigma, slope, low, step, d, ns, (int)p[20], p[21]};
    double integral = 0.;
    if (p[3] < p[5]) {
        if (d > (double)(float).99 * p[8]) {
            if (p[5] * (1. - 1.e-7) <= p[4]) integral = f.weighted_mass(p[5]) / exp(p[5]);
        } else if (method == 2) {
            double gval = 0.;
            if (!threadIdx.x) {
                gval = hmf_integral_gamma_approx([&](double lnM) { return f.interp(sigma, lnM); },
                                                 p[3], p[4], p[6], d, p[7], p[9], log(p[10]), 1.e20,
                                                 p[11] + p[12], 3);
            }
            integral = __shfl_sync(0xffffffff, gval, 0);
        } else {
            integral =
                conditional_quadrature(f, (method == 0) || d > 1.2, p[3], p[4], xi, wi, rule);
        }
    }
    if (!threadIdx.x) {
        // Preserve NaN for diagnosis; the native code floors -infinity.
        float y = log(integral * p[17]);
        out[i] = y < (sfr ? -50.f : -40.f) ? (sfr ? -50.f : -40.f) : y;
    }
}

struct GlobalIntegral {
    SourceIntegral base;
    double omm, oml, omr;
    __device__ double weighted_hmf(double x, double weight) const {
        const double *p = base.p;
        return weight * hmf_unconditional(base.hmf, p[0], p[6], x, base.interp(base.sigma, x),
                                          pow(10., base.interp(base.slope, x)), omm, oml, omr, p[9],
                                          base.delta_c_delos);
    }
    __device__ double operator()(double x) const { return weighted_hmf(x, base.weighted_mass(x)); }
};
extern "C" __global__ void global_source_table(const double *rows, const float *sigma,
                                               const float *slope, double low, double step, int ns,
                                               const double *rule, double *out) {
    int i = blockIdx.x;
    const double *p = rows + 23 * i;
    GlobalIntegral f = {
        {p, sigma, slope, low, step, 0., ns, (int)p[18], p[19]}, p[20], p[21], p[22]};
    double val = table_integral(p[3], p[4], .001, f, rule);
    if (threadIdx.x == 0) out[i] = val * p[17];
}

__device__ double virial_mass(double z, double temp, double mu, const double *c) {
    double x = 1.f + (float)z, matter = c[0] * x * x * x;
    double om = matter / (matter + c[1] + c[2] * x * x * x * x + c[3] * x * x);
    double d = om - 1., delta = 177.65287921960845 + 82. * d - 39. * d * d;
    return 7030.97 / c[4] * sqrt(om / (c[0] * delta)) * pow(temp / (mu * (1. + z)), 1.5);
}
__device__ double growth_amplitude(double om) {
    return 2.5 * om / (1. / 70. + om * (209. - om) / 140. + pow(om, 4. / 7.));
}
extern "C" __global__ void global_rows(int n, double lo, double step, int sfr, const double *c,
                                       const double *base, double *rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double z = lo + i * step, x = 1. + z, matter = c[0] * x * x * x, rad = c[2] * x * x * x * x;
    double *p = rows + 23 * i;
    for (int k = 0; k < 23; k++) p[k] = base[k];
    p[0] = z;
    if (fabs(c[0] - 1.) < 1.e-4)
        p[6] = 1. / x;
    else if (c[1] > -1.e-4 && fabs(c[1] + c[0] + c[2] - 1.) < .01 && fabs(c[5] + 1.) < 1.e-4)
        p[6] = growth_amplitude(matter / (c[1] + matter + rad)) /
               ((2.5 * c[0] /
                 (1. / 70. + (float)((float)c[0] * (209.f - (float)c[0])) / 140. +
                  pow(c[0], 4. / 7.))) *
                x);
    else if (c[11] < 1. + 1.e-4 && fabs(c[1]) < 1.e-4) {
        double x_0 = 1. / c[0] - 1.,
               d_0 = 1. + 3. / x_0 +
                     3. * log(sqrt(1. + x_0) - sqrt(x_0)) * sqrt(1. + x_0) / pow(x_0, 1.5);
        double x_z = fabs(1. / c[0] - 1.) / x,
               d_z = 1. + 3. / x_z +
                     3. * log(sqrt(1. + x_z) - sqrt(x_z)) * sqrt(1. + x_z) / pow(x_z, 1.5);
        p[6] = d_z / d_0;
    } else
        p[6] = nan("");
    p[10] = fmax(c[6], virial_mass((float)z, 1.e4, .59, c));
    p[3] = log((c[7] ? c[6] : virial_mass(z, c[8], c[8] < 9999.99 ? 1.22 : .6, c)) / 50.);
    if (sfr) {
        double y = 1.f + (float)z;
        double hubble = c[4] * 3.2407e-18 * sqrt(c[0] * y * y * y + c[2] * y * y * y * y + c[1]);
        p[17] = c[9] * base[17] / (c[10] / hubble);
    }
}

struct DndmIntegral {
    SourceIntegral source;
    bool weighted;
    __device__ double operator()(double x) const {
        double s = source.interp(source.sigma, x);
        if (s < source.p[7]) return 0.;
        double cmf = source.mass_function(x, s);
        return weighted ? exp(x) * cmf : cmf;
    }
};

extern "C" __global__ void grid_dndm_table(const double *p, int n, const float *sigma,
        const float *slope, double low, double step, int ns, const double *rule,
        double *nhalo_out, double *fcoll_out) {
    int i = blockIdx.x;
    double delta = p[0] + (p[1] - p[0]) * ((double)i / (double)(n - 1));
    double nhalo = 0., fcoll = 0.;
    if (p[2] < p[5]) {
        if (delta > 0.99 * p[11]) {
            if (p[5] * (1. - 1.e-7) <= p[3]) {
                nhalo = p[8] / exp(p[5]);
                fcoll = 1.;
            }
        } else {
            __shared__ double params[10];
            if (threadIdx.x == 0) {
                params[6] = p[4];
                params[7] = p[6];
                params[9] = p[7];
            }
            __syncwarp();
            SourceIntegral s = {params, sigma, slope, low, step, delta, ns, (int)p[9], p[10]};
            DndmIntegral fn = {s, false};
            double vn = table_integral(p[2], p[3], .001, fn, rule);
            fn.weighted = true;
            double vf = table_integral(p[2], p[3], .001, fn, rule);
            nhalo = vn * p[8];
            fcoll = vf;
        }
    }
    if (threadIdx.x == 0) {
        nhalo_out[i] = nhalo;
        fcoll_out[i] = fcoll;
    }
}
