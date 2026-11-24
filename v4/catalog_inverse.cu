// EvaluateRGTable1D_f in the native 4.3.2 release order.
__device__ double catalog_float_table(const float *values, double x, double low, double step,
                                      int length) {
    double bin = floor((x - low) / step);
    if (!(bin >= 0. && bin < length - 1)) return nan("");
    int i = (int)bin;
    double t = ((x - low) - step * i) / step;
    return values[i] * (1. - t) + values[i + 1] * t;
}

// Conditional Sheth–Tormen integrand in the native 4.3.2 release order.
struct InverseMassFunction {
    const float *sigma, *slope;
    int length;
    double low, step, sigma_cond, delta, growth, delta_c;
    int hmf;
    double delta_c_delos;

    __device__ double interpolate(const float *values, double x) const {
        return catalog_float_table(values, x, low, step, length);
    }

    __device__ double operator()(double x) const {
        double s = interpolate(sigma, x);
        if (s < sigma_cond) return 0.;
        double variance = s == sigma_cond ? 1.e-6 : s * s - sigma_cond * sigma_cond;
        double inverse = 1. / variance, inverse_sigma = 1. / (s * s);
        double deriv = -reference_exp(2.302585092994046 * interpolate(slope, x));
        if (hmf == 4) {
            double nu = (delta_c_delos - delta) * sqrt(inverse) / growth;
            return 0.519 * reference_pow(nu, 0.582) * reference_exp((-0.469 * nu) * nu) *
                   fabs(0.5 * deriv) * inverse;
        }
        if (hmf != 1) {
            double del = (delta_c - delta) / growth;
            return ((reference_exp(((-del * del) * .5) * inverse) * .3989422804014327) *
                    (sqrt(inverse) * inverse)) *
                   ((-del) * deriv);
        }
        double t1 = -variance * (.81 * inverse_sigma);
        double t2 = (.09499999999999997 * inverse_sigma) * (t1 * variance);
        double t3 = (.3966666666666666 * inverse_sigma) * (t2 * variance);
        double t4 = (.5475 * inverse_sigma) * (t3 * variance);
        double t5 = (t4 * variance) * (.638 * inverse_sigma);
        double sum = ((t3 + t5) + t1) + ((t2 + t4) + 1.);
        double critical = delta_c / growth;
        double c = .8544003745317531 * critical;
        double b = .34 * reference_pow((.73 * inverse_sigma) * (critical * critical), -.81);
        double offset = delta / growth;
        double barrier = c * (1. + b), factor = c * (1. + b * sum);
        double exponent = ((barrier - offset) * inverse) * ((offset - barrier) * .5);
        return ((reference_exp(exponent) * (sqrt(inverse) * inverse)) *
                (deriv * .3989422804014327)) *
               (offset - factor);
    }
};

struct CatalogCDF {
    InverseMassFunction mass;
    const double *rule;
    double hi, barrier, density;

    __device__ double integrate(double low) const {
        if (low >= hi) return 0.;
        if (mass.delta > barrier) return density / reference_exp(hi);
        double area = table_integral(low, hi, .001, mass, rule);
        return __shfl_sync(0xffffffff, area, 0) * density;
    }

    __device__ double residual(double x, double norm, double target, double min_lp) const {
        double area = integrate(x);
        return (area == 0. ? 2. * min_lp : reference_log(area / norm)) - target;
    }
};

__device__ bool same_sign_nonzero(double a, double b) {
    return (a < 0. && b < 0.) || (a > 0. && b > 0.);
}
// Match the GSL Brent solver's interpolation, interval test, and carried
// lower bound. Bisection by itself changes mass samples at fixed RNG.
__device__ double catalog_quantile(const CatalogCDF &cdf, double &low, double target, double norm,
                                   double min_lp) {
    double a = low, b = cdf.hi, c = b;
    double fa = cdf.residual(a, norm, target, min_lp), fb = 2. * min_lp - target, fc = fb;
    double d = b - a, e = d;
    for (int iteration = 0; iteration < 200; ++iteration) {
        bool ac_equal = false;
        if (same_sign_nonzero(fb, fc)) {
            ac_equal = true;
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if (fabs(fc) < fabs(fb)) {
            ac_equal = true;
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        double midpoint = .5 * (c - b);
        double tol = 1.1102230246251565e-16 * fabs(b);
        if (fb == 0.) {
            low = b;
            return b;
        }
        if (fabs(midpoint) <= tol) {
            low = fmin(b, c);
            return b;
        }
        if (fabs(e) < tol || fabs(fa) <= fabs(fb)) {
            d = midpoint;
            e = d;
        } else {
            double s = fb / fa;
            double p, q;
            if (ac_equal) {
                p = 2. * midpoint * s;
                q = 1. - s;
            } else {
                q = fa / fc;
                double r = fb / fc;
                p = s * (2. * midpoint * q * (q - r) - (b - a) * (r - 1.));
                q = (q - 1.) * (r - 1.) * (s - 1.);
            }
            if (p > 0.)
                q = -q;
            else
                p = -p;
            if (2. * p < fmin(3. * midpoint * q - fabs(tol * q), fabs(e * q))) {
                e = d;
                d = p / q;
            } else {
                d = midpoint;
                e = d;
            }
        }
        a = b;
        fa = fb;
        b += fabs(d) > tol ? d : copysign(tol, midpoint);
        fb = cdf.residual(b, norm, target, min_lp);
        double bound = same_sign_nonzero(fb, fc) ? a : c;
        if (fabs(b - bound) < 1.e-4) {
            low = fmin(b, bound);
            return b;
        }
    }
    return nan("");
}

extern "C" __global__ void build_inverse_catalog(const double *bounds, const double *conditions,
                                                 int bins, const float *sigma, const float *slope,
                                                 int length, double low, double step,
                                                 const double *rule, double *result,
                                                 const double *targets) {
    const double *row = conditions + 4 * blockIdx.x;
    CatalogCDF cdf = {{sigma, slope, length, low, step, row[1], row[2], bounds[1], bounds[2],
                       (int)bounds[4], bounds[5]},
                      rule,
                      row[0],
                      row[3],
                      bounds[3]};
    double *out = result + blockIdx.x * bins;
    double lower = bounds[0], min_lp = targets[0], mcond = reference_exp(cdf.hi);
    double norm = cdf.integrate(lower);
    double floor_mass = reference_exp(lower) / mcond;
    if (norm == 0.) {
        for (int k = 1 + threadIdx.x; k < bins - 1; k += blockDim.x) out[k] = floor_mass;
        return;
    }
    if (!threadIdx.x) out[bins - 1] = floor_mass;
    double inverse_mass = 1. / mcond;
    for (int k = bins - 2; k >= 0; --k) {
        double root = catalog_quantile(cdf, lower, targets[k], norm, min_lp);
        if (!threadIdx.x) out[k] = reference_exp(root) * inverse_mass;
    }
}

extern "C" __global__ void verify_catalog_math(int count, const double *points, double *results) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    results[3 * i] = reference_exp(points[3 * i]);
    results[3 * i + 1] = reference_pow(points[3 * i + 1], -.81);
    results[3 * i + 2] = reference_log(points[3 * i + 2]);
}
