struct V4Moment {
    const double *p;
    double r, rho, gaussian_mass_derivative;
    bool deriv;
#ifdef USE_CLASS_POWER
    ClassDensity classy;
#elif defined(USE_ANALYTIC_POWER)
    AnalyticDensity analytic;
#endif
    __device__ double operator()(double k) const {
#ifdef USE_CLASS_POWER
        double power = classy.power(k, p);
#elif defined(USE_ANALYTIC_POWER)
        double power = analytic.power(k, p);
#else
        double q = k * p[6] * p[6] / p[3], a = sqrt(p[1]);
        double qe = q / (a + (1. - a) / (1. + pow(.43 * k * p[0], 4.)));
        double tf = log(2.718281828459045 + 1.84 * p[2] * a * qe);
        tf /= tf + qe * qe * (14.4 + 325. / (1. + 60.5 * pow(qe, 1.11)));
        double qn = 3.92 * q / sqrt(p[4]);
        tf *= 1. + 1.2 * pow(p[4], .64) / (pow(qn, -1.6) + pow(qn, .8));
        tf *= k * k;
        double power = p[7] * p[8] * pow(k / .05, p[9] - 1.) * tf * tf / (k * k * k);
#endif
        double x = k * r, w;
        if (gaussian_mass_derivative != 0.) {
            // The native Gaussian variance and derivative use different widths.
            if (!deriv) {
                w = exp(-0.643 * 0.643 * (x * x) / 2.);
                return k * k * power * w * w /
                       (2. * 3.14159265358979323846 * 3.14159265358979323846);
            }
            w = exp(-x * x / 2.);
            double dwdr = -k * x * w;
            double drdm = 1. / (gaussian_mass_derivative * r * r);
            double dw2dm = 2. * w * dwdr * drdm;
            return k * k * power * dw2dm / (2. * 3.14159265358979323846 * 3.14159265358979323846);
        }
        if (!deriv) {
            w = x < 1.e-4 ? 1. - .1 * x * x : 3. / (x * x * x) * (sin(x) - cos(x) * x);
            return k * k * power * w * w / (2. * 3.14159265358979323846 * 3.14159265358979323846);
        }
#if defined(USE_CLASS_POWER) || defined(USE_ANALYTIC_POWER)
        // 4.3.2's native top-hat derivative reuses the reciprocal of x cubed.
        double xx = x * x, inv_cube = 1. / (xx * x);
        double sine = sin(x), cosine = cos(x);
        w = x < 1.e-4 ? 1. : 3. * (sine * inv_cube - cosine * (1. / xx));
        double dw = x < 1.e-10
                        ? 0.
                        : (cosine * inv_cube) * (9. * k) + (1. - 3. / xx) * (sine * 3.) / (x * r);
#else
        w = x < 1.e-4 ? 1. : 3. * (sin(x) / (x * x * x) - cos(x) / (x * x));
        double dw = x < 1.e-10 ? 0.
                               : 9. * cos(x) * k / (x * x * x) +
                                     3. * sin(x) * (1. - 3. / (x * x)) / (x * r);
#endif
        return k * k * power * 2. * w * dw / (4. * 3.14159265358979323846 * rho * r * r) /
               (2. * 3.14159265358979323846 * 3.14159265358979323846);
    }
};
extern "C" __global__ void moments(const double *p, double rho, double min_log, double step,
                                   double norm_radius, const double *rule, double *s, double *d,
                                   double gaussian_mass_derivative, const double *radii
#ifdef USE_CLASS_POWER
                                   ,
                                   const double *class_table, int class_rows, int suppress_streaming
#elif defined(USE_ANALYTIC_POWER)
                                   ,
                                   int spectrum, float om, float h, float ob
#endif
) {
    int i = blockIdx.x;
    double m = (float)exp(min_log + i * step);
    double r =
        norm_radius > 0 ? norm_radius : pow(3. * m / (4. * 3.14159265358979323846 * rho), 1. / 3.);
    if (radii) r = radii[i];
    double lo = 1.e-99 / r, hi = 350. / r;
    V4Moment moment{p,
                    r,
                    rho,
                    gaussian_mass_derivative,
                    false
#ifdef USE_CLASS_POWER
                    ,
                    {class_table, class_rows, suppress_streaming}
#elif defined(USE_ANALYTIC_POWER)
                    ,
                    {spectrum, om, h, ob}
#endif
    };
    double var = table_integral(lo, hi, 1.e-6, moment, rule);
    if (!threadIdx.x) s[i] = sqrt(var);
    if (norm_radius <= 0) {
        moment.deriv = true;
        double slope = table_integral(lo, hi, 1.e-6, moment, rule);
        if (!threadIdx.x) d[i] = log10(-slope);
    }
}
