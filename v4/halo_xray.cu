// The source-weight formula in hmf.c uses a capped power law for the integrated
// ACG population, including when individual catalogue halos use upper turnover.
struct XrayIntegral {
    SourceIntegral source;

    __device__ double luminosity(double lnM) const {
        const double *p = source.p;
        double mass = exp(lnM);
        double log_star = log_scaling_PL_limit(lnM, p[13], p[11], 23.02585092994045684, p[15]);
        double fstar = exp(log_star - p[10] / mass + p[13]);
        double stellar = mass * fstar * p[14] / p[16];
        double sfr = stellar / p[19];
        double metallicity;
        get_halo_metallicity(sfr, stellar, p[0], &metallicity, p[21]);
        return sfr * lx_on_sfr_doublePL(metallicity, p[20]);
    }

    __device__ double operator()(double lnM) const {
        double sigma = source.interp(source.sigma, lnM);
        if (sigma < source.p[7]) return 0.;
        return luminosity(lnM) * source.mass_function(lnM, sigma);
    }
};

extern "C" __global__ void xray_conditional_table(const double *p, int n, int kind,
                                                  const float *sigma, const float *slope,
                                                  double low, double step, int ns, const double *xi,
                                                  const double *wi, const double *rule,
                                                  float *out) {
    int i = blockIdx.x;
    double delta = p[1] + (float)i / ((float)n - 1.) * (p[2] - p[1]);
    XrayIntegral f = {{p, sigma, slope, low, step, delta, ns, (int)p[23], p[24]}};
    double result = 0.;
    if (p[3] < p[5]) {
        if (delta > (double)(float).99 * p[8]) {
            // The native single-halo limit returns Lx(Mcell)/Mcell directly.
            if (p[5] * (1. - 1.e-7) <= p[4]) result = f.luminosity(p[5]) / exp(p[5]);
        } else {
            double integral =
                conditional_quadrature(f, (int)p[22] == 0 || delta > 1.2, p[3], p[4], xi, wi, rule);
            result = integral * p[17] * p[16];
        }
    }
    if (!threadIdx.x) {
        float value = log(result);
        out[i] = value < -50.f ? -50.f : value;
    }
}
