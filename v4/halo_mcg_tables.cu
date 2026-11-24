// Each warp fills one (density, log10 turnover) cell. Preserve the ACG
// integrands and the branch-specific normalization of a collapsed cell.
struct PopulationIntegral {
    SourceIntegral base;
    int kind;
    bool mini;

    __device__ double weight(double ln_mass) const {
        const double *p = base.p;
        if (!mini)
            return kind == 2 ? XrayIntegral{base}.luminosity(ln_mass) : base.weighted_mass(ln_mass);
        double mass = exp(ln_mass);
        double pivot = 7. * 2.30258509299404568402;
        double star = log_scaling_PL_limit(ln_mass, p[13], p[11], pivot, p[15]);
        if (kind == 2) {
            double fraction = exp(star - p[10] / mass - mass / p[23] + p[13]);
            double stellar = mass * fraction * p[14] / p[16];
            double sfr = stellar / p[19], metallicity;
            get_halo_metallicity(sfr, stellar, p[0], &metallicity, p[21]);
            return sfr * lx_on_sfr_doublePL(metallicity, p[20]);
        }
        double escape = log_scaling_PL_limit(ln_mass, p[14], p[12], pivot, p[16]);
        return molecular_weight(ln_mass, mass, star, escape, p[10], p[23]);
    }

    __device__ double operator()(double ln_mass) const {
        double sigma = base.interp(base.sigma, ln_mass);
        if (sigma < base.p[7]) return 0.;
        return weight(ln_mass) * base.mass_function(ln_mass, sigma);
    }
};

extern "C" __global__ void population_table(const double *rows, int nx, int ny, int kind, int mini,
                                            const float *sigma, const float *slope, double low,
                                            double step, int ns, const double *xi, const double *wi,
                                            const double *rule, float *out) {
    int entry = blockIdx.x, i = entry / ny, j = entry % ny;
    const double *p = rows + 30 * j;
    double delta = p[1] + (float)i / ((float)nx - 1.) * (p[2] - p[1]);
    PopulationIntegral f = {
        {p, sigma, slope, low, step, delta, ns, (int)p[27], p[28]}, kind, (bool)mini};
    int method = (int)p[22];
    double result = 0.;
    if (!p[24] && p[3] < p[5]) {
        if (delta > (double)(float).99 * p[8]) {
            if (p[5] * (1. - 1.e-7) <= p[4]) {
                result = f.weight(p[5]) / exp(p[5]);
                if (kind != 2) result *= p[17];
            }
        } else if (method == 2 && kind != 2) {
            double integral = 0.;
            if (!threadIdx.x) {
                integral = hmf_integral_gamma_approx(
                    [&](double lnM) { return f.base.interp(sigma, lnM); }, p[3], p[4], p[6], delta,
                    p[7], p[9], mini ? -1.e30 : log(p[10]), p[23], p[11] + p[12], mini ? 4 : 3);
            }
            result = __shfl_sync(0xffffffff, integral, 0) * p[17];
        } else {
            result =
                conditional_quadrature(f, (method == 0) || delta > 1.2, p[3], p[4], xi, wi, rule);
            result *= p[17];
            if (kind == 2) result *= p[16];
        }
    }
    if (!threadIdx.x) {
        float value = log(result), floor_value = kind == 0 ? -40.f : -50.f;
        out[entry] = value < floor_value ? floor_value : value;
    }
}
