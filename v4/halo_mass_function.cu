struct CatalogMassFunction {
    SourceIntegral source;
    bool weighted;
    __device__ double operator()(double x) const {
        double sigma = source.interp(source.sigma, x);
        if (sigma < source.p[7]) return 0.;
        double dndm = source.mass_function(x, sigma);
        return weighted ? exp(x) * dndm : dndm;
    }
};
struct CatalogIntegral {
    CatalogMassFunction mass;
    const double *rule;
    double hi, critical, density, rho_crit, omega_m;
    __device__ double integrate(double lo, double upper) const {
        if (lo >= hi) return 0.;
        if (mass.source.density > critical) {
            if (hi * (1. - 1.e-7) > upper) return 0.;
            return mass.weighted ? 1. : density / exp(hi);
        }
        double result = table_integral(lo, upper, .001, mass, rule);
        result = __shfl_sync(0xffffffff, result, 0);
        return mass.weighted ? result : result * rho_crit * omega_m;
    }
};
extern "C" __global__ void catalog_number_and_fraction(const double *bounds,
                                                       const double *conditions, const float *sigma,
                                                       const float *slope, double lo, double step,
                                                       int ns, const double *rule, double *result) {
    int i = blockIdx.x;
    const double *row = conditions + 4 * i;
    __shared__ double parameters[10];
    if (threadIdx.x == 0) {
        parameters[6] = bounds[4];
        parameters[7] = row[1];
        parameters[9] = bounds[5];
    }
    __syncwarp();
    SourceIntegral source = {parameters, sigma, slope,          lo,       step,
                             row[2],     ns,    (int)bounds[8], bounds[9]};
    CatalogIntegral cdf = {{source, false},       rule,      row[0],   row[3],
                           bounds[6] * bounds[7], bounds[6], bounds[7]};
    double number = cdf.integrate(bounds[2], bounds[3]);
    cdf.mass.weighted = true;
    double fraction = cdf.integrate(bounds[2], bounds[3]);
    if (!threadIdx.x) {
        result[2 * i] = number;
        result[2 * i + 1] = fraction;
    }
}

// Each native bilinear lookup uses floor(q) and floor(q)+1. Widen this
// pair by one at both ends so libm and device logarithm roundoff at an
// exact bin edge cannot remove a row needed by a descendant.
extern "C" __global__ void select_condition_rows(const float *values, unsigned long long count,
                                                 int from_catalog, double growth, double xmin,
                                                 double width, int nx, int *used, int *invalid) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    double value = (double)values[i];
    double x = from_catalog ? log(value) : value * growth;
    double bin = (x - xmin) / width;
    if (!isfinite(bin)) {
        atomicExch(invalid, 1);
        return;
    }
    int base = (int)floor(fmax(0., fmin((double)(nx - 1), bin)));
    for (int row = max(0, base - 1); row <= min(nx - 1, base + 2); ++row) atomicExch(used + row, 1);
}
