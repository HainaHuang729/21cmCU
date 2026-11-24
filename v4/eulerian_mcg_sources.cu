struct GlobalEmissivity {
    PopulationIntegral population;
    __device__ double operator()(double log_mass) const {
        const double *p = population.base.p;
        return GlobalIntegral{population.base, p[27], p[28], p[29]}.weighted_hmf(
            log_mass, population.weight(log_mass));
    }
};

extern "C" __global__ void eulerian_targets(const double *rows, const float *sigma,
                                            const float *slope, double low, double step, int ns,
                                            const double *rule, double *out) {
    int idx = blockIdx.x;
    const double *p = rows + 30 * idx;
    double integral = 0.;
    if (!p[24]) {
        GlobalEmissivity f = {
            {{p, sigma, slope, low, step, 0., ns, (int)p[25], p[26]}, idx / 2, (bool)(idx % 2)}};
        integral = table_integral(p[3], p[4], .001, f, rule);
    }
    if (!threadIdx.x) out[idx] = integral * p[17];
}

extern "C" __global__ void eulerian_cells(int n, const double *meta, const double *values,
                                          const float *density, const float *acg, const float *mcg,
                                          const double *p, float *out, int *bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool metal = p[3];
    int stride = metal ? 3 : 2;
    double delta = density[i], a = acg[i], m = mcg[i];
    bool active = pow(10., a) <= p[0];
    double photons = molecular_emissivity(delta, a, 0, meta, values, bad);
    if (active) photons += molecular_emissivity(delta, m, stride, meta, values, bad);
    out[i] = photons;
    out[n + i] = (1. + delta) * molecular_emissivity(delta, a, 1, meta, values, bad);
    out[2 * n + i] =
        active ? (1. + delta) * molecular_emissivity(delta, m, stride + 1, meta, values, bad) : 0.;
    if (metal) {
        double flux = molecular_emissivity(delta, a, 2, meta, values, bad);
        if (active) flux += molecular_emissivity(delta, m, stride + 2, meta, values, bad);
        out[3 * n + i] = (1. + delta) * flux;
    } else {
        // map_mass.c stores the ACG contribution in float before adding MCGs.
        float acg_flux = p[1] * out[n + i];
        out[3 * n + i] = acg_flux + p[2] * out[2 * n + i];
    }
}
