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

__device__ double eulerian_acg_emissivity(double delta, double turn, bool photo, int field,
                                         const double *meta, const double *values, int *bad) {
    if (photo) return molecular_emissivity(delta, turn, field, meta, values, bad);
    const double *m = meta + 7 * field;
    int j = (int)floor((delta - m[0]) / m[2]);
    if (j < 0 || j >= (int)m[5] - 1) {
        atomicExch(bad, 1);
        return 0.;
    }
    double w = (delta - (m[0] + m[2] * j)) / m[2];
    const double *y = values + (int)m[4];
    return exp(y[j] * (1. - w) + y[j + 1] * w);
}

extern "C" __global__ void eulerian_cells(int n, const double *meta, const double *values,
                                          const double *dndm_meta, const double *dndm_values,
                                          const float *density, const float *acg, const float *mcg,
                                          const double *p, float *out, int *bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool metal = p[3] != 0.;
    bool extra = p[5] != 0.;
    bool thermal = p[9] != 0.;
    bool use_mcg = p[10] != 0.;
    bool photo = p[11] != 0.;
    bool need_sfr = thermal || extra;
    int stride = (need_sfr ? 1 : 0) + (thermal && metal ? 1 : 0) + 1;
    double delta = density[i];
    double a = photo ? acg[i] : 0.;
    double m = use_mcg ? mcg[i] : 0.;
    double a_lin = photo ? pow(10., a) : p[12];
    bool active = use_mcg && (a_lin <= p[0]);

    double photons = eulerian_acg_emissivity(delta, a, photo, 0, meta, values, bad);
    if (active) photons += molecular_emissivity(delta, m, stride, meta, values, bad);
    out[i] = photons;

    double sfr_a = 0., sfr_m = 0.;
    if (need_sfr) {
        sfr_a = (1. + delta) * eulerian_acg_emissivity(delta, a, photo, 1, meta, values, bad);
        if (use_mcg) {
            sfr_m = active ? (1. + delta) * molecular_emissivity(delta, m, stride + 1, meta, values, bad) : 0.;
        }
        out[n + i] = sfr_a;
        if (use_mcg) out[2 * n + i] = sfr_m;
    }
    if (thermal) {
        if (metal) {
            double flux = eulerian_acg_emissivity(delta, a, photo, 2, meta, values, bad);
            if (active) flux += molecular_emissivity(delta, m, stride + 2, meta, values, bad);
            out[3 * n + i] = (1. + delta) * flux;
        } else {
            // map_mass.c stores the ACG contribution in float before adding MCGs.
            float acg_flux = p[1] * out[n + i];
            out[3 * n + i] = use_mcg ? (acg_flux + p[2] * out[2 * n + i]) : acg_flux;
        }
    }
    if (extra) {
        out[4 * n + i] = linear_lookup(delta, dndm_meta, dndm_values, bad) * p[6] * (1. + delta);
        out[5 * n + i] = linear_lookup(delta, dndm_meta + 4, dndm_values, bad) * p[7] * (1. + delta);
        out[6 * n + i] = out[n + i] * p[8];
        if (use_mcg) out[7 * n + i] = out[2 * n + i] * p[8];
    }
}
