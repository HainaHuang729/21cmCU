struct GlobalPopulation {
    GlobalIntegral hmf;
    double turnover;
    bool mini;

    __device__ double operator()(double x) const {
        if (!mini) return hmf(x);
        const double *p = hmf.base.p;
        double mass = exp(x), pivot = 7. * 2.30258509299404568402;
        double stellar = log_scaling_PL_limit(x, p[13], p[11], pivot, p[15]);
        double escape = log_scaling_PL_limit(x, p[14], p[12], pivot, p[16]);
        return hmf.weighted_hmf(x, molecular_weight(x, mass, stellar, escape, turnover, p[23]));
    }
};

extern "C" __global__ void mcg_nion_history(const double *rows, const double *turns, int nt,
                                            const float *sigma, const float *slope, double low,
                                            double step, int ns, const double *rule, double *out) {
    int idx = blockIdx.x, z = idx / (nt + 1), col = idx % (nt + 1);
    bool mini = col != 0;
    const double *p = rows + (z * 2 + mini) * 25;
    double val = 0.;
    if (!p[24]) {
        GlobalPopulation f = {
            {{p, sigma, slope, low, step, 0., ns, (int)p[18], p[19]}, p[20], p[21], p[22]},
            mini ? turns[col - 1] : p[10],
            mini};
        val = table_integral(p[3], p[4], .001, f, rule);
    }
    if (!threadIdx.x) out[idx] = val * p[17];
}
