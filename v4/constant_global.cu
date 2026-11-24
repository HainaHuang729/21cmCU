// 4.3.2's constant-efficiency model integrates M dn/dlnM over the same
// finite mass interval for Nion and both samples of the SFRD derivative.
struct ConstantCollapsedMass {
    SourceIntegral grid;
    double z, growth, delta_c, omm, oml, omr, delta_c_delos;
    int hmf;
    __device__ double operator()(double logmass) const {
        double sigma0 = grid.interp(grid.sigma, logmass);
        double pos_ds2 = pow(10., grid.interp(grid.slope, logmass));
        return exp(logmass) * hmf_unconditional(hmf, z, growth, logmass, sigma0, pos_ds2, omm, oml,
                                                omr, delta_c, delta_c_delos);
    }
};

extern "C" __global__ void constant_global_integrals(const double *rows, const float *sigma,
                                                     const float *slope, double low, double step,
                                                     int ns, const double *rule, double *out) {
    const int i = blockIdx.x;
    const double *row = rows + 12 * i;
    double values[2];
    const bool derivative = row[3] != 0.;
    const int hmf = (int)row[7];
    SourceIntegral grid = {nullptr, sigma, slope, low, step, 0., ns, hmf, row[8]};
    for (int j = 0; j < (derivative ? 2 : 1); ++j) {
        double z = derivative ? (j == 0 ? row[6] + 0.001 : row[6] - 0.001) : row[6];
        ConstantCollapsedMass integrand = {grid,    z,       row[2 + j], row[4], row[9],
                                           row[10], row[11], row[8],     hmf};
        values[j] = table_integral(row[0], row[1], .001, integrand, rule);
    }
    if (!threadIdx.x) {
        double value = derivative ? (values[0] - values[1]) / .002 : values[0];
        out[i] = value * row[5];
    }
}
