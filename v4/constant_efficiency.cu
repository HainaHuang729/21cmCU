// erfcc(float) above is taken from the linked 4.3.2 hmf.c. It also appears
// in v3 ps.c. The native approximation rounds both its argument and result.
__device__ double eps_fraction(float growth, float delta, float sigma_min, float sigma_cell,
                               double delta_c) {
    if (sigma_cell == sigma_min) return 0.;
    double sigma = sqrt((double)(sigma_min * sigma_min - sigma_cell * sigma_cell));
    double barrier = (delta_c - delta) / growth;
    double x = barrier / (sqrt(2.) * sigma);
    return x < 0. ? 1. : erfcc((float)x);
}

extern "C" __global__ void eps_source_table(const double *p, int n, int sfr, float *out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float delta = (float)(p[1] + (float)i / ((float)n - 1.) * (p[2] - p[1]));
    double value;
    if (sfr) {
        double above = eps_fraction((float)p[6], delta, (float)p[3], (float)p[4], p[8]);
        double below = eps_fraction((float)p[7], delta, (float)p[3], (float)p[4], p[8]);
        float derivative = (float)((above - below) / (2. * .001));
        value = derivative * p[9] * p[10] * p[11] / p[12];
    } else {
        value = eps_fraction((float)p[5], delta, (float)p[3], (float)p[4], p[8]) * p[13];
    }
    float logged = (float)log(value);
    float minimum = sfr ? -50.f : -40.f;
    out[i] = logged < minimum ? minimum : logged;
}
