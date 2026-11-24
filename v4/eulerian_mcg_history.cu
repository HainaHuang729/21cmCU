// Upstream EvaluateRGTable2D_f performs bilinear interpolation in double
// precision on float32 ln(Nion), first along log10(Mturn), then density.
__device__ double photons(double d, double m, const float *table, const double *axes, int nx,
                          int ny, int *bad) {
    int x = (int)floor((d - axes[0]) / axes[1]), y = (int)floor((m - axes[2]) / axes[3]);
    // The clipped turnover can land exactly at the last native knot. Use its
    // left interval, evaluating that endpoint without an out-of-bounds read.
    if (x == nx - 1 && d == axes[0] + (nx - 1) * axes[1]) --x;
    if (y == ny - 1 && m == axes[2] + (ny - 1) * axes[3]) --y;
    if (x < 0 || x + 1 >= nx || y < 0 || y + 1 >= ny) {
        atomicExch(bad, 1);
        return 0.;
    }
    double fx = (d - (axes[0] + x * axes[1])) / axes[1],
           fy = (m - (axes[2] + y * axes[3])) / axes[3];
    const float *left = table + x * ny + y, *right = left + ny;
    double a = left[0] * (1. - fy) + left[1] * fy, b = right[0] * (1. - fy) + right[1] * fy;
    return exp(a * (1. - fx) + b * fx);
}

__device__ float advance(double now, double previous, double accumulated, double ceiling) {
    now = now > ceiling ? ceiling : (now < 0. ? 1.e-40 : now);
    previous = previous > ceiling ? ceiling : (previous < 0. ? 1.e-40 : previous);
    // 4.3.2 rounds the update into the float output before the ceiling test.
    float result = accumulated + now - previous;
    return result > ceiling ? (float)ceiling : result;
}

extern "C" __global__ void ion_history(int n, int nx, int ny, const double *axes, const double *p,
                                       const float *current_a, const float *current_m,
                                       const float *previous_a, const float *previous_m,
                                       float *density, const float *old_density,
                                       const float *turn_a, const float *turn_m, const float *old_a,
                                       const float *old_m, float *electrons, float *recombinations,
                                       float *out_a, float *out_m, int *bad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double d = density[i] = fmaxf(density[i], (float)p[0]);
    if (electrons) electrons[i] = fminf(fmaxf(electrons[i], 0.f), .999f);
    if (recombinations) recombinations[i] = fmaxf(recombinations[i], 0.f);
    double ta = turn_a[i], tm = turn_m[i];
    bool active = !(pow(10., ta) > p[1]);
    double acg = photons(d, ta, current_a, axes, nx, ny, bad);
    double mcg = active ? photons(d, tm, current_m, axes + 4, nx, ny, bad) : 0.;
    double prev_a = 0., prev_m = 0.;
    if (previous_a) {
        double prev_d = old_density[i];
        prev_a = photons(prev_d, ta, previous_a, axes + 8, nx, ny, bad);
        if (active) prev_m = photons(prev_d, tm, previous_m, axes + 12, nx, ny, bad);
    }
    out_a[i] = advance(acg, prev_a, old_a[i], p[2]);
    out_m[i] = advance(mcg, prev_m, old_m[i], p[3]);
}
