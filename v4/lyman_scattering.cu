__device__ float2 apply_scattering_weight(float2 mode, double weight, long long total) {
    float inverse_pixels = 1.f / (float)total;
    return make_float2((mode.x * inverse_pixels) * weight, (mode.y * inverse_pixels) * weight);
}

extern "C" __global__ void scattering_window(float2 *modes, int nx, int ny, int nz, double dk_xy,
                                             double dk_z, double inner, double outer,
                                             double alpha_outer, double beta_outer,
                                             double alpha_inner, double beta_inner,
                                             double *weights) {
    int half = nz / 2 + 1;
    long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= (long long)nx * ny * half) return;

    int x = (int)(index / ((long long)ny * half));
    int y = (int)(index / half % ny);
    int z = (int)(index % half);
    float kx = (x > nx / 2 ? x - nx : x) * dk_xy;
    float ky = (y > ny / 2 ? y - ny : y) * dk_xy;
    float kz = z * dk_z;
    float k_squared = kx * kx + ky * ky + kz * kz;
    multiple_scattering_params parameters{alpha_outer, beta_outer, alpha_inner, beta_inner};
    double weight = multiple_scattering_filter(sqrt((double)k_squared), inner, outer, &parameters);
    if (weights) weights[index] = weight;
    modes[index] = apply_scattering_weight(modes[index], weight, (long long)nx * ny * nz);
}

extern "C" __global__ void reuse_scattering_window(float2 *modes, int nx, int ny, int nz,
                                                   const double *weights) {
    long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= (long long)nx * ny * (nz / 2 + 1)) return;
    modes[index] = apply_scattering_weight(modes[index], weights[index], (long long)nx * ny * nz);
}
