__device__ float2 apply_scattering_weight(float2 mode, double weight, int side) {
    float inverse_pixels = 1.f / (float)(side * side * side);
    return make_float2((mode.x * inverse_pixels) * weight, (mode.y * inverse_pixels) * weight);
}

extern "C" __global__ void scattering_window(float2 *modes, int side, double dk, double inner,
                                             double outer, double alpha_outer, double beta_outer,
                                             double alpha_inner, double beta_inner,
                                             double *weights) {
    int half = side / 2 + 1;
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= side * side * half) return;

    int x = index / (side * half);
    int y = index / half % side;
    int z = index % half;
    float kx = (x > side / 2 ? x - side : x) * dk;
    float ky = (y > side / 2 ? y - side : y) * dk;
    float kz = z * dk;
    float k_squared = kx * kx + ky * ky + kz * kz;
    multiple_scattering_params parameters{alpha_outer, beta_outer, alpha_inner, beta_inner};
    double weight = multiple_scattering_filter(sqrt((double)k_squared), inner, outer, &parameters);
    if (weights) weights[index] = weight;
    modes[index] = apply_scattering_weight(modes[index], weight, side);
}

extern "C" __global__ void reuse_scattering_window(float2 *modes, int side, const double *weights) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= side * side * (side / 2 + 1)) return;
    modes[index] = apply_scattering_weight(modes[index], weights[index], side);
}
