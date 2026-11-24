// The four native filter functions are supplied from the installed 4.3.2 source.
extern "C" __global__ void halo_window(float2 *spectrum, int side, double dk, float radius,
                                       float mfp, double decay, int kind) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half_side = side / 2 + 1;
    if (i >= side * side * half_side) return;
    int x = i / (side * half_side), y = i / half_side % side;
    float kx = (x > side / 2 ? x - side : x) * dk;
    float ky = (y > side / 2 ? y - side : y) * dk;
    float kz = (i % half_side) * dk;
    float k2 = kx * kx + ky * ky + kz * kz;
    double k = sqrt((double)k2);
    float kr = k * radius;
    double weight;
    if (kind == 0)
        weight = real_tophat_filter(kr);
    else if (kind == 1)
        weight = sharp_k_filter(kr);
    else if (kind == 2)
        weight = gaussian_filter(k2 * radius * radius);
    else
        weight = exp_mfp_filter(k, radius, mfp, decay);
    spectrum[i] = make_float2(spectrum[i].x * weight, spectrum[i].y * weight);
}
