// The four native filter functions are supplied from the installed 4.3.2 source.
extern "C" __global__ void halo_window(float2 *spectrum, int nx, int ny, int nz, double dk_xy,
                                       double dk_z, float radius, float mfp, double decay,
                                       int kind) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int half_z = nz / 2 + 1;
    if (i >= (long long)nx * ny * half_z) return;
    int x = (int)(i / ((long long)ny * half_z)), y = (int)(i / half_z % ny);
    float kx = (x > nx / 2 ? x - nx : x) * dk_xy;
    float ky = (y > ny / 2 ? y - ny : y) * dk_xy;
    float kz = (i % half_z) * dk_z;
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
