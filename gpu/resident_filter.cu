extern "C" __global__ void filter_spectrum(
    const float2* input, float2* output, const int side, const double dk,
    const float radius, const int apply_filter
) {
    const long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const int half = side / 2 + 1;
    if (index >= (long)side * side * half) return;

    const int iz = index % half;
    const int iy = (index / half) % side;
    const int ix = index / (half * side);
    const float kx = (float)((ix > side / 2 ? ix - side : ix) * dk);
    const float ky = (float)((iy > side / 2 ? iy - side : iy) * dk);
    const float kz = (float)(iz * dk);
    const float kr = sqrtf(kx*kx + ky*ky + kz*kz) * radius;
    const float norm = (float)((long)side * side * side);
    const float2 f = input[index];
    double window = 1.;
    if (apply_filter && (double)kr > 1.e-4) {
        const double q = kr;
        window = 3. * pow(q, -3.) * (sin(q) - cos(q) * q);
    }
    output[index] = make_float2((float)((f.x / norm) * window),
                                (float)((f.y / norm) * window));
}

extern "C" __global__ void extrapolate_density(
    const float* input, float* output, const long size,
    const float inverse_growth, const float low
) {
    const long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;

    const float value = input[index];
    output[index] = (value <= -1.f ? low : value) * inverse_growth;
}
