// Inputs use the reference code's half-complex layout and float wavevectors.
extern "C" __global__ void ic_modes(const float2 *source, float2 *output,
    int side, double dk, double volume, float radius, int operation, int a, int b) {
    const long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const int half = side / 2;
    const int stride = half + 1;
    if (index >= (long)side * side * stride) return;
    const int z = index % stride;
    const int y = (index / stride) % side;
    const int x = index / stride / side;
    const int mode[3] = {x, y, z};
    const float k[3] = {(float)((x > half ? x - side : x) * dk),
                        (float)((y > half ? y - side : y) * dk), (float)(z * dk)};
    const float square = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
    float2 value = source[index];
    if (operation != 0) {
        if (index == 0) {
            output[index] = make_float2(0, 0);
            return;
        }
        if (operation == 1) {
            const double scale = (double)(k[a]/square) / volume;
            // A self-conjugate derivative vanishes on the real transform planes.
            if ((side % 2) == 0 && (z == 0 || z == half) && mode[a] == half) {
                value = make_float2(0, 0);
            } else {
                value = make_float2(-value.y * scale, value.x * scale);
            }
        } else {
            const float scale = -k[a] * k[b];
            if ((side % 2) == 0 && (z == 0 || z == half) && a != b &&
                ((mode[a] == half) != (mode[b] == half))) {
                value = make_float2(0, 0);
            } else {
                value = make_float2((double)((scale*value.x)/square)/volume,
                                    (double)((scale*value.y)/square)/volume);
            }
        }
    }
    const float kr = sqrtf(square) * radius;
    if (kr > 1e-4) {
        const double r = kr;
        const double weight = 3.0 * pow(r, -3) * (sin(r) - cos(r)*r);
        value = make_float2(value.x * weight, value.y * weight);
    }
    output[index] = value;
}

extern "C" __global__ void ic_source(float *source, const float *x, const float *y,
    const float *cross, long n) {
    const long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) source[index] = (source[index] + x[index]*y[index]) - cross[index]*cross[index];
}
