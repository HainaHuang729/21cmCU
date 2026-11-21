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

    double curr = (double)input[index];
    if (curr <= -1.0) curr = -1.0 + 1.0e-7;
    output[index] = (float)(curr * (double)inverse_growth);
}

extern "C" __global__ void unpack_extrapolate_bounds_pass1(
    const float* __restrict__ padded_in,
    float* __restrict__ out,
    float* __restrict__ block_min,
    float* __restrict__ block_max,
    const int side,
    const int nr,
    const float inverse_growth,
    const double fract_err
) {
    const int j = (int)blockIdx.y;
    if (j >= nr) return;
    const int padded = 2 * (side / 2 + 1);
    const long n = (long)side * (long)side * (long)side;
    const long padded_vol = (long)side * (long)side * (long)padded;
    const float* src_j = padded_in + (long)j * padded_vol;
    float* dst_j = out + (long)j * n;
    float lo = 0.0f;
    float hi = 0.0f;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += (long)gridDim.x * blockDim.x) {
        const long xy = idx / side;
        const int z = (int)(idx - xy * side);
        double curr = (double)src_j[xy * padded + z];
        if (curr <= -1.0) curr = -1.0 + fract_err;
        curr *= (double)inverse_growth;
        const float fv = (float)curr;
        dst_j[idx] = fv;
        if (curr < (double)lo) lo = fv;
        if (curr > (double)hi) hi = fv;
    }
    __shared__ float sh_min[256];
    __shared__ float sh_max[256];
    sh_min[threadIdx.x] = lo;
    sh_max[threadIdx.x] = hi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh_min[threadIdx.x] = fminf(sh_min[threadIdx.x], sh_min[threadIdx.x + s]);
            sh_max[threadIdx.x] = fmaxf(sh_max[threadIdx.x], sh_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_min[j * 256 + blockIdx.x] = sh_min[0];
        block_max[j * 256 + blockIdx.x] = sh_max[0];
    }
}

extern "C" __global__ void unpack_extrapolate_bounds_pass2(
    const float* __restrict__ block_min,
    const float* __restrict__ block_max,
    float* __restrict__ out_min,
    float* __restrict__ out_max,
    const int nr
) {
    const int j = (int)blockIdx.x;
    if (j >= nr) return;
    __shared__ float sh_min[256];
    __shared__ float sh_max[256];
    sh_min[threadIdx.x] = block_min[j * 256 + threadIdx.x];
    sh_max[threadIdx.x] = block_max[j * 256 + threadIdx.x];
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh_min[threadIdx.x] = fminf(sh_min[threadIdx.x], sh_min[threadIdx.x + s]);
            sh_max[threadIdx.x] = fmaxf(sh_max[threadIdx.x], sh_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_min[j] = sh_min[0];
        out_max[j] = sh_max[0];
    }
}
