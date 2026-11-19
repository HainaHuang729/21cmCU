extern "C" __global__ void ic_source_padded(float *out, const float *left,
    const float *right, const float *cross, long n, int nlos, int stride) {
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long padded = (i/nlos)*stride + i%nlos;
    out[padded] = (out[padded] + left[i]*right[i]) - cross[padded]*cross[padded];
}
