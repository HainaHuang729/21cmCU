extern "C" __global__ void collapse_candidates(
    const float *density, const float *growth, const long size, const int scales,
    const double threshold, const double margin, unsigned int *count,
    long *candidates, const unsigned int capacity
) {
    const long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size * scales) return;
    const float d = density[index] * growth[index / size];
    if (fabs((double)d - threshold) <= margin) {
        const unsigned int offset = atomicAdd(count, 1);
        if (offset < capacity) candidates[offset] = index;
    }
}
