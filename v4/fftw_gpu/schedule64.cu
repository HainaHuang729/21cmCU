// One independent FFTW direct-64 codelet per line on a8.
template <bool inv, bool real>
__global__ void line64(float *src, float *dst, int lines) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= lines) return;
    if (real) {
        float *r = inv ? dst + row * 64 : src + row * 64;
        float *c = inv ? src + row * 66 : dst + row * 66;
        if (inv)
            r2cb_64(r, r + 1, c, c + 1, 2, 2, 2, 1, 0, 0);
        else {
            r2cf_64(r, r + 1, c, c + 1, 2, 2, 2, 1, 0, 0);
            c[1] = c[65] = 0;
        }
    } else {
        float *a = src + row * 128, *b = dst + row * 128;
        if (inv)
            n1bv_64(a, a, b, b, 2, 2, 1, 0, 0);
        else
            n1fv_64(a, a, b, b, 2, 2, 1, 0, 0);
    }
}
