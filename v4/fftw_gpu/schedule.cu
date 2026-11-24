// One thread transforms one contiguous line. Strides below are in floats.
template <bool inverse>
DEV void direct(int n, float *in, float *out, int is, int os) {
    if (n == 15) {
        if (inverse)
            n1bv_15(in, in, out, out, is, os, 1, 0, 0);
        else
            n1fv_15(in, in, out, out, is, os, 1, 0, 0);
    } else if (n == 5) {
        if (inverse)
            n1bv_5(in, in, out, out, is, os, 1, 0, 0);
        else
            n1fv_5(in, in, out, out, is, os, 1, 0, 0);
    } else {
        if (inverse)
            n1bv_2(in, in, out, out, is, os, 1, 0, 0);
        else
            n1fv_2(in, in, out, out, is, os, 1, 0, 0);
    }
}
template <bool inv>
DEV void combine(int r, int m, float *a, const float *w) {
    if (r == 5) {
        if (inv)
            t1buv_5(a, a, w, 2 * m, 0, m, 2);
        else
            t1fuv_5(a, a, w, 2 * m, 0, m, 2);
    } else {
        if (inv)
            t1buv_10(a, a, w, 2 * m, 0, m, 2);
        else
            t1fuv_10(a, a, w, 2 * m, 0, m, 2);
    }
}
template <int N, bool inv>
DEV void complex_line(float *a, float *b, const float *w, const float *w10) {
    if (N == 50) {
        if (inv)
            q1bv_5(a, a, w10, 20, 2, 0, 2, 10);
        else
            q1fv_5(a, a, w10, 20, 2, 0, 2, 10);
        for (int row = 0; row < 5; row++)
            for (int k = 0; k < 5; k++)
                direct<inv>(2, a + 20 * row + 2 * k, b + 20 * row + 2 * k, 10, 10);
        combine<inv>(5, 10, b, w);
    } else {
        for (int j = 0; j < 10; j++) direct<inv>(15, a + 2 * j, b + 30 * j, 20, 2);
        combine<inv>(10, 15, b, w);
    }
}
template <int N, bool inv>
__global__ void axis_fft(const float *src, float *dst, int lines, const float *w,
                         const float *w10) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= lines) return;
    float a[2 * N], b[2 * N];
    for (int i = 0; i < 2 * N; i++) a[i] = src[row * 2 * N + i];
    complex_line<N, inv>(a, b, w, w10);
    for (int i = 0; i < 2 * N; i++) dst[row * 2 * N + i] = b[i];
}
template <int N, bool inv>
__global__ void real_fft(const float *src, float *dst, int lines, const float *w,
                         const float *w25) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= lines) return;
    constexpr int r = N == 50 ? 2 : 10, m = N / r;
    float a[N + 2], b[N + 2];
    for (int i = 0; i < (inv ? N + 2 : N); i++) a[i] = src[row * (inv ? N + 2 : N) + i];
    if (inv) {
        if (r == 2)
            r2cb_2(a, a + 1, a, a + 1, 2 * m, 2 * m, 2 * m, 1, 0, 0);
        else
            r2cb_10(a, a + 1, a, a + 1, 2 * m, 2 * m, 2 * m, 1, 0, 0);
        if (r == 2)
            hc2cbdftv_2(a + 2, a + 3, a + 2 * (m - 1), a + 2 * (m - 1) + 1, w, 2 * m, 1,
                        (m + 1) / 2, 2);
        else
            hc2cbdftv_10(a + 2, a + 3, a + 2 * (m - 1), a + 2 * (m - 1) + 1, w, 2 * m, 1,
                         (m + 1) / 2, 2);
    }
    if (N == 150)
        for (int j = 0; j < 5; j++)
            direct<inv>(15, inv ? a + 30 * j : a + 2 * j, inv ? b + 2 * j : b + 30 * j,
                        inv ? 2 : 10, inv ? 10 : 2);
    else {
        for (int j = 0; j < 5; j++) direct<inv>(5, a + 2 * j, b + 10 * j, 10, 2);
        combine<inv>(5, 5, b, w25);
    }
    if (!inv) {
        if (r == 2)
            r2cf_2(b, b + 1, b, b + 1, 2 * m, 2 * m, 2 * m, 1, 0, 0);
        else
            r2cf_10(b, b + 1, b, b + 1, 2 * m, 2 * m, 2 * m, 1, 0, 0);
        b[1] = b[N + 1] = 0;
        if (r == 2)
            hc2cfdftv_2(b + 2, b + 3, b + 2 * (m - 1), b + 2 * (m - 1) + 1, w, 2 * m, 1,
                        (m + 1) / 2, 2);
        else
            hc2cfdftv_10(b + 2, b + 3, b + 2 * (m - 1), b + 2 * (m - 1) + 1, w, 2 * m, 1,
                         (m + 1) / 2, 2);
    }
    for (int i = 0; i < (inv ? N : N + 2); i++) dst[row * (inv ? N : N + 2) + i] = b[i];
}
