// Scalar CUDA implementation of the independent complex lanes of the FFTW SSE2 codelets.
typedef float R;
typedef float E;
typedef int stride;
typedef int INT;
typedef float2 V;
#define VL 1
#define TWVL 1
#define WS(a, b) ((a) * (b))
#define MAKE_VOLATILE_STRIDE(n, s) (void)0
#define VLEAVE() (void)0
#define DK(k, c) const float k = (float)(c)
#define DVK(k, c) DK(k, c)
#define LDK(k) make_float2(k, k)
#define FMA(a, b, c) ((c) + (a) * (b))
#define FNMS(a, b, c) ((c) - (a) * (b))
#define FNMA(a, b, c) (-((a) * (b) + (c)))
#define FMS(a, b, c) ((a) * (b) - (c))
#define DEV __device__ __forceinline__
DEV V VADD(V a, V b) { return make_float2(a.x + b.x, a.y + b.y); }
DEV V VSUB(V a, V b) { return make_float2(a.x - b.x, a.y - b.y); }
DEV V VMUL(V a, V b) { return make_float2(a.x * b.x, a.y * b.y); }
DEV V VCONJ(V a) { return make_float2(a.x, -a.y); }
DEV V VBYI(V a) { return make_float2(-a.y, a.x); }
#define VFMA(a, b, c) VADD(c, VMUL(a, b))
#define VFMS(a, b, c) VSUB(VMUL(a, b), c)
#define VFNMS(a, b, c) VSUB(c, VMUL(a, b))
DEV V LDW(const float *p) { return make_float2(p[0], p[1]); }
#define LD(p, s, a) LDW(p)
DEV void store(float *p, V v) {
    p[0] = v.x;
    p[1] = v.y;
}
#define ST(p, v, s, a) store(p, v)
DEV V VZMUL(V w, V a) {
    return VADD(make_float2(w.x * a.x, w.x * a.y), make_float2(-w.y * a.y, w.y * a.x));
}
DEV V VZMULJ(V w, V a) {
    return VSUB(make_float2(w.x * a.x, w.x * a.y), make_float2(-w.y * a.y, w.y * a.x));
}
DEV V VZMULI(V w, V a) { return make_float2(-w.x * a.y - w.y * a.x, w.x * a.x - w.y * a.y); }
DEV V VZMULIJ(V w, V a) { return make_float2(-w.x * a.y + w.y * a.x, w.x * a.x + w.y * a.y); }
#define BYTW(w, x) VZMUL(LDW(w), x)
#define BYTWJ(w, x) VZMULJ(LDW(w), x)
#define STM2(p, v, s, a) store(p, v)
#define STN2(p, v0, v1, s) (void)0
