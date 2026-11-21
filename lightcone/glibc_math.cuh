#ifndef GLIBC_MATH_FMA
#define GLIBC_MATH_FMA 0
#endif

__device__ __constant__ unsigned long long GLIBC_EXP_T[32] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL, 0x3fef9301d0125b51ULL,
    0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL, 0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL,
    0x3fef06fe0a31b715ULL, 0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL, 0x3feea47eb03a5585ULL,
    0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL, 0x3feea11473eb0187ULL, 0x3feea589994cce13ULL,
    0x3feeace5422aa0dbULL, 0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL, 0x3fef3720dcef9069ULL,
    0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL, 0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL
};

__device__ __constant__ unsigned long long GLIBC_LOG_T[32] = {
    0x3ff661ec79f8f3beULL, 0xbfd57bf7808caadeULL, 0x3ff571ed4aaf883dULL, 0xbfd2bef0a7c06ddbULL,
    0x3ff49539f0f010b0ULL, 0xbfd01eae7f513a67ULL, 0x3ff3c995b0b80385ULL, 0xbfcb31d8a68224e9ULL,
    0x3ff30d190c8864a5ULL, 0xbfc6574f0ac07758ULL, 0x3ff25e227b0b8ea0ULL, 0xbfc1aa2bc79c8100ULL,
    0x3ff1bb4a4a1a343fULL, 0xbfba4e76ce8c0e5eULL, 0x3ff12358f08ae5baULL, 0xbfb1973c5a611cccULL,
    0x3ff0953f419900a7ULL, 0xbfa252f438e10c1eULL, 0x3ff0000000000000ULL, 0x0000000000000000ULL,
    0x3fee608cfd9a47acULL, 0x3faaa5aa5df25984ULL, 0x3feca4b31f026aa0ULL, 0x3fbc5e53aa362eb4ULL,
    0x3feb2036576afce6ULL, 0x3fc526e57720db08ULL, 0x3fe9c2d163a1aa2dULL, 0x3fcbc2860d224770ULL,
    0x3fe886e6037841edULL, 0x3fd1058bc8a07ee1ULL, 0x3fe767dcf5534862ULL, 0x3fd4043057b6ee09ULL
};

__device__ __forceinline__ float glibc_logf(float x) {
    unsigned int ix = __float_as_uint(x);
    if (ix == 0x3f800000u) return 0.0f;
    if (ix < 0x00800000u) {
        if (ix == 0u) return -1.0f / 0.0f;
        x *= 8388608.0f;
        ix = __float_as_uint(x) - 0x0b800000u;
    }
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 0xfu);
    int k = ((int)tmp) >> 23;
    unsigned int iz = ix - (tmp & 0xff800000u);
    double z = (double)__uint_as_float(iz);
    double invc = __longlong_as_double((long long)GLIBC_LOG_T[2 * i]);
    double logc = __longlong_as_double((long long)GLIBC_LOG_T[2 * i + 1]);
    double ln2 = __longlong_as_double((long long)0x3fe62e42fefa39efULL);
    double A0  = __longlong_as_double((long long)0xbfd00ea348b88334ULL);
    double A1  = __longlong_as_double((long long)0x3fd5575b0be00b6aULL);
    double A2  = __longlong_as_double((long long)0xbfdffffef20a4123ULL);
#if GLIBC_MATH_FMA
    double y = fma((double)k, ln2, logc);
    double r = fma(z, invc, -1.0);
    double r2 = r * r;
    double p = fma(A1, r, A2);
    double yr = r + y;
    p = fma(A0, r2, p);
    return (float)fma(r2, p, yr);
#else
    double r = z * invc - 1.0;
    double r2 = r * r;
    double p = (A0 * r2) + (A1 * r + A2);
    double y = (double)k * ln2 + logc;
    return (float)((y + r) + r2 * p);
#endif
}

__device__ __forceinline__ float glibc_log10f(float x) {
    int ix = __float_as_int(x);
    int k = 0;
    if (ix <= 0x007fffff) {
        if ((ix & 0x7fffffff) == 0) return -33554432.0f / fabsf(x);
        if (ix < 0) return (x - x) / (x - x);
        x *= 33554432.0f;
        k = -25;
        ix = __float_as_int(x);
    }
    if (ix > 0x7f7fffff) return x + x;
    k += (ix >> 23) - 127;
    int sgn = (int)(((unsigned int)k) >> 31);
    k += sgn;
    float fk = (float)k;
    float m = __int_as_float(((127 - sgn) << 23) | (ix & 0x7fffff));
    float y = fk * __int_as_float(0x355427db);
    float hi = fk * __int_as_float(0x3e9a2080);
    float z = glibc_logf(m) * __int_as_float(0x3ede5bd9);
    return (z + y) + hi;
}

__device__ __forceinline__ float glibc_expf(float x) {
    if (x < -103.97207641601562f) return 0.0f;
    if (x > 88.72283172607422f) return __int_as_float(0x7f800000);
    double xd = (double)x;
    double SHIFT    = __longlong_as_double((long long)0x4338000000000000ULL);
    double INVLN2_N = __longlong_as_double((long long)0x40471547652b82feULL);
    double C0       = __longlong_as_double((long long)0x3ebc6af84b912394ULL);
    double C1       = __longlong_as_double((long long)0x3f2ebfce50fac4f3ULL);
    double C2       = __longlong_as_double((long long)0x3f962e42ff0c52d6ULL);
#if GLIBC_MATH_FMA
    double z = fma(INVLN2_N, xd, SHIFT);
    unsigned long long ki = (unsigned long long)__double_as_longlong(z);
    double kd = z - SHIFT;
    double r = fma(INVLN2_N, xd, -kd);
    unsigned long long t = GLIBC_EXP_T[ki & 0x1fu] + (ki << 47);
    double s = __longlong_as_double((long long)t);
    double r2 = r * r;
    double p = fma(C0, r, C1);
    double q = fma(C2, r, 1.0);
    double y = fma(p, r2, q);
    return (float)(y * s);
#else
    double z = INVLN2_N * xd;
    double tmp = z + SHIFT;
    unsigned long long ki = (unsigned long long)__double_as_longlong(tmp);
    double kd = tmp - SHIFT;
    double r = z - kd;
    unsigned long long t = GLIBC_EXP_T[ki & 0x1fu] + (ki << 47);
    double s = __longlong_as_double((long long)t);
    double r2 = r * r;
    double y = ((C0 * r + C1) * r2) + (C2 * r + 1.0);
    return (float)(y * s);
#endif
}
