// Emit whole MT19937 blocks. The wraparound uses already updated words,
// so the twist must advance in three dependency-ordered intervals.
extern "C" __global__ void mt_stream(unsigned *state, unsigned *out, int blocks, unsigned seed,
                                     int init) {
    __shared__ unsigned mt[624];
    int lane = threadIdx.x;
    if (init && lane == 0) {
        mt[0] = seed;
        for (int j = 1; j < 624; ++j) mt[j] = 1812433253u * (mt[j - 1] ^ (mt[j - 1] >> 30)) + j;
    } else if (!init) {
        for (int j = lane; j < 624; j += blockDim.x) mt[j] = state[j];
    }
    __syncthreads();
    for (int block = 0; block < blocks; ++block) {
        for (int first = 0; first < 623; first += 227) {
            int j = first + lane;
            unsigned next = 0;
            if (lane < 227 && j < 623) {
                unsigned mix = (mt[j] & 0x80000000u) | (mt[j + 1] & 0x7fffffffu);
                next = mt[(j + 397) % 624] ^ (mix >> 1) ^ ((0u - (mix & 1u)) & 0x9908b0dfu);
            }
            __syncthreads();
            if (lane < 227 && j < 623) mt[j] = next;
            __syncthreads();
        }
        if (lane == 0) {
            unsigned mix = (mt[623] & 0x80000000u) | (mt[0] & 0x7fffffffu);
            mt[623] = mt[396] ^ (mix >> 1) ^ ((0u - (mix & 1u)) & 0x9908b0dfu);
        }
        __syncthreads();
        for (int j = lane; j < 624; j += blockDim.x) {
            unsigned v = mt[j];
            v ^= v >> 11;
            v ^= (v << 7) & 0x9d2c5680u;
            v ^= (v << 15) & 0xefc60000u;
            v ^= v >> 18;
            out[block * 624 + j] = v;
        }
        __syncthreads();
    }
    for (int j = lane; j < 624; j += blockDim.x) state[j] = mt[j];
}

extern "C" __global__ void gfsr_stream(unsigned *ring, unsigned *out, int count, unsigned seed,
                                       int init, int offset) {
    int lane = threadIdx.x;
    if (init) {
        // Jump to the first LCG bit of this word by binary exponentiation.
        for (int j = lane; j < 16384; j += blockDim.x) {
            unsigned v = seed, base = 69069u;
            for (int skip = 32 * j; skip; skip >>= 1, base *= base)
                if (skip & 1) v *= base;
            unsigned word = 0;
            for (int b = 0; b < 32; b++) {
                v *= 69069u;
                word = (word << 1) | (v >> 31);
            }
            if (j >= 7 && (j - 7) % 3 == 0 && (j - 7) / 3 < 32) {
                int b = (j - 7) / 3;
                word = (word & (0xffffffffu >> b)) | (0x80000000u >> b);
            }
            ring[j] = word;
        }
    }
    __syncthreads();
    for (int off = 0; off < count; off += blockDim.x) {
        int j = off + lane, at = (33 + offset + j) & 16383;
        if (j < count) {
            unsigned v = ring[(at - 471) & 16383] ^ ring[(at - 1586) & 16383] ^
                         ring[(at - 6988) & 16383] ^ ring[(at - 9689) & 16383];
            ring[at] = v;
            out[j] = v;
        }
        __syncthreads();
    }
}

extern "C" __global__ void cmrg_stream(unsigned *state, unsigned *out, int count, unsigned seed,
                                       int init) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long m1 = 2147483647LL, m2 = 2145483479LL;
    long long x1, x2, x3, y1, y2, y3;
    int warm = 0;
    if (init) {
        unsigned s = seed ? seed : 1u;
        s *= 69069u;
        x1 = (long long)(s % 2147483647u);
        s *= 69069u;
        x2 = (long long)(s % 2147483647u);
        s *= 69069u;
        x3 = (long long)(s % 2147483647u);
        s *= 69069u;
        y1 = (long long)(s % 2145483479u);
        s *= 69069u;
        y2 = (long long)(s % 2145483479u);
        s *= 69069u;
        y3 = (long long)(s % 2145483479u);
        warm = 7;
    } else {
        x1 = (long long)state[0];
        x2 = (long long)state[1];
        x3 = (long long)state[2];
        y1 = (long long)state[3];
        y2 = (long long)state[4];
        y3 = (long long)state[5];
    }
    for (int step = 0; step < warm + count; ++step) {
        long long h3 = x3 / 11714LL;
        long long p3 = 183326LL * (x3 - h3 * 11714LL) - h3 * 2883LL;
        long long h2 = x2 / 33921LL;
        long long p2 = 63308LL * (x2 - h2 * 33921LL) - h2 * 12979LL;
        if (p3 < 0) p3 += m1;
        if (p2 < 0) p2 += m1;
        x3 = x2;
        x2 = x1;
        x1 = p2 - p3;
        if (x1 < 0) x1 += m1;
        long long k3 = y3 / 3976LL;
        long long q3 = 539608LL * (y3 - k3 * 3976LL) - k3 * 2071LL;
        long long k1 = y1 / 24919LL;
        long long q1 = 86098LL * (y1 - k1 * 24919LL) - k1 * 7417LL;
        if (q3 < 0) q3 += m2;
        if (q1 < 0) q1 += m2;
        y3 = y2;
        y2 = y1;
        y1 = q1 - q3;
        if (y1 < 0) y1 += m2;
        long long ans = x1 - y1;
        if (ans < 0) ans += m1;
        if (step >= warm) out[step - warm] = (unsigned)ans;
    }
    state[0] = (unsigned)x1;
    state[1] = (unsigned)x2;
    state[2] = (unsigned)x3;
    state[3] = (unsigned)y1;
    state[4] = (unsigned)y2;
    state[5] = (unsigned)y3;
}

extern "C" __global__ void mrg_stream(unsigned *state, unsigned *out, int count, unsigned seed,
                                      int init) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long m = 2147483647LL;
    long long x1, x2, x3, x4, x5;
    int warm = 0;
    if (init) {
        unsigned s = seed ? seed : 1u;
        s *= 69069u;
        x1 = (long long)(s % 2147483647u);
        s *= 69069u;
        x2 = (long long)(s % 2147483647u);
        s *= 69069u;
        x3 = (long long)(s % 2147483647u);
        s *= 69069u;
        x4 = (long long)(s % 2147483647u);
        s *= 69069u;
        x5 = (long long)(s % 2147483647u);
        warm = 6;
    } else {
        x1 = (long long)state[0];
        x2 = (long long)state[1];
        x3 = (long long)state[2];
        x4 = (long long)state[3];
        x5 = (long long)state[4];
    }
    for (int step = 0; step < warm + count; ++step) {
        long long h5 = x5 / 20554LL;
        long long p5 = 104480LL * (x5 - h5 * 20554LL) - h5 * 1727LL;
        if (p5 > 0) p5 -= m;
        long long h1 = x1 / 20LL;
        long long p1 = 107374182LL * (x1 - h1 * 20LL) - h1 * 7LL;
        if (p1 < 0) p1 += m;
        p1 += p5;
        if (p1 < 0) p1 += m;
        x5 = x4;
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = p1;
        if (step >= warm) out[step - warm] = (unsigned)x1;
    }
    state[0] = (unsigned)x1;
    state[1] = (unsigned)x2;
    state[2] = (unsigned)x3;
    state[3] = (unsigned)x4;
    state[4] = (unsigned)x5;
}

extern "C" __global__ void taus2_stream(unsigned *state, unsigned *out, int count, unsigned seed,
                                        int init) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned s1, s2, s3;
    int warm = 0;
    if (init) {
        s1 = 69069u * (seed ? seed : 1u);
        if (s1 < 2u) s1 += 2u;
        s2 = 69069u * s1;
        if (s2 < 8u) s2 += 8u;
        s3 = 69069u * s2;
        if (s3 < 16u) s3 += 16u;
        warm = 6;
    } else {
        s1 = state[0];
        s2 = state[1];
        s3 = state[2];
    }
    for (int step = 0; step < warm + count; ++step) {
        unsigned b1 = ((s1 << 13) ^ s1) >> 19;
        s1 = ((s1 & 4294967294u) << 12) ^ b1;
        unsigned b2 = ((s2 << 2) ^ s2) >> 25;
        s2 = ((s2 & 4294967288u) << 4) ^ b2;
        unsigned b3 = ((s3 << 3) ^ s3) >> 11;
        s3 = ((s3 & 4294967280u) << 17) ^ b3;
        if (step >= warm) out[step - warm] = s1 ^ s2 ^ s3;
    }
    state[0] = s1;
    state[1] = s2;
    state[2] = s3;
}

extern "C" __global__ void hessian_modes(const float2 *modes, float2 *out, int n, double dk,
                                         double dk_y, int a, int b) {
    int h = n / 2 + 1, j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j >= n * n * h) return;
    int x = j / (n * h), y = j / h % n, z = j % h;
    double k[3] = {(x > n / 2 ? x - n : x) * dk, (y > n / 2 ? y - n : y) * dk_y, z * dk};
    double kk =
        __dadd_rn(__dadd_rn(__dmul_rn(k[0], k[0]), __dmul_rn(k[1], k[1])), __dmul_rn(k[2], k[2]));
    double weight = __dmul_rn(-k[a], k[b]);
    float2 v = modes[j];
    out[j] = j ? make_float2(__ddiv_rn(__dmul_rn(v.x, weight), kk),
                             __ddiv_rn(__dmul_rn(v.y, weight), kk))
               : make_float2(0, 0);
}
