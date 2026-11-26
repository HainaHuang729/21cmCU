#ifndef P21_CUDA_PHILOX_CUH
#define P21_CUDA_PHILOX_CUH

#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Deterministic counter-based Philox4x32-10 PRNG for parallel CUDA streams. */
struct P21PhiloxRng {
    uint32_t k0, k1;
    uint32_t c0, c1, c2, c3;
    uint32_t buf[4];
    int idx;

    __device__ void refill() {
        uint32_t v0 = c0, v1 = c1, v2 = c2, v3 = c3;
        uint32_t rk0 = k0, rk1 = k1;
        constexpr uint32_t kM0 = 0xD2511F53U;
        constexpr uint32_t kM1 = 0xCD9E8D57U;
        constexpr uint32_t kW0 = 0x9E3779B9U;
        constexpr uint32_t kW1 = 0xBB67AE85U;

        for (int r = 0; r < 10; ++r) {
            uint32_t hi0 = __umulhi(kM0, v0);
            uint32_t lo0 = kM0 * v0;
            uint32_t hi1 = __umulhi(kM1, v2);
            uint32_t lo1 = kM1 * v2;
            uint32_t nv0 = hi1 ^ v1 ^ rk0;
            uint32_t nv1 = lo1;
            uint32_t nv2 = hi0 ^ v3 ^ rk1;
            uint32_t nv3 = lo0;
            v0 = nv0;
            v1 = nv1;
            v2 = nv2;
            v3 = nv3;
            rk0 += kW0;
            rk1 += kW1;
        }
        buf[0] = v0;
        buf[1] = v1;
        buf[2] = v2;
        buf[3] = v3;
        idx = 0;
        if (++c0 == 0U) {
            c1 = (c1 & 0xFF000000U) | ((c1 + 1U) & 0x00FFFFFFU);
        }
    }

    __device__ void init(uint64_t seed, uint64_t stream_id, uint32_t substream = 0) {
        k0 = (uint32_t)(seed & 0xFFFFFFFFULL);
        k1 = (uint32_t)(seed >> 32);
        c0 = 0U;
        c1 = (substream & 0xFFU) << 24;
        c2 = (uint32_t)(stream_id & 0xFFFFFFFFULL);
        c3 = (uint32_t)(stream_id >> 32);
        idx = 4;
    }

    __device__ uint32_t next_u32() {
        if (idx >= 4) {
            refill();
        }
        return buf[idx++];
    }

    __device__ double uniform() {
        uint64_t hi = (uint64_t)next_u32();
        uint64_t lo = (uint64_t)next_u32();
        uint64_t u53 = ((hi << 21) ^ (lo >> 11)) & ((1ULL << 53) - 1ULL);
        return (double)u53 * (1.0 / 9007199254740992.0);
    }

    __device__ void ugaussian_pair(double *z0, double *z1) {
        double u1 = uniform();
        if (u1 < 1e-300) {
            u1 = 1e-300;
        }
        double u2 = uniform();
        double r = sqrt(-2.0 * log(u1));
        double ang = 2.0 * M_PI * u2;
        *z0 = r * cos(ang);
        *z1 = r * sin(ang);
    }

    __device__ double ugaussian() {
        double u1 = uniform();
        if (u1 < 1e-300) {
            u1 = 1e-300;
        }
        double u2 = uniform();
        return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    }

    __device__ double ugaussian_tail(double a) {
        if (a <= 0.0) {
            double z = ugaussian();
            return fabs(z);
        }
        if (a < 1.0) {
            while (true) {
                double z = ugaussian();
                if (z > a) {
                    return z;
                }
            }
        }
        double s = a * a;
        while (true) {
            double u = uniform();
            double v = uniform();
            double d = 1.0 - u;
            if (d < 1e-300) {
                d = 1e-300;
            }
            double x = sqrt(s - 2.0 * log(d));
            if (x * v <= a) {
                return x;
            }
        }
    }

    __device__ int poisson(double lam) {
        if (lam <= 0.0) {
            return 0;
        }
        if (lam < 30.0) {
            double L = exp(-lam);
            int k = 0;
            double p = 1.0;
            do {
                ++k;
                double u = 1.0 - uniform();
                p *= u;
            } while (p > L);
            return k - 1;
        }
        /* Hormann PTRS rejection algorithm for large lambda >= 30. */
        double slam = sqrt(lam);
        double loglam = log(lam);
        double b = 0.931 + 2.53 * slam;
        double a = -0.059 + 0.02483 * b;
        double inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
        double v_r = 0.9277 - 3.6224 / (b - 2.0);
        while (true) {
            double u = uniform() - 0.5;
            double v = uniform();
            double us = 0.5 - fabs(u);
            double kf = floor((2.0 * a / us + b) * u + lam + 0.43);
            if (kf < 0.0) {
                continue;
            }
            int k = (int)kf;
            if (us >= 0.07 && v <= v_r) {
                return k;
            }
            if (us < 0.013 && v > us) {
                continue;
            }
            if ((log(v) + log(inv_alpha) - log(a / (us * us) + b)) <=
                (-lam + (double)k * loglam - lgamma((double)k + 1.0))) {
                return k;
            }
        }
    }
};

#endif /* P21_CUDA_PHILOX_CUH */
