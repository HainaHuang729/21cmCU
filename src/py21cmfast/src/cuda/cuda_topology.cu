#include <algorithm>
#include <cmath>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "cuda_analysis_common.cuh"
#include "cuda_philox.cuh"

namespace {

__global__ void bispectrum_mask_shell_kernel(const cufftComplex *__restrict__ d_src,
                                             cufftComplex *__restrict__ d_out, int nx, int ny,
                                             int nz, double lx, double ly, double lz, double k_lo,
                                             double k_hi, int remove_mean, int unit_mode) {
    int hz = nz / 2 + 1;
    size_t n_freq = (size_t)nx * (size_t)ny * (size_t)hz;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    double dkx = 2.0 * M_PI / lx;
    double dky = 2.0 * M_PI / ly;
    double dkz = 2.0 * M_PI / lz;

    for (size_t idx = tid; idx < n_freq; idx += stride) {
        int iz = (int)(idx % (size_t)hz);
        size_t q = idx / (size_t)hz;
        int iy = (int)(q % (size_t)ny);
        int ix = (int)(q / (size_t)ny);

        if (remove_mean && ix == 0 && iy == 0 && iz == 0) {
            d_out[idx] = cufftComplex{0.0f, 0.0f};
            continue;
        }
        int sx = (ix <= nx / 2) ? ix : (ix - nx);
        int sy = (iy <= ny / 2) ? iy : (iy - ny);
        double kx = (double)sx * dkx;
        double ky = (double)sy * dky;
        double kz = (double)iz * dkz;
        double kmag = sqrt(kx * kx + ky * ky + kz * kz);
        if (kmag >= k_lo && kmag < k_hi) {
            d_out[idx] = unit_mode ? cufftComplex{1.0f, 0.0f} : d_src[idx];
        } else {
            d_out[idx] = cufftComplex{0.0f, 0.0f};
        }
    }
}

__global__ void bispectrum_reduce_kernel(const float *__restrict__ D0, const float *__restrict__ I0,
                                         const float *__restrict__ D1, const float *__restrict__ I1,
                                         const float *__restrict__ D2, const float *__restrict__ I2,
                                         size_t n_tot, double inv_n, double *d_acc) {
    __shared__ double s_buf[8 * 256];
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    double s_d = 0.0, s_i = 0.0;
    double d0_sq = 0.0, i0_sq = 0.0;
    double d1_sq = 0.0, i1_sq = 0.0;
    double d2_sq = 0.0, i2_sq = 0.0;

    for (size_t idx = tid; idx < n_tot; idx += stride) {
        double d0 = (double)D0[idx] * inv_n;
        double w0 = (double)I0[idx] * inv_n;
        double d1 = (double)D1[idx] * inv_n;
        double w1 = (double)I1[idx] * inv_n;
        double d2 = (double)D2[idx] * inv_n;
        double w2 = (double)I2[idx] * inv_n;

        s_d += d0 * d1 * d2;
        s_i += w0 * w1 * w2;
        d0_sq += d0 * d0;
        i0_sq += w0 * w0;
        d1_sq += d1 * d1;
        i1_sq += w1 * w1;
        d2_sq += d2 * d2;
        i2_sq += w2 * w2;
    }

    int t = threadIdx.x;
    s_buf[0 * 256 + t] = s_d;
    s_buf[1 * 256 + t] = s_i;
    s_buf[2 * 256 + t] = d0_sq;
    s_buf[3 * 256 + t] = i0_sq;
    s_buf[4 * 256 + t] = d1_sq;
    s_buf[5 * 256 + t] = i1_sq;
    s_buf[6 * 256 + t] = d2_sq;
    s_buf[7 * 256 + t] = i2_sq;
    __syncthreads();

    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) {
            for (int c = 0; c < 8; ++c) {
                s_buf[c * 256 + t] += s_buf[c * 256 + t + off];
            }
        }
        __syncthreads();
    }
    if (t == 0) {
        for (int c = 0; c < 8; ++c) {
            p21_atomic_add_double(&d_acc[c], s_buf[c * 256]);
        }
    }
}


__global__ void mfp_count_ionized_kernel(const float *__restrict__ d_in, size_t n_tot,
                                         float threshold, int below_threshold,
                                         unsigned long long *d_count) {
    __shared__ unsigned long long s_cnt[256];
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned long long c = 0;
    for (size_t idx = tid; idx < n_tot; idx += stride) {
        float v = d_in[idx];
        bool ion = below_threshold ? (v < threshold) : (v >= threshold);
        if (ion) {
            ++c;
        }
    }
    s_cnt[threadIdx.x] = c;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            s_cnt[threadIdx.x] += s_cnt[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && s_cnt[0] > 0) {
        atomicAdd(d_count, s_cnt[0]);
    }
}

__global__ void mfp_collect_ionized_kernel(const float *__restrict__ d_in, size_t n_tot,
                                           float threshold, int below_threshold,
                                           unsigned long long *d_ion_idx,
                                           unsigned long long *d_write_pos) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = tid; idx < n_tot; idx += stride) {
        float v = d_in[idx];
        bool ion = below_threshold ? (v < threshold) : (v >= threshold);
        unsigned mask = __ballot_sync(0xFFFFFFFFU, ion);
        if (ion) {
            int lane = threadIdx.x & 31;
            unsigned prefix = mask & ((1U << lane) - 1U);
            int local_off = __popc(prefix);
            int warp_tot = __popc(mask);
            unsigned long long base = 0;
            if (local_off == 0) {
                base = atomicAdd(d_write_pos, (unsigned long long)warp_tot);
            }
            base = __shfl_sync(mask, base, __ffs(mask) - 1);
            d_ion_idx[base + (unsigned long long)local_off] = (unsigned long long)idx;
        }
    }
}

__global__ void mfp_trace_rays_kernel(const float *__restrict__ d_in,
                                      const unsigned long long *__restrict__ d_ion_idx,
                                      unsigned long long n_ion, int nx, int ny, int nz, double dx,
                                      double dy, double dz, float threshold, int below_threshold,
                                      unsigned long long n_rays, double ds, int max_steps,
                                      double max_dist, int n_bins, double r_min, double r_max,
                                      int log_bins, unsigned long long seed,
                                      unsigned long long *d_hist, double *d_sums) {
    __shared__ double s_r[256];
    __shared__ double s_r2[256];

    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    double inv_dx = 1.0 / dx;
    double inv_dy = 1.0 / dy;
    double inv_dz = 1.0 / dz;
    double log_rmin = log_bins ? log(r_min) : 0.0;
    double inv_span =
        log_bins ? ((double)n_bins / (log(r_max) - log_rmin)) : ((double)n_bins / (r_max - r_min));

    double acc_r = 0.0;
    double acc_r2 = 0.0;

    for (unsigned long long ray = tid; ray < n_rays; ray += stride) {
        P21PhiloxRng rng;
        rng.init((uint64_t)seed, (uint64_t)ray, 0U);
        unsigned long long pick =
            (((unsigned long long)rng.next_u32()) << 32) | (unsigned long long)rng.next_u32();
        unsigned long long vox = d_ion_idx[pick % n_ion];
        int iz = (int)(vox % (unsigned long long)nz);
        unsigned long long q = vox / (unsigned long long)nz;
        int iy = (int)(q % (unsigned long long)ny);
        int ix = (int)(q / (unsigned long long)ny);

        double cx = ((double)ix + 0.5) * dx;
        double cy = ((double)iy + 0.5) * dy;
        double cz = ((double)iz + 0.5) * dz;

        double cos_t = 2.0 * rng.uniform() - 1.0;
        double sin_t = sqrt(fmax(0.0, 1.0 - cos_t * cos_t));
        double phi = 2.0 * M_PI * rng.uniform();
        double vx = sin_t * cos(phi);
        double vy = sin_t * sin(phi);
        double vz = cos_t;

        double r_hit = max_dist;
        for (int s = 1; s <= max_steps; ++s) {
            cx += vx * ds;
            cy += vy * ds;
            cz += vz * ds;
            int gx = (int)floor(cx * inv_dx) % nx;
            int gy = (int)floor(cy * inv_dy) % ny;
            int gz = (int)floor(cz * inv_dz) % nz;
            if (gx < 0) gx += nx;
            if (gy < 0) gy += ny;
            if (gz < 0) gz += nz;
            size_t gidx = ((size_t)gx * (size_t)ny + (size_t)gy) * (size_t)nz + (size_t)gz;
            float val = d_in[gidx];
            bool ion = below_threshold ? (val < threshold) : (val >= threshold);
            if (!ion) {
                r_hit = ((double)s - 0.5) * ds;
                break;
            }
        }
        r_hit = fmin(r_hit, max_dist);
        acc_r += r_hit;
        acc_r2 += r_hit * r_hit;

        if (r_hit >= r_min && r_hit <= r_max) {
            double coord = log_bins ? (log(fmax(r_hit, 1e-300)) - log_rmin) : (r_hit - r_min);
            int b = (int)floor(coord * inv_span);
            if (b >= n_bins) b = n_bins - 1;
            if (b < 0) b = 0;
            atomicAdd(&d_hist[b], 1ULL);
        }
    }

    int t = threadIdx.x;
    s_r[t] = acc_r;
    s_r2[t] = acc_r2;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) {
            s_r[t] += s_r[t + off];
            s_r2[t] += s_r2[t + off];
        }
        __syncthreads();
    }
    if (t == 0) {
        p21_atomic_add_double(&d_sums[0], s_r[0]);
        p21_atomic_add_double(&d_sums[1], s_r2[0]);
    }
}


__global__ void minkowski_chunk_kernel(const float *__restrict__ d_in, int nx, int ny, int nz,
                                       const double *__restrict__ d_thr, int n_thr,
                                       unsigned long long *d_counts) {
    __shared__ unsigned long long s_warp[8 * (4 * kMkChunk)];
    unsigned long long loc[4 * kMkChunk] = {};

    size_t n_tot = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t ny_nz = (size_t)ny * (size_t)nz;

    for (size_t idx = tid; idx < n_tot; idx += stride) {
        int iz = (int)(idx % (size_t)nz);
        size_t q = idx / (size_t)nz;
        int iy = (int)(q % (size_t)ny);
        int ix = (int)(q / (size_t)ny);

        int ix1 = (ix + 1 < nx) ? (ix + 1) : 0;
        int iy1 = (iy + 1 < ny) ? (iy + 1) : 0;
        int iz1 = (iz + 1 < nz) ? (iz + 1) : 0;

        float v000 = d_in[idx];
        float v100 = d_in[(size_t)ix1 * ny_nz + (size_t)iy * (size_t)nz + (size_t)iz];
        float v010 = d_in[(size_t)ix * ny_nz + (size_t)iy1 * (size_t)nz + (size_t)iz];
        float v001 = d_in[(size_t)ix * ny_nz + (size_t)iy * (size_t)nz + (size_t)iz1];
        float v110 = d_in[(size_t)ix1 * ny_nz + (size_t)iy1 * (size_t)nz + (size_t)iz];
        float v101 = d_in[(size_t)ix1 * ny_nz + (size_t)iy * (size_t)nz + (size_t)iz1];
        float v011 = d_in[(size_t)ix * ny_nz + (size_t)iy1 * (size_t)nz + (size_t)iz1];
        float v111 = d_in[(size_t)ix1 * ny_nz + (size_t)iy1 * (size_t)nz + (size_t)iz1];

        for (int t = 0; t < n_thr; ++t) {
            double thr = d_thr[t];
            unsigned b000 = ((double)v000 >= thr) ? 1U : 0U;
            unsigned b100 = ((double)v100 >= thr) ? 1U : 0U;
            unsigned b010 = ((double)v010 >= thr) ? 1U : 0U;
            unsigned b001 = ((double)v001 >= thr) ? 1U : 0U;
            unsigned b110 = ((double)v110 >= thr) ? 1U : 0U;
            unsigned b101 = ((double)v101 >= thr) ? 1U : 0U;
            unsigned b011 = ((double)v011 >= thr) ? 1U : 0U;
            unsigned b111 = ((double)v111 >= thr) ? 1U : 0U;

            unsigned n3 = b000;
            unsigned n2 = (b000 | b100) + (b000 | b010) + (b000 | b001);
            unsigned n1 = (b000 | b010 | b001 | b011) + (b000 | b100 | b001 | b101) +
                          (b000 | b100 | b010 | b110);
            unsigned n0 = b000 | b100 | b010 | b001 | b110 | b101 | b011 | b111;

            loc[4 * t + 0] += n0;
            loc[4 * t + 1] += n1;
            loc[4 * t + 2] += n2;
            loc[4 * t + 3] += n3;
        }
    }

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int n_counters = 4 * n_thr;
    for (int c = 0; c < n_counters; ++c) {
        unsigned long long val = loc[c];
        for (int off = 16; off > 0; off >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFFU, val, off);
        }
        if (lane == 0) {
            s_warp[warp * (4 * kMkChunk) + c] = val;
        }
    }
    __syncthreads();

    if (threadIdx.x < n_counters) {
        int c = threadIdx.x;
        unsigned long long blk_sum = 0;
        for (int w = 0; w < 8; ++w) {
            blk_sum += s_warp[w * (4 * kMkChunk) + c];
        }
        if (blk_sum > 0) {
            atomicAdd(&d_counts[c], blk_sum);
        }
    }
}


}  // namespace

extern "C" {

int p21_cuda_bispectrum_3d(const float *h_field, int nx, int ny, int nz, double lx, double ly,
                           double lz, int remove_mean, int n_triangles, const double *h_k1_lo,
                           const double *h_k1_hi, const double *h_k2_lo, const double *h_k2_hi,
                           const double *h_k3_lo, const double *h_k3_hi, double *h_b_out,
                           double *h_q_out, double *h_pk1_out, double *h_pk2_out, double *h_pk3_out,
                           double *h_ntri_out) {
    if (!h_field || nx <= 0 || ny <= 0 || nz <= 0 || lx <= 0.0 || ly <= 0.0 || lz <= 0.0 ||
        n_triangles < 0) {
        return -1;
    }
    if (n_triangles == 0) {
        return 0;
    }
    if (!h_k1_lo || !h_k1_hi || !h_k2_lo || !h_k2_hi || !h_k3_lo || !h_k3_hi || !h_b_out ||
        !h_q_out || !h_pk1_out || !h_pk2_out || !h_pk3_out || !h_ntri_out) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    size_t n_tot = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t box_bytes = n_tot * sizeof(float);
    size_t hz = (size_t)(nz / 2 + 1);
    size_t n_freq = (size_t)nx * (size_t)ny * hz;
    size_t cplx_bytes = n_freq * sizeof(cufftComplex);

    DeviceFloatView v_in = acquire_float_view(h_field, box_bytes, stream);
    cufftComplex *d_freq_src = (cufftComplex *)p21_cuda_pool_alloc(cplx_bytes, stream);
    cufftComplex *d_work_c = (cufftComplex *)p21_cuda_pool_alloc(cplx_bytes, stream);
    float *d_D[3] = {nullptr, nullptr, nullptr};
    float *d_I[3] = {nullptr, nullptr, nullptr};
    for (int s = 0; s < 3; ++s) {
        d_D[s] = (float *)p21_cuda_pool_alloc(box_bytes, stream);
        d_I[s] = (float *)p21_cuda_pool_alloc(box_bytes, stream);
    }
    double *d_acc = (double *)p21_cuda_pool_alloc(8ULL * sizeof(double), stream);

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_acc, stream);
        for (int s = 2; s >= 0; --s) {
            p21_cuda_pool_free(d_I[s], stream);
            p21_cuda_pool_free(d_D[s], stream);
        }
        p21_cuda_pool_free(d_work_c, stream);
        p21_cuda_pool_free(d_freq_src, stream);
        release_float_view(v_in, stream);
    };

    bool alloc_ok = v_in.d_ptr && d_freq_src && d_work_c && d_acc;
    for (int s = 0; s < 3; ++s) {
        alloc_ok = alloc_ok && d_D[s] && d_I[s];
    }
    if (!alloc_ok || !p21_cuda_fft_r2c(v_in.d_ptr, d_freq_src, nx, ny, nz, false, stream)) {
        cleanup();
        return -1;
    }

    constexpr int kTpb = 256;
    int freq_blocks = p21_grid_blocks(n_freq, kTpb);
    int real_blocks = p21_grid_blocks(n_tot, kTpb);
    double inv_n = 1.0 / (double)n_tot;
    double n_tot_d = (double)n_tot;
    double vol = lx * ly * lz;

    struct ShellSlot {
        double lo = 0.0;
        double hi = -1.0;
        bool valid = false;
    } slots[3];

    auto eval_shell = [&](int slot_idx, double k_lo, double k_hi) -> bool {
        bispectrum_mask_shell_kernel<<<freq_blocks, kTpb, 0, stream>>>(
            d_freq_src, d_work_c, nx, ny, nz, lx, ly, lz, k_lo, k_hi, remove_mean, 0);
        if (!P21_CUDA_OK(cudaGetLastError()) ||
            !p21_cuda_fft_c2r(d_work_c, d_D[slot_idx], nx, ny, nz, false, stream)) {
            return false;
        }
        bispectrum_mask_shell_kernel<<<freq_blocks, kTpb, 0, stream>>>(
            d_freq_src, d_work_c, nx, ny, nz, lx, ly, lz, k_lo, k_hi, remove_mean, 1);
        if (!P21_CUDA_OK(cudaGetLastError()) ||
            !p21_cuda_fft_c2r(d_work_c, d_I[slot_idx], nx, ny, nz, false, stream)) {
            return false;
        }
        slots[slot_idx].lo = k_lo;
        slots[slot_idx].hi = k_hi;
        slots[slot_idx].valid = true;
        return true;
    };

    bool ok = true;
    for (int t = 0; t < n_triangles && ok; ++t) {
        double klo[3] = {h_k1_lo[t], h_k2_lo[t], h_k3_lo[t]};
        double khi[3] = {h_k1_hi[t], h_k2_hi[t], h_k3_hi[t]};
        int slot_idx[3] = {-1, -1, -1};
        bool pinned[3] = {false, false, false};

        for (int m = 0; m < 3; ++m) {
            for (int s = 0; s < 3; ++s) {
                if (slots[s].valid && slots[s].lo == klo[m] && slots[s].hi == khi[m]) {
                    slot_idx[m] = s;
                    pinned[s] = true;
                    break;
                }
            }
        }
        for (int m = 0; m < 3 && ok; ++m) {
            if (slot_idx[m] >= 0) {
                continue;
            }
            for (int pm = 0; pm < m; ++pm) {
                if (klo[pm] == klo[m] && khi[pm] == khi[m]) {
                    slot_idx[m] = slot_idx[pm];
                    break;
                }
            }
            if (slot_idx[m] >= 0) {
                continue;
            }
            int pick = -1;
            for (int s = 0; s < 3; ++s) {
                if (!pinned[s]) {
                    pick = s;
                    break;
                }
            }
            if (pick < 0) {
                ok = false;
                break;
            }
            pinned[pick] = true;
            slot_idx[m] = pick;
            ok = eval_shell(pick, klo[m], khi[m]);
        }

        if (!ok) {
            break;
        }

        double h_acc[8] = {};
        ok = P21_CUDA_OK(cudaMemsetAsync(d_acc, 0, 8ULL * sizeof(double), stream));
        if (ok) {
            bispectrum_reduce_kernel<<<real_blocks, kTpb, 0, stream>>>(
                d_D[slot_idx[0]], d_I[slot_idx[0]], d_D[slot_idx[1]], d_I[slot_idx[1]],
                d_D[slot_idx[2]], d_I[slot_idx[2]], n_tot, inv_n, d_acc);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 P21_CUDA_OK(cudaMemcpyAsync(h_acc, d_acc, 8ULL * sizeof(double),
                                             cudaMemcpyDeviceToHost, stream)) &&
                 P21_CUDA_OK(cudaStreamSynchronize(stream));
        }
        if (!ok) {
            break;
        }

        double s_d = h_acc[0];
        double s_i = h_acc[1];
        double ntri = n_tot_d * n_tot_d * s_i;
        double b_val = (s_i > 0.0)
                           ? ((vol * vol) / (n_tot_d * n_tot_d * n_tot_d)) * (s_d / s_i)
                           : NAN;
        double pk1 = (h_acc[3] > 0.0) ? (vol / (n_tot_d * n_tot_d)) * (h_acc[2] / h_acc[3]) : NAN;
        double pk2 = (h_acc[5] > 0.0) ? (vol / (n_tot_d * n_tot_d)) * (h_acc[4] / h_acc[5]) : NAN;
        double pk3 = (h_acc[7] > 0.0) ? (vol / (n_tot_d * n_tot_d)) * (h_acc[6] / h_acc[7]) : NAN;
        double q_denom = pk1 * pk2 + pk2 * pk3 + pk3 * pk1;
        double q_val = (isfinite(b_val) && isfinite(q_denom) && fabs(q_denom) > 0.0)
                           ? (b_val / q_denom)
                           : NAN;

        h_ntri_out[t] = ntri;
        h_b_out[t] = b_val;
        h_pk1_out[t] = pk1;
        h_pk2_out[t] = pk2;
        h_pk3_out[t] = pk3;
        h_q_out[t] = q_val;
    }

    cleanup();
    return ok ? 0 : -1;
}


int p21_cuda_mfp_bsd(const float *h_field, int nx, int ny, int nz, double lx, double ly, double lz,
                     float threshold, int below_threshold, unsigned long long n_rays,
                     double step_cells, double max_dist, int n_bins, double r_min, double r_max,
                     int log_bins, unsigned long long seed, unsigned long long *h_hist_out,
                     double *mean_r_out, double *std_r_out, unsigned long long *n_ion_out) {
    if (!h_field || !h_hist_out || !mean_r_out || !std_r_out || !n_ion_out || nx <= 0 || ny <= 0 ||
        nz <= 0 || lx <= 0.0 || ly <= 0.0 || lz <= 0.0 || n_rays == 0 || step_cells <= 0.0 ||
        max_dist <= 0.0 || n_bins <= 0 || r_max <= r_min || (log_bins && r_min <= 0.0)) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    size_t n_tot = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t box_bytes = n_tot * sizeof(float);

    DeviceFloatView v_in = acquire_float_view(h_field, box_bytes, stream);
    unsigned long long *d_count =
        (unsigned long long *)p21_cuda_pool_alloc(sizeof(unsigned long long), stream);
    unsigned long long *d_ion_idx = nullptr;
    unsigned long long *d_hist =
        (unsigned long long *)p21_cuda_pool_alloc((size_t)n_bins * sizeof(unsigned long long), stream);
    double *d_sums = (double *)p21_cuda_pool_alloc(2ULL * sizeof(double), stream);

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_sums, stream);
        p21_cuda_pool_free(d_hist, stream);
        p21_cuda_pool_free(d_ion_idx, stream);
        p21_cuda_pool_free(d_count, stream);
        release_float_view(v_in, stream);
    };

    if (!v_in.d_ptr || !d_count || !d_hist || !d_sums ||
        !P21_CUDA_OK(cudaMemsetAsync(d_count, 0, sizeof(unsigned long long), stream))) {
        cleanup();
        return -1;
    }

    constexpr int kTpb = 256;
    int real_blocks = p21_grid_blocks(n_tot, kTpb);
    mfp_count_ionized_kernel<<<real_blocks, kTpb, 0, stream>>>(v_in.d_ptr, n_tot, threshold,
                                                               below_threshold, d_count);
    unsigned long long n_ion = 0;
    if (!P21_CUDA_OK(cudaGetLastError()) ||
        !P21_CUDA_OK(cudaMemcpyAsync(&n_ion, d_count, sizeof(unsigned long long),
                                     cudaMemcpyDeviceToHost, stream)) ||
        !P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        cleanup();
        return -1;
    }

    *n_ion_out = n_ion;
    if (n_ion == 0) {
        memset(h_hist_out, 0, (size_t)n_bins * sizeof(unsigned long long));
        *mean_r_out = 0.0;
        *std_r_out = 0.0;
        cleanup();
        return 0;
    }

    d_ion_idx = (unsigned long long *)p21_cuda_pool_alloc(
        (size_t)n_ion * sizeof(unsigned long long), stream);
    if (!d_ion_idx ||
        !P21_CUDA_OK(cudaMemsetAsync(d_count, 0, sizeof(unsigned long long), stream)) ||
        !P21_CUDA_OK(
            cudaMemsetAsync(d_hist, 0, (size_t)n_bins * sizeof(unsigned long long), stream)) ||
        !P21_CUDA_OK(cudaMemsetAsync(d_sums, 0, 2ULL * sizeof(double), stream))) {
        cleanup();
        return -1;
    }

    mfp_collect_ionized_kernel<<<real_blocks, kTpb, 0, stream>>>(
        v_in.d_ptr, n_tot, threshold, below_threshold, d_ion_idx, d_count);
    if (!P21_CUDA_OK(cudaGetLastError())) {
        cleanup();
        return -1;
    }

    double dx = lx / (double)nx;
    double dy = ly / (double)ny;
    double dz = lz / (double)nz;
    double ds = step_cells * fmin(dx, fmin(dy, dz));
    int max_steps = (int)ceil(max_dist / ds);
    if (max_steps < 1) max_steps = 1;

    int ray_blocks = p21_grid_blocks((size_t)n_rays, kTpb);
    mfp_trace_rays_kernel<<<ray_blocks, kTpb, 0, stream>>>(
        v_in.d_ptr, d_ion_idx, n_ion, nx, ny, nz, dx, dy, dz, threshold, below_threshold, n_rays,
        ds, max_steps, max_dist, n_bins, r_min, r_max, log_bins, seed, d_hist, d_sums);

    double h_sums[2] = {0.0, 0.0};
    bool ok = P21_CUDA_OK(cudaGetLastError()) &&
              P21_CUDA_OK(cudaMemcpyAsync(h_hist_out, d_hist,
                                          (size_t)n_bins * sizeof(unsigned long long),
                                          cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaMemcpyAsync(h_sums, d_sums, 2ULL * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaStreamSynchronize(stream));
    cleanup();
    if (!ok) {
        return -1;
    }

    double inv_rays = 1.0 / (double)n_rays;
    double mean_r = h_sums[0] * inv_rays;
    double var_r = fmax(0.0, h_sums[1] * inv_rays - mean_r * mean_r);
    *mean_r_out = mean_r;
    *std_r_out = sqrt(var_r);
    return 0;
}

constexpr int kMkChunk = 8;


int p21_cuda_minkowski_3d(const float *h_field, int nx, int ny, int nz, double lx, double ly,
                          double lz, int n_thresholds, const double *h_thresholds,
                          double *h_v0_out, double *h_v1_out, double *h_v2_out, double *h_v3_out,
                          long long *h_euler_out) {
    if (!h_field || nx <= 0 || ny <= 0 || nz <= 0 || lx <= 0.0 || ly <= 0.0 || lz <= 0.0 ||
        n_thresholds < 0) {
        return -1;
    }
    if (n_thresholds == 0) {
        return 0;
    }
    if (!h_thresholds || !h_v0_out || !h_v1_out || !h_v2_out || !h_v3_out || !h_euler_out) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    size_t n_tot = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t box_bytes = n_tot * sizeof(float);
    DeviceFloatView v_in = acquire_float_view(h_field, box_bytes, stream);
    double *d_thr = (double *)p21_cuda_pool_alloc((size_t)kMkChunk * sizeof(double), stream);
    unsigned long long *d_counts = (unsigned long long *)p21_cuda_pool_alloc(
        (size_t)(4 * kMkChunk) * sizeof(unsigned long long), stream);

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_counts, stream);
        p21_cuda_pool_free(d_thr, stream);
        release_float_view(v_in, stream);
    };

    if (!v_in.d_ptr || !d_thr || !d_counts) {
        cleanup();
        return -1;
    }

    constexpr int kTpb = 256;
    int real_blocks = p21_grid_blocks(n_tot, kTpb);
    double dx = lx / (double)nx;
    double dy = ly / (double)ny;
    double dz = lz / (double)nz;
    double a = cbrt(dx * dy * dz);
    double vol = lx * ly * lz;
    double n_tot_d = (double)n_tot;

    bool ok = true;
    for (int t0 = 0; t0 < n_thresholds && ok; t0 += kMkChunk) {
        int cur_n = std::min(kMkChunk, n_thresholds - t0);
        unsigned long long h_counts[4 * kMkChunk] = {};
        ok = P21_CUDA_OK(cudaMemcpyAsync(d_thr, h_thresholds + t0, (size_t)cur_n * sizeof(double),
                                         cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemsetAsync(d_counts, 0,
                                         (size_t)(4 * cur_n) * sizeof(unsigned long long), stream));
        if (ok) {
            minkowski_chunk_kernel<<<real_blocks, kTpb, 0, stream>>>(v_in.d_ptr, nx, ny, nz, d_thr,
                                                                     cur_n, d_counts);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 P21_CUDA_OK(cudaMemcpyAsync(h_counts, d_counts,
                                             (size_t)(4 * cur_n) * sizeof(unsigned long long),
                                             cudaMemcpyDeviceToHost, stream)) &&
                 P21_CUDA_OK(cudaStreamSynchronize(stream));
        }
        if (!ok) {
            break;
        }

        for (int i = 0; i < cur_n; ++i) {
            double n0 = (double)h_counts[4 * i + 0];
            double n1 = (double)h_counts[4 * i + 1];
            double n2 = (double)h_counts[4 * i + 2];
            double n3 = (double)h_counts[4 * i + 3];
            long long euler = (long long)h_counts[4 * i + 0] - (long long)h_counts[4 * i + 1] +
                              (long long)h_counts[4 * i + 2] - (long long)h_counts[4 * i + 3];
            int out_i = t0 + i;
            h_v0_out[out_i] = n3 / n_tot_d;
            h_v1_out[out_i] = (2.0 / (9.0 * a * n_tot_d)) * (n2 - 3.0 * n3);
            h_v2_out[out_i] = (2.0 / (9.0 * a * a * n_tot_d)) * (n1 - 2.0 * n2 + 3.0 * n3);
            h_euler_out[out_i] = euler;
            h_v3_out[out_i] = (double)euler / vol;
        }
    }

    cleanup();
    return ok ? 0 : -1;
}

}  // extern "C"
