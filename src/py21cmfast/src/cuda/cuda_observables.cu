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

__global__ void wedge_filter_kernel(cufftComplex *d_freq, int nx, int ny, int nz, double lx,
                                    double ly, double lz, int los_axis, double slope,
                                    double buffer_mpc, float inv_n_tot) {
    size_t hz = (size_t)(nz / 2 + 1);
    size_t n_freq = (size_t)nx * (size_t)ny * hz;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t ny_hz = (size_t)ny * hz;

    for (size_t idx = tid; idx < n_freq; idx += stride) {
        int ix = (int)(idx / ny_hz);
        size_t rem = idx - (size_t)ix * ny_hz;
        int iy = (int)(rem / hz);
        int iz = (int)(rem - (size_t)iy * hz);

        double kx = p21_index_to_k(ix, lx, nx);
        double ky = p21_index_to_k(iy, ly, ny);
        double kz = (double)iz * (2.0 * M_PI / lz);

        double k_para = 0.0;
        double k_perp = 0.0;
        if (los_axis == 0) {
            k_para = fabs(kx);
            k_perp = sqrt(ky * ky + kz * kz);
        } else if (los_axis == 1) {
            k_para = fabs(ky);
            k_perp = sqrt(kx * kx + kz * kz);
        } else {
            k_para = fabs(kz);
            k_perp = sqrt(kx * kx + ky * ky);
        }

        if (k_para < slope * k_perp + buffer_mpc) {
            d_freq[idx].x = 0.0f;
            d_freq[idx].y = 0.0f;
        } else {
            d_freq[idx].x *= inv_n_tot;
            d_freq[idx].y *= inv_n_tot;
        }
    }
}

__global__ void thermal_noise_philox_kernel(const float *d_in, float *d_out, size_t n_elems,
                                            float sigma_mk, unsigned long long seed) {
    size_t n_pairs = (n_elems + 1ULL) / 2ULL;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t pair = tid; pair < n_pairs; pair += stride) {
        P21PhiloxRng rng;
        rng.init((uint64_t)seed, (uint64_t)pair, 0U);
        double z0 = 0.0;
        double z1 = 0.0;
        rng.ugaussian_pair(&z0, &z1);
        size_t i0 = 2ULL * pair;
        if (i0 < n_elems) {
            d_out[i0] = d_in[i0] + sigma_mk * (float)z0;
        }
        size_t i1 = i0 + 1ULL;
        if (i1 < n_elems) {
            d_out[i1] = d_in[i1] + sigma_mk * (float)z1;
        }
    }
}

__global__ void moments_pass1_sum_kernel(const float *d_in, size_t n_elems, double *d_sum) {
    __shared__ double s_sum[256];
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    double acc = 0.0;
    for (size_t i = tid; i < n_elems; i += stride) {
        acc += (double)d_in[i];
    }
    s_sum[threadIdx.x] = acc;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        p21_atomic_add_double(d_sum, s_sum[0]);
    }
}

__global__ void moments_pass2_kernel(const float *d_in, size_t n_elems, double mean, int n_bins,
                                     double vmin, double vmax, double *d_m2, double *d_m3,
                                     double *d_m4, unsigned long long *d_hist) {
    __shared__ double s_m2[256];
    __shared__ double s_m3[256];
    __shared__ double s_m4[256];

    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    double a2 = 0.0;
    double a3 = 0.0;
    double a4 = 0.0;
    double inv_span = (vmax > vmin) ? ((double)n_bins / (vmax - vmin)) : 0.0;

    for (size_t i = tid; i < n_elems; i += stride) {
        double v = (double)d_in[i];
        double d = v - mean;
        double d2 = d * d;
        a2 += d2;
        a3 += d2 * d;
        a4 += d2 * d2;

        if (d_hist && n_bins > 0 && v >= vmin && v <= vmax) {
            int b = (vmax > vmin) ? (int)floor((v - vmin) * inv_span) : 0;
            if (b >= n_bins) {
                b = n_bins - 1;
            }
            if (b < 0) {
                b = 0;
            }
            atomicAdd(&d_hist[b], 1ULL);
        }
    }

    s_m2[threadIdx.x] = a2;
    s_m3[threadIdx.x] = a3;
    s_m4[threadIdx.x] = a4;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s_m2[threadIdx.x] += s_m2[threadIdx.x + offset];
            s_m3[threadIdx.x] += s_m3[threadIdx.x + offset];
            s_m4[threadIdx.x] += s_m4[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        p21_atomic_add_double(d_m2, s_m2[0]);
        p21_atomic_add_double(d_m3, s_m3[0]);
        p21_atomic_add_double(d_m4, s_m4[0]);
    }
}


__global__ void interferometer_mock_kernel(cufftComplex *d_freq, int nx, int ny, int nz, double lx,
                                           double ly, double lz, int los_axis,
                                           double sigma_perp_mpc, double channel_dr_mpc,
                                           int channel_mode, int apply_uv_mask, double k_perp_min,
                                           double k_perp_max, int apply_wedge, double wedge_slope,
                                           double wedge_buffer_mpc, int add_noise,
                                           int n_noise_table, double noise_kperp_max,
                                           const float *d_sigma_table, unsigned long long seed,
                                           float inv_n_tot) {
    size_t hz = (size_t)(nz / 2 + 1);
    size_t n_freq = (size_t)nx * (size_t)ny * hz;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t ny_hz = (size_t)ny * hz;
    bool even_z = (nz % 2) == 0;

    for (size_t idx = tid; idx < n_freq; idx += stride) {
        int ix = (int)(idx / ny_hz);
        size_t rem = idx - (size_t)ix * ny_hz;
        int iy = (int)(rem / hz);
        int iz = (int)(rem - (size_t)iy * hz);

        double kx = p21_index_to_k(ix, lx, nx);
        double ky = p21_index_to_k(iy, ly, ny);
        double kz = (double)iz * (2.0 * M_PI / lz);

        double k_para = 0.0;
        double k_perp = 0.0;
        if (los_axis == 0) {
            k_para = fabs(kx);
            k_perp = sqrt(ky * ky + kz * kz);
        } else if (los_axis == 1) {
            k_para = fabs(ky);
            k_perp = sqrt(kx * kx + kz * kz);
        } else {
            k_para = fabs(kz);
            k_perp = sqrt(kx * kx + ky * ky);
        }

        if (apply_uv_mask && (k_perp < k_perp_min || k_perp > k_perp_max)) {
            d_freq[idx].x = 0.0f;
            d_freq[idx].y = 0.0f;
            continue;
        }
        if (apply_wedge && k_para < wedge_slope * k_perp + wedge_buffer_mpc) {
            d_freq[idx].x = 0.0f;
            d_freq[idx].y = 0.0f;
            continue;
        }

        double resp = 1.0;
        if (sigma_perp_mpc > 0.0) {
            double arg_p = k_perp * sigma_perp_mpc;
            resp *= exp(-0.5 * arg_p * arg_p);
        }
        if (channel_dr_mpc > 0.0 && channel_mode != 0 && k_para > 0.0) {
            if (channel_mode == 1) {
                double x = 0.5 * k_para * channel_dr_mpc;
                resp *= (x > 1e-8) ? (sin(x) / x) : 1.0;
            } else {
                double arg_z = k_para * channel_dr_mpc;
                resp *= exp(-0.5 * arg_z * arg_z);
            }
        }

        double re = (double)d_freq[idx].x * resp;
        double im = (double)d_freq[idx].y * resp;

        if (add_noise && d_sigma_table && n_noise_table >= 2 && noise_kperp_max > 0.0) {
            double u_idx = (k_perp / noise_kperp_max) * (double)(n_noise_table - 1);
            if (u_idx < 0.0) {
                u_idx = 0.0;
            } else if (u_idx > (double)(n_noise_table - 1)) {
                u_idx = (double)(n_noise_table - 1);
            }
            int i0 = (int)floor(u_idx);
            int i1 = (i0 + 1 < n_noise_table) ? (i0 + 1) : i0;
            double frac = u_idx - (double)i0;
            double sigma_re =
                (1.0 - frac) * (double)d_sigma_table[i0] + frac * (double)d_sigma_table[i1];

            if (sigma_re > 0.0) {
                if (iz == 0 || (even_z && iz == nz / 2)) {
                    int ix_c = (ix == 0) ? 0 : (nx - ix);
                    int iy_c = (iy == 0) ? 0 : (ny - iy);
                    size_t p = (size_t)ix * (size_t)ny + (size_t)iy;
                    size_t pc = (size_t)ix_c * (size_t)ny + (size_t)iy_c;
                    uint32_t sub = (iz == 0) ? 1U : 2U;
                    if (p == pc) {
                        if (!(iz == 0 && ix == 0 && iy == 0)) {
                            P21PhiloxRng rng;
                            rng.init((uint64_t)seed, (uint64_t)p, sub);
                            double z0 = 0.0;
                            double z1 = 0.0;
                            rng.ugaussian_pair(&z0, &z1);
                            re += sqrt(2.0) * sigma_re * z0;
                            im = 0.0;
                        }
                    } else if (p < pc) {
                        P21PhiloxRng rng;
                        rng.init((uint64_t)seed, (uint64_t)p, sub);
                        double z0 = 0.0;
                        double z1 = 0.0;
                        rng.ugaussian_pair(&z0, &z1);
                        re += sigma_re * z0;
                        im += sigma_re * z1;
                    } else {
                        P21PhiloxRng rng;
                        rng.init((uint64_t)seed, (uint64_t)pc, sub);
                        double z0 = 0.0;
                        double z1 = 0.0;
                        rng.ugaussian_pair(&z0, &z1);
                        re += sigma_re * z0;
                        im -= sigma_re * z1;
                    }
                } else {
                    P21PhiloxRng rng;
                    rng.init((uint64_t)seed, (uint64_t)idx, 0U);
                    double z0 = 0.0;
                    double z1 = 0.0;
                    rng.ugaussian_pair(&z0, &z1);
                    re += sigma_re * z0;
                    im += sigma_re * z1;
                }
            }
        }

        d_freq[idx].x = (float)(re * (double)inv_n_tot);
        d_freq[idx].y = (float)(im * (double)inv_n_tot);
    }
}


}  // namespace

extern "C" {

int p21_cuda_wedge_filter_3d(const float *h_in, float *h_out, int nx, int ny, int nz, double lx,
                             double ly, double lz, int los_axis, double slope, double buffer_mpc) {
    if (!h_in || !h_out || nx <= 0 || ny <= 0 || nz <= 0 || lx <= 0.0 || ly <= 0.0 ||
        lz <= 0.0 || los_axis < 0 || los_axis > 2) {
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

    DeviceFloatView v_in = acquire_float_view(h_in, box_bytes, stream);
    cufftComplex *d_freq =
        (cufftComplex *)p21_cuda_pool_alloc(n_freq * sizeof(cufftComplex), stream);
    float *d_out = (float *)p21_cuda_pool_alloc(box_bytes, stream);

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_out, stream);
        p21_cuda_pool_free(d_freq, stream);
        release_float_view(v_in, stream);
    };

    if (!v_in.d_ptr || !d_freq || !d_out) {
        cleanup();
        return -1;
    }
    if (!p21_cuda_fft_r2c(v_in.d_ptr, d_freq, nx, ny, nz, false, stream)) {
        cleanup();
        return -1;
    }

    constexpr int kTpb = 256;
    int blocks = p21_grid_blocks(n_freq, kTpb);
    float inv_n = (float)(1.0 / (double)n_tot);
    wedge_filter_kernel<<<blocks, kTpb, 0, stream>>>(d_freq, nx, ny, nz, lx, ly, lz, los_axis,
                                                     slope, buffer_mpc, inv_n);
    bool ok = P21_CUDA_OK(cudaGetLastError()) &&
              p21_cuda_fft_c2r(d_freq, d_out, nx, ny, nz, false, stream) &&
              P21_CUDA_OK(cudaMemcpyAsync(h_out, d_out, box_bytes, cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaStreamSynchronize(stream));
    if (ok) {
        p21_cuda_stats_add_d2h(box_bytes);
        if (box_bytes <= p21_cuda_registry_max_entry_bytes()) {
            p21_cuda_registry_put_copy(h_out, d_out, box_bytes, stream);
        }
    }
    cleanup();
    return ok ? 0 : -1;
}

int p21_cuda_add_thermal_noise_3d(const float *h_in, float *h_out, size_t n_elems, float sigma_mk,
                                  unsigned long long seed) {
    if (!h_in || !h_out || n_elems == 0 || sigma_mk < 0.0f) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    size_t box_bytes = n_elems * sizeof(float);
    DeviceFloatView v_in = acquire_float_view(h_in, box_bytes, stream);
    float *d_out = (float *)p21_cuda_pool_alloc(box_bytes, stream);

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_out, stream);
        release_float_view(v_in, stream);
    };

    if (!v_in.d_ptr || !d_out) {
        cleanup();
        return -1;
    }

    bool ok = true;
    if (sigma_mk == 0.0f) {
        ok = P21_CUDA_OK(
            cudaMemcpyAsync(d_out, v_in.d_ptr, box_bytes, cudaMemcpyDeviceToDevice, stream));
    } else {
        constexpr int kTpb = 256;
        size_t n_pairs = (n_elems + 1ULL) / 2ULL;
        int blocks = p21_grid_blocks(n_pairs, kTpb);
        thermal_noise_philox_kernel<<<blocks, kTpb, 0, stream>>>(v_in.d_ptr, d_out, n_elems,
                                                                 sigma_mk, seed);
        ok = P21_CUDA_OK(cudaGetLastError());
    }

    ok = ok &&
         P21_CUDA_OK(cudaMemcpyAsync(h_out, d_out, box_bytes, cudaMemcpyDeviceToHost, stream)) &&
         P21_CUDA_OK(cudaStreamSynchronize(stream));
    if (ok) {
        p21_cuda_stats_add_d2h(box_bytes);
        if (box_bytes <= p21_cuda_registry_max_entry_bytes()) {
            p21_cuda_registry_put_copy(h_out, d_out, box_bytes, stream);
        }
    }
    cleanup();
    return ok ? 0 : -1;
}

int p21_cuda_field_moments_and_pdf(const float *h_in, size_t n_elems, int n_bins, double vmin,
                                   double vmax, double *mean_out, double *var_out, double *skew_out,
                                   double *kurt_out, unsigned long long *hist_out) {
    if (!h_in || n_elems == 0 || n_bins <= 0 || !mean_out || !var_out || !skew_out || !kurt_out ||
        !hist_out) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    size_t box_bytes = n_elems * sizeof(float);
    DeviceFloatView v_in = acquire_float_view(h_in, box_bytes, stream);
    double *d_acc = (double *)p21_cuda_pool_alloc(4ULL * sizeof(double), stream);
    unsigned long long *d_hist =
        (unsigned long long *)p21_cuda_pool_alloc((size_t)n_bins * sizeof(unsigned long long), stream);

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_hist, stream);
        p21_cuda_pool_free(d_acc, stream);
        release_float_view(v_in, stream);
    };

    if (!v_in.d_ptr || !d_acc || !d_hist) {
        cleanup();
        return -1;
    }
    if (!P21_CUDA_OK(cudaMemsetAsync(d_acc, 0, 4ULL * sizeof(double), stream)) ||
        !P21_CUDA_OK(
            cudaMemsetAsync(d_hist, 0, (size_t)n_bins * sizeof(unsigned long long), stream))) {
        cleanup();
        return -1;
    }

    constexpr int kTpb = 256;
    int blocks = p21_grid_blocks(n_elems, kTpb);
    moments_pass1_sum_kernel<<<blocks, kTpb, 0, stream>>>(v_in.d_ptr, n_elems, &d_acc[0]);
    double total_sum = 0.0;
    bool ok =
        P21_CUDA_OK(cudaGetLastError()) &&
        P21_CUDA_OK(cudaMemcpyAsync(&total_sum, &d_acc[0], sizeof(double), cudaMemcpyDeviceToHost,
                                    stream)) &&
        P21_CUDA_OK(cudaStreamSynchronize(stream));
    if (!ok) {
        cleanup();
        return -1;
    }

    double mean = total_sum / (double)n_elems;
    moments_pass2_kernel<<<blocks, kTpb, 0, stream>>>(v_in.d_ptr, n_elems, mean, n_bins, vmin, vmax,
                                                      &d_acc[1], &d_acc[2], &d_acc[3], d_hist);
    double h_moments[4] = {0.0, 0.0, 0.0, 0.0};
    ok = P21_CUDA_OK(cudaGetLastError()) &&
         P21_CUDA_OK(cudaMemcpyAsync(h_moments, d_acc, 4ULL * sizeof(double), cudaMemcpyDeviceToHost,
                                     stream)) &&
         P21_CUDA_OK(cudaMemcpyAsync(hist_out, d_hist, (size_t)n_bins * sizeof(unsigned long long),
                                     cudaMemcpyDeviceToHost, stream)) &&
         P21_CUDA_OK(cudaStreamSynchronize(stream));
    cleanup();
    if (!ok) {
        return -1;
    }

    double var = h_moments[1] / (double)n_elems;
    double skew = 0.0;
    double kurt = 0.0;
    if (var > 0.0) {
        double m3 = h_moments[2] / (double)n_elems;
        double m4 = h_moments[3] / (double)n_elems;
        double std_dev = sqrt(var);
        skew = m3 / (std_dev * var);
        kurt = (m4 / (var * var)) - 3.0;
    }
    *mean_out = mean;
    *var_out = var;
    *skew_out = skew;
    *kurt_out = kurt;
    return 0;
}


int p21_cuda_mock_observation_3d(const float *h_in, float *h_out, int nx, int ny, int nz,
                                 double lx, double ly, double lz, int los_axis,
                                 double sigma_perp_mpc, double channel_dr_mpc, int channel_mode,
                                 int apply_uv_mask, double k_perp_min, double k_perp_max,
                                 int apply_wedge, double wedge_slope, double wedge_buffer_mpc,
                                 int add_noise, int n_noise_table, double noise_kperp_max,
                                 const float *h_sigma_re_table, unsigned long long seed) {
    if (!h_in || !h_out || nx <= 0 || ny <= 0 || nz <= 0 || lx <= 0.0 || ly <= 0.0 ||
        lz <= 0.0 || los_axis < 0 || los_axis > 2) {
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
    bool use_noise = (add_noise != 0 && h_sigma_re_table != nullptr && n_noise_table >= 2 &&
                      noise_kperp_max > 0.0);

    DeviceFloatView v_in = acquire_float_view(h_in, box_bytes, stream);
    cufftComplex *d_freq =
        (cufftComplex *)p21_cuda_pool_alloc(n_freq * sizeof(cufftComplex), stream);
    float *d_out = (float *)p21_cuda_pool_alloc(box_bytes, stream);
    float *d_sigma_table =
        use_noise ? (float *)p21_cuda_pool_alloc((size_t)n_noise_table * sizeof(float), stream)
                  : nullptr;

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_sigma_table, stream);
        p21_cuda_pool_free(d_out, stream);
        p21_cuda_pool_free(d_freq, stream);
        release_float_view(v_in, stream);
    };

    if (!v_in.d_ptr || !d_freq || !d_out || (use_noise && !d_sigma_table)) {
        cleanup();
        return -1;
    }
    if (use_noise &&
        !P21_CUDA_OK(cudaMemcpyAsync(d_sigma_table, h_sigma_re_table,
                                     (size_t)n_noise_table * sizeof(float), cudaMemcpyHostToDevice,
                                     stream))) {
        cleanup();
        return -1;
    }
    if (!p21_cuda_fft_r2c(v_in.d_ptr, d_freq, nx, ny, nz, false, stream)) {
        cleanup();
        return -1;
    }

    constexpr int kTpb = 256;
    int blocks = p21_grid_blocks(n_freq, kTpb);
    float inv_n = (float)(1.0 / (double)n_tot);
    interferometer_mock_kernel<<<blocks, kTpb, 0, stream>>>(
        d_freq, nx, ny, nz, lx, ly, lz, los_axis, sigma_perp_mpc, channel_dr_mpc, channel_mode,
        apply_uv_mask, k_perp_min, k_perp_max, apply_wedge, wedge_slope, wedge_buffer_mpc,
        use_noise ? 1 : 0, n_noise_table, noise_kperp_max, d_sigma_table, seed, inv_n);

    bool ok = P21_CUDA_OK(cudaGetLastError()) &&
              p21_cuda_fft_c2r(d_freq, d_out, nx, ny, nz, false, stream) &&
              P21_CUDA_OK(cudaMemcpyAsync(h_out, d_out, box_bytes, cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaStreamSynchronize(stream));
    if (ok) {
        p21_cuda_stats_add_d2h(box_bytes);
        if (box_bytes <= p21_cuda_registry_max_entry_bytes()) {
            p21_cuda_registry_put_copy(h_out, d_out, box_bytes, stream);
        }
    }
    cleanup();
    return ok ? 0 : -1;
}

int p21_cuda_field_mean_f32(const float *h_in, size_t n_elems, double *mean_out) {
    if (!h_in || n_elems == 0 || !mean_out) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    size_t box_bytes = n_elems * sizeof(float);
    DeviceFloatView v_in = acquire_float_view(h_in, box_bytes, stream);
    double *d_sum = (double *)p21_cuda_pool_alloc(sizeof(double), stream);

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_sum, stream);
        release_float_view(v_in, stream);
    };

    if (!v_in.d_ptr || !d_sum || !P21_CUDA_OK(cudaMemsetAsync(d_sum, 0, sizeof(double), stream))) {
        cleanup();
        return -1;
    }

    constexpr int kTpb = 256;
    int blocks = p21_grid_blocks(n_elems, kTpb);
    moments_pass1_sum_kernel<<<blocks, kTpb, 0, stream>>>(v_in.d_ptr, n_elems, d_sum);

    double total_sum = 0.0;
    bool ok =
        P21_CUDA_OK(cudaGetLastError()) &&
        P21_CUDA_OK(cudaMemcpyAsync(&total_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost,
                                    stream)) &&
        P21_CUDA_OK(cudaStreamSynchronize(stream));
    cleanup();
    if (!ok) {
        return -1;
    }
    *mean_out = total_sum / (double)n_elems;
    return 0;
}

}  // extern "C"
