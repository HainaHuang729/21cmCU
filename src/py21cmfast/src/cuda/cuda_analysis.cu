#include <algorithm>
#include <cmath>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "cuda_common.cuh"
#include "cuda_mgpu.cuh"
#include "cuda_philox.cuh"

namespace {

struct DeviceFloatView {
    const float *d_ptr = nullptr;
    bool owned = false;
};

DeviceFloatView acquire_float_view(const float *h_ptr, size_t nbytes, cudaStream_t stream) {
    DeviceFloatView view{};
    if (!h_ptr || nbytes == 0) {
        return view;
    }
    if (nbytes <= p21_cuda_registry_max_entry_bytes()) {
        view.d_ptr = (const float *)p21_cuda_registry_get_or_upload(h_ptr, nbytes, stream);
        if (view.d_ptr) {
            view.owned = false;
            return view;
        }
    }
    float *d_tmp = (float *)p21_cuda_pool_alloc(nbytes, stream);
    if (!d_tmp) {
        return view;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(d_tmp, h_ptr, nbytes, cudaMemcpyHostToDevice, stream))) {
        p21_cuda_pool_free(d_tmp, stream);
        return view;
    }
    p21_cuda_stats_mut()->registry_h2d_bytes += (unsigned long long int)nbytes;
    view.d_ptr = d_tmp;
    view.owned = true;
    return view;
}

void release_float_view(DeviceFloatView &view, cudaStream_t stream) {
    if (view.owned && view.d_ptr) {
        p21_cuda_pool_free(const_cast<float *>(view.d_ptr), stream);
    }
    view.d_ptr = nullptr;
    view.owned = false;
}

__device__ static inline double eval_dvdr_factor(double tb, double tau21, double vg, double H_z,
                                                 double max_dvdr, int use_ts_fluct) {
    if (!use_ts_fluct) {
        double max_v = max_dvdr * H_z;
        double dvdx = fmin(fmax(vg, -max_v), max_v);
        double grad_comp = fabs(1.0 + dvdx / H_z);
        return tb / grad_comp;
    }
    double grad_comp = fabs(1.0 + vg / H_z);
    double gfac = 1.0;
    if (tau21 >= 1e-10) {
        gfac = (1.0 - exp(-tau21 / grad_comp)) / (1.0 - exp(-tau21));
    }
    /* Match rsds.py float32 cast of gradient_factor. */
    gfac = (double)(float)gfac;
    return tb * gfac;
}

__global__ void dvdr_nonperiodic_kernel(const float *d_tb, const float *d_vel,
                                        const float *d_tau21, const double *d_H, size_t n_los,
                                        int nz, double dx_los_mpc, double max_dvdr,
                                        int use_ts_fluct, float *d_out) {
    size_t total = n_los * (size_t)nz;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    double inv_2dx = 0.5 / dx_los_mpc;

    for (size_t idx = tid; idx < total; idx += stride) {
        int iz = (int)(idx % (size_t)nz);
        double vg;
        if (iz == 0) {
            vg = (-3.0 * (double)d_vel[idx] + 4.0 * (double)d_vel[idx + 1] -
                  (double)d_vel[idx + 2]) *
                 inv_2dx;
        } else if (iz == nz - 1) {
            vg = (3.0 * (double)d_vel[idx] - 4.0 * (double)d_vel[idx - 1] +
                  (double)d_vel[idx - 2]) *
                 inv_2dx;
        } else {
            vg = ((double)d_vel[idx + 1] - (double)d_vel[idx - 1]) * inv_2dx;
        }

        double tau = (use_ts_fluct && d_tau21) ? (double)d_tau21[idx] : 0.0;
        double out_val =
            eval_dvdr_factor((double)d_tb[idx], tau, vg, d_H[iz], max_dvdr, use_ts_fluct);
        d_out[idx] = (float)out_val;
    }
}

__global__ void mul_ikz_r2c_kernel(cufftComplex *d_freq, int nx, int ny, int nz,
                                   double dx_los_mpc) {
    int hz = nz / 2 + 1;
    size_t total = (size_t)nx * (size_t)ny * (size_t)hz;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    double dk_z = 2.0 * M_PI / ((double)nz * dx_los_mpc);
    double inv_vol = 1.0 / ((double)nx * (double)ny * (double)nz);

    for (size_t idx = tid; idx < total; idx += stride) {
        int iz = (int)(idx % (size_t)hz);
        double kz_norm = (double)iz * dk_z * inv_vol;
        cufftComplex v = d_freq[idx];
        cufftComplex out;
        out.x = (float)(-(double)v.y * kz_norm);
        out.y = (float)((double)v.x * kz_norm);
        d_freq[idx] = out;
    }
}

__global__ void apply_velgrad_dvdr_kernel(const float *d_tb, const float *d_vel_grad,
                                          const float *d_tau21, const double *d_H, size_t n_los,
                                          int nz, double max_dvdr, int use_ts_fluct,
                                          float *d_out) {
    size_t total = n_los * (size_t)nz;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        int iz = (int)(idx % (size_t)nz);
        double tau = (use_ts_fluct && d_tau21) ? (double)d_tau21[idx] : 0.0;
        double out_val = eval_dvdr_factor((double)d_tb[idx], tau, (double)d_vel_grad[idx], d_H[iz],
                                          max_dvdr, use_ts_fluct);
        d_out[idx] = (float)out_val;
    }
}

__global__ void rsds_shift_kernel(const float *d_field, const float *d_disp, int n_los, int nz,
                                  int n_rsd_subcells, int periodic, double *d_out_acc) {
    size_t total = (size_t)n_los * (size_t)nz;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    long long n_fine = (long long)nz * (long long)n_rsd_subcells;
    double S = (double)n_rsd_subcells;

    for (size_t idx = tid; idx < total; idx += stride) {
        size_t c = idx / (size_t)nz;
        int k = (int)(idx % (size_t)nz);
        const float *disp_col = d_disp + c * (size_t)nz;
        double *out_col = d_out_acc + c * (size_t)nz;
        double weight = (double)d_field[idx] / S;
        if (weight == 0.0) {
            continue;
        }

        for (int s = 0; s < n_rsd_subcells; ++s) {
            long long ii = (long long)k * (long long)n_rsd_subcells + (long long)s;
            double fine_x = ((double)ii + 0.5) / S;
            double u = fine_x - 0.5;

            double d_interp;
            if (periodic) {
                int j0 = (int)floor(u);
                double frac = u - (double)j0;
                int j0w = ((j0 % nz) + nz) % nz;
                int j1w = (j0w + 1) % nz;
                d_interp = (1.0 - frac) * (double)disp_col[j0w] + frac * (double)disp_col[j1w];
            } else {
                int j0 = (int)floor(u);
                if (j0 < 0) j0 = 0;
                if (j0 > nz - 2) j0 = nz - 2;
                double frac = u - (double)j0;
                d_interp = (1.0 - frac) * (double)disp_col[j0] + frac * (double)disp_col[j0 + 1];
            }

            double x = (double)ii + d_interp * S;
            long long i0 = (long long)floor(x);
            long long i1 = i0 + 1;
            double tx = (double)i1 - x;
            double w1 = 1.0 - tx;

            if (periodic) {
                long long i0w = ((i0 % n_fine) + n_fine) % n_fine;
                long long i1w = ((i1 % n_fine) + n_fine) % n_fine;
                int dst0 = (int)(i0w / n_rsd_subcells);
                int dst1 = (int)(i1w / n_rsd_subcells);
                if (dst0 == dst1) {
                    p21_atomic_add_double(&out_col[dst0], weight);
                } else {
                    p21_atomic_add_double(&out_col[dst0], tx * weight);
                    p21_atomic_add_double(&out_col[dst1], w1 * weight);
                }
            } else {
                bool in0 = (i0 >= 0 && i0 < n_fine);
                bool in1 = (i1 >= 0 && i1 < n_fine);
                int dst0 = in0 ? (int)(i0 / n_rsd_subcells) : -1;
                int dst1 = in1 ? (int)(i1 / n_rsd_subcells) : -1;
                if (in0 && in1 && dst0 == dst1) {
                    p21_atomic_add_double(&out_col[dst0], weight);
                } else {
                    if (in0) {
                        p21_atomic_add_double(&out_col[dst0], tx * weight);
                    }
                    if (in1) {
                        p21_atomic_add_double(&out_col[dst1], w1 * weight);
                    }
                }
            }
        }
    }
}

__global__ void double_to_float_kernel(const double *src, float *dst, size_t n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < n; i += stride) {
        dst[i] = (float)src[i];
    }
}

__global__ void rectilinear_slices_kernel(const float *d_box_a, const float *d_box_b, int nx,
                                          int ny, int nz, int n_slices, const int *d_z_indices,
                                          const float *d_w_a, const float *d_w_b, int kind,
                                          float *d_out) {
    size_t plane_size = (size_t)nx * (size_t)ny;
    size_t total = (size_t)n_slices * plane_size;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;

    for (size_t idx = tid; idx < total; idx += stride) {
        int s = (int)(idx / plane_size);
        size_t xy = idx % plane_size;
        int iz = d_z_indices[s];
        size_t src_idx = xy * (size_t)nz + (size_t)iz;

        float a = d_box_a[src_idx];
        float b = d_box_b[src_idx];
        float val = d_w_a[s] * a + d_w_b[s] * b;
        if (kind == 1 && (a * b < 0.0f)) {
            val = fmaxf(a, b);
        }
        d_out[idx] = val;
    }
}

__device__ static inline int binary_search_bin(double val, const double *edges, int n_bins) {
    if (n_bins <= 0 || val < edges[0] || val >= edges[n_bins]) {
        return -1;
    }
    int lo = 0;
    int hi = n_bins - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (val < edges[mid]) {
            hi = mid - 1;
        } else if (val >= edges[mid + 1]) {
            lo = mid + 1;
        } else {
            return mid;
        }
    }
    return -1;
}

__global__ void ps_1d_bin_kernel(const cufftComplex *d_fa, const cufftComplex *d_fb, int nx, int ny,
                                 int nz, double lx, double ly, double lz, double norm, int skip_dc,
                                 int n_bins, const double *d_edges, double *d_k_sum,
                                 double *d_pk_sum, unsigned long long *d_counts) {
    int hz = nz / 2 + 1;
    size_t total = (size_t)nx * (size_t)ny * (size_t)hz;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    bool even_z = (nz % 2 == 0);
    double dk_z = 2.0 * M_PI / lz;

    for (size_t idx = tid; idx < total; idx += stride) {
        int iz = (int)(idx % (size_t)hz);
        size_t q = idx / (size_t)hz;
        int iy = (int)(q % (size_t)ny);
        int ix = (int)(q / (size_t)ny);
        if (skip_dc && ix == 0 && iy == 0 && iz == 0) {
            continue;
        }

        double kx = p21_index_to_k(ix, lx, nx);
        double ky = p21_index_to_k(iy, ly, ny);
        double kz = (double)iz * dk_z;
        double k = sqrt(kx * kx + ky * ky + kz * kz);
        int b = binary_search_bin(k, d_edges, n_bins);
        if (b < 0) {
            continue;
        }

        unsigned long long mult = (iz == 0 || (even_z && iz == nz / 2)) ? 1ULL : 2ULL;
        cufftComplex fa = d_fa[idx];
        double p;
        if (!d_fb) {
            p = (double)fa.x * (double)fa.x + (double)fa.y * (double)fa.y;
        } else {
            cufftComplex fb = d_fb[idx];
            p = (double)fa.x * (double)fb.x + (double)fa.y * (double)fb.y;
        }
        p *= norm;

        double m_d = (double)mult;
        p21_atomic_add_double(&d_k_sum[b], m_d * k);
        p21_atomic_add_double(&d_pk_sum[b], m_d * p);
        atomicAdd(&d_counts[b], mult);
    }
}

__global__ void ps_2d_bin_kernel(const cufftComplex *d_fa, const cufftComplex *d_fb, int nx, int ny,
                                 int nz, double lx, double ly, double lz, int los_axis,
                                 double norm, int skip_dc, int n_kperp,
                                 const double *d_kperp_edges, int n_kpara,
                                 const double *d_kpara_edges, double *d_kperp_sum,
                                 double *d_kpara_sum, double *d_pk_sum,
                                 unsigned long long *d_counts) {
    int hz = nz / 2 + 1;
    size_t total = (size_t)nx * (size_t)ny * (size_t)hz;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    bool even_z = (nz % 2 == 0);
    double dk_z = 2.0 * M_PI / lz;

    for (size_t idx = tid; idx < total; idx += stride) {
        int iz = (int)(idx % (size_t)hz);
        size_t q = idx / (size_t)hz;
        int iy = (int)(q % (size_t)ny);
        int ix = (int)(q / (size_t)ny);
        if (skip_dc && ix == 0 && iy == 0 && iz == 0) {
            continue;
        }

        double kx = p21_index_to_k(ix, lx, nx);
        double ky = p21_index_to_k(iy, ly, ny);
        double kz = (double)iz * dk_z;

        double k_para;
        double k_perp;
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

        int b_perp = binary_search_bin(k_perp, d_kperp_edges, n_kperp);
        if (b_perp < 0) {
            continue;
        }
        int b_para = binary_search_bin(k_para, d_kpara_edges, n_kpara);
        if (b_para < 0) {
            continue;
        }

        int b = b_perp * n_kpara + b_para;
        unsigned long long mult = (iz == 0 || (even_z && iz == nz / 2)) ? 1ULL : 2ULL;
        cufftComplex fa = d_fa[idx];
        double p;
        if (!d_fb) {
            p = (double)fa.x * (double)fa.x + (double)fa.y * (double)fa.y;
        } else {
            cufftComplex fb = d_fb[idx];
            p = (double)fa.x * (double)fb.x + (double)fa.y * (double)fb.y;
        }
        p *= norm;

        double m_d = (double)mult;
        p21_atomic_add_double(&d_kperp_sum[b], m_d * k_perp);
        p21_atomic_add_double(&d_kpara_sum[b], m_d * k_para);
        p21_atomic_add_double(&d_pk_sum[b], m_d * p);
        atomicAdd(&d_counts[b], mult);
    }
}

__global__ void ps_1d_bin_yslab_kernel(const cufftComplex *d_fa, const cufftComplex *d_fb, int y0,
                                       int sy, int nx, int ny, int nz, double lx, double ly,
                                       double lz, double norm, int skip_dc, int n_bins,
                                       const double *d_edges, double *d_k_sum, double *d_pk_sum,
                                       unsigned long long *d_counts) {
    int hz = nz / 2 + 1;
    size_t total = (size_t)sy * (size_t)hz * (size_t)nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    bool even_z = (nz % 2 == 0);
    double dk_z = 2.0 * M_PI / lz;

    for (size_t idx = tid; idx < total; idx += stride) {
        int ix = (int)(idx % (size_t)nx);
        size_t q = idx / (size_t)nx;
        int iz = (int)(q % (size_t)hz);
        int iy = y0 + (int)(q / (size_t)hz);
        if (skip_dc && ix == 0 && iy == 0 && iz == 0) {
            continue;
        }

        double kx = p21_index_to_k(ix, lx, nx);
        double ky = p21_index_to_k(iy, ly, ny);
        double kz = (double)iz * dk_z;
        double k = sqrt(kx * kx + ky * ky + kz * kz);
        int b = binary_search_bin(k, d_edges, n_bins);
        if (b < 0) {
            continue;
        }

        unsigned long long mult = (iz == 0 || (even_z && iz == nz / 2)) ? 1ULL : 2ULL;
        cufftComplex fa = d_fa[idx];
        double p;
        if (!d_fb) {
            p = (double)fa.x * (double)fa.x + (double)fa.y * (double)fa.y;
        } else {
            cufftComplex fb = d_fb[idx];
            p = (double)fa.x * (double)fb.x + (double)fa.y * (double)fb.y;
        }
        p *= norm;

        double m_d = (double)mult;
        p21_atomic_add_double(&d_k_sum[b], m_d * k);
        p21_atomic_add_double(&d_pk_sum[b], m_d * p);
        atomicAdd(&d_counts[b], mult);
    }
}

__global__ void ps_2d_bin_yslab_kernel(const cufftComplex *d_fa, const cufftComplex *d_fb, int y0,
                                       int sy, int nx, int ny, int nz, double lx, double ly,
                                       double lz, int los_axis, double norm, int skip_dc,
                                       int n_kperp, const double *d_kperp_edges, int n_kpara,
                                       const double *d_kpara_edges, double *d_kperp_sum,
                                       double *d_kpara_sum, double *d_pk_sum,
                                       unsigned long long *d_counts) {
    int hz = nz / 2 + 1;
    size_t total = (size_t)sy * (size_t)hz * (size_t)nx;
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    bool even_z = (nz % 2 == 0);
    double dk_z = 2.0 * M_PI / lz;

    for (size_t idx = tid; idx < total; idx += stride) {
        int ix = (int)(idx % (size_t)nx);
        size_t q = idx / (size_t)nx;
        int iz = (int)(q % (size_t)hz);
        int iy = y0 + (int)(q / (size_t)hz);
        if (skip_dc && ix == 0 && iy == 0 && iz == 0) {
            continue;
        }

        double kx = p21_index_to_k(ix, lx, nx);
        double ky = p21_index_to_k(iy, ly, ny);
        double kz = (double)iz * dk_z;

        double k_para;
        double k_perp;
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

        int b_perp = binary_search_bin(k_perp, d_kperp_edges, n_kperp);
        if (b_perp < 0) {
            continue;
        }
        int b_para = binary_search_bin(k_para, d_kpara_edges, n_kpara);
        if (b_para < 0) {
            continue;
        }

        int b = b_perp * n_kpara + b_para;
        unsigned long long mult = (iz == 0 || (even_z && iz == nz / 2)) ? 1ULL : 2ULL;
        cufftComplex fa = d_fa[idx];
        double p;
        if (!d_fb) {
            p = (double)fa.x * (double)fa.x + (double)fa.y * (double)fa.y;
        } else {
            cufftComplex fb = d_fb[idx];
            p = (double)fa.x * (double)fb.x + (double)fa.y * (double)fb.y;
        }
        p *= norm;

        double m_d = (double)mult;
        p21_atomic_add_double(&d_kperp_sum[b], m_d * k_perp);
        p21_atomic_add_double(&d_kpara_sum[b], m_d * k_para);
        p21_atomic_add_double(&d_pk_sum[b], m_d * p);
        atomicAdd(&d_counts[b], mult);
    }
}

int compute_grid_blocks(size_t total, int threads) {
    size_t max_blocks = 65535ULL;
    size_t b = (total + (size_t)threads - 1) / (size_t)threads;
    return (int)std::min(b, max_blocks);
}

bool mgpu_ps_force_enabled(void) {
    const char *env = getenv("P21_CUDA_FORCE_MGPU");
    return env && strcmp(env, "1") == 0;
}

int mgpu_power_spectrum_1d(const float *h_field_a, const float *h_field_b, int nx, int ny, int nz,
                           double lx, double ly, double lz, int n_bins, const double *h_k_edges,
                           int skip_dc, double *h_k_mean_out, double *h_pk_out,
                           double *h_delta_sq_out, unsigned long long *h_counts_out) {
    p21_cuda_registry_sync_to_host(h_field_a);
    if (h_field_b && h_field_b != h_field_a) {
        p21_cuda_registry_sync_to_host(h_field_b);
    }

    P21MgpuGrid grid;
    if (!grid.init(nx, ny, nz)) {
        return -1;
    }
    const int G = grid.n_gpus;
    const bool is_cross = (h_field_b != nullptr && h_field_b != h_field_a);

    size_t max_slab_bytes = 0;
    for (int g = 0; g < G; ++g) {
        max_slab_bytes = std::max(max_slab_bytes, grid.cplx_elems_y(g) * sizeof(cufftComplex));
    }
    const size_t min_free = p21_mgpu_min_free_mem(grid);
    void *h_stage = nullptr;
    if (min_free < (is_cross ? 3ULL : 2ULL) * max_slab_bytes + (768ULL << 20)) {
        h_stage = malloc(grid.total_cplx_elems() * sizeof(cufftComplex));
        if (!h_stage) {
            grid.destroy();
            return -1;
        }
    }

    std::vector<cufftComplex *> d_fa((size_t)G, nullptr);
    std::vector<cufftComplex *> d_fb((size_t)G, nullptr);
    bool ok = true;
    for (int g = 0; g < G; ++g) {
        size_t bytes_y = grid.cplx_elems_y(g) * sizeof(cufftComplex);
        d_fa[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(grid.devs[g], bytes_y, grid.streams[g]);
        if (!d_fa[(size_t)g]) {
            ok = false;
        }
        if (is_cross) {
            d_fb[(size_t)g] =
                (cufftComplex *)p21_mgpu_malloc(grid.devs[g], bytes_y, grid.streams[g]);
            if (!d_fb[(size_t)g]) {
                ok = false;
            }
        }
    }

    if (ok) {
        ok = p21_mgpu_fft_r2c_from_host(grid, h_field_a, 1.0, d_fa.data(), h_stage);
    }
    if (ok && is_cross) {
        ok = p21_mgpu_fft_r2c_from_host(grid, h_field_b, 1.0, d_fb.data(), h_stage);
    }

    if (h_stage) {
        free(h_stage);
        h_stage = nullptr;
    }

    std::vector<double> tot_k((size_t)n_bins, 0.0);
    std::vector<double> tot_pk((size_t)n_bins, 0.0);
    memset(h_counts_out, 0, (size_t)n_bins * sizeof(unsigned long long));

    if (ok) {
        double vol = lx * ly * lz;
        size_t n_tot = (size_t)nx * (size_t)ny * (size_t)nz;
        double norm = vol / ((double)n_tot * (double)n_tot);
        constexpr int kTpb = 256;

        std::vector<double *> d_edges((size_t)G, nullptr);
        std::vector<double *> d_ks((size_t)G, nullptr);
        std::vector<double *> d_pks((size_t)G, nullptr);
        std::vector<unsigned long long *> d_cnt((size_t)G, nullptr);

        for (int g = 0; g < G && ok; ++g) {
            cudaSetDevice(grid.devs[g]);
            d_edges[(size_t)g] = (double *)p21_mgpu_malloc(
                grid.devs[g], (size_t)(n_bins + 1) * sizeof(double), grid.streams[g]);
            d_ks[(size_t)g] = (double *)p21_mgpu_malloc(
                grid.devs[g], (size_t)n_bins * sizeof(double), grid.streams[g]);
            d_pks[(size_t)g] = (double *)p21_mgpu_malloc(
                grid.devs[g], (size_t)n_bins * sizeof(double), grid.streams[g]);
            d_cnt[(size_t)g] = (unsigned long long *)p21_mgpu_malloc(
                grid.devs[g], (size_t)n_bins * sizeof(unsigned long long), grid.streams[g]);
            if (!d_edges[(size_t)g] || !d_ks[(size_t)g] || !d_pks[(size_t)g] ||
                !d_cnt[(size_t)g] ||
                !P21_CUDA_OK(cudaMemcpyAsync(d_edges[(size_t)g], h_k_edges,
                                             (size_t)(n_bins + 1) * sizeof(double),
                                             cudaMemcpyHostToDevice, grid.streams[g])) ||
                !P21_CUDA_OK(cudaMemsetAsync(d_ks[(size_t)g], 0, (size_t)n_bins * sizeof(double),
                                           grid.streams[g])) ||
                !P21_CUDA_OK(cudaMemsetAsync(d_pks[(size_t)g], 0, (size_t)n_bins * sizeof(double),
                                           grid.streams[g])) ||
                !P21_CUDA_OK(cudaMemsetAsync(d_cnt[(size_t)g], 0,
                                           (size_t)n_bins * sizeof(unsigned long long),
                                           grid.streams[g]))) {
                ok = false;
                break;
            }
            int blocks = compute_grid_blocks(grid.cplx_elems_y(g), kTpb);
            ps_1d_bin_yslab_kernel<<<blocks, kTpb, 0, grid.streams[g]>>>(
                d_fa[(size_t)g], is_cross ? d_fb[(size_t)g] : nullptr, grid.y0[g], grid.sy[g], nx,
                ny, nz, lx, ly, lz, norm, skip_dc, n_bins, d_edges[(size_t)g], d_ks[(size_t)g],
                d_pks[(size_t)g], d_cnt[(size_t)g]);
            ok = P21_CUDA_OK(cudaGetLastError());
        }

        std::vector<double> lk((size_t)n_bins);
        std::vector<double> lpk((size_t)n_bins);
        std::vector<unsigned long long> lc((size_t)n_bins);
        for (int g = 0; g < G && ok; ++g) {
            cudaSetDevice(grid.devs[g]);
            ok = P21_CUDA_OK(cudaMemcpyAsync(lk.data(), d_ks[(size_t)g],
                                             (size_t)n_bins * sizeof(double),
                                             cudaMemcpyDeviceToHost, grid.streams[g])) &&
                 P21_CUDA_OK(cudaMemcpyAsync(lpk.data(), d_pks[(size_t)g],
                                             (size_t)n_bins * sizeof(double),
                                             cudaMemcpyDeviceToHost, grid.streams[g])) &&
                 P21_CUDA_OK(cudaMemcpyAsync(lc.data(), d_cnt[(size_t)g],
                                             (size_t)n_bins * sizeof(unsigned long long),
                                             cudaMemcpyDeviceToHost, grid.streams[g])) &&
                 P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
            if (ok) {
                for (int b = 0; b < n_bins; ++b) {
                    tot_k[(size_t)b] += lk[(size_t)b];
                    tot_pk[(size_t)b] += lpk[(size_t)b];
                    h_counts_out[b] += lc[(size_t)b];
                }
            }
        }
        for (int g = 0; g < G; ++g) {
            p21_mgpu_free(grid.devs[g], d_cnt[(size_t)g], grid.streams[g]);
            p21_mgpu_free(grid.devs[g], d_pks[(size_t)g], grid.streams[g]);
            p21_mgpu_free(grid.devs[g], d_ks[(size_t)g], grid.streams[g]);
            p21_mgpu_free(grid.devs[g], d_edges[(size_t)g], grid.streams[g]);
        }
    }

    for (int g = 0; g < G; ++g) {
        if (is_cross) p21_mgpu_free(grid.devs[g], d_fb[(size_t)g], grid.streams[g]);
        p21_mgpu_free(grid.devs[g], d_fa[(size_t)g], grid.streams[g]);
    }
    grid.destroy();
    if (!ok) {
        return -1;
    }

    const double inv_2pi2 = 1.0 / (2.0 * M_PI * M_PI);
    for (int b = 0; b < n_bins; ++b) {
        unsigned long long cnt = h_counts_out[b];
        if (cnt > 0ULL) {
            double km = tot_k[(size_t)b] / (double)cnt;
            double pk = tot_pk[(size_t)b] / (double)cnt;
            h_k_mean_out[b] = km;
            h_pk_out[b] = pk;
            h_delta_sq_out[b] = km * km * km * pk * inv_2pi2;
        } else {
            h_k_mean_out[b] = NAN;
            h_pk_out[b] = NAN;
            h_delta_sq_out[b] = NAN;
        }
    }
    return 0;
}

int mgpu_power_spectrum_2d(const float *h_field_a, const float *h_field_b, int nx, int ny, int nz,
                           double lx, double ly, double lz, int los_axis, int n_kperp,
                           const double *h_kperp_edges, int n_kpara, const double *h_kpara_edges,
                           int skip_dc, double *h_kperp_mean_out, double *h_kpara_mean_out,
                           double *h_pk_out, unsigned long long *h_counts_out) {
    p21_cuda_registry_sync_to_host(h_field_a);
    if (h_field_b && h_field_b != h_field_a) {
        p21_cuda_registry_sync_to_host(h_field_b);
    }

    P21MgpuGrid grid;
    if (!grid.init(nx, ny, nz)) {
        return -1;
    }
    const int G = grid.n_gpus;
    const bool is_cross = (h_field_b != nullptr && h_field_b != h_field_a);
    const size_t n_bins = (size_t)n_kperp * (size_t)n_kpara;

    size_t max_slab_bytes = 0;
    for (int g = 0; g < G; ++g) {
        max_slab_bytes = std::max(max_slab_bytes, grid.cplx_elems_y(g) * sizeof(cufftComplex));
    }
    const size_t min_free = p21_mgpu_min_free_mem(grid);
    void *h_stage = nullptr;
    if (min_free < (is_cross ? 3ULL : 2ULL) * max_slab_bytes + (768ULL << 20)) {
        h_stage = malloc(grid.total_cplx_elems() * sizeof(cufftComplex));
        if (!h_stage) {
            grid.destroy();
            return -1;
        }
    }

    std::vector<cufftComplex *> d_fa((size_t)G, nullptr);
    std::vector<cufftComplex *> d_fb((size_t)G, nullptr);
    bool ok = true;
    for (int g = 0; g < G; ++g) {
        size_t bytes_y = grid.cplx_elems_y(g) * sizeof(cufftComplex);
        d_fa[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(grid.devs[g], bytes_y, grid.streams[g]);
        if (!d_fa[(size_t)g]) ok = false;
        if (is_cross) {
            d_fb[(size_t)g] =
                (cufftComplex *)p21_mgpu_malloc(grid.devs[g], bytes_y, grid.streams[g]);
            if (!d_fb[(size_t)g]) ok = false;
        }
    }

    if (ok) ok = p21_mgpu_fft_r2c_from_host(grid, h_field_a, 1.0, d_fa.data(), h_stage);
    if (ok && is_cross) ok = p21_mgpu_fft_r2c_from_host(grid, h_field_b, 1.0, d_fb.data(), h_stage);

    if (h_stage) {
        free(h_stage);
        h_stage = nullptr;
    }

    std::vector<double> tot_kp(n_bins, 0.0);
    std::vector<double> tot_kz(n_bins, 0.0);
    std::vector<double> tot_pk(n_bins, 0.0);
    memset(h_counts_out, 0, n_bins * sizeof(unsigned long long));

    if (ok) {
        double vol = lx * ly * lz;
        size_t n_tot = (size_t)nx * (size_t)ny * (size_t)nz;
        double norm = vol / ((double)n_tot * (double)n_tot);
        constexpr int kTpb = 256;

        std::vector<double *> d_ekp((size_t)G, nullptr), d_ekz((size_t)G, nullptr);
        std::vector<double *> d_kp((size_t)G, nullptr), d_kz((size_t)G, nullptr),
            d_pk((size_t)G, nullptr);
        std::vector<unsigned long long *> d_cnt((size_t)G, nullptr);

        for (int g = 0; g < G && ok; ++g) {
            cudaSetDevice(grid.devs[g]);
            d_ekp[(size_t)g] = (double *)p21_mgpu_malloc(
                grid.devs[g], (size_t)(n_kperp + 1) * sizeof(double), grid.streams[g]);
            d_ekz[(size_t)g] = (double *)p21_mgpu_malloc(
                grid.devs[g], (size_t)(n_kpara + 1) * sizeof(double), grid.streams[g]);
            d_kp[(size_t)g] =
                (double *)p21_mgpu_malloc(grid.devs[g], n_bins * sizeof(double), grid.streams[g]);
            d_kz[(size_t)g] =
                (double *)p21_mgpu_malloc(grid.devs[g], n_bins * sizeof(double), grid.streams[g]);
            d_pk[(size_t)g] =
                (double *)p21_mgpu_malloc(grid.devs[g], n_bins * sizeof(double), grid.streams[g]);
            d_cnt[(size_t)g] = (unsigned long long *)p21_mgpu_malloc(
                grid.devs[g], n_bins * sizeof(unsigned long long), grid.streams[g]);

            if (!d_ekp[(size_t)g] || !d_ekz[(size_t)g] || !d_kp[(size_t)g] || !d_kz[(size_t)g] ||
                !d_pk[(size_t)g] || !d_cnt[(size_t)g] ||
                !P21_CUDA_OK(cudaMemcpyAsync(d_ekp[(size_t)g], h_kperp_edges,
                                             (size_t)(n_kperp + 1) * sizeof(double),
                                             cudaMemcpyHostToDevice, grid.streams[g])) ||
                !P21_CUDA_OK(cudaMemcpyAsync(d_ekz[(size_t)g], h_kpara_edges,
                                             (size_t)(n_kpara + 1) * sizeof(double),
                                             cudaMemcpyHostToDevice, grid.streams[g])) ||
                !P21_CUDA_OK(
                    cudaMemsetAsync(d_kp[(size_t)g], 0, n_bins * sizeof(double), grid.streams[g])) ||
                !P21_CUDA_OK(
                    cudaMemsetAsync(d_kz[(size_t)g], 0, n_bins * sizeof(double), grid.streams[g])) ||
                !P21_CUDA_OK(
                    cudaMemsetAsync(d_pk[(size_t)g], 0, n_bins * sizeof(double), grid.streams[g])) ||
                !P21_CUDA_OK(cudaMemsetAsync(d_cnt[(size_t)g], 0,
                                           n_bins * sizeof(unsigned long long), grid.streams[g]))) {
                ok = false;
                break;
            }
            int blocks = compute_grid_blocks(grid.cplx_elems_y(g), kTpb);
            ps_2d_bin_yslab_kernel<<<blocks, kTpb, 0, grid.streams[g]>>>(
                d_fa[(size_t)g], is_cross ? d_fb[(size_t)g] : nullptr, grid.y0[g], grid.sy[g], nx,
                ny, nz, lx, ly, lz, los_axis, norm, skip_dc, n_kperp, d_ekp[(size_t)g], n_kpara,
                d_ekz[(size_t)g], d_kp[(size_t)g], d_kz[(size_t)g], d_pk[(size_t)g],
                d_cnt[(size_t)g]);
            ok = P21_CUDA_OK(cudaGetLastError());
        }

        std::vector<double> lkp(n_bins), lkz(n_bins), lpk(n_bins);
        std::vector<unsigned long long> lc(n_bins);
        for (int g = 0; g < G && ok; ++g) {
            cudaSetDevice(grid.devs[g]);
            ok = P21_CUDA_OK(cudaMemcpyAsync(lkp.data(), d_kp[(size_t)g], n_bins * sizeof(double),
                                             cudaMemcpyDeviceToHost, grid.streams[g])) &&
                 P21_CUDA_OK(cudaMemcpyAsync(lkz.data(), d_kz[(size_t)g], n_bins * sizeof(double),
                                             cudaMemcpyDeviceToHost, grid.streams[g])) &&
                 P21_CUDA_OK(cudaMemcpyAsync(lpk.data(), d_pk[(size_t)g], n_bins * sizeof(double),
                                             cudaMemcpyDeviceToHost, grid.streams[g])) &&
                 P21_CUDA_OK(cudaMemcpyAsync(lc.data(), d_cnt[(size_t)g],
                                             n_bins * sizeof(unsigned long long),
                                             cudaMemcpyDeviceToHost, grid.streams[g])) &&
                 P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
            if (ok) {
                for (size_t b = 0; b < n_bins; ++b) {
                    tot_kp[b] += lkp[b];
                    tot_kz[b] += lkz[b];
                    tot_pk[b] += lpk[b];
                    h_counts_out[b] += lc[b];
                }
            }
        }
        for (int g = 0; g < G; ++g) {
            p21_mgpu_free(grid.devs[g], d_cnt[(size_t)g], grid.streams[g]);
            p21_mgpu_free(grid.devs[g], d_pk[(size_t)g], grid.streams[g]);
            p21_mgpu_free(grid.devs[g], d_kz[(size_t)g], grid.streams[g]);
            p21_mgpu_free(grid.devs[g], d_kp[(size_t)g], grid.streams[g]);
            p21_mgpu_free(grid.devs[g], d_ekz[(size_t)g], grid.streams[g]);
            p21_mgpu_free(grid.devs[g], d_ekp[(size_t)g], grid.streams[g]);
        }
    }

    for (int g = 0; g < G; ++g) {
        if (is_cross) p21_mgpu_free(grid.devs[g], d_fb[(size_t)g], grid.streams[g]);
        p21_mgpu_free(grid.devs[g], d_fa[(size_t)g], grid.streams[g]);
    }
    grid.destroy();
    if (!ok) {
        return -1;
    }

    for (size_t b = 0; b < n_bins; ++b) {
        unsigned long long cnt = h_counts_out[b];
        if (cnt > 0ULL) {
            double c = (double)cnt;
            h_kperp_mean_out[b] = tot_kp[b] / c;
            h_kpara_mean_out[b] = tot_kz[b] / c;
            h_pk_out[b] = tot_pk[b] / c;
        } else {
            h_kperp_mean_out[b] = NAN;
            h_kpara_mean_out[b] = NAN;
            h_pk_out[b] = NAN;
        }
    }
    return 0;
}

__device__ static inline int wrap_coord_ll(long long v, int d) {
    long long m = v % (long long)d;
    return (int)(m < 0 ? m + (long long)d : m);
}

__global__ void angular_slice_kernel(const float *d_box_a, const float *d_box_b, int nx, int ny,
                                     int nz, int n_coords, const double *d_coords_xyz, float w_a,
                                     float w_b, int order, int kind, float *d_out) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t nc = (size_t)n_coords;
    double wa = (double)w_a;
    double wb = (double)w_b;

    for (size_t i = tid; i < nc; i += stride) {
        double cx = d_coords_xyz[i];
        double cy = d_coords_xyz[nc + i];
        double cz = d_coords_xyz[2 * nc + i];

        if (order == 0) {
            int ix = wrap_coord_ll((long long)floor(cx + 0.5), nx);
            int iy = wrap_coord_ll((long long)floor(cy + 0.5), ny);
            int iz = wrap_coord_ll((long long)floor(cz + 0.5), nz);
            size_t idx = p21_grid_idx(ix, iy, iz, ny, nz);
            float a = d_box_a[idx];
            float b = d_box_b[idx];
            float v = w_a * a + w_b * b;
            if (kind == 1 && (a * b < 0.0f)) {
                v = fmaxf(a, b);
            }
            d_out[i] = v;
        } else {
            double fx = floor(cx);
            double fy = floor(cy);
            double fz = floor(cz);
            double tx = cx - fx;
            double ty = cy - fy;
            double tz = cz - fz;

            int x0 = wrap_coord_ll((long long)fx, nx);
            int y0 = wrap_coord_ll((long long)fy, ny);
            int z0 = wrap_coord_ll((long long)fz, nz);
            int x1 = (x0 + 1 < nx) ? (x0 + 1) : 0;
            int y1 = (y0 + 1 < ny) ? (y0 + 1) : 0;
            int z1 = (z0 + 1 < nz) ? (z0 + 1) : 0;

            int xs[2] = {x0, x1};
            int ys[2] = {y0, y1};
            int zs[2] = {z0, z1};
            double wxs[2] = {1.0 - tx, tx};
            double wys[2] = {1.0 - ty, ty};
            double wzs[2] = {1.0 - tz, tz};

            double acc = 0.0;
            for (int dx = 0; dx < 2; ++dx) {
                for (int dy = 0; dy < 2; ++dy) {
                    double wxy = wxs[dx] * wys[dy];
                    for (int dz = 0; dz < 2; ++dz) {
                        size_t idx = p21_grid_idx(xs[dx], ys[dy], zs[dz], ny, nz);
                        float a = d_box_a[idx];
                        float b = d_box_b[idx];
                        double v = wa * (double)a + wb * (double)b;
                        if (kind == 1 && (a * b < 0.0f)) {
                            v = (double)fmaxf(a, b);
                        }
                        acc += (wxy * wzs[dz]) * v;
                    }
                }
            }
            d_out[i] = (float)acc;
        }
    }
}

__global__ void angular_los_vel_kernel(const float *d_vxa, const float *d_vya, const float *d_vza,
                                       const float *d_vxb, const float *d_vyb, const float *d_vzb,
                                       int nx, int ny, int nz, int n_coords,
                                       const double *d_coords_xyz, const double *d_unit_xyz,
                                       float w_a, float w_b, int order, float *d_out) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t nc = (size_t)n_coords;
    double wa = (double)w_a;
    double wb = (double)w_b;

    for (size_t i = tid; i < nc; i += stride) {
        double cx = d_coords_xyz[i];
        double cy = d_coords_xyz[nc + i];
        double cz = d_coords_xyz[2 * nc + i];
        double ux = d_unit_xyz[i];
        double uy = d_unit_xyz[nc + i];
        double uz = d_unit_xyz[2 * nc + i];

        if (order == 0) {
            int ix = wrap_coord_ll((long long)floor(cx + 0.5), nx);
            int iy = wrap_coord_ll((long long)floor(cy + 0.5), ny);
            int iz = wrap_coord_ll((long long)floor(cz + 0.5), nz);
            size_t idx = p21_grid_idx(ix, iy, iz, ny, nz);
            double la = ux * (double)d_vxa[idx] + uy * (double)d_vya[idx] + uz * (double)d_vza[idx];
            double lb = ux * (double)d_vxb[idx] + uy * (double)d_vyb[idx] + uz * (double)d_vzb[idx];
            d_out[i] = (float)(wa * la + wb * lb);
        } else {
            double fx = floor(cx);
            double fy = floor(cy);
            double fz = floor(cz);
            double tx = cx - fx;
            double ty = cy - fy;
            double tz = cz - fz;

            int x0 = wrap_coord_ll((long long)fx, nx);
            int y0 = wrap_coord_ll((long long)fy, ny);
            int z0 = wrap_coord_ll((long long)fz, nz);
            int x1 = (x0 + 1 < nx) ? (x0 + 1) : 0;
            int y1 = (y0 + 1 < ny) ? (y0 + 1) : 0;
            int z1 = (z0 + 1 < nz) ? (z0 + 1) : 0;

            int xs[2] = {x0, x1};
            int ys[2] = {y0, y1};
            int zs[2] = {z0, z1};
            double wxs[2] = {1.0 - tx, tx};
            double wys[2] = {1.0 - ty, ty};
            double wzs[2] = {1.0 - tz, tz};

            double acc = 0.0;
            for (int dx = 0; dx < 2; ++dx) {
                for (int dy = 0; dy < 2; ++dy) {
                    double wxy = wxs[dx] * wys[dy];
                    for (int dz = 0; dz < 2; ++dz) {
                        size_t idx = p21_grid_idx(xs[dx], ys[dy], zs[dz], ny, nz);
                        double vx = wa * (double)d_vxa[idx] + wb * (double)d_vxb[idx];
                        double vy = wa * (double)d_vya[idx] + wb * (double)d_vyb[idx];
                        double vz = wa * (double)d_vza[idx] + wb * (double)d_vzb[idx];
                        acc += (wxy * wzs[dz]) * (ux * vx + uy * vy + uz * vz);
                    }
                }
            }
            d_out[i] = (float)acc;
        }
    }
}

}  // namespace

extern "C" {

int p21_cuda_include_dvdr_in_tau21(const float *h_tb, const float *h_los_vel,
                                   const float *h_tau_21, const double *h_H_s_inv, int nx, int ny,
                                   int nz, double dx_los_mpc, double max_dvdr, int use_ts_fluct,
                                   int periodic, float *h_out) {
    if (!h_tb || !h_los_vel || !h_H_s_inv || !h_out || nx <= 0 || ny <= 0 || nz < 3 ||
        dx_los_mpc <= 0.0) {
        return -1;
    }
    if (use_ts_fluct && !h_tau_21) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    size_t n_los = (size_t)nx * (size_t)ny;
    size_t total = n_los * (size_t)nz;
    size_t box_bytes = total * sizeof(float);

    DeviceFloatView v_tb = acquire_float_view(h_tb, box_bytes, stream);
    DeviceFloatView v_vel = acquire_float_view(h_los_vel, box_bytes, stream);
    DeviceFloatView v_tau{};
    if (use_ts_fluct) {
        v_tau = acquire_float_view(h_tau_21, box_bytes, stream);
    }

    double *d_H = (double *)p21_cuda_pool_alloc((size_t)nz * sizeof(double), stream);
    float *d_out = (float *)p21_cuda_pool_alloc(box_bytes, stream);
    float *d_vel_grad = nullptr;
    cufftComplex *d_freq = nullptr;

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_freq, stream);
        p21_cuda_pool_free(d_vel_grad, stream);
        p21_cuda_pool_free(d_out, stream);
        p21_cuda_pool_free(d_H, stream);
        release_float_view(v_tau, stream);
        release_float_view(v_vel, stream);
        release_float_view(v_tb, stream);
    };

    if (!v_tb.d_ptr || !v_vel.d_ptr || (use_ts_fluct && !v_tau.d_ptr) || !d_H || !d_out) {
        cleanup();
        return -1;
    }

    if (!P21_CUDA_OK(cudaMemcpyAsync(d_H, h_H_s_inv, (size_t)nz * sizeof(double),
                                     cudaMemcpyHostToDevice, stream))) {
        cleanup();
        return -1;
    }

    constexpr int kTpb = 256;
    if (!periodic) {
        int blocks = compute_grid_blocks(total, kTpb);
        dvdr_nonperiodic_kernel<<<blocks, kTpb, 0, stream>>>(
            v_tb.d_ptr, v_vel.d_ptr, v_tau.d_ptr, d_H, n_los, nz, dx_los_mpc, max_dvdr,
            use_ts_fluct, d_out);
        if (!P21_CUDA_OK(cudaGetLastError())) {
            cleanup();
            return -1;
        }
    } else {
        size_t hz = (size_t)(nz / 2 + 1);
        size_t n_freq = n_los * hz;
        d_freq = (cufftComplex *)p21_cuda_pool_alloc(n_freq * sizeof(cufftComplex), stream);
        d_vel_grad = (float *)p21_cuda_pool_alloc(box_bytes, stream);
        if (!d_freq || !d_vel_grad) {
            cleanup();
            return -1;
        }
        if (!p21_cuda_fft_r2c(v_vel.d_ptr, d_freq, nx, ny, nz, false, stream)) {
            cleanup();
            return -1;
        }
        mul_ikz_r2c_kernel<<<compute_grid_blocks(n_freq, kTpb), kTpb, 0, stream>>>(
            d_freq, nx, ny, nz, dx_los_mpc);
        if (!P21_CUDA_OK(cudaGetLastError())) {
            cleanup();
            return -1;
        }
        if (!p21_cuda_fft_c2r(d_freq, d_vel_grad, nx, ny, nz, false, stream)) {
            cleanup();
            return -1;
        }
        apply_velgrad_dvdr_kernel<<<compute_grid_blocks(total, kTpb), kTpb, 0, stream>>>(
            v_tb.d_ptr, d_vel_grad, v_tau.d_ptr, d_H, n_los, nz, max_dvdr, use_ts_fluct, d_out);
        if (!P21_CUDA_OK(cudaGetLastError())) {
            cleanup();
            return -1;
        }
    }

    bool ok =
        P21_CUDA_OK(cudaMemcpyAsync(h_out, d_out, box_bytes, cudaMemcpyDeviceToHost, stream)) &&
        P21_CUDA_OK(cudaStreamSynchronize(stream));
    if (ok) {
        p21_cuda_stats_add_d2h(box_bytes);
    }
    cleanup();
    return ok ? 0 : -1;
}

int p21_cuda_rsds_shift(const float *h_field, const float *h_disp, int n_los, int nz,
                        int n_rsd_subcells, int periodic, float *h_out) {
    if (!h_field || !h_disp || !h_out || n_los <= 0 || nz < 2 || n_rsd_subcells < 1) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    size_t per_los_bytes = (size_t)nz * (sizeof(float) * 3 + sizeof(double));
    size_t free_b = p21_cuda_free_mem_bytes();
    size_t budget = (free_b > 0) ? std::min<size_t>(free_b / 2, 1ULL << 30) : (256ULL << 20);
    int chunk_los = n_los;
    if ((size_t)n_los * per_los_bytes > budget) {
        chunk_los = (int)std::max<size_t>(1, budget / std::max<size_t>(per_los_bytes, 1));
    }

    size_t max_elems = (size_t)chunk_los * (size_t)nz;
    float *d_field = (float *)p21_cuda_pool_alloc(max_elems * sizeof(float), stream);
    float *d_disp = (float *)p21_cuda_pool_alloc(max_elems * sizeof(float), stream);
    double *d_acc = (double *)p21_cuda_pool_alloc(max_elems * sizeof(double), stream);
    float *d_out = (float *)p21_cuda_pool_alloc(max_elems * sizeof(float), stream);

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_out, stream);
        p21_cuda_pool_free(d_acc, stream);
        p21_cuda_pool_free(d_disp, stream);
        p21_cuda_pool_free(d_field, stream);
    };

    if (!d_field || !d_disp || !d_acc || !d_out) {
        cleanup();
        return -1;
    }

    constexpr int kTpb = 256;
    size_t total_d2h = 0;
    for (int c0 = 0; c0 < n_los; c0 += chunk_los) {
        int cur_los = std::min(chunk_los, n_los - c0);
        size_t offset = (size_t)c0 * (size_t)nz;
        size_t elems = (size_t)cur_los * (size_t)nz;
        size_t fbytes = elems * sizeof(float);
        size_t dbytes = elems * sizeof(double);

        if (!P21_CUDA_OK(cudaMemcpyAsync(d_field, h_field + offset, fbytes,
                                         cudaMemcpyHostToDevice, stream)) ||
            !P21_CUDA_OK(cudaMemcpyAsync(d_disp, h_disp + offset, fbytes,
                                         cudaMemcpyHostToDevice, stream)) ||
            !P21_CUDA_OK(cudaMemsetAsync(d_acc, 0, dbytes, stream))) {
            cleanup();
            return -1;
        }
        p21_cuda_stats_mut()->registry_h2d_bytes += (unsigned long long int)(2ULL * fbytes);

        int blocks = compute_grid_blocks(elems, kTpb);
        rsds_shift_kernel<<<blocks, kTpb, 0, stream>>>(d_field, d_disp, cur_los, nz,
                                                       n_rsd_subcells, periodic, d_acc);
        if (!P21_CUDA_OK(cudaGetLastError())) {
            cleanup();
            return -1;
        }

        double_to_float_kernel<<<blocks, kTpb, 0, stream>>>(d_acc, d_out, elems);
        if (!P21_CUDA_OK(cudaGetLastError()) ||
            !P21_CUDA_OK(cudaMemcpyAsync(h_out + offset, d_out, fbytes,
                                         cudaMemcpyDeviceToHost, stream))) {
            cleanup();
            return -1;
        }
        total_d2h += fbytes;
    }

    bool ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
    if (ok) {
        p21_cuda_stats_add_d2h(total_d2h);
    }
    cleanup();
    return ok ? 0 : -1;
}

int p21_cuda_rectilinear_slices(const float *h_box_a, const float *h_box_b, int nx, int ny, int nz,
                                int n_slices, const int *h_z_indices, const float *h_w_a,
                                const float *h_w_b, int kind, float *h_out) {
    if (!h_box_a || !h_box_b || nx <= 0 || ny <= 0 || nz <= 0) {
        return -1;
    }
    if (n_slices <= 0) {
        return 0;
    }
    if (!h_z_indices || !h_w_a || !h_w_b || !h_out || (kind != 0 && kind != 1)) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    size_t box_bytes = (size_t)nx * (size_t)ny * (size_t)nz * sizeof(float);
    size_t out_elems = (size_t)n_slices * (size_t)nx * (size_t)ny;
    size_t out_bytes = out_elems * sizeof(float);

    DeviceFloatView v_a = acquire_float_view(h_box_a, box_bytes, stream);
    DeviceFloatView v_b = acquire_float_view(h_box_b, box_bytes, stream);
    int *d_z_idx = (int *)p21_cuda_pool_alloc((size_t)n_slices * sizeof(int), stream);
    float *d_wa = (float *)p21_cuda_pool_alloc((size_t)n_slices * sizeof(float), stream);
    float *d_wb = (float *)p21_cuda_pool_alloc((size_t)n_slices * sizeof(float), stream);
    float *d_out = (float *)p21_cuda_pool_alloc(out_bytes, stream);

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_out, stream);
        p21_cuda_pool_free(d_wb, stream);
        p21_cuda_pool_free(d_wa, stream);
        p21_cuda_pool_free(d_z_idx, stream);
        release_float_view(v_b, stream);
        release_float_view(v_a, stream);
    };

    if (!v_a.d_ptr || !v_b.d_ptr || !d_z_idx || !d_wa || !d_wb || !d_out) {
        cleanup();
        return -1;
    }

    if (!P21_CUDA_OK(cudaMemcpyAsync(d_z_idx, h_z_indices, (size_t)n_slices * sizeof(int),
                                     cudaMemcpyHostToDevice, stream)) ||
        !P21_CUDA_OK(cudaMemcpyAsync(d_wa, h_w_a, (size_t)n_slices * sizeof(float),
                                     cudaMemcpyHostToDevice, stream)) ||
        !P21_CUDA_OK(cudaMemcpyAsync(d_wb, h_w_b, (size_t)n_slices * sizeof(float),
                                     cudaMemcpyHostToDevice, stream))) {
        cleanup();
        return -1;
    }

    constexpr int kTpb = 256;
    int blocks = compute_grid_blocks(out_elems, kTpb);
    rectilinear_slices_kernel<<<blocks, kTpb, 0, stream>>>(v_a.d_ptr, v_b.d_ptr, nx, ny, nz,
                                                           n_slices, d_z_idx, d_wa, d_wb, kind,
                                                           d_out);
    bool ok =
        P21_CUDA_OK(cudaGetLastError()) &&
        P21_CUDA_OK(cudaMemcpyAsync(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost, stream)) &&
        P21_CUDA_OK(cudaStreamSynchronize(stream));
    if (ok) {
        p21_cuda_stats_add_d2h(out_bytes);
    }
    cleanup();
    return ok ? 0 : -1;
}

int p21_cuda_power_spectrum_1d(const float *h_field_a, const float *h_field_b, int nx, int ny,
                               int nz, double lx, double ly, double lz, int n_bins,
                               const double *h_k_edges, int skip_dc, double *h_k_mean_out,
                               double *h_pk_out, double *h_delta_sq_out,
                               unsigned long long *h_counts_out) {
    if (!h_field_a || nx <= 0 || ny <= 0 || nz <= 0 || lx <= 0.0 || ly <= 0.0 || lz <= 0.0 ||
        n_bins <= 0 || !h_k_edges || !h_k_mean_out || !h_pk_out || !h_delta_sq_out ||
        !h_counts_out) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    size_t n_tot = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t box_bytes = n_tot * sizeof(float);
    if (p21_cuda_get_num_devices() >= 2 &&
        (box_bytes >= 512ULL * 1024 * 1024 || mgpu_ps_force_enabled())) {
        int mres = mgpu_power_spectrum_1d(h_field_a, h_field_b, nx, ny, nz, lx, ly, lz, n_bins,
                                          h_k_edges, skip_dc, h_k_mean_out, h_pk_out,
                                          h_delta_sq_out, h_counts_out);
        if (mres == 0) {
            return 0;
        }
    }
    size_t hz = (size_t)(nz / 2 + 1);
    size_t n_freq = (size_t)nx * (size_t)ny * hz;
    bool is_cross = (h_field_b != nullptr && h_field_b != h_field_a);

    DeviceFloatView v_a = acquire_float_view(h_field_a, box_bytes, stream);
    DeviceFloatView v_b{};
    if (is_cross) {
        v_b = acquire_float_view(h_field_b, box_bytes, stream);
    }

    cufftComplex *d_fa = (cufftComplex *)p21_cuda_pool_alloc(n_freq * sizeof(cufftComplex), stream);
    cufftComplex *d_fb =
        is_cross ? (cufftComplex *)p21_cuda_pool_alloc(n_freq * sizeof(cufftComplex), stream)
                 : nullptr;
    double *d_edges = (double *)p21_cuda_pool_alloc((size_t)(n_bins + 1) * sizeof(double), stream);
    double *d_k_sum = (double *)p21_cuda_pool_alloc((size_t)n_bins * sizeof(double), stream);
    double *d_pk_sum = (double *)p21_cuda_pool_alloc((size_t)n_bins * sizeof(double), stream);
    unsigned long long *d_counts = (unsigned long long *)p21_cuda_pool_alloc(
        (size_t)n_bins * sizeof(unsigned long long), stream);

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_counts, stream);
        p21_cuda_pool_free(d_pk_sum, stream);
        p21_cuda_pool_free(d_k_sum, stream);
        p21_cuda_pool_free(d_edges, stream);
        p21_cuda_pool_free(d_fb, stream);
        p21_cuda_pool_free(d_fa, stream);
        release_float_view(v_b, stream);
        release_float_view(v_a, stream);
    };

    if (!v_a.d_ptr || (is_cross && !v_b.d_ptr) || !d_fa || (is_cross && !d_fb) || !d_edges ||
        !d_k_sum || !d_pk_sum || !d_counts) {
        cleanup();
        return -1;
    }

    if (!P21_CUDA_OK(cudaMemcpyAsync(d_edges, h_k_edges, (size_t)(n_bins + 1) * sizeof(double),
                                     cudaMemcpyHostToDevice, stream)) ||
        !P21_CUDA_OK(cudaMemsetAsync(d_k_sum, 0, (size_t)n_bins * sizeof(double), stream)) ||
        !P21_CUDA_OK(cudaMemsetAsync(d_pk_sum, 0, (size_t)n_bins * sizeof(double), stream)) ||
        !P21_CUDA_OK(cudaMemsetAsync(d_counts, 0, (size_t)n_bins * sizeof(unsigned long long),
                                     stream))) {
        cleanup();
        return -1;
    }

    if (!p21_cuda_fft_r2c(v_a.d_ptr, d_fa, nx, ny, nz, false, stream)) {
        cleanup();
        return -1;
    }
    if (is_cross && !p21_cuda_fft_r2c(v_b.d_ptr, d_fb, nx, ny, nz, false, stream)) {
        cleanup();
        return -1;
    }

    double vol = lx * ly * lz;
    double norm = vol / ((double)n_tot * (double)n_tot);
    constexpr int kTpb = 256;
    int blocks = compute_grid_blocks(n_freq, kTpb);
    ps_1d_bin_kernel<<<blocks, kTpb, 0, stream>>>(d_fa, d_fb, nx, ny, nz, lx, ly, lz, norm,
                                                  skip_dc, n_bins, d_edges, d_k_sum, d_pk_sum,
                                                  d_counts);

    std::vector<double> h_k_sum((size_t)n_bins, 0.0);
    std::vector<double> h_pk_sum((size_t)n_bins, 0.0);
    bool ok = P21_CUDA_OK(cudaGetLastError()) &&
              P21_CUDA_OK(cudaMemcpyAsync(h_k_sum.data(), d_k_sum, (size_t)n_bins * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaMemcpyAsync(h_pk_sum.data(), d_pk_sum, (size_t)n_bins * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaMemcpyAsync(h_counts_out, d_counts,
                                          (size_t)n_bins * sizeof(unsigned long long),
                                          cudaMemcpyDeviceToHost, stream)) &&
              P21_CUDA_OK(cudaStreamSynchronize(stream));
    cleanup();
    if (!ok) {
        return -1;
    }

    const double inv_2pi2 = 1.0 / (2.0 * M_PI * M_PI);
    for (int b = 0; b < n_bins; ++b) {
        unsigned long long cnt = h_counts_out[b];
        if (cnt > 0ULL) {
            double km = h_k_sum[b] / (double)cnt;
            double pk = h_pk_sum[b] / (double)cnt;
            h_k_mean_out[b] = km;
            h_pk_out[b] = pk;
            h_delta_sq_out[b] = km * km * km * pk * inv_2pi2;
        } else {
            h_k_mean_out[b] = NAN;
            h_pk_out[b] = NAN;
            h_delta_sq_out[b] = NAN;
        }
    }
    return 0;
}

int p21_cuda_power_spectrum_2d(const float *h_field_a, const float *h_field_b, int nx, int ny,
                               int nz, double lx, double ly, double lz, int los_axis,
                               int n_kperp, const double *h_kperp_edges, int n_kpara,
                               const double *h_kpara_edges, int skip_dc,
                               double *h_kperp_mean_out, double *h_kpara_mean_out,
                               double *h_pk_out, unsigned long long *h_counts_out) {
    if (!h_field_a || nx <= 0 || ny <= 0 || nz <= 0 || lx <= 0.0 || ly <= 0.0 || lz <= 0.0 ||
        los_axis < 0 || los_axis > 2 || n_kperp <= 0 || n_kpara <= 0 || !h_kperp_edges ||
        !h_kpara_edges || !h_kperp_mean_out || !h_kpara_mean_out || !h_pk_out || !h_counts_out) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    size_t n_tot = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t box_bytes = n_tot * sizeof(float);
    if (p21_cuda_get_num_devices() >= 2 &&
        (box_bytes >= 512ULL * 1024 * 1024 || mgpu_ps_force_enabled())) {
        int mres = mgpu_power_spectrum_2d(h_field_a, h_field_b, nx, ny, nz, lx, ly, lz, los_axis,
                                          n_kperp, h_kperp_edges, n_kpara, h_kpara_edges, skip_dc,
                                          h_kperp_mean_out, h_kpara_mean_out, h_pk_out,
                                          h_counts_out);
        if (mres == 0) {
            return 0;
        }
    }
    size_t hz = (size_t)(nz / 2 + 1);
    size_t n_freq = (size_t)nx * (size_t)ny * hz;
    size_t n_bins = (size_t)n_kperp * (size_t)n_kpara;
    bool is_cross = (h_field_b != nullptr && h_field_b != h_field_a);

    DeviceFloatView v_a = acquire_float_view(h_field_a, box_bytes, stream);
    DeviceFloatView v_b{};
    if (is_cross) {
        v_b = acquire_float_view(h_field_b, box_bytes, stream);
    }

    cufftComplex *d_fa = (cufftComplex *)p21_cuda_pool_alloc(n_freq * sizeof(cufftComplex), stream);
    cufftComplex *d_fb =
        is_cross ? (cufftComplex *)p21_cuda_pool_alloc(n_freq * sizeof(cufftComplex), stream)
                 : nullptr;
    double *d_kp_edges =
        (double *)p21_cuda_pool_alloc((size_t)(n_kperp + 1) * sizeof(double), stream);
    double *d_kz_edges =
        (double *)p21_cuda_pool_alloc((size_t)(n_kpara + 1) * sizeof(double), stream);
    double *d_kp_sum = (double *)p21_cuda_pool_alloc(n_bins * sizeof(double), stream);
    double *d_kz_sum = (double *)p21_cuda_pool_alloc(n_bins * sizeof(double), stream);
    double *d_pk_sum = (double *)p21_cuda_pool_alloc(n_bins * sizeof(double), stream);
    unsigned long long *d_counts =
        (unsigned long long *)p21_cuda_pool_alloc(n_bins * sizeof(unsigned long long), stream);

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_counts, stream);
        p21_cuda_pool_free(d_pk_sum, stream);
        p21_cuda_pool_free(d_kz_sum, stream);
        p21_cuda_pool_free(d_kp_sum, stream);
        p21_cuda_pool_free(d_kz_edges, stream);
        p21_cuda_pool_free(d_kp_edges, stream);
        p21_cuda_pool_free(d_fb, stream);
        p21_cuda_pool_free(d_fa, stream);
        release_float_view(v_b, stream);
        release_float_view(v_a, stream);
    };

    if (!v_a.d_ptr || (is_cross && !v_b.d_ptr) || !d_fa || (is_cross && !d_fb) || !d_kp_edges ||
        !d_kz_edges || !d_kp_sum || !d_kz_sum || !d_pk_sum || !d_counts) {
        cleanup();
        return -1;
    }

    if (!P21_CUDA_OK(cudaMemcpyAsync(d_kp_edges, h_kperp_edges,
                                     (size_t)(n_kperp + 1) * sizeof(double),
                                     cudaMemcpyHostToDevice, stream)) ||
        !P21_CUDA_OK(cudaMemcpyAsync(d_kz_edges, h_kpara_edges,
                                     (size_t)(n_kpara + 1) * sizeof(double),
                                     cudaMemcpyHostToDevice, stream)) ||
        !P21_CUDA_OK(cudaMemsetAsync(d_kp_sum, 0, n_bins * sizeof(double), stream)) ||
        !P21_CUDA_OK(cudaMemsetAsync(d_kz_sum, 0, n_bins * sizeof(double), stream)) ||
        !P21_CUDA_OK(cudaMemsetAsync(d_pk_sum, 0, n_bins * sizeof(double), stream)) ||
        !P21_CUDA_OK(
            cudaMemsetAsync(d_counts, 0, n_bins * sizeof(unsigned long long), stream))) {
        cleanup();
        return -1;
    }

    if (!p21_cuda_fft_r2c(v_a.d_ptr, d_fa, nx, ny, nz, false, stream)) {
        cleanup();
        return -1;
    }
    if (is_cross && !p21_cuda_fft_r2c(v_b.d_ptr, d_fb, nx, ny, nz, false, stream)) {
        cleanup();
        return -1;
    }

    double vol = lx * ly * lz;
    double norm = vol / ((double)n_tot * (double)n_tot);
    constexpr int kTpb = 256;
    int blocks = compute_grid_blocks(n_freq, kTpb);
    ps_2d_bin_kernel<<<blocks, kTpb, 0, stream>>>(d_fa, d_fb, nx, ny, nz, lx, ly, lz, los_axis,
                                                  norm, skip_dc, n_kperp, d_kp_edges, n_kpara,
                                                  d_kz_edges, d_kp_sum, d_kz_sum, d_pk_sum,
                                                  d_counts);

    std::vector<double> h_kp_sum(n_bins, 0.0);
    std::vector<double> h_kz_sum(n_bins, 0.0);
    std::vector<double> h_pk_sum(n_bins, 0.0);
    bool ok =
        P21_CUDA_OK(cudaGetLastError()) &&
        P21_CUDA_OK(cudaMemcpyAsync(h_kp_sum.data(), d_kp_sum, n_bins * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream)) &&
        P21_CUDA_OK(cudaMemcpyAsync(h_kz_sum.data(), d_kz_sum, n_bins * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream)) &&
        P21_CUDA_OK(cudaMemcpyAsync(h_pk_sum.data(), d_pk_sum, n_bins * sizeof(double),
                                    cudaMemcpyDeviceToHost, stream)) &&
        P21_CUDA_OK(cudaMemcpyAsync(h_counts_out, d_counts, n_bins * sizeof(unsigned long long),
                                    cudaMemcpyDeviceToHost, stream)) &&
        P21_CUDA_OK(cudaStreamSynchronize(stream));
    cleanup();
    if (!ok) {
        return -1;
    }

    for (size_t b = 0; b < n_bins; ++b) {
        unsigned long long cnt = h_counts_out[b];
        if (cnt > 0ULL) {
            double c = (double)cnt;
            h_kperp_mean_out[b] = h_kp_sum[b] / c;
            h_kpara_mean_out[b] = h_kz_sum[b] / c;
            h_pk_out[b] = h_pk_sum[b] / c;
        } else {
            h_kperp_mean_out[b] = NAN;
            h_kpara_mean_out[b] = NAN;
            h_pk_out[b] = NAN;
        }
    }
    return 0;
}

int p21_cuda_angular_slice(const float *h_box_a, const float *h_box_b, int nx, int ny, int nz,
                           int n_coords, const double *h_coords_xyz, float w_a, float w_b,
                           int order, int kind, float *h_out) {
    if (!h_box_a || !h_box_b || !h_coords_xyz || !h_out || nx <= 0 || ny <= 0 || nz <= 0 ||
        n_coords <= 0 || (order != 0 && order != 1) || (kind != 0 && kind != 1)) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    size_t box_bytes = (size_t)nx * (size_t)ny * (size_t)nz * sizeof(float);
    size_t coords_bytes = 3ULL * (size_t)n_coords * sizeof(double);
    size_t out_bytes = (size_t)n_coords * sizeof(float);

    DeviceFloatView v_a = acquire_float_view(h_box_a, box_bytes, stream);
    DeviceFloatView v_b = acquire_float_view(h_box_b, box_bytes, stream);
    double *d_coords = (double *)p21_cuda_pool_alloc(coords_bytes, stream);
    float *d_out = (float *)p21_cuda_pool_alloc(out_bytes, stream);

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_out, stream);
        p21_cuda_pool_free(d_coords, stream);
        release_float_view(v_b, stream);
        release_float_view(v_a, stream);
    };

    if (!v_a.d_ptr || !v_b.d_ptr || !d_coords || !d_out) {
        cleanup();
        return -1;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(d_coords, h_coords_xyz, coords_bytes,
                                     cudaMemcpyHostToDevice, stream))) {
        cleanup();
        return -1;
    }

    constexpr int kTpb = 256;
    int blocks = compute_grid_blocks((size_t)n_coords, kTpb);
    angular_slice_kernel<<<blocks, kTpb, 0, stream>>>(
        v_a.d_ptr, v_b.d_ptr, nx, ny, nz, n_coords, d_coords, w_a, w_b, order, kind, d_out);
    bool ok =
        P21_CUDA_OK(cudaGetLastError()) &&
        P21_CUDA_OK(cudaMemcpyAsync(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost, stream)) &&
        P21_CUDA_OK(cudaStreamSynchronize(stream));
    if (ok) {
        p21_cuda_stats_add_d2h(out_bytes);
    }
    cleanup();
    return ok ? 0 : -1;
}

int p21_cuda_angular_los_velocity(const float *h_vxa, const float *h_vya, const float *h_vza,
                                  const float *h_vxb, const float *h_vyb, const float *h_vzb,
                                  int nx, int ny, int nz, int n_coords,
                                  const double *h_coords_xyz, const double *h_unit_xyz, float w_a,
                                  float w_b, int order, float *h_out) {
    if (!h_vxa || !h_vya || !h_vza || !h_vxb || !h_vyb || !h_vzb || !h_coords_xyz ||
        !h_unit_xyz || !h_out || nx <= 0 || ny <= 0 || nz <= 0 || n_coords <= 0 ||
        (order != 0 && order != 1)) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    size_t box_bytes = (size_t)nx * (size_t)ny * (size_t)nz * sizeof(float);
    size_t vec_bytes = 3ULL * (size_t)n_coords * sizeof(double);
    size_t out_bytes = (size_t)n_coords * sizeof(float);

    DeviceFloatView vxa = acquire_float_view(h_vxa, box_bytes, stream);
    DeviceFloatView vya = acquire_float_view(h_vya, box_bytes, stream);
    DeviceFloatView vza = acquire_float_view(h_vza, box_bytes, stream);
    DeviceFloatView vxb = acquire_float_view(h_vxb, box_bytes, stream);
    DeviceFloatView vyb = acquire_float_view(h_vyb, box_bytes, stream);
    DeviceFloatView vzb = acquire_float_view(h_vzb, box_bytes, stream);
    double *d_coords = (double *)p21_cuda_pool_alloc(vec_bytes, stream);
    double *d_unit = (double *)p21_cuda_pool_alloc(vec_bytes, stream);
    float *d_out = (float *)p21_cuda_pool_alloc(out_bytes, stream);

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_out, stream);
        p21_cuda_pool_free(d_unit, stream);
        p21_cuda_pool_free(d_coords, stream);
        release_float_view(vzb, stream);
        release_float_view(vyb, stream);
        release_float_view(vxb, stream);
        release_float_view(vza, stream);
        release_float_view(vya, stream);
        release_float_view(vxa, stream);
    };

    if (!vxa.d_ptr || !vya.d_ptr || !vza.d_ptr || !vxb.d_ptr || !vyb.d_ptr || !vzb.d_ptr ||
        !d_coords || !d_unit || !d_out) {
        cleanup();
        return -1;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(d_coords, h_coords_xyz, vec_bytes, cudaMemcpyHostToDevice,
                                     stream)) ||
        !P21_CUDA_OK(
            cudaMemcpyAsync(d_unit, h_unit_xyz, vec_bytes, cudaMemcpyHostToDevice, stream))) {
        cleanup();
        return -1;
    }

    constexpr int kTpb = 256;
    int blocks = compute_grid_blocks((size_t)n_coords, kTpb);
    angular_los_vel_kernel<<<blocks, kTpb, 0, stream>>>(
        vxa.d_ptr, vya.d_ptr, vza.d_ptr, vxb.d_ptr, vyb.d_ptr, vzb.d_ptr, nx, ny, nz, n_coords,
        d_coords, d_unit, w_a, w_b, order, d_out);
    bool ok =
        P21_CUDA_OK(cudaGetLastError()) &&
        P21_CUDA_OK(cudaMemcpyAsync(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost, stream)) &&
        P21_CUDA_OK(cudaStreamSynchronize(stream));
    if (ok) {
        p21_cuda_stats_add_d2h(out_bytes);
    }
    cleanup();
    return ok ? 0 : -1;
}

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
    int blocks = compute_grid_blocks(n_freq, kTpb);
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
        int blocks = compute_grid_blocks(n_pairs, kTpb);
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
    int blocks = compute_grid_blocks(n_elems, kTpb);
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
    int blocks = compute_grid_blocks(n_freq, kTpb);
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
    int blocks = compute_grid_blocks(n_elems, kTpb);
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

/* ========================================================================= */
/* Phase 10 (v0.9.0-dev): 3D Bispectrum, MFP BSD & 3D Minkowski Functionals   */
/* ========================================================================= */

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
    int freq_blocks = compute_grid_blocks(n_freq, kTpb);
    int real_blocks = compute_grid_blocks(n_tot, kTpb);
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
    int real_blocks = compute_grid_blocks(n_tot, kTpb);
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

    int ray_blocks = compute_grid_blocks((size_t)n_rays, kTpb);
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
    int real_blocks = compute_grid_blocks(n_tot, kTpb);
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

__global__ void cic_deposit_3d_kernel(const float *__restrict__ pos_xyz,
                                      const float *__restrict__ weights,
                                      unsigned long long n_particles, int nx, int ny, int nz,
                                      double inv_dx, double inv_dy, double inv_dz,
                                      float *__restrict__ grid) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; idx < n_particles; idx += stride) {
        double ux = fmod((double)pos_xyz[3 * idx + 0] * inv_dx, (double)nx);
        double uy = fmod((double)pos_xyz[3 * idx + 1] * inv_dy, (double)ny);
        double uz = fmod((double)pos_xyz[3 * idx + 2] * inv_dz, (double)nz);
        if (ux < 0.0) ux += (double)nx;
        if (uy < 0.0) uy += (double)ny;
        if (uz < 0.0) uz += (double)nz;

        double fx_floor = floor(ux);
        double fy_floor = floor(uy);
        double fz_floor = floor(uz);
        int ix0 = (((int)fx_floor) % nx + nx) % nx;
        int iy0 = (((int)fy_floor) % ny + ny) % ny;
        int iz0 = (((int)fz_floor) % nz + nz) % nz;
        int ix1 = (ix0 + 1) % nx;
        int iy1 = (iy0 + 1) % ny;
        int iz1 = (iz0 + 1) % nz;

        float fx = (float)(ux - fx_floor);
        float fy = (float)(uy - fy_floor);
        float fz = (float)(uz - fz_floor);
        float wx0 = 1.0f - fx;
        float wy0 = 1.0f - fy;
        float wz0 = 1.0f - fz;
        float w = weights ? weights[idx] : 1.0f;

        size_t row_x0 = (size_t)ix0 * (size_t)ny;
        size_t row_x1 = (size_t)ix1 * (size_t)ny;
        float w00 = w * wx0 * wy0;
        float w01 = w * wx0 * fy;
        float w10 = w * fx * wy0;
        float w11 = w * fx * fy;

        atomicAdd(&grid[(row_x0 + (size_t)iy0) * (size_t)nz + (size_t)iz0], w00 * wz0);
        atomicAdd(&grid[(row_x0 + (size_t)iy0) * (size_t)nz + (size_t)iz1], w00 * fz);
        atomicAdd(&grid[(row_x0 + (size_t)iy1) * (size_t)nz + (size_t)iz0], w01 * wz0);
        atomicAdd(&grid[(row_x0 + (size_t)iy1) * (size_t)nz + (size_t)iz1], w01 * fz);
        atomicAdd(&grid[(row_x1 + (size_t)iy0) * (size_t)nz + (size_t)iz0], w10 * wz0);
        atomicAdd(&grid[(row_x1 + (size_t)iy0) * (size_t)nz + (size_t)iz1], w10 * fz);
        atomicAdd(&grid[(row_x1 + (size_t)iy1) * (size_t)nz + (size_t)iz0], w11 * wz0);
        atomicAdd(&grid[(row_x1 + (size_t)iy1) * (size_t)nz + (size_t)iz1], w11 * fz);
    }
}

__global__ void cic_to_overdensity_kernel(float *__restrict__ grid, size_t n_tot,
                                          double inv_mean_w) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; idx < n_tot; idx += stride) {
        grid[idx] = (float)((double)grid[idx] * inv_mean_w - 1.0);
    }
}

int p21_cuda_cic_deposit_3d(const float *h_pos_xyz, const float *h_weights,
                            unsigned long long n_particles, int nx, int ny, int nz, double lx,
                            double ly, double lz, int overdensity, float *h_grid_out) {
    if (!h_grid_out || nx <= 0 || ny <= 0 || nz <= 0 || lx <= 0.0 || ly <= 0.0 || lz <= 0.0) {
        return -1;
    }
    size_t n_tot = (size_t)nx * (size_t)ny * (size_t)nz;
    size_t grid_bytes = n_tot * sizeof(float);
    if (n_particles == 0ULL) {
        memset(h_grid_out, 0, grid_bytes);
        return 0;
    }
    if (!h_pos_xyz) {
        return -1;
    }
    if (!p21_cuda_ensure_init()) {
        return -1;
    }

    cudaStream_t stream = p21_cuda_stream();
    size_t pos_bytes = (size_t)n_particles * 3ULL * sizeof(float);
    size_t w_bytes = (size_t)n_particles * sizeof(float);

    float *d_pos = (float *)p21_cuda_pool_alloc(pos_bytes, stream);
    float *d_weights = h_weights ? (float *)p21_cuda_pool_alloc(w_bytes, stream) : nullptr;
    float *d_grid = (float *)p21_cuda_pool_alloc(grid_bytes, stream);

    auto cleanup = [&]() {
        p21_cuda_pool_free(d_grid, stream);
        p21_cuda_pool_free(d_weights, stream);
        p21_cuda_pool_free(d_pos, stream);
    };

    if (!d_pos || !d_grid || (h_weights && !d_weights)) {
        cleanup();
        return -1;
    }

    bool ok = P21_CUDA_OK(cudaMemsetAsync(d_grid, 0, grid_bytes, stream)) &&
              P21_CUDA_OK(cudaMemcpyAsync(d_pos, h_pos_xyz, pos_bytes, cudaMemcpyHostToDevice,
                                          stream));
    if (ok && h_weights) {
        ok = P21_CUDA_OK(
            cudaMemcpyAsync(d_weights, h_weights, w_bytes, cudaMemcpyHostToDevice, stream));
    }

    constexpr int kTpb = 256;
    if (ok) {
        int part_blocks = compute_grid_blocks((size_t)n_particles, kTpb);
        cic_deposit_3d_kernel<<<part_blocks, kTpb, 0, stream>>>(
            d_pos, d_weights, n_particles, nx, ny, nz, (double)nx / lx, (double)ny / ly,
            (double)nz / lz, d_grid);
        ok = P21_CUDA_OK(cudaGetLastError());
    }

    if (ok && overdensity != 0) {
        double total_w = (double)n_particles;
        if (h_weights) {
            total_w = 0.0;
            for (unsigned long long i = 0; i < n_particles; ++i) {
                total_w += (double)h_weights[i];
            }
        }
        double mean_w = total_w / (double)n_tot;
        if (mean_w > 0.0) {
            int grid_blocks = compute_grid_blocks(n_tot, kTpb);
            cic_to_overdensity_kernel<<<grid_blocks, kTpb, 0, stream>>>(d_grid, n_tot,
                                                                        1.0 / mean_w);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
    }

    if (ok) {
        ok = P21_CUDA_OK(
                 cudaMemcpyAsync(h_grid_out, d_grid, grid_bytes, cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }

    cleanup();
    return ok ? 0 : -1;
}

}  // extern "C"
