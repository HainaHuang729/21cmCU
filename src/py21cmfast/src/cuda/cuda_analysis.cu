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

}  // extern "C"
