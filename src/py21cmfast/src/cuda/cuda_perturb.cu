#include <algorithm>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "cuda_common.cuh"

namespace {

__device__ static inline void p21_atomic_add_accum(double *addr, double val) {
    p21_atomic_add_double(addr, val);
}

__device__ static inline void p21_atomic_add_accum(float *addr, double val) {
    atomicAdd(addr, (float)val);
}

__global__ void linear_density_kernel(const float *dens_in, float *out_real, size_t n,
                                      double growth_factor) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out_real[idx] = (float)(growth_factor * (double)dens_in[idx]);
    }
}

template <typename AccumT>
__global__ void move_grid_masses_slab_kernel(
    const float *hires_dens, const float *vel_x, const float *vel_y, const float *vel_z,
    const float *vel_x_2lpt, const float *vel_y_2lpt, const float *vel_z_2lpt, int x_start,
    int slab_nx, int hi_y, int hi_z, int vel_x_dim, int vel_y_dim, int vel_z_dim, int out_x,
    int out_y, int out_z, double dim_ratio_vel, double dim_ratio_out, double vdf0, double vdf1,
    double vdf2, double vdf2_0, double vdf2_1, double vdf2_2, double init_growth, bool use_2lpt,
    bool vel_is_slab, AccumT *resampled_out) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t slab_total = (size_t)slab_nx * hi_y * hi_z;
    if (idx >= slab_total) {
        return;
    }

    int k = (int)(idx % hi_z);
    size_t q = idx / hi_z;
    int j = (int)(q % hi_y);
    int i = x_start + (int)(q / hi_y);

    size_t vel_idx;
    if (vel_is_slab) {
        vel_idx = idx;
    } else {
        int vx_i = p21_wrap_coord(p21_resample_coord(i, dim_ratio_vel), vel_x_dim);
        int vx_j = p21_wrap_coord(p21_resample_coord(j, dim_ratio_vel), vel_y_dim);
        int vx_k = p21_wrap_coord(p21_resample_coord(k, dim_ratio_vel), vel_z_dim);
        vel_idx = p21_grid_idx(vx_i, vx_j, vx_k, vel_y_dim, vel_z_dim);
    }

    double pos0 = (double)i + (double)vel_x[vel_idx] * vdf0;
    double pos1 = (double)j + (double)vel_y[vel_idx] * vdf1;
    double pos2 = (double)k + (double)vel_z[vel_idx] * vdf2;
    if (use_2lpt) {
        pos0 -= (double)vel_x_2lpt[vel_idx] * vdf2_0;
        pos1 -= (double)vel_y_2lpt[vel_idx] * vdf2_1;
        pos2 -= (double)vel_z_2lpt[vel_idx] * vdf2_2;
    }
    pos0 *= dim_ratio_out;
    pos1 *= dim_ratio_out;
    pos2 *= dim_ratio_out;

    int ip0 = (int)floor(pos0);
    int ip1 = (int)floor(pos1);
    int ip2 = (int)floor(pos2);
    double d0 = pos0 - (double)ip0;
    double d1 = pos1 - (double)ip1;
    double d2 = pos2 - (double)ip2;

    int ipp0 = p21_wrap_coord(ip0 + 1, out_x);
    int ipp1 = p21_wrap_coord(ip1 + 1, out_y);
    int ipp2 = p21_wrap_coord(ip2 + 1, out_z);
    ip0 = p21_wrap_coord(ip0, out_x);
    ip1 = p21_wrap_coord(ip1, out_y);
    ip2 = p21_wrap_coord(ip2, out_z);

    double curr_dens = 1.0 + (double)hires_dens[idx] * init_growth;
    double om0 = 1.0 - d0;
    double om1 = 1.0 - d1;
    double om2 = 1.0 - d2;

    p21_atomic_add_accum(&resampled_out[p21_grid_idx(ip0, ip1, ip2, out_y, out_z)],
                         curr_dens * om0 * om1 * om2);
    p21_atomic_add_accum(&resampled_out[p21_grid_idx(ipp0, ip1, ip2, out_y, out_z)],
                         curr_dens * d0 * om1 * om2);
    p21_atomic_add_accum(&resampled_out[p21_grid_idx(ip0, ipp1, ip2, out_y, out_z)],
                         curr_dens * om0 * d1 * om2);
    p21_atomic_add_accum(&resampled_out[p21_grid_idx(ipp0, ipp1, ip2, out_y, out_z)],
                         curr_dens * d0 * d1 * om2);
    p21_atomic_add_accum(&resampled_out[p21_grid_idx(ip0, ip1, ipp2, out_y, out_z)],
                         curr_dens * om0 * om1 * d2);
    p21_atomic_add_accum(&resampled_out[p21_grid_idx(ipp0, ip1, ipp2, out_y, out_z)],
                         curr_dens * d0 * om1 * d2);
    p21_atomic_add_accum(&resampled_out[p21_grid_idx(ip0, ipp1, ipp2, out_y, out_z)],
                         curr_dens * om0 * d1 * d2);
    p21_atomic_add_accum(&resampled_out[p21_grid_idx(ipp0, ipp1, ipp2, out_y, out_z)],
                         curr_dens * d0 * d1 * d2);
}

__global__ void double_to_float_kernel(const double *src, float *dst, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = (float)src[idx];
    }
}

__global__ void resample_box_kernel(const float *hi_real, int hi_y, int hi_z, float *lo_real,
                                    int lo_x, int lo_y, int lo_z, double dim_ratio, float denom) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)lo_x * lo_y * lo_z;
    if (idx >= total) {
        return;
    }
    int k = (int)(idx % lo_z);
    size_t q = idx / lo_z;
    int j = (int)(q % lo_y);
    int i = (int)(q / lo_y);
    int ri = p21_resample_coord(i, dim_ratio);
    int rj = p21_resample_coord(j, dim_ratio);
    int rk = p21_resample_coord(k, dim_ratio);
    lo_real[idx] = hi_real[p21_grid_idx(ri, rj, rk, hi_y, hi_z)] / denom;
}

__global__ void normalise_delta_kernel(float *grid, size_t n, double mass_factor) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = grid[idx];
        v = (float)((double)v * mass_factor);
        v -= 1.0f;
        grid[idx] = v;
    }
}

__global__ void normalize_and_clip_density_kernel(float *grid, size_t n, float norm_denom,
                                                  float min_val) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = grid[idx] / norm_denom;
        if (v < min_val) {
            v = min_val;
        }
        grid[idx] = v;
    }
}

__global__ void perturb_velocity_kspace_kernel(const cufftComplex *density_saved, cufftComplex *out,
                                               int bx, int by, int bz, double len_x, double len_y,
                                               double len_z, int axis, double dDdt_over_D,
                                               double n_r_pixels) {
    int hz = bz / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)bx * by * hz;
    if (idx >= total) {
        return;
    }
    if (idx == 0) {
        out[0] = make_float2(0.0f, 0.0f);
        return;
    }

    int n_z = (int)(idx % hz);
    size_t q = idx / hz;
    int n_y = (int)(q % by);
    int n_x = (int)(q / by);

    /* Match PerturbedField.c lines 287, 326-346 float k_x, k_y, k_z, k_sq. */
    float kx = (float)p21_index_to_k(n_x, len_x, bx);
    float ky = (float)p21_index_to_k(n_y, len_y, by);
    float kz = (float)p21_index_to_k(n_z, len_z, bz);
    float k_sq = kx * kx + ky * ky + kz * kz;
    float k_axis = (axis == 0) ? kx : ((axis == 1) ? ky : kz);
    double factor = dDdt_over_D * (double)k_axis / (double)k_sq / n_r_pixels;

    cufftComplex v = density_saved[idx];
    out[idx] = make_float2((float)(-(double)v.y * factor), (float)((double)v.x * factor));
}

bool upload_and_register_d2h(float *host_ptr, const float *d_src, size_t nbytes,
                             cudaStream_t stream) {
    if (!host_ptr || !d_src || nbytes == 0) {
        return false;
    }
    if (!P21_CUDA_OK(cudaMemcpyAsync(host_ptr, d_src, nbytes, cudaMemcpyDeviceToHost, stream))) {
        return false;
    }
    p21_cuda_stats_add_d2h(nbytes);
    if (!P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        return false;
    }
    if (!p21_cuda_registry_put_copy(host_ptr, d_src, nbytes, stream)) {
        p21_cuda_invalidate_host_ptr(host_ptr);
    }
    return true;
}

int read_perturb_slab_x_env(void) {
    const char *env = getenv("P21_CUDA_PERTURB_SLAB_X");
    if (!env || !*env) {
        return 0;
    }
    char *end = nullptr;
    unsigned long long val = strtoull(env, &end, 10);
    if (end != env && *end == '\0' && val > 0ULL) {
        return (int)val;
    }
    return 0;
}

}  // namespace

extern "C" int p21_cuda_compute_perturbed_field(float redshift, InitialConditions *boxes,
                                                PerturbedField *perturbed_field) {
    if (!boxes || !perturbed_field || !p21_cuda_get_enabled() || !p21_cuda_ensure_init()) {
        return -1;
    }

    const int nx = simulation_options_global->DIM;
    const int ny = simulation_options_global->DIM;
    const int nz = D_PARA;
    const int lx = simulation_options_global->HII_DIM;
    const int ly = simulation_options_global->HII_DIM;
    const int lz = HII_D_PARA;

    if (nx <= 1 || ny <= 1 || nz <= 1 || lx <= 1 || ly <= 1 || lz <= 1) {
        return -1;
    }

    const int alg = matter_options_global->PERTURB_ALGORITHM;
    if (alg != PERTURB_ALGORITHM_LINEAR && alg != PERTURB_ALGORITHM_ZELDOVICH &&
        alg != PERTURB_ALGORITHM_2LPT) {
        return -1;
    }

    const bool perturb_hires = matter_options_global->PERTURB_ON_HIGH_RES;
    const int bx = perturb_hires ? nx : lx;
    const int by = perturb_hires ? ny : ly;
    const int bz = perturb_hires ? nz : lz;

    const size_t hi_pixels = (size_t)nx * ny * nz;
    const size_t lo_pixels = (size_t)lx * ly * lz;
    const size_t box_pixels = (size_t)bx * by * bz;
    const size_t lo_k_pixels = (size_t)lx * ly * (lz / 2 + 1);
    const size_t box_k_pixels = (size_t)bx * by * (bz / 2 + 1);

    const size_t hi_bytes = hi_pixels * sizeof(float);
    const size_t lo_bytes = lo_pixels * sizeof(float);
    const size_t box_bytes = box_pixels * sizeof(float);
    const size_t lo_k_bytes = lo_k_pixels * sizeof(cufftComplex);
    const size_t box_k_bytes = box_k_pixels * sizeof(cufftComplex);

    const double box_len = simulation_options_global->BOX_LEN;
    const double ncf = simulation_options_global->NON_CUBIC_FACTOR;
    const double len_z = ncf * box_len;
    const float tophat_R = (float)(physconst.l_factor * box_len / (lx + 0.0));
    const float smooth_R =
        (float)(simulation_options_global->DENSITY_SMOOTH_RADIUS * box_len / (float)lx);
    const int tpb = 256;

    cudaStream_t stream = p21_cuda_stream();
    if (box_bytes >= 256ULL * 1024 * 1024) {
        size_t total_mem = p21_cuda_total_mem_bytes();
        size_t target_free = box_bytes * 4;
        if (total_mem > 0) {
            target_free = std::min<size_t>(target_free, (total_mem * 3) / 4);
        }
        p21_cuda_registry_evict_to_free(target_free, stream);
    }

    float *d_fft_real = (float *)p21_cuda_pool_alloc(box_bytes, stream);
    float *d_real_lo = nullptr;
    cufftComplex *d_density_saved = nullptr;
    cufftComplex *d_k_lo = nullptr;
    cufftComplex *d_k_work = nullptr;

    bool ok = (d_fft_real != nullptr);

    /* 1. make_density_grid */
    if (ok && alg == PERTURB_ALGORITHM_LINEAR) {
        const float *h_dens = perturb_hires ? boxes->hires_density : boxes->lowres_density;
        double gf = dicke(redshift);
        int g = (int)((box_pixels + tpb - 1) / tpb);
        if (box_bytes > p21_cuda_registry_max_entry_bytes()) {
            if (!P21_CUDA_OK(cudaMemcpyAsync(d_fft_real, h_dens, box_bytes, cudaMemcpyHostToDevice,
                                             stream))) {
                ok = false;
            } else {
                linear_density_kernel<<<g, tpb, 0, stream>>>(d_fft_real, d_fft_real, box_pixels,
                                                            gf);
                ok = P21_CUDA_OK(cudaGetLastError());
            }
        } else {
            float *d_dens = (float *)p21_cuda_registry_get_or_upload(h_dens, box_bytes, stream);
            if (!d_dens) {
                ok = false;
            } else {
                linear_density_kernel<<<g, tpb, 0, stream>>>(d_dens, d_fft_real, box_pixels, gf);
                ok = P21_CUDA_OK(cudaGetLastError());
            }
        }
    } else if (ok) {
        const float *h_vx = perturb_hires ? boxes->hires_vx : boxes->lowres_vx;
        const float *h_vy = perturb_hires ? boxes->hires_vy : boxes->lowres_vy;
        const float *h_vz = perturb_hires ? boxes->hires_vz : boxes->lowres_vz;
        const float *h_vx2 = perturb_hires ? boxes->hires_vx_2LPT : boxes->lowres_vx_2LPT;
        const float *h_vy2 = perturb_hires ? boxes->hires_vy_2LPT : boxes->lowres_vy_2LPT;
        const float *h_vz2 = perturb_hires ? boxes->hires_vz_2LPT : boxes->lowres_vz_2LPT;
        const bool use_2lpt = (alg == PERTURB_ALGORITHM_2LPT);
        const bool use_float_accum =
            (p21_cuda_get_fast_ic() != 0) &&
            ((box_bytes >= (2ULL << 30)) ||
             (p21_cuda_free_mem_bytes() < box_pixels * sizeof(double) + (3ULL << 30)));

        double gf = dicke(redshift);
        double init_gf = dicke(simulation_options_global->INITIAL_REDSHIFT);
        double df2 = -(3.0 / 7.0) * gf * gf;
        double init_df2 = -(3.0 / 7.0) * init_gf * init_gf;
        double vdf0 = (gf - init_gf) / box_len * nx;
        double vdf1 = (gf - init_gf) / box_len * ny;
        double vdf2 = (gf - init_gf) / len_z * nz;
        double vdf2_0 = (df2 - init_df2) / box_len * nx;
        double vdf2_1 = (df2 - init_df2) / box_len * ny;
        double vdf2_2 = (df2 - init_df2) / len_z * nz;
        double dim_ratio = (double)bx / (double)nx;

        double *d_resampled = nullptr;
        if (use_float_accum) {
            if (!P21_CUDA_OK(cudaMemsetAsync(d_fft_real, 0, box_bytes, stream))) {
                ok = false;
            }
        } else {
            d_resampled = (double *)p21_cuda_pool_alloc(box_pixels * sizeof(double), stream);
            if (!d_resampled ||
                !P21_CUDA_OK(cudaMemsetAsync(d_resampled, 0, box_pixels * sizeof(double), stream))) {
                ok = false;
            }
        }

        if (ok) {
            int forced_slab_x = read_perturb_slab_x_env();
            size_t max_entry = p21_cuda_registry_max_entry_bytes();
            size_t n_vel_grids = use_2lpt ? 6ULL : 3ULL;
            size_t resident_need = hi_bytes + n_vel_grids * box_bytes + 1ULL * 1024 * 1024 * 1024;
            bool use_full_resident = (forced_slab_x <= 0) && (hi_bytes <= max_entry) &&
                                     (box_bytes <= max_entry) &&
                                     (p21_cuda_free_mem_bytes() >= resident_need);

            if (use_full_resident) {
                float *d_hi_dens =
                    (float *)p21_cuda_registry_get_or_upload(boxes->hires_density, hi_bytes, stream);
                float *d_vx = (float *)p21_cuda_registry_get_or_upload(h_vx, box_bytes, stream);
                float *d_vy = (float *)p21_cuda_registry_get_or_upload(h_vy, box_bytes, stream);
                float *d_vz = (float *)p21_cuda_registry_get_or_upload(h_vz, box_bytes, stream);
                float *d_vx2 =
                    use_2lpt ? (float *)p21_cuda_registry_get_or_upload(h_vx2, box_bytes, stream)
                             : nullptr;
                float *d_vy2 =
                    use_2lpt ? (float *)p21_cuda_registry_get_or_upload(h_vy2, box_bytes, stream)
                             : nullptr;
                float *d_vz2 =
                    use_2lpt ? (float *)p21_cuda_registry_get_or_upload(h_vz2, box_bytes, stream)
                             : nullptr;

                if (!d_hi_dens || !d_vx || !d_vy || !d_vz ||
                    (use_2lpt && (!d_vx2 || !d_vy2 || !d_vz2))) {
                    ok = false;
                } else {
                    int hg = (int)((hi_pixels + tpb - 1) / tpb);
                    if (use_float_accum) {
                        move_grid_masses_slab_kernel<float><<<hg, tpb, 0, stream>>>(
                            d_hi_dens, d_vx, d_vy, d_vz, d_vx2, d_vy2, d_vz2, 0, nx, ny, nz, bx,
                            by, bz, bx, by, bz, dim_ratio, dim_ratio, vdf0, vdf1, vdf2, vdf2_0,
                            vdf2_1, vdf2_2, init_gf, use_2lpt, false, d_fft_real);
                    } else {
                        move_grid_masses_slab_kernel<double><<<hg, tpb, 0, stream>>>(
                            d_hi_dens, d_vx, d_vy, d_vz, d_vx2, d_vy2, d_vz2, 0, nx, ny, nz, bx,
                            by, bz, bx, by, bz, dim_ratio, dim_ratio, vdf0, vdf1, vdf2, vdf2_0,
                            vdf2_1, vdf2_2, init_gf, use_2lpt, false, d_resampled);
                    }
                    ok = P21_CUDA_OK(cudaGetLastError());
                }
            } else {
                const bool same_dims = (bx == nx && by == ny && bz == nz);
                const size_t yz_bytes = (size_t)ny * (size_t)nz * sizeof(float);

                if (!same_dims) {
                    /* Mode A (!perturb_hires with HII_DIM < DIM): low-res vels resident, stream hires_density. */
                    if (box_bytes >= 64ULL * 1024 * 1024) {
                        p21_cuda_registry_evict_to_free(
                            n_vel_grids * box_bytes + 1536ULL * 1024 * 1024, stream);
                    }
                    float *d_vx = (float *)p21_cuda_registry_get_or_upload(h_vx, box_bytes, stream);
                    float *d_vy = (float *)p21_cuda_registry_get_or_upload(h_vy, box_bytes, stream);
                    float *d_vz = (float *)p21_cuda_registry_get_or_upload(h_vz, box_bytes, stream);
                    float *d_vx2 =
                        use_2lpt
                            ? (float *)p21_cuda_registry_get_or_upload(h_vx2, box_bytes, stream)
                            : nullptr;
                    float *d_vy2 =
                        use_2lpt
                            ? (float *)p21_cuda_registry_get_or_upload(h_vy2, box_bytes, stream)
                            : nullptr;
                    float *d_vz2 =
                        use_2lpt
                            ? (float *)p21_cuda_registry_get_or_upload(h_vz2, box_bytes, stream)
                            : nullptr;

                    int slab_x = forced_slab_x > 0
                                     ? std::min(forced_slab_x, nx)
                                     : std::max(1, std::min(nx, (int)((256ULL << 20) / yz_bytes)));
                    size_t slab_bytes = (size_t)slab_x * yz_bytes;
                    float *d_hi_slab = (float *)p21_cuda_pool_alloc(slab_bytes, stream);
                    if (!d_vx || !d_vy || !d_vz ||
                        (use_2lpt && (!d_vx2 || !d_vy2 || !d_vz2)) || !d_hi_slab) {
                        ok = false;
                    } else {
                        for (int x0 = 0; ok && x0 < nx; x0 += slab_x) {
                            int cur_nx = std::min(slab_x, nx - x0);
                            size_t cur_pix = (size_t)cur_nx * (size_t)ny * (size_t)nz;
                            size_t cur_bytes = cur_pix * sizeof(float);
                            size_t off = (size_t)x0 * (size_t)ny * (size_t)nz;
                            if (!P21_CUDA_OK(cudaMemcpyAsync(d_hi_slab, boxes->hires_density + off,
                                                             cur_bytes, cudaMemcpyHostToDevice,
                                                             stream))) {
                                ok = false;
                                break;
                            }
                            int sg = (int)((cur_pix + tpb - 1) / tpb);
                            if (use_float_accum) {
                                move_grid_masses_slab_kernel<float><<<sg, tpb, 0, stream>>>(
                                    d_hi_slab, d_vx, d_vy, d_vz, d_vx2, d_vy2, d_vz2, x0, cur_nx,
                                    ny, nz, bx, by, bz, bx, by, bz, dim_ratio, dim_ratio, vdf0,
                                    vdf1, vdf2, vdf2_0, vdf2_1, vdf2_2, init_gf, use_2lpt, false,
                                    d_fft_real);
                            } else {
                                move_grid_masses_slab_kernel<double><<<sg, tpb, 0, stream>>>(
                                    d_hi_slab, d_vx, d_vy, d_vz, d_vx2, d_vy2, d_vz2, x0, cur_nx,
                                    ny, nz, bx, by, bz, bx, by, bz, dim_ratio, dim_ratio, vdf0,
                                    vdf1, vdf2, vdf2_0, vdf2_1, vdf2_2, init_gf, use_2lpt, false,
                                    d_resampled);
                            }
                            ok = P21_CUDA_OK(cudaGetLastError());
                        }
                    }
                    p21_cuda_pool_free(d_hi_slab, stream);
                } else {
                    /* Mode B (perturb_hires or bx==nx): stream hires_density + velocities in x-slabs. */
                    size_t free_b = p21_cuda_free_mem_bytes();
                    size_t max_slab_budget = std::min<size_t>(
                        896ULL << 20, std::max<size_t>(64ULL << 20, free_b / 2));
                    size_t per_slice_all = (1ULL + n_vel_grids) * yz_bytes;
                    int auto_slab_x =
                        std::max(1, std::min(nx, (int)(max_slab_budget / per_slice_all)));
                    int slab_x = (forced_slab_x > 0) ? std::min(forced_slab_x, nx) : auto_slab_x;
                    size_t slab_bytes = (size_t)slab_x * yz_bytes;

                    float *d_hi_slab = (float *)p21_cuda_pool_alloc(slab_bytes, stream);
                    float *d_vx_slab = (float *)p21_cuda_pool_alloc(slab_bytes, stream);
                    float *d_vy_slab = (float *)p21_cuda_pool_alloc(slab_bytes, stream);
                    float *d_vz_slab = (float *)p21_cuda_pool_alloc(slab_bytes, stream);
                    float *d_vx2_slab =
                        use_2lpt ? (float *)p21_cuda_pool_alloc(slab_bytes, stream) : nullptr;
                    float *d_vy2_slab =
                        use_2lpt ? (float *)p21_cuda_pool_alloc(slab_bytes, stream) : nullptr;
                    float *d_vz2_slab =
                        use_2lpt ? (float *)p21_cuda_pool_alloc(slab_bytes, stream) : nullptr;

                    if (!d_hi_slab || !d_vx_slab || !d_vy_slab || !d_vz_slab ||
                        (use_2lpt && (!d_vx2_slab || !d_vy2_slab || !d_vz2_slab))) {
                        ok = false;
                    } else {
                        for (int x0 = 0; ok && x0 < nx; x0 += slab_x) {
                            int cur_nx = std::min(slab_x, nx - x0);
                            size_t cur_pix = (size_t)cur_nx * (size_t)ny * (size_t)nz;
                            size_t cur_bytes = cur_pix * sizeof(float);
                            size_t off = (size_t)x0 * (size_t)ny * (size_t)nz;
                            bool cp_ok =
                                P21_CUDA_OK(cudaMemcpyAsync(d_hi_slab, boxes->hires_density + off,
                                                            cur_bytes, cudaMemcpyHostToDevice,
                                                            stream)) &&
                                P21_CUDA_OK(cudaMemcpyAsync(d_vx_slab, h_vx + off, cur_bytes,
                                                            cudaMemcpyHostToDevice, stream)) &&
                                P21_CUDA_OK(cudaMemcpyAsync(d_vy_slab, h_vy + off, cur_bytes,
                                                            cudaMemcpyHostToDevice, stream)) &&
                                P21_CUDA_OK(cudaMemcpyAsync(d_vz_slab, h_vz + off, cur_bytes,
                                                            cudaMemcpyHostToDevice, stream));
                            if (cp_ok && use_2lpt) {
                                cp_ok =
                                    P21_CUDA_OK(cudaMemcpyAsync(d_vx2_slab, h_vx2 + off, cur_bytes,
                                                                cudaMemcpyHostToDevice, stream)) &&
                                    P21_CUDA_OK(cudaMemcpyAsync(d_vy2_slab, h_vy2 + off, cur_bytes,
                                                                cudaMemcpyHostToDevice, stream)) &&
                                    P21_CUDA_OK(cudaMemcpyAsync(d_vz2_slab, h_vz2 + off, cur_bytes,
                                                                cudaMemcpyHostToDevice, stream));
                            }
                            if (!cp_ok) {
                                ok = false;
                                break;
                            }
                            int sg = (int)((cur_pix + tpb - 1) / tpb);
                            if (use_float_accum) {
                                move_grid_masses_slab_kernel<float><<<sg, tpb, 0, stream>>>(
                                    d_hi_slab, d_vx_slab, d_vy_slab, d_vz_slab, d_vx2_slab,
                                    d_vy2_slab, d_vz2_slab, x0, cur_nx, ny, nz, bx, by, bz, bx, by,
                                    bz, dim_ratio, dim_ratio, vdf0, vdf1, vdf2, vdf2_0, vdf2_1,
                                    vdf2_2, init_gf, use_2lpt, true, d_fft_real);
                            } else {
                                move_grid_masses_slab_kernel<double><<<sg, tpb, 0, stream>>>(
                                    d_hi_slab, d_vx_slab, d_vy_slab, d_vz_slab, d_vx2_slab,
                                    d_vy2_slab, d_vz2_slab, x0, cur_nx, ny, nz, bx, by, bz, bx, by,
                                    bz, dim_ratio, dim_ratio, vdf0, vdf1, vdf2, vdf2_0, vdf2_1,
                                    vdf2_2, init_gf, use_2lpt, true, d_resampled);
                            }
                            ok = P21_CUDA_OK(cudaGetLastError());
                        }
                    }

                    p21_cuda_pool_free(d_vz2_slab, stream);
                    p21_cuda_pool_free(d_vy2_slab, stream);
                    p21_cuda_pool_free(d_vx2_slab, stream);
                    p21_cuda_pool_free(d_vz_slab, stream);
                    p21_cuda_pool_free(d_vy_slab, stream);
                    p21_cuda_pool_free(d_vx_slab, stream);
                    p21_cuda_pool_free(d_hi_slab, stream);
                }
            }
        }

        if (ok && !use_float_accum) {
            int bg = (int)((box_pixels + tpb - 1) / tpb);
            double_to_float_kernel<<<bg, tpb, 0, stream>>>(d_resampled, d_fft_real, box_pixels);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
        p21_cuda_pool_free(d_resampled, stream);
    }

    /* Post-Step-1 deferred allocation (safe boundary: zero borrowed registry pointers held). */
    if (ok && perturb_hires && box_k_bytes >= 256ULL * 1024 * 1024) {
        size_t total_mem = p21_cuda_total_mem_bytes();
        size_t target_free = box_k_bytes * 3 + lo_bytes;
        if (total_mem > 0) {
            target_free = std::min<size_t>(target_free, (total_mem * 3) / 4);
        }
        p21_cuda_registry_evict_to_free(target_free, stream);
    }
    if (ok) {
        d_density_saved = (cufftComplex *)p21_cuda_pool_alloc(box_k_bytes, stream);
        d_k_work = (cufftComplex *)p21_cuda_pool_alloc(box_k_bytes, stream);
        d_real_lo = perturb_hires ? (float *)p21_cuda_pool_alloc(lo_bytes, stream) : d_fft_real;
        d_k_lo = perturb_hires ? (cufftComplex *)p21_cuda_pool_alloc(lo_k_bytes, stream) : d_k_work;
        if (!d_density_saved || !d_k_work || !d_real_lo || !d_k_lo) {
            ok = false;
        }
    }

    /* 2. assign_to_lowres_grid if PERTURB_ON_HIGH_RES */
    if (ok && perturb_hires) {
        ok = p21_cuda_fft_r2c(d_fft_real, d_k_work, nx, ny, nz, false, stream) &&
             P21_CUDA_OK(cudaMemcpyAsync(d_density_saved, d_k_work, box_k_bytes,
                                         cudaMemcpyDeviceToDevice, stream)) &&
             p21_cuda_filter_box(d_k_work, nx, ny, nz, box_len, ncf, FILTER_TOPHAT, tophat_R, 0.0f,
                                 0.0f, stream) &&
             p21_cuda_fft_c2r(d_k_work, d_fft_real, nx, ny, nz, false, stream);
        if (ok) {
            int lg = (int)((lo_pixels + tpb - 1) / tpb);
            double dim_ratio = (double)nx / (double)lx;
            resample_box_kernel<<<lg, tpb, 0, stream>>>(d_fft_real, ny, nz, d_real_lo, lx, ly, lz,
                                                        dim_ratio, (float)hi_pixels);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
    }

    /* 3. normalise_delta_grid */
    if (ok && alg > PERTURB_ALGORITHM_LINEAR) {
        double mass_factor = perturb_hires ? 1.0 : ((double)lo_pixels / (double)hi_pixels);
        int lg = (int)((lo_pixels + tpb - 1) / tpb);
        normalise_delta_kernel<<<lg, tpb, 0, stream>>>(d_real_lo, lo_pixels, mass_factor);
        ok = P21_CUDA_OK(cudaGetLastError());
    }

    /* 4. smooth_and_clip_density */
    if (ok) {
        ok = p21_cuda_fft_r2c(d_real_lo, d_k_lo, lx, ly, lz, false, stream);
        if (ok && matter_options_global->SMOOTH_EVOLVED_DENSITY_FIELD) {
            ok = p21_cuda_filter_box(d_k_lo, lx, ly, lz, box_len, ncf, FILTER_GAUSSIAN, smooth_R,
                                     0.0f, 0.0f, stream);
        }
        if (ok && !perturb_hires) {
            ok = P21_CUDA_OK(cudaMemcpyAsync(d_density_saved, d_k_lo, lo_k_bytes,
                                             cudaMemcpyDeviceToDevice, stream));
        }
        if (ok) {
            ok = p21_cuda_fft_c2r(d_k_lo, d_real_lo, lx, ly, lz, false, stream);
        }
        if (ok) {
            int lg = (int)((lo_pixels + tpb - 1) / tpb);
            normalize_and_clip_density_kernel<<<lg, tpb, 0, stream>>>(
                d_real_lo, lo_pixels, (float)lo_pixels, (float)(-1.0 + FRACT_FLOAT_ERR));
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 upload_and_register_d2h(perturbed_field->density, d_real_lo, lo_bytes, stream);
        }
    }
    if (perturb_hires && d_k_lo) {
        p21_cuda_pool_free(d_k_lo, stream);
        d_k_lo = nullptr;
    }

    /* 5. compute_perturbed_velocities */
    if (ok) {
        double dDdt_over_D = ddickedt(redshift) / dicke(redshift);
        double dim_ratio = (double)bx / (double)lx;
        int kg = (int)((box_k_pixels + tpb - 1) / tpb);
        int lg = (int)((lo_pixels + tpb - 1) / tpb);

        int axes[3] = {0, 1, 2};
        float *vel_out[3] = {perturbed_field->velocity_x, perturbed_field->velocity_y,
                             perturbed_field->velocity_z};
        int start_idx = matter_options_global->KEEP_3D_VELOCITIES ? 0 : 2;

        for (int idx = start_idx; ok && idx < 3; ++idx) {
            int axis = axes[idx];
            perturb_velocity_kspace_kernel<<<kg, tpb, 0, stream>>>(
                d_density_saved, d_k_work, bx, by, bz, box_len, box_len, len_z, axis, dDdt_over_D,
                (double)box_pixels);
            ok = P21_CUDA_OK(cudaGetLastError());
            if (ok && perturb_hires && nx != lx) {
                ok = p21_cuda_filter_box(d_k_work, bx, by, bz, box_len, ncf, FILTER_TOPHAT, tophat_R,
                                         0.0f, 0.0f, stream);
            }
            if (ok) {
                ok = p21_cuda_fft_c2r(d_k_work, d_fft_real, bx, by, bz, false, stream);
            }
            if (ok) {
                if (perturb_hires) {
                    resample_box_kernel<<<lg, tpb, 0, stream>>>(d_fft_real, by, bz, d_real_lo, lx,
                                                                ly, lz, dim_ratio, 1.0f);
                    ok = P21_CUDA_OK(cudaGetLastError()) &&
                         upload_and_register_d2h(vel_out[idx], d_real_lo, lo_bytes, stream);
                } else {
                    ok = upload_and_register_d2h(vel_out[idx], d_fft_real, lo_bytes, stream);
                }
            }
        }
    }

    p21_cuda_pool_free(d_k_work, stream);
    if (perturb_hires) {
        p21_cuda_pool_free(d_k_lo, stream);
        p21_cuda_pool_free(d_real_lo, stream);
    }
    p21_cuda_pool_free(d_density_saved, stream);
    p21_cuda_pool_free(d_fft_real, stream);

    if (ok && P21_CUDA_OK(cudaStreamSynchronize(stream))) {
        p21_cuda_stats_mut()->perturb_runs++;
        return 0;
    }
    return -1;
}
