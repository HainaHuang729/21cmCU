#include <math.h>
#include <stdint.h>
#include <string.h>

#include "cuda_analysis_common.cuh"

namespace {

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
    P21_GRID_STRIDE_LOOP(idx, n_tot) {
        grid[idx] = (float)((double)grid[idx] * inv_mean_w - 1.0);
    }
}

}  // namespace

extern "C" {

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
    ScopedPoolBuf<float> d_pos((size_t)n_particles * 3ULL, stream);
    ScopedPoolBuf<float> d_weights(h_weights ? (size_t)n_particles : 0ULL, stream);
    ScopedPoolBuf<float> d_grid(n_tot, stream);
    if (!d_pos || !d_grid || (h_weights && !d_weights)) {
        return -1;
    }

    size_t pos_bytes = (size_t)n_particles * 3ULL * sizeof(float);
    size_t w_bytes = (size_t)n_particles * sizeof(float);
    bool ok = P21_CUDA_OK(cudaMemsetAsync(d_grid.ptr(), 0, grid_bytes, stream)) &&
              P21_CUDA_OK(cudaMemcpyAsync(d_pos.ptr(), h_pos_xyz, pos_bytes,
                                          cudaMemcpyHostToDevice, stream));
    if (ok && h_weights) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(d_weights.ptr(), h_weights, w_bytes,
                                         cudaMemcpyHostToDevice, stream));
    }

    constexpr int kTpb = 256;
    if (ok) {
        int part_blocks = p21_grid_blocks((size_t)n_particles, kTpb);
        cic_deposit_3d_kernel<<<part_blocks, kTpb, 0, stream>>>(
            d_pos.ptr(), h_weights ? d_weights.ptr() : nullptr, n_particles, nx, ny, nz,
            (double)nx / lx, (double)ny / ly, (double)nz / lz, d_grid.ptr());
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
            int grid_blocks = p21_grid_blocks(n_tot, kTpb);
            cic_to_overdensity_kernel<<<grid_blocks, kTpb, 0, stream>>>(d_grid.ptr(), n_tot,
                                                                        1.0 / mean_w);
            ok = P21_CUDA_OK(cudaGetLastError());
        }
    }

    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(h_grid_out, d_grid.ptr(), grid_bytes,
                                         cudaMemcpyDeviceToHost, stream)) &&
             P21_CUDA_OK(cudaStreamSynchronize(stream));
    }
    return ok ? 0 : -1;
}

}  // extern "C"
