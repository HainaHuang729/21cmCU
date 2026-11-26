#include "cuda_mgpu.cuh"

#include <algorithm>
#include <string.h>
#include <vector>

namespace {

constexpr size_t kMgpuMaxStripBytes = 256ULL << 20;
constexpr int kMgpuThreads = 256;

struct MgpuPart {
    int n_parts = 0;
    int off[kMgpuMaxDevices]{};
    int len[kMgpuMaxDevices]{};
};

__device__ static inline int find_part(const MgpuPart &p, int coord) {
    for (int g = 0; g < p.n_parts; ++g) {
        if (coord < p.off[g] + p.len[g]) {
            return g;
        }
    }
    return p.n_parts - 1;
}

__global__ void mgpu_pack_xy_tile_kernel(const cufftComplex *__restrict__ x_slab, int sx, int ny,
                                         int hz, int y0, int sy,
                                         cufftComplex *__restrict__ tile_out) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)sy * (size_t)hz * (size_t)sx;
    if (idx >= total) {
        return;
    }
    int x_loc = (int)(idx % (size_t)sx);
    size_t q = idx / (size_t)sx;
    int z = (int)(q % (size_t)hz);
    int y_loc = (int)(q / (size_t)hz);
    size_t src_idx = ((size_t)x_loc * (size_t)ny + (size_t)(y0 + y_loc)) * (size_t)hz + (size_t)z;
    tile_out[idx] = x_slab[src_idx];
}

__global__ void mgpu_unpack_xy_tile_kernel(const cufftComplex *__restrict__ tile_in, int x0,
                                           int sx, int sy, int hz, int nx,
                                           cufftComplex *__restrict__ y_slab) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)sy * (size_t)hz * (size_t)sx;
    if (idx >= total) {
        return;
    }
    int x_loc = (int)(idx % (size_t)sx);
    size_t q = idx / (size_t)sx;
    int z = (int)(q % (size_t)hz);
    int y_loc = (int)(q / (size_t)hz);
    size_t dst_idx = ((size_t)y_loc * (size_t)hz + (size_t)z) * (size_t)nx + (size_t)(x0 + x_loc);
    y_slab[dst_idx] = tile_in[idx];
}

__global__ void mgpu_unpack_xy_full_kernel(const cufftComplex *__restrict__ recv_buf, int sy,
                                           int hz, int nx, MgpuPart xpart,
                                           cufftComplex *__restrict__ y_slab) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)sy * (size_t)hz * (size_t)nx;
    if (idx >= total) {
        return;
    }
    int x = (int)(idx % (size_t)nx);
    size_t q = idx / (size_t)nx;
    int z = (int)(q % (size_t)hz);
    int y_loc = (int)(q / (size_t)hz);

    int s = find_part(xpart, x);
    int x0_s = xpart.off[s];
    int sx_s = xpart.len[s];
    int x_loc = x - x0_s;
    size_t tile_base = (size_t)x0_s * (size_t)sy * (size_t)hz;
    size_t src_idx = tile_base + ((size_t)y_loc * (size_t)hz + (size_t)z) * (size_t)sx_s +
                     (size_t)x_loc;
    y_slab[idx] = recv_buf[src_idx];
}

__global__ void mgpu_pack_yx_tile_kernel(const cufftComplex *__restrict__ y_slab, int sy, int hz,
                                         int nx, int x0, int sx,
                                         cufftComplex *__restrict__ tile_out) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)sx * (size_t)sy * (size_t)hz;
    if (idx >= total) {
        return;
    }
    int z = (int)(idx % (size_t)hz);
    size_t q = idx / (size_t)hz;
    int y_loc = (int)(q % (size_t)sy);
    int x_loc = (int)(q / (size_t)sy);
    size_t src_idx = ((size_t)y_loc * (size_t)hz + (size_t)z) * (size_t)nx + (size_t)(x0 + x_loc);
    tile_out[idx] = y_slab[src_idx];
}

__global__ void mgpu_unpack_yx_tile_kernel(const cufftComplex *__restrict__ tile_in, int sx,
                                           int y0, int sy, int ny, int hz,
                                           cufftComplex *__restrict__ x_slab) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)sx * (size_t)sy * (size_t)hz;
    if (idx >= total) {
        return;
    }
    int z = (int)(idx % (size_t)hz);
    size_t q = idx / (size_t)hz;
    int y_loc = (int)(q % (size_t)sy);
    int x_loc = (int)(q / (size_t)sy);
    size_t dst_idx = ((size_t)x_loc * (size_t)ny + (size_t)(y0 + y_loc)) * (size_t)hz + (size_t)z;
    x_slab[dst_idx] = tile_in[idx];
}

__global__ void mgpu_unpack_yx_full_kernel(const cufftComplex *__restrict__ recv_buf, int sx,
                                           int ny, int hz, MgpuPart ypart,
                                           cufftComplex *__restrict__ x_slab) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)sx * (size_t)ny * (size_t)hz;
    if (idx >= total) {
        return;
    }
    int z = (int)(idx % (size_t)hz);
    size_t q = idx / (size_t)hz;
    int y = (int)(q % (size_t)ny);
    int x_loc = (int)(q / (size_t)ny);

    int d = find_part(ypart, y);
    int y0_d = ypart.off[d];
    int sy_d = ypart.len[d];
    int y_loc = y - y0_d;
    size_t tile_base = (size_t)sx * (size_t)y0_d * (size_t)hz;
    size_t src_idx = tile_base + ((size_t)x_loc * (size_t)sy_d + (size_t)y_loc) * (size_t)hz +
                     (size_t)z;
    x_slab[idx] = recv_buf[src_idx];
}

__global__ void scale_strip_f32_kernel(float *buf, size_t n, double scale) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        buf[idx] = (float)((double)buf[idx] * scale);
    }
}

int grid_blocks_u32(size_t total) {
    size_t b = (total + kMgpuThreads - 1ULL) / kMgpuThreads;
    if (b == 0) {
        return 1;
    }
    if (b > 2147483647ULL) {
        b = 2147483647ULL;
    }
    return (int)b;
}

bool mgpu_local_2d_r2c(int dev, cudaStream_t stream, int sx, int ny, int nz, const float *d_real,
                       cufftComplex *d_cplx) {
    if (sx <= 0) {
        return true;
    }
    if (!P21_CUDA_OK(cudaSetDevice(dev))) {
        return false;
    }
    const int hz = nz / 2 + 1;
    const size_t plane_real = (size_t)ny * (size_t)nz;
    const size_t plane_cplx = (size_t)ny * (size_t)hz;
    const size_t plane_bytes = plane_cplx * sizeof(cufftComplex);
    const int strip =
        std::max(1, std::min(sx, (int)(kMgpuMaxStripBytes / std::max<size_t>(1, plane_bytes))));

    int n2d[2] = {ny, nz};
    int inembed[2] = {ny, nz};
    int onembed[2] = {ny, hz};

    auto run_strip = [&](int batch, int x_off) -> bool {
        cufftHandle plan = 0;
        if (!P21_CUFFT_OK(cufftCreate(&plan))) {
            return false;
        }
        size_t work_size = 0;
        bool ok = P21_CUFFT_OK(cufftSetAutoAllocation(plan, 0)) &&
                  P21_CUFFT_OK(cufftMakePlanMany(plan, 2, n2d, inembed, 1, (int)plane_real, onembed,
                                                 1, (int)plane_cplx, CUFFT_R2C, batch, &work_size)) &&
                  P21_CUFFT_OK(cufftSetStream(plan, stream));
        void *d_work = ok && work_size > 0 ? p21_mgpu_malloc(dev, work_size, stream) : nullptr;
        if (ok && work_size > 0 && !d_work) {
            ok = false;
        }
        if (ok) {
            ok = P21_CUFFT_OK(cufftSetWorkArea(plan, d_work)) &&
                 P21_CUFFT_OK(cufftExecR2C(
                     plan, (cufftReal *)const_cast<float *>(d_real + (size_t)x_off * plane_real),
                     d_cplx + (size_t)x_off * plane_cplx));
        }
        if (d_work) {
            p21_mgpu_free(dev, d_work, stream);
        }
        cufftDestroy(plan);
        return ok;
    };

    int full_end = (sx / strip) * strip;
    if (full_end > 0) {
        cufftHandle plan = 0;
        if (!P21_CUFFT_OK(cufftCreate(&plan))) {
            return false;
        }
        size_t work_size = 0;
        bool ok = P21_CUFFT_OK(cufftSetAutoAllocation(plan, 0)) &&
                  P21_CUFFT_OK(cufftMakePlanMany(plan, 2, n2d, inembed, 1, (int)plane_real, onembed,
                                                 1, (int)plane_cplx, CUFFT_R2C, strip, &work_size)) &&
                  P21_CUFFT_OK(cufftSetStream(plan, stream));
        void *d_work = ok && work_size > 0 ? p21_mgpu_malloc(dev, work_size, stream) : nullptr;
        if (ok && work_size > 0 && !d_work) {
            ok = false;
        }
        if (ok) {
            ok = P21_CUFFT_OK(cufftSetWorkArea(plan, d_work));
        }
        for (int x = 0; ok && x < full_end; x += strip) {
            ok = P21_CUFFT_OK(cufftExecR2C(
                plan, (cufftReal *)const_cast<float *>(d_real + (size_t)x * plane_real),
                d_cplx + (size_t)x * plane_cplx));
        }
        if (d_work) {
            p21_mgpu_free(dev, d_work, stream);
        }
        cufftDestroy(plan);
        if (!ok) {
            return false;
        }
    }
    if (full_end < sx) {
        return run_strip(sx - full_end, full_end);
    }
    return true;
}

bool mgpu_local_2d_c2r(int dev, cudaStream_t stream, int sx, int ny, int nz, cufftComplex *d_cplx,
                       float *d_real) {
    if (sx <= 0) {
        return true;
    }
    if (!P21_CUDA_OK(cudaSetDevice(dev))) {
        return false;
    }
    const int hz = nz / 2 + 1;
    const size_t plane_real = (size_t)ny * (size_t)nz;
    const size_t plane_cplx = (size_t)ny * (size_t)hz;
    const size_t plane_bytes = plane_cplx * sizeof(cufftComplex);
    const int strip =
        std::max(1, std::min(sx, (int)(kMgpuMaxStripBytes / std::max<size_t>(1, plane_bytes))));

    int n2d[2] = {ny, nz};
    int inembed[2] = {ny, hz};
    int onembed[2] = {ny, nz};

    auto run_tail = [&](int batch, int x_off) -> bool {
        cufftHandle plan = 0;
        if (!P21_CUFFT_OK(cufftCreate(&plan))) {
            return false;
        }
        size_t work_size = 0;
        bool ok = P21_CUFFT_OK(cufftSetAutoAllocation(plan, 0)) &&
                  P21_CUFFT_OK(cufftMakePlanMany(plan, 2, n2d, inembed, 1, (int)plane_cplx, onembed,
                                                 1, (int)plane_real, CUFFT_C2R, batch, &work_size)) &&
                  P21_CUFFT_OK(cufftSetStream(plan, stream));
        void *d_work = ok && work_size > 0 ? p21_mgpu_malloc(dev, work_size, stream) : nullptr;
        if (ok && work_size > 0 && !d_work) {
            ok = false;
        }
        if (ok) {
            ok = P21_CUFFT_OK(cufftSetWorkArea(plan, d_work)) &&
                 P21_CUFFT_OK(cufftExecC2R(plan, d_cplx + (size_t)x_off * plane_cplx,
                                           (cufftReal *)(d_real + (size_t)x_off * plane_real)));
        }
        if (d_work) {
            p21_mgpu_free(dev, d_work, stream);
        }
        cufftDestroy(plan);
        return ok;
    };

    int full_end = (sx / strip) * strip;
    if (full_end > 0) {
        cufftHandle plan = 0;
        if (!P21_CUFFT_OK(cufftCreate(&plan))) {
            return false;
        }
        size_t work_size = 0;
        bool ok = P21_CUFFT_OK(cufftSetAutoAllocation(plan, 0)) &&
                  P21_CUFFT_OK(cufftMakePlanMany(plan, 2, n2d, inembed, 1, (int)plane_cplx, onembed,
                                                 1, (int)plane_real, CUFFT_C2R, strip, &work_size)) &&
                  P21_CUFFT_OK(cufftSetStream(plan, stream));
        void *d_work = ok && work_size > 0 ? p21_mgpu_malloc(dev, work_size, stream) : nullptr;
        if (ok && work_size > 0 && !d_work) {
            ok = false;
        }
        if (ok) {
            ok = P21_CUFFT_OK(cufftSetWorkArea(plan, d_work));
        }
        for (int x = 0; ok && x < full_end; x += strip) {
            ok = P21_CUFFT_OK(cufftExecC2R(plan, d_cplx + (size_t)x * plane_cplx,
                                           (cufftReal *)(d_real + (size_t)x * plane_real)));
        }
        if (d_work) {
            p21_mgpu_free(dev, d_work, stream);
        }
        cufftDestroy(plan);
        if (!ok) {
            return false;
        }
    }
    if (full_end < sx) {
        return run_tail(sx - full_end, full_end);
    }
    return true;
}

bool mgpu_local_1d_c2c(int dev, cudaStream_t stream, int n_rows, int nx, cufftComplex *d_y,
                       int dir) {
    if (n_rows <= 0) {
        return true;
    }
    if (!P21_CUDA_OK(cudaSetDevice(dev))) {
        return false;
    }
    const size_t row_bytes = (size_t)nx * sizeof(cufftComplex);
    const int strip =
        std::max(1, std::min(n_rows, (int)(kMgpuMaxStripBytes / std::max<size_t>(1, row_bytes))));

    auto run_tail = [&](int batch, int row_off) -> bool {
        cufftHandle plan = 0;
        if (!P21_CUFFT_OK(cufftCreate(&plan))) {
            return false;
        }
        size_t work_size = 0;
        bool ok =
            P21_CUFFT_OK(cufftSetAutoAllocation(plan, 0)) &&
            P21_CUFFT_OK(cufftMakePlanMany(plan, 1, &nx, &nx, 1, nx, &nx, 1, nx, CUFFT_C2C, batch,
                                           &work_size)) &&
            P21_CUFFT_OK(cufftSetStream(plan, stream));
        void *d_work = ok && work_size > 0 ? p21_mgpu_malloc(dev, work_size, stream) : nullptr;
        if (ok && work_size > 0 && !d_work) {
            ok = false;
        }
        if (ok) {
            cufftComplex *ptr = d_y + (size_t)row_off * (size_t)nx;
            ok = P21_CUFFT_OK(cufftSetWorkArea(plan, d_work)) &&
                 P21_CUFFT_OK(cufftExecC2C(plan, ptr, ptr, dir));
        }
        if (d_work) {
            p21_mgpu_free(dev, d_work, stream);
        }
        cufftDestroy(plan);
        return ok;
    };

    int full_end = (n_rows / strip) * strip;
    if (full_end > 0) {
        cufftHandle plan = 0;
        if (!P21_CUFFT_OK(cufftCreate(&plan))) {
            return false;
        }
        size_t work_size = 0;
        bool ok =
            P21_CUFFT_OK(cufftSetAutoAllocation(plan, 0)) &&
            P21_CUFFT_OK(cufftMakePlanMany(plan, 1, &nx, &nx, 1, nx, &nx, 1, nx, CUFFT_C2C, strip,
                                           &work_size)) &&
            P21_CUFFT_OK(cufftSetStream(plan, stream));
        void *d_work = ok && work_size > 0 ? p21_mgpu_malloc(dev, work_size, stream) : nullptr;
        if (ok && work_size > 0 && !d_work) {
            ok = false;
        }
        if (ok) {
            ok = P21_CUFFT_OK(cufftSetWorkArea(plan, d_work));
        }
        for (int r = 0; ok && r < full_end; r += strip) {
            cufftComplex *ptr = d_y + (size_t)r * (size_t)nx;
            ok = P21_CUFFT_OK(cufftExecC2C(plan, ptr, ptr, dir));
        }
        if (d_work) {
            p21_mgpu_free(dev, d_work, stream);
        }
        cufftDestroy(plan);
        if (!ok) {
            return false;
        }
    }
    if (full_end < n_rows) {
        return run_tail(n_rows - full_end, full_end);
    }
    return true;
}

bool mgpu_transpose_x_to_y(const P21MgpuGrid &grid, cufftComplex *const *d_cplx_x,
                           cufftComplex **d_cplx_y, void *h_stage_opt) {
    const int G = grid.n_gpus;
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int hz = grid.hz;

    size_t max_slab_bytes = 0;
    for (int g = 0; g < G; ++g) {
        max_slab_bytes = std::max(max_slab_bytes, grid.cplx_elems_y(g) * sizeof(cufftComplex));
    }
    const size_t min_free = p21_mgpu_min_free_mem(grid);
    const bool can_peer = (h_stage_opt == nullptr) && (min_free >= 2 * max_slab_bytes + (512ULL << 20));

    if (can_peer) {
        std::vector<cufftComplex *> d_pack((size_t)G, nullptr);
        std::vector<cufftComplex *> d_recv((size_t)G, nullptr);
        bool ok = true;
        for (int g = 0; g < G; ++g) {
            d_pack[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(
                grid.devs[g], grid.cplx_elems_x(g) * sizeof(cufftComplex), grid.streams[g]);
            d_recv[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(
                grid.devs[g], grid.cplx_elems_y(g) * sizeof(cufftComplex), grid.streams[g]);
            if (!d_pack[(size_t)g] || !d_recv[(size_t)g]) {
                ok = false;
            }
        }
        if (ok) {
            for (int s = 0; s < G && ok; ++s) {
                if (!P21_CUDA_OK(cudaSetDevice(grid.devs[s]))) {
                    ok = false;
                    break;
                }
                for (int d = 0; d < G && ok; ++d) {
                    size_t tile_elems = (size_t)grid.sx[s] * (size_t)grid.sy[d] * (size_t)hz;
                    cufftComplex *tile_ptr =
                        d_pack[(size_t)s] + (size_t)grid.sx[s] * (size_t)grid.y0[d] * (size_t)hz;
                    mgpu_pack_xy_tile_kernel<<<grid_blocks_u32(tile_elems), kMgpuThreads, 0,
                                             grid.streams[s]>>>(
                        d_cplx_x[s], grid.sx[s], ny, hz, grid.y0[d], grid.sy[d], tile_ptr);
                    ok = P21_CUDA_OK(cudaGetLastError());
                }
            }
            for (int s = 0; s < G && ok; ++s) {
                P21_CUDA_OK(cudaSetDevice(grid.devs[s]));
                ok = P21_CUDA_OK(cudaStreamSynchronize(grid.streams[s]));
            }
            for (int s = 0; s < G && ok; ++s) {
                for (int d = 0; d < G && ok; ++d) {
                    size_t tile_bytes =
                        (size_t)grid.sx[s] * (size_t)grid.sy[d] * (size_t)hz * sizeof(cufftComplex);
                    const cufftComplex *src_ptr =
                        d_pack[(size_t)s] + (size_t)grid.sx[s] * (size_t)grid.y0[d] * (size_t)hz;
                    cufftComplex *dst_ptr =
                        d_recv[(size_t)d] + (size_t)grid.x0[s] * (size_t)grid.sy[d] * (size_t)hz;
                    if (s == d) {
                        P21_CUDA_OK(cudaSetDevice(grid.devs[s]));
                        ok = P21_CUDA_OK(cudaMemcpyAsync(dst_ptr, src_ptr, tile_bytes,
                                                         cudaMemcpyDeviceToDevice, grid.streams[d]));
                    } else {
                        ok = P21_CUDA_OK(cudaMemcpyPeerAsync(dst_ptr, grid.devs[d], src_ptr,
                                                             grid.devs[s], tile_bytes,
                                                             grid.streams[d]));
                    }
                }
            }
            MgpuPart xpart{};
            xpart.n_parts = G;
            for (int s = 0; s < G; ++s) {
                xpart.off[s] = grid.x0[s];
                xpart.len[s] = grid.sx[s];
            }
            for (int d = 0; d < G && ok; ++d) {
                P21_CUDA_OK(cudaSetDevice(grid.devs[d]));
                size_t elems_y = grid.cplx_elems_y(d);
                mgpu_unpack_xy_full_kernel<<<grid_blocks_u32(elems_y), kMgpuThreads, 0,
                                           grid.streams[d]>>>(d_recv[(size_t)d], grid.sy[d], hz,
                                                              nx, xpart, d_cplx_y[d]);
                ok = P21_CUDA_OK(cudaGetLastError());
            }
            for (int d = 0; d < G && ok; ++d) {
                P21_CUDA_OK(cudaSetDevice(grid.devs[d]));
                ok = P21_CUDA_OK(cudaStreamSynchronize(grid.streams[d]));
            }
        }
        for (int g = 0; g < G; ++g) {
            p21_mgpu_free(grid.devs[g], d_pack[(size_t)g], grid.streams[g]);
            p21_mgpu_free(grid.devs[g], d_recv[(size_t)g], grid.streams[g]);
        }
        if (ok) {
            return true;
        }
    }

    /* Low-VRAM / fallback host-staged path (safe for 2048^3 on 4x 24 GB GPUs). */
    size_t max_tile_bytes = (size_t)grid.max_sx * (size_t)grid.max_sy * (size_t)hz * sizeof(cufftComplex);
    cufftComplex *h_stage = (cufftComplex *)h_stage_opt;
    bool own_stage = false;
    if (!h_stage) {
        h_stage = (cufftComplex *)malloc(grid.total_cplx_elems() * sizeof(cufftComplex));
        if (!h_stage) {
            return false;
        }
        own_stage = true;
    }

    std::vector<cufftComplex *> d_tile((size_t)G, nullptr);
    bool ok = true;
    for (int g = 0; g < G; ++g) {
        d_tile[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(grid.devs[g], max_tile_bytes, grid.streams[g]);
        if (!d_tile[(size_t)g]) {
            ok = false;
        }
    }

    for (int s = 0; s < G && ok; ++s) {
        P21_CUDA_OK(cudaSetDevice(grid.devs[s]));
        for (int d = 0; d < G && ok; ++d) {
            size_t tile_elems = (size_t)grid.sx[s] * (size_t)grid.sy[d] * (size_t)hz;
            cufftComplex *h_slot = h_stage + ((size_t)grid.y0[d] * (size_t)hz * (size_t)nx) +
                                   (size_t)grid.x0[s] * (size_t)grid.sy[d] * (size_t)hz;
            mgpu_pack_xy_tile_kernel<<<grid_blocks_u32(tile_elems), kMgpuThreads, 0,
                                     grid.streams[s]>>>(
                d_cplx_x[s], grid.sx[s], ny, hz, grid.y0[d], grid.sy[d], d_tile[(size_t)s]);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 P21_CUDA_OK(cudaMemcpyAsync(h_slot, d_tile[(size_t)s],
                                             tile_elems * sizeof(cufftComplex),
                                             cudaMemcpyDeviceToHost, grid.streams[s])) &&
                 P21_CUDA_OK(cudaStreamSynchronize(grid.streams[s]));
        }
    }

    for (int d = 0; d < G && ok; ++d) {
        P21_CUDA_OK(cudaSetDevice(grid.devs[d]));
        for (int s = 0; s < G && ok; ++s) {
            size_t tile_elems = (size_t)grid.sx[s] * (size_t)grid.sy[d] * (size_t)hz;
            const cufftComplex *h_slot = h_stage + ((size_t)grid.y0[d] * (size_t)hz * (size_t)nx) +
                                         (size_t)grid.x0[s] * (size_t)grid.sy[d] * (size_t)hz;
            ok = P21_CUDA_OK(cudaMemcpyAsync(d_tile[(size_t)d], h_slot,
                                             tile_elems * sizeof(cufftComplex),
                                             cudaMemcpyHostToDevice, grid.streams[d]));
            if (ok) {
                mgpu_unpack_xy_tile_kernel<<<grid_blocks_u32(tile_elems), kMgpuThreads, 0,
                                           grid.streams[d]>>>(
                    d_tile[(size_t)d], grid.x0[s], grid.sx[s], grid.sy[d], hz, nx, d_cplx_y[d]);
                ok = P21_CUDA_OK(cudaGetLastError()) &&
                     P21_CUDA_OK(cudaStreamSynchronize(grid.streams[d]));
            }
        }
    }

    for (int g = 0; g < G; ++g) {
        p21_mgpu_free(grid.devs[g], d_tile[(size_t)g], grid.streams[g]);
    }
    if (own_stage) {
        free(h_stage);
    }
    return ok;
}

bool mgpu_transpose_y_to_x(const P21MgpuGrid &grid, cufftComplex *const *d_cplx_y,
                           cufftComplex **d_cplx_x, void *h_stage_opt) {
    const int G = grid.n_gpus;
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int hz = grid.hz;

    size_t max_slab_bytes = 0;
    for (int g = 0; g < G; ++g) {
        max_slab_bytes = std::max(max_slab_bytes, grid.cplx_elems_x(g) * sizeof(cufftComplex));
    }
    const size_t min_free = p21_mgpu_min_free_mem(grid);
    const bool can_peer = (h_stage_opt == nullptr) && (min_free >= 2 * max_slab_bytes + (512ULL << 20));

    if (can_peer) {
        std::vector<cufftComplex *> d_pack((size_t)G, nullptr);
        std::vector<cufftComplex *> d_recv((size_t)G, nullptr);
        bool ok = true;
        for (int g = 0; g < G; ++g) {
            d_pack[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(
                grid.devs[g], grid.cplx_elems_y(g) * sizeof(cufftComplex), grid.streams[g]);
            d_recv[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(
                grid.devs[g], grid.cplx_elems_x(g) * sizeof(cufftComplex), grid.streams[g]);
            if (!d_pack[(size_t)g] || !d_recv[(size_t)g]) {
                ok = false;
            }
        }
        if (ok) {
            for (int d = 0; d < G && ok; ++d) {
                P21_CUDA_OK(cudaSetDevice(grid.devs[d]));
                for (int s = 0; s < G && ok; ++s) {
                    size_t tile_elems = (size_t)grid.sx[s] * (size_t)grid.sy[d] * (size_t)hz;
                    cufftComplex *tile_ptr =
                        d_pack[(size_t)d] + (size_t)grid.x0[s] * (size_t)grid.sy[d] * (size_t)hz;
                    mgpu_pack_yx_tile_kernel<<<grid_blocks_u32(tile_elems), kMgpuThreads, 0,
                                             grid.streams[d]>>>(
                        d_cplx_y[d], grid.sy[d], hz, nx, grid.x0[s], grid.sx[s], tile_ptr);
                    ok = P21_CUDA_OK(cudaGetLastError());
                }
            }
            for (int d = 0; d < G && ok; ++d) {
                P21_CUDA_OK(cudaSetDevice(grid.devs[d]));
                ok = P21_CUDA_OK(cudaStreamSynchronize(grid.streams[d]));
            }
            for (int d = 0; d < G && ok; ++d) {
                for (int s = 0; s < G && ok; ++s) {
                    size_t tile_bytes =
                        (size_t)grid.sx[s] * (size_t)grid.sy[d] * (size_t)hz * sizeof(cufftComplex);
                    const cufftComplex *src_ptr =
                        d_pack[(size_t)d] + (size_t)grid.x0[s] * (size_t)grid.sy[d] * (size_t)hz;
                    cufftComplex *dst_ptr =
                        d_recv[(size_t)s] + (size_t)grid.sx[s] * (size_t)grid.y0[d] * (size_t)hz;
                    if (s == d) {
                        P21_CUDA_OK(cudaSetDevice(grid.devs[s]));
                        ok = P21_CUDA_OK(cudaMemcpyAsync(dst_ptr, src_ptr, tile_bytes,
                                                         cudaMemcpyDeviceToDevice, grid.streams[s]));
                    } else {
                        ok = P21_CUDA_OK(cudaMemcpyPeerAsync(dst_ptr, grid.devs[s], src_ptr,
                                                             grid.devs[d], tile_bytes,
                                                             grid.streams[s]));
                    }
                }
            }
            MgpuPart ypart{};
            ypart.n_parts = G;
            for (int d = 0; d < G; ++d) {
                ypart.off[d] = grid.y0[d];
                ypart.len[d] = grid.sy[d];
            }
            for (int s = 0; s < G && ok; ++s) {
                P21_CUDA_OK(cudaSetDevice(grid.devs[s]));
                size_t elems_x = grid.cplx_elems_x(s);
                mgpu_unpack_yx_full_kernel<<<grid_blocks_u32(elems_x), kMgpuThreads, 0,
                                           grid.streams[s]>>>(d_recv[(size_t)s], grid.sx[s], ny,
                                                              hz, ypart, d_cplx_x[s]);
                ok = P21_CUDA_OK(cudaGetLastError());
            }
            for (int s = 0; s < G && ok; ++s) {
                P21_CUDA_OK(cudaSetDevice(grid.devs[s]));
                ok = P21_CUDA_OK(cudaStreamSynchronize(grid.streams[s]));
            }
        }
        for (int g = 0; g < G; ++g) {
            p21_mgpu_free(grid.devs[g], d_pack[(size_t)g], grid.streams[g]);
            p21_mgpu_free(grid.devs[g], d_recv[(size_t)g], grid.streams[g]);
        }
        if (ok) {
            return true;
        }
    }

    size_t max_tile_bytes = (size_t)grid.max_sx * (size_t)grid.max_sy * (size_t)hz * sizeof(cufftComplex);
    cufftComplex *h_stage = (cufftComplex *)h_stage_opt;
    bool own_stage = false;
    if (!h_stage) {
        h_stage = (cufftComplex *)malloc(grid.total_cplx_elems() * sizeof(cufftComplex));
        if (!h_stage) {
            return false;
        }
        own_stage = true;
    }

    std::vector<cufftComplex *> d_tile((size_t)G, nullptr);
    bool ok = true;
    for (int g = 0; g < G; ++g) {
        d_tile[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(grid.devs[g], max_tile_bytes, grid.streams[g]);
        if (!d_tile[(size_t)g]) {
            ok = false;
        }
    }

    for (int d = 0; d < G && ok; ++d) {
        P21_CUDA_OK(cudaSetDevice(grid.devs[d]));
        for (int s = 0; s < G && ok; ++s) {
            size_t tile_elems = (size_t)grid.sx[s] * (size_t)grid.sy[d] * (size_t)hz;
            cufftComplex *h_slot = h_stage + ((size_t)grid.x0[s] * (size_t)ny * (size_t)hz) +
                                   (size_t)grid.sx[s] * (size_t)grid.y0[d] * (size_t)hz;
            mgpu_pack_yx_tile_kernel<<<grid_blocks_u32(tile_elems), kMgpuThreads, 0,
                                     grid.streams[d]>>>(
                d_cplx_y[d], grid.sy[d], hz, nx, grid.x0[s], grid.sx[s], d_tile[(size_t)d]);
            ok = P21_CUDA_OK(cudaGetLastError()) &&
                 P21_CUDA_OK(cudaMemcpyAsync(h_slot, d_tile[(size_t)d],
                                             tile_elems * sizeof(cufftComplex),
                                             cudaMemcpyDeviceToHost, grid.streams[d])) &&
                 P21_CUDA_OK(cudaStreamSynchronize(grid.streams[d]));
        }
    }

    for (int s = 0; s < G && ok; ++s) {
        P21_CUDA_OK(cudaSetDevice(grid.devs[s]));
        for (int d = 0; d < G && ok; ++d) {
            size_t tile_elems = (size_t)grid.sx[s] * (size_t)grid.sy[d] * (size_t)hz;
            const cufftComplex *h_slot = h_stage + ((size_t)grid.x0[s] * (size_t)ny * (size_t)hz) +
                                         (size_t)grid.sx[s] * (size_t)grid.y0[d] * (size_t)hz;
            ok = P21_CUDA_OK(cudaMemcpyAsync(d_tile[(size_t)s], h_slot,
                                             tile_elems * sizeof(cufftComplex),
                                             cudaMemcpyHostToDevice, grid.streams[s]));
            if (ok) {
                mgpu_unpack_yx_tile_kernel<<<grid_blocks_u32(tile_elems), kMgpuThreads, 0,
                                           grid.streams[s]>>>(
                    d_tile[(size_t)s], grid.sx[s], grid.y0[d], grid.sy[d], ny, hz, d_cplx_x[s]);
                ok = P21_CUDA_OK(cudaGetLastError()) &&
                     P21_CUDA_OK(cudaStreamSynchronize(grid.streams[s]));
            }
        }
    }

    for (int g = 0; g < G; ++g) {
        p21_mgpu_free(grid.devs[g], d_tile[(size_t)g], grid.streams[g]);
    }
    if (own_stage) {
        free(h_stage);
    }
    return ok;
}

}  // namespace

void *p21_mgpu_malloc(int dev, size_t nbytes, cudaStream_t stream) {
    size_t bytes = nbytes > 0 ? nbytes : 256;
    if (!P21_CUDA_OK(cudaSetDevice(dev))) {
        return nullptr;
    }
    void *ptr = nullptr;
    cudaError_t err = cudaErrorUnknown;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    if (stream != nullptr) {
        err = cudaMallocAsync(&ptr, bytes, stream);
        if (err != cudaSuccess) {
            cudaGetLastError();
            cudaMemPool_t pool = nullptr;
            if (cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess && pool != nullptr) {
                cudaStreamSynchronize(stream);
                cudaMemPoolTrimTo(pool, 0);
            }
            cudaGetLastError();
            err = cudaMalloc(&ptr, bytes);
        }
    } else {
        err = cudaMalloc(&ptr, bytes);
    }
#else
    (void)stream;
    err = cudaMalloc(&ptr, bytes);
#endif
    if (err != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return ptr;
}

void p21_mgpu_free(int dev, void *ptr, cudaStream_t stream) {
    if (!ptr) {
        return;
    }
    cudaSetDevice(dev);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
    if (stream != nullptr && cudaFreeAsync(ptr, stream) == cudaSuccess) {
        return;
    }
    cudaGetLastError();
#else
    (void)stream;
#endif
    cudaFree(ptr);
    cudaGetLastError();
}

size_t p21_mgpu_min_free_mem(const P21MgpuGrid &grid) {
    size_t min_free = SIZE_MAX;
    for (int g = 0; g < grid.n_gpus; ++g) {
        if (cudaSetDevice(grid.devs[g]) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        size_t fb = 0, tb = 0;
        if (cudaMemGetInfo(&fb, &tb) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        min_free = std::min(min_free, fb);
    }
    cudaSetDevice(grid.primary_dev);
    return min_free == SIZE_MAX ? 0 : min_free;
}

bool P21MgpuGrid::init(int nx_, int ny_, int nz_) {
    primary_dev = p21_cuda_current_device();
    if (primary_dev < 0) {
        primary_dev = 0;
    }
    nx = nx_;
    ny = ny_;
    nz = nz_;
    hz = nz_ / 2 + 1;
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        return false;
    }

    int count = p21_cuda_get_devices(devs, kMgpuMaxDevices);
    int limit = std::min(nx, ny);
    n_gpus = std::max(1, std::min(count, std::min(limit, kMgpuMaxDevices)));

    int rx = nx % n_gpus, bx = nx / n_gpus, cx = 0;
    int ry = ny % n_gpus, by = ny / n_gpus, cy = 0;
    max_sx = 0;
    max_sy = 0;
    for (int g = 0; g < n_gpus; ++g) {
        sx[g] = bx + (g < rx ? 1 : 0);
        x0[g] = cx;
        cx += sx[g];
        sy[g] = by + (g < ry ? 1 : 0);
        y0[g] = cy;
        cy += sy[g];
        max_sx = std::max(max_sx, sx[g]);
        max_sy = std::max(max_sy, sy[g]);
        streams[g] = nullptr;
    }

    for (int g = 0; g < n_gpus; ++g) {
        if (!P21_CUDA_OK(cudaSetDevice(devs[g]))) {
            destroy();
            return false;
        }
        if (!P21_CUDA_OK(cudaStreamCreateWithFlags(&streams[g], cudaStreamNonBlocking))) {
            destroy();
            return false;
        }
        for (int j = 0; j < n_gpus; ++j) {
            peer_enabled[g][j] = (g == j);
            if (g != j) {
                int can = 0;
                if (cudaDeviceCanAccessPeer(&can, devs[g], devs[j]) == cudaSuccess && can) {
                    cudaError_t perr = cudaDeviceEnablePeerAccess(devs[j], 0);
                    if (perr == cudaSuccess || perr == cudaErrorPeerAccessAlreadyEnabled) {
                        peer_enabled[g][j] = true;
                    }
                    cudaGetLastError();
                } else {
                    cudaGetLastError();
                }
            }
        }
    }
    cudaSetDevice(primary_dev);
    return true;
}

void P21MgpuGrid::destroy(void) {
    const size_t total_elems = total_cplx_elems();
    for (int g = 0; g < n_gpus; ++g) {
        if (streams[g]) {
            cudaSetDevice(devs[g]);
            cudaStreamSynchronize(streams[g]);
            cudaStreamDestroy(streams[g]);
            streams[g] = nullptr;
            cudaGetLastError();
        }
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
        if (total_elems * sizeof(cufftComplex) >= (1ULL << 30)) {
            cudaSetDevice(devs[g]);
            cudaMemPool_t pool = nullptr;
            if (cudaDeviceGetDefaultMemPool(&pool, devs[g]) == cudaSuccess && pool != nullptr) {
                cudaMemPoolTrimTo(pool, 0);
            }
            cudaGetLastError();
        }
#endif
    }
    cudaSetDevice(primary_dev);
    cudaGetLastError();
    n_gpus = 0;
}

bool p21_mgpu_fft_r2c_slabs(const P21MgpuGrid &grid, float *const *d_real_x,
                            cufftComplex **d_cplx_y, void *h_stage_opt) {
    const int G = grid.n_gpus;
    std::vector<cufftComplex *> d_cplx_x((size_t)G, nullptr);
    std::vector<bool> owns_x((size_t)G, false);

    bool ok = true;
    for (int g = 0; g < G; ++g) {
        size_t need_x = grid.cplx_elems_x(g);
        if (need_x <= grid.cplx_elems_y(g)) {
            d_cplx_x[(size_t)g] = d_cplx_y[g];
        } else {
            d_cplx_x[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(
                grid.devs[g], need_x * sizeof(cufftComplex), grid.streams[g]);
            owns_x[(size_t)g] = true;
            if (!d_cplx_x[(size_t)g]) {
                ok = false;
            }
        }
    }

    for (int g = 0; g < G && ok; ++g) {
        ok = mgpu_local_2d_r2c(grid.devs[g], grid.streams[g], grid.sx[g], grid.ny, grid.nz,
                               d_real_x[g], d_cplx_x[(size_t)g]);
    }
    for (int g = 0; g < G && ok; ++g) {
        cudaSetDevice(grid.devs[g]);
        ok = P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
    }

    if (ok) {
        ok = mgpu_transpose_x_to_y(grid, d_cplx_x.data(), d_cplx_y, h_stage_opt);
    }

    for (int g = 0; g < G; ++g) {
        if (owns_x[(size_t)g] && d_cplx_x[(size_t)g]) {
            p21_mgpu_free(grid.devs[g], d_cplx_x[(size_t)g], grid.streams[g]);
        }
    }

    for (int g = 0; g < G && ok; ++g) {
        ok = mgpu_local_1d_c2c(grid.devs[g], grid.streams[g], grid.sy[g] * grid.hz, grid.nx,
                               d_cplx_y[g], CUFFT_FORWARD);
    }
    for (int g = 0; g < G && ok; ++g) {
        cudaSetDevice(grid.devs[g]);
        ok = P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
    }
    cudaSetDevice(grid.primary_dev);
    if (ok) {
        p21_cuda_stats_mut()->mgpu_fft_runs++;
        p21_cuda_stats_mut()->cufft_exec_r2c++;
    }
    return ok;
}

bool p21_mgpu_fft_r2c_from_host(const P21MgpuGrid &grid, const float *h_real_in, double pre_scale,
                                cufftComplex **d_cplx_y, void *h_stage_opt) {
    if (!h_real_in) {
        return false;
    }
    const int G = grid.n_gpus;
    const int ny = grid.ny;
    const int nz = grid.nz;
    const int hz = grid.hz;
    const size_t plane_real = (size_t)ny * (size_t)nz;
    const size_t plane_cplx = (size_t)ny * (size_t)hz;
    const size_t strip_cap_bytes = 128ULL << 20;
    const int strip_max = std::max(
        1, std::min(grid.max_sx, (int)(strip_cap_bytes / std::max<size_t>(1, plane_real * sizeof(float)))));

    std::vector<cufftComplex *> d_cplx_x((size_t)G, nullptr);
    std::vector<bool> owns_x((size_t)G, false);
    bool ok = true;

    for (int g = 0; g < G; ++g) {
        size_t need_x = grid.cplx_elems_x(g);
        if (need_x <= grid.cplx_elems_y(g)) {
            d_cplx_x[(size_t)g] = d_cplx_y[g];
        } else {
            d_cplx_x[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(
                grid.devs[g], need_x * sizeof(cufftComplex), grid.streams[g]);
            owns_x[(size_t)g] = true;
            if (!d_cplx_x[(size_t)g]) {
                ok = false;
            }
        }
    }

    for (int g = 0; g < G && ok; ++g) {
        int dev = grid.devs[g];
        cudaStream_t stream = grid.streams[g];
        int sx_g = grid.sx[g];
        if (sx_g <= 0) {
            continue;
        }
        if (!P21_CUDA_OK(cudaSetDevice(dev))) {
            ok = false;
            break;
        }
        int strip = std::min(strip_max, sx_g);
        float *d_strip = (float *)p21_mgpu_malloc(dev, (size_t)strip * plane_real * sizeof(float), stream);
        if (!d_strip) {
            ok = false;
            break;
        }
        for (int x = 0; ok && x < sx_g; x += strip) {
            int cur_nx = std::min(strip, sx_g - x);
            size_t elems = (size_t)cur_nx * plane_real;
            const float *src = h_real_in + (size_t)(grid.x0[g] + x) * plane_real;
            ok = P21_CUDA_OK(cudaMemcpyAsync(d_strip, src, elems * sizeof(float),
                                             cudaMemcpyHostToDevice, stream));
            if (ok && pre_scale != 1.0) {
                scale_strip_f32_kernel<<<grid_blocks_u32(elems), kMgpuThreads, 0, stream>>>(
                    d_strip, elems, pre_scale);
                ok = P21_CUDA_OK(cudaGetLastError());
            }
            if (ok) {
                ok = mgpu_local_2d_r2c(dev, stream, cur_nx, ny, nz, d_strip,
                                       d_cplx_x[(size_t)g] + (size_t)x * plane_cplx);
            }
        }
        p21_mgpu_free(dev, d_strip, stream);
        if (ok) {
            ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
        }
    }

    if (ok) {
        ok = mgpu_transpose_x_to_y(grid, d_cplx_x.data(), d_cplx_y, h_stage_opt);
    }
    for (int g = 0; g < G; ++g) {
        if (owns_x[(size_t)g] && d_cplx_x[(size_t)g]) {
            p21_mgpu_free(grid.devs[g], d_cplx_x[(size_t)g], grid.streams[g]);
        }
    }
    for (int g = 0; g < G && ok; ++g) {
        ok = mgpu_local_1d_c2c(grid.devs[g], grid.streams[g], grid.sy[g] * hz, grid.nx, d_cplx_y[g],
                               CUFFT_FORWARD);
    }
    for (int g = 0; g < G && ok; ++g) {
        cudaSetDevice(grid.devs[g]);
        ok = P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
    }
    cudaSetDevice(grid.primary_dev);
    if (ok) {
        p21_cuda_stats_mut()->mgpu_fft_runs++;
        p21_cuda_stats_mut()->cufft_exec_r2c++;
    }
    return ok;
}

bool p21_mgpu_fft_c2r_strips(const P21MgpuGrid &grid, cufftComplex **d_cplx_y, void *h_stage_opt,
                             const P21MgpuRealStripFn &on_strip) {
    const int G = grid.n_gpus;
    const int ny = grid.ny;
    const int nz = grid.nz;
    const int hz = grid.hz;
    const size_t plane_real = (size_t)ny * (size_t)nz;
    const size_t plane_cplx = (size_t)ny * (size_t)hz;
    const size_t strip_cap_bytes = 128ULL << 20;
    const int strip_max = std::max(
        1, std::min(grid.max_sx, (int)(strip_cap_bytes / std::max<size_t>(1, plane_real * sizeof(float)))));

    bool ok = true;
    for (int g = 0; g < G && ok; ++g) {
        ok = mgpu_local_1d_c2c(grid.devs[g], grid.streams[g], grid.sy[g] * hz, grid.nx, d_cplx_y[g],
                               CUFFT_INVERSE);
    }
    for (int g = 0; g < G && ok; ++g) {
        cudaSetDevice(grid.devs[g]);
        ok = P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
    }

    std::vector<cufftComplex *> d_cplx_x((size_t)G, nullptr);
    std::vector<bool> owns_x((size_t)G, false);
    for (int g = 0; g < G && ok; ++g) {
        size_t need_x = grid.cplx_elems_x(g);
        if (need_x <= grid.cplx_elems_y(g)) {
            d_cplx_x[(size_t)g] = d_cplx_y[g];
        } else {
            d_cplx_x[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(
                grid.devs[g], need_x * sizeof(cufftComplex), grid.streams[g]);
            owns_x[(size_t)g] = true;
            if (!d_cplx_x[(size_t)g]) {
                ok = false;
            }
        }
    }

    if (ok) {
        ok = mgpu_transpose_y_to_x(grid, d_cplx_y, d_cplx_x.data(), h_stage_opt);
    }

    for (int g = 0; g < G && ok; ++g) {
        int dev = grid.devs[g];
        cudaStream_t stream = grid.streams[g];
        int sx_g = grid.sx[g];
        if (sx_g <= 0) {
            continue;
        }
        if (!P21_CUDA_OK(cudaSetDevice(dev))) {
            ok = false;
            break;
        }
        int strip = std::min(strip_max, sx_g);
        float *d_strip = (float *)p21_mgpu_malloc(dev, (size_t)strip * plane_real * sizeof(float), stream);
        if (!d_strip) {
            ok = false;
            break;
        }
        for (int x = 0; ok && x < sx_g; x += strip) {
            int cur_nx = std::min(strip, sx_g - x);
            ok = mgpu_local_2d_c2r(dev, stream, cur_nx, ny, nz,
                                   d_cplx_x[(size_t)g] + (size_t)x * plane_cplx, d_strip) &&
                 on_strip(g, grid.x0[g] + x, cur_nx, d_strip, stream);
        }
        if (ok) {
            ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
        }
        p21_mgpu_free(dev, d_strip, stream);
    }

    for (int g = 0; g < G; ++g) {
        if (owns_x[(size_t)g] && d_cplx_x[(size_t)g]) {
            p21_mgpu_free(grid.devs[g], d_cplx_x[(size_t)g], grid.streams[g]);
        }
    }
    cudaSetDevice(grid.primary_dev);
    if (ok) {
        p21_cuda_stats_mut()->mgpu_fft_runs++;
        p21_cuda_stats_mut()->cufft_exec_c2r++;
    }
    return ok;
}

bool p21_mgpu_fft_c2r_slabs(const P21MgpuGrid &grid, cufftComplex **d_cplx_y, float **d_real_x,
                            void *h_stage_opt) {
    const int G = grid.n_gpus;
    bool ok = true;
    for (int g = 0; g < G && ok; ++g) {
        ok = mgpu_local_1d_c2c(grid.devs[g], grid.streams[g], grid.sy[g] * grid.hz, grid.nx,
                               d_cplx_y[g], CUFFT_INVERSE);
    }
    for (int g = 0; g < G && ok; ++g) {
        cudaSetDevice(grid.devs[g]);
        ok = P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
    }

    std::vector<cufftComplex *> d_cplx_x((size_t)G, nullptr);
    std::vector<bool> owns_x((size_t)G, false);
    for (int g = 0; g < G && ok; ++g) {
        size_t need_x = grid.cplx_elems_x(g);
        if (need_x <= grid.cplx_elems_y(g)) {
            d_cplx_x[(size_t)g] = d_cplx_y[g];
        } else {
            d_cplx_x[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(
                grid.devs[g], need_x * sizeof(cufftComplex), grid.streams[g]);
            owns_x[(size_t)g] = true;
            if (!d_cplx_x[(size_t)g]) {
                ok = false;
            }
        }
    }

    if (ok) {
        ok = mgpu_transpose_y_to_x(grid, d_cplx_y, d_cplx_x.data(), h_stage_opt);
    }
    for (int g = 0; g < G && ok; ++g) {
        ok = mgpu_local_2d_c2r(grid.devs[g], grid.streams[g], grid.sx[g], grid.ny, grid.nz,
                               d_cplx_x[(size_t)g], d_real_x[g]);
    }
    for (int g = 0; g < G && ok; ++g) {
        cudaSetDevice(grid.devs[g]);
        ok = P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
    }

    for (int g = 0; g < G; ++g) {
        if (owns_x[(size_t)g] && d_cplx_x[(size_t)g]) {
            p21_mgpu_free(grid.devs[g], d_cplx_x[(size_t)g], grid.streams[g]);
        }
    }
    cudaSetDevice(grid.primary_dev);
    if (ok) {
        p21_cuda_stats_mut()->mgpu_fft_runs++;
        p21_cuda_stats_mut()->cufft_exec_c2r++;
    }
    return ok;
}

bool p21_mgpu_enforce_real_planes_yslabs(const P21MgpuGrid &grid, cufftComplex **d_cplx_y) {
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;
    const int hz = grid.hz;
    const int mx = nx / 2;
    const int my = ny / 2;
    const int mz = nz / 2;
    const int G = grid.n_gpus;

    int planes_k[2] = {0, mz};
    int n_planes = (mz > 0) ? 2 : 1;
    std::vector<cufftComplex> plane((size_t)nx * (size_t)ny);

    for (int p_idx = 0; p_idx < n_planes; ++p_idx) {
        int k = planes_k[p_idx];
        for (int g = 0; g < G; ++g) {
            cudaSetDevice(grid.devs[g]);
            const cufftComplex *src_base = d_cplx_y[g] + (size_t)k * (size_t)nx;
            size_t dpitch = (size_t)nx * sizeof(cufftComplex);
            size_t spitch = (size_t)hz * (size_t)nx * sizeof(cufftComplex);
            cufftComplex *dst_host = plane.data() + (size_t)grid.y0[g] * (size_t)nx;
            if (!P21_CUDA_OK(cudaMemcpy2DAsync(dst_host, dpitch, src_base, spitch, dpitch,
                                               (size_t)grid.sy[g], cudaMemcpyDeviceToHost,
                                               grid.streams[g])) ||
                !P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]))) {
                cudaSetDevice(grid.primary_dev);
                return false;
            }
        }

        auto at = [&](int i, int j) -> cufftComplex & {
            return plane[(size_t)j * (size_t)nx + (size_t)i];
        };

        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                if (i == 0 && j == 0 && k == 0) {
                    at(0, 0) = make_float2(0.0f, 0.0f);
                } else if ((i == 0 || i == mx) && (j == 0 || j == my)) {
                    at(i, j).y = 0.0f;
                } else if (i > 0 && i < mx) {
                    if (j == 0 || j == my) {
                        cufftComplex v = at(nx - i, j);
                        at(i, j) = make_float2(v.x, -v.y);
                    } else if (j < my || j > ny - my) {
                        cufftComplex v = at(nx - i, ny - j);
                        at(i, j) = make_float2(v.x, -v.y);
                    }
                } else if ((i == 0 || i == mx) && j > 0 && j < my) {
                    cufftComplex v = at(i, ny - j);
                    at(i, j) = make_float2(v.x, -v.y);
                }
            }
        }

        for (int g = 0; g < G; ++g) {
            cudaSetDevice(grid.devs[g]);
            cufftComplex *dst_base = d_cplx_y[g] + (size_t)k * (size_t)nx;
            size_t spitch = (size_t)nx * sizeof(cufftComplex);
            size_t dpitch = (size_t)hz * (size_t)nx * sizeof(cufftComplex);
            const cufftComplex *src_host = plane.data() + (size_t)grid.y0[g] * (size_t)nx;
            if (!P21_CUDA_OK(cudaMemcpy2DAsync(dst_base, dpitch, src_host, spitch, spitch,
                                               (size_t)grid.sy[g], cudaMemcpyHostToDevice,
                                               grid.streams[g])) ||
                !P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]))) {
                cudaSetDevice(grid.primary_dev);
                return false;
            }
        }
    }

    cudaSetDevice(grid.primary_dev);
    return true;
}

extern "C" int p21_cuda_mgpu_fft_r2c_host(const float *h_in, float *h_out_complex, int nx, int ny,
                                          int nz) {
    if (!h_in || !h_out_complex || nx <= 0 || ny <= 0 || nz <= 0 || !p21_cuda_ensure_init()) {
        return -1;
    }
    P21MgpuGrid grid;
    if (!grid.init(nx, ny, nz)) {
        return -1;
    }
    const int G = grid.n_gpus;
    std::vector<float *> d_x((size_t)G, nullptr);
    std::vector<cufftComplex *> d_y((size_t)G, nullptr);
    bool ok = true;

    for (int g = 0; g < G; ++g) {
        d_x[(size_t)g] = (float *)p21_mgpu_malloc(
            grid.devs[g], grid.real_elems_x(g) * sizeof(float), grid.streams[g]);
        d_y[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(
            grid.devs[g], grid.cplx_elems_y(g) * sizeof(cufftComplex), grid.streams[g]);
        if (!d_x[(size_t)g] || !d_y[(size_t)g]) {
            ok = false;
        }
    }

    for (int g = 0; g < G && ok; ++g) {
        cudaSetDevice(grid.devs[g]);
        const float *src = h_in + (size_t)grid.x0[g] * (size_t)ny * (size_t)nz;
        ok = P21_CUDA_OK(cudaMemcpyAsync(d_x[(size_t)g], src,
                                         grid.real_elems_x(g) * sizeof(float),
                                         cudaMemcpyHostToDevice, grid.streams[g])) &&
             P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
    }

    if (ok) {
        ok = p21_mgpu_fft_r2c_slabs(grid, d_x.data(), d_y.data(), nullptr);
    }

    /* Transpose back? Caller expects standard (nx, ny, hz) complex64 layout -> download Y-slabs into
     * host and scatter to (nx, ny, hz). */
    cufftComplex *h_out = (cufftComplex *)h_out_complex;
    if (ok) {
        for (int g = 0; g < G && ok; ++g) {
            cudaSetDevice(grid.devs[g]);
            std::vector<cufftComplex> y_host(grid.cplx_elems_y(g));
            ok = P21_CUDA_OK(cudaMemcpyAsync(y_host.data(), d_y[(size_t)g],
                                             y_host.size() * sizeof(cufftComplex),
                                             cudaMemcpyDeviceToHost, grid.streams[g])) &&
                 P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
            if (ok) {
                for (int y_loc = 0; y_loc < grid.sy[g]; ++y_loc) {
                    int y = grid.y0[g] + y_loc;
                    for (int z = 0; z < grid.hz; ++z) {
                        const cufftComplex *row =
                            y_host.data() + ((size_t)y_loc * (size_t)grid.hz + (size_t)z) * (size_t)nx;
                        for (int x = 0; x < nx; ++x) {
                            h_out[((size_t)x * (size_t)ny + (size_t)y) * (size_t)grid.hz + (size_t)z] =
                                row[x];
                        }
                    }
                }
            }
        }
    }

    for (int g = 0; g < G; ++g) {
        p21_mgpu_free(grid.devs[g], d_y[(size_t)g], grid.streams[g]);
        p21_mgpu_free(grid.devs[g], d_x[(size_t)g], grid.streams[g]);
    }
    grid.destroy();
    return ok ? 0 : -1;
}

extern "C" int p21_cuda_mgpu_fft_c2r_host(const float *h_in_complex, float *h_out_real, int nx,
                                          int ny, int nz) {
    if (!h_in_complex || !h_out_real || nx <= 0 || ny <= 0 || nz <= 0 || !p21_cuda_ensure_init()) {
        return -1;
    }
    P21MgpuGrid grid;
    if (!grid.init(nx, ny, nz)) {
        return -1;
    }
    const int G = grid.n_gpus;
    const cufftComplex *h_in = (const cufftComplex *)h_in_complex;
    std::vector<cufftComplex *> d_y((size_t)G, nullptr);
    std::vector<float *> d_x((size_t)G, nullptr);
    bool ok = true;

    for (int g = 0; g < G; ++g) {
        d_y[(size_t)g] = (cufftComplex *)p21_mgpu_malloc(
            grid.devs[g], grid.cplx_elems_y(g) * sizeof(cufftComplex), grid.streams[g]);
        d_x[(size_t)g] = (float *)p21_mgpu_malloc(
            grid.devs[g], grid.real_elems_x(g) * sizeof(float), grid.streams[g]);
        if (!d_y[(size_t)g] || !d_x[(size_t)g]) {
            ok = false;
        }
    }

    for (int g = 0; g < G && ok; ++g) {
        cudaSetDevice(grid.devs[g]);
        std::vector<cufftComplex> y_host(grid.cplx_elems_y(g));
        for (int y_loc = 0; y_loc < grid.sy[g]; ++y_loc) {
            int y = grid.y0[g] + y_loc;
            for (int z = 0; z < grid.hz; ++z) {
                cufftComplex *row =
                    y_host.data() + ((size_t)y_loc * (size_t)grid.hz + (size_t)z) * (size_t)nx;
                for (int x = 0; x < nx; ++x) {
                    row[x] = h_in[((size_t)x * (size_t)ny + (size_t)y) * (size_t)grid.hz + (size_t)z];
                }
            }
        }
        ok = P21_CUDA_OK(cudaMemcpyAsync(d_y[(size_t)g], y_host.data(),
                                         y_host.size() * sizeof(cufftComplex),
                                         cudaMemcpyHostToDevice, grid.streams[g])) &&
             P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
    }

    if (ok) {
        ok = p21_mgpu_fft_c2r_slabs(grid, d_y.data(), d_x.data(), nullptr);
    }

    for (int g = 0; g < G && ok; ++g) {
        cudaSetDevice(grid.devs[g]);
        float *dst = h_out_real + (size_t)grid.x0[g] * (size_t)ny * (size_t)nz;
        ok = P21_CUDA_OK(cudaMemcpyAsync(dst, d_x[(size_t)g],
                                         grid.real_elems_x(g) * sizeof(float),
                                         cudaMemcpyDeviceToHost, grid.streams[g])) &&
             P21_CUDA_OK(cudaStreamSynchronize(grid.streams[g]));
    }

    for (int g = 0; g < G; ++g) {
        p21_mgpu_free(grid.devs[g], d_y[(size_t)g], grid.streams[g]);
        p21_mgpu_free(grid.devs[g], d_x[(size_t)g], grid.streams[g]);
    }
    grid.destroy();
    return ok ? 0 : -1;
}
