// Work in nearest periodic images, using the strict native centre-distance test.
struct DexmSphere {
    int nx, ny, nz, lx, wx, ly, wy, lz, wz, x, y, z;
    float radius_squared;

    __device__ DexmSphere(long long centre, int nx_, int ny_, int nz_, int range, float r2)
        : nx(nx_),
          ny(ny_),
          nz(nz_),
          lx(max(-range, -nx_ / 2)),
          wx(min(range, (nx_ - 1) / 2) - lx + 1),
          ly(max(-range, -ny_ / 2)),
          wy(min(range, (ny_ - 1) / 2) - ly + 1),
          lz(max(-range, -nz_ / 2)),
          wz(min(range, (nz_ - 1) / 2) - lz + 1),
          x((int)(centre / ((long long)ny_ * nz_))),
          y((int)(centre / nz_ % ny_)),
          z((int)(centre % nz_)),
          radius_squared(r2) {}

    __device__ long long size() const { return (long long)wx * wy * wz; }

    __device__ long long cell(long long i) const {
        long long wyz = (long long)wy * wz;
        int dx = (int)(i / wyz) + lx;
        int dy = (int)((i / wz) % wy) + ly, dz = (int)(i % wz) + lz;
        if ((float)dx * dx + (float)dy * dy + (float)dz * dz >= radius_squared) return -1;
        return (((long long)(x + dx + nx) % nx) * ny + (y + dy + ny) % ny) * nz +
               (z + dz + nz) % nz;
    }

    __device__ void paint_occupied(unsigned char *grid) const {
        for (long long i = threadIdx.x; i < size(); i += blockDim.x) {
            long long pix = cell(i);
            if (pix >= 0) grid[pix] = 1;
        }
    }

    __device__ void paint_forbidden(unsigned int *bits) const {
        for (long long i = threadIdx.x; i < size(); i += blockDim.x) {
            long long pix = cell(i);
            if (pix >= 0) atomicOr(bits + (pix >> 5), 1u << (pix & 31));
        }
    }
};

// One block accepts candidate centres in the native x/y/z order.
extern "C" __global__ void ordered_dexm(const long long *candidates, long long count,
                                        unsigned char *occupied, float *mass_field,
                                        unsigned long long *accepted, int nx, int ny, int nz,
                                        int check_range, float check_sq, int paint_range,
                                        float paint_sq, float mass) {
    __shared__ int blocked;
    for (long long j = 0; j < count; ++j) {
        long long centre = candidates[j];
        if (threadIdx.x == 0) blocked = occupied[centre];
        __syncthreads();
        if (!blocked) {
            DexmSphere check(centre, nx, ny, nz, check_range, check_sq);
            bool overlaps = false;
            for (long long i = threadIdx.x; i < check.size() && !overlaps; i += blockDim.x) {
                long long pix = check.cell(i);
                overlaps = pix >= 0 && occupied[pix];
            }
            if (!__syncthreads_count(overlaps)) {
                if (threadIdx.x == 0) {
                    mass_field[centre] = mass;
                    ++*accepted;
                }
                DexmSphere(centre, nx, ny, nz, paint_range, paint_sq).paint_occupied(occupied);
            }
        }
        __syncthreads();
    }
}

// One block per previous halo. Overlapping exclusion spheres share packed bits.
extern "C" __global__ void rebuild_forbidden(const long long *centres, unsigned int *forbidden,
                                             int nx, int ny, int nz, int range, float squared) {
    DexmSphere(centres[blockIdx.x], nx, ny, nz, range, squared).paint_forbidden(forbidden);
}

extern "C" __global__ void optimized_dexm(const long long *candidates, long long count,
                                          unsigned char *occupied, float *mass_field,
                                          unsigned long long *accepted, int nx, int ny, int nz,
                                          int paint_range, float paint_sq, float mass,
                                          int forbidden_range, float forbidden_sq,
                                          unsigned int *forbidden, long long *new_centres) {
    __shared__ int blocked;
    for (long long j = 0; j < count; ++j) {
        long long centre = candidates[j];
        if (threadIdx.x == 0) blocked = ((forbidden[centre >> 5] >> (centre & 31)) & 1u);
        __syncthreads();
        if (!blocked) {
            if (threadIdx.x == 0) {
                mass_field[centre] = mass;
                new_centres[(*accepted)++] = centre;
            }
            DexmSphere(centre, nx, ny, nz, paint_range, paint_sq).paint_occupied(occupied);
            DexmSphere(centre, nx, ny, nz, forbidden_range, forbidden_sq)
                .paint_forbidden(forbidden);
        }
        __syncthreads();
    }
}
