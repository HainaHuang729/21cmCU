// Work in nearest periodic images, using the strict native centre-distance test.
struct DexmSphere {
    int side, lower, width, x, y, z;
    float radius_squared;

    __device__ DexmSphere(long long centre, int n, int range, float r2)
        : side(n),
          lower(max(-range, -n / 2)),
          width(min(range, (n - 1) / 2) - lower + 1),
          x(centre / (n * n)),
          y(centre / n % n),
          z(centre % n),
          radius_squared(r2) {}

    __device__ long long size() const { return (long long)width * width * width; }

    __device__ int cell(long long i) const {
        int dx = i / ((long long)width * width) + lower;
        int dy = (i / width) % width + lower, dz = i % width + lower;
        if ((float)dx * dx + (float)dy * dy + (float)dz * dz >= radius_squared) return -1;
        return (((x + dx + side) % side) * side + (y + dy + side) % side) * side +
               (z + dz + side) % side;
    }

    __device__ void paint_occupied(unsigned char *grid) const {
        for (long long i = threadIdx.x; i < size(); i += blockDim.x) {
            int pix = cell(i);
            if (pix >= 0) grid[pix] = 1;
        }
    }

    __device__ void paint_forbidden(unsigned int *bits) const {
        for (long long i = threadIdx.x; i < size(); i += blockDim.x) {
            int pix = cell(i);
            if (pix >= 0) atomicOr(bits + (pix >> 5), 1u << (pix & 31));
        }
    }
};

// One block accepts candidate centres in the native x/y/z order.
extern "C" __global__ void ordered_dexm(const long long *candidates, long long count,
                                        unsigned char *occupied, float *mass_field,
                                        unsigned long long *accepted, int side, int check_range,
                                        float check_sq, int paint_range, float paint_sq,
                                        float mass) {
    __shared__ int blocked;
    for (long long j = 0; j < count; ++j) {
        long long centre = candidates[j];
        if (threadIdx.x == 0) blocked = occupied[centre];
        __syncthreads();
        if (!blocked) {
            DexmSphere check(centre, side, check_range, check_sq);
            bool overlaps = false;
            for (long long i = threadIdx.x; i < check.size() && !overlaps; i += blockDim.x) {
                int pix = check.cell(i);
                overlaps = pix >= 0 && occupied[pix];
            }
            if (!__syncthreads_count(overlaps)) {
                if (threadIdx.x == 0) {
                    mass_field[centre] = mass;
                    ++*accepted;
                }
                DexmSphere(centre, side, paint_range, paint_sq).paint_occupied(occupied);
            }
        }
        __syncthreads();
    }
}

// One block per previous halo. Overlapping exclusion spheres share packed bits.
extern "C" __global__ void rebuild_forbidden(const long long *centres, unsigned int *forbidden,
                                             int side, int range, float squared) {
    DexmSphere(centres[blockIdx.x], side, range, squared).paint_forbidden(forbidden);
}

extern "C" __global__ void optimized_dexm(const long long *candidates, long long count,
                                          unsigned char *occupied, float *mass_field,
                                          unsigned long long *accepted, int side, int paint_range,
                                          float paint_sq, float mass, int forbidden_range,
                                          float forbidden_sq, unsigned int *forbidden,
                                          long long *new_centres) {
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
            DexmSphere(centre, side, paint_range, paint_sq).paint_occupied(occupied);
            DexmSphere(centre, side, forbidden_range, forbidden_sq).paint_forbidden(forbidden);
        }
        __syncthreads();
    }
}
