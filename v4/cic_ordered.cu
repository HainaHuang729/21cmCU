// The CPU adds contributions in particle order within each OpenMP slab.
// Revisit cells whose double-precision summation error could change the
// rounded float grid. Both passes use the same position and weight arithmetic.
struct CICGrid {
    const float *rho, *vel, *second;
    int nx, ny, nz, mx, my, mz;
    double ratio, growth;
    double first[3], correction[3];
    bool two;
    // The native x expression adds the coordinate before the 2LPT correction.
    // The y and z expressions combine the two displacements first.
    __device__ double lower_and_fraction(int axis, int coord, double first_shift,
                                         double second_shift, int &lower) const {
        double pos = coord + first_shift;
        if (two) pos = axis == 0 ? pos - second_shift : coord + (first_shift - second_shift);
        pos *= ratio;
        int m_axis = axis == 0 ? mx : (axis == 1 ? my : mz);
        int cell = (int)floor(pos);
        lower = (cell % m_axis + m_axis) % m_axis;
        return pos - cell;
    }
    __device__ void particle(int x, int y, int z, int *base, double *fraction) const {
        const int coords[] = {x, y, z};
        const int mdims[] = {mx, my, mz};
        const long long msize = (long long)mx * my * mz;
        long long source = 0;
        for (int a = 0; a < 3; ++a)
            source = source * mdims[a] + ((int)(coords[a] * ratio + .5)) % mdims[a];
        for (int a = 0; a < 3; ++a) {
            long long j = source + (long long)a * msize;
            fraction[a] = lower_and_fraction(a, coords[a], vel[j] * first[a],
                                             two ? second[j] * correction[a] : 0., base[a]);
        }
    }
    __device__ long long corner(int bits, const int *base, const double *frac, double &w) const {
        const int mdims[] = {mx, my, mz};
        long long cell = 0;
        w = 1.;
        for (int a = 0; a < 3; ++a) {
            int bit = (bits >> a) & 1;
            int ix = base[a] + bit;
            cell = cell * mdims[a] + (ix == mdims[a] ? 0 : ix);
            w *= bit ? frac[a] : 1. - frac[a];
        }
        return cell;
    }
    __device__ void scatter_site(int x, int y, int z, double *mass, int *counts) const {
        int base[3];
        double frac[3];
        particle(x, y, z, base, frac);
        double particle_mass = 1. + rho[((long long)x * ny + y) * nz + z] * growth;
        for (int b = 0; b < 8; ++b) {
            double w;
            long long dst = corner(b, base, frac, w);
            atomicAdd(mass + dst, w * particle_mass);
            atomicAdd(counts + dst, 1);
        }
    }
};

extern "C" __global__ void scatter(const float *rho, const float *vel, const float *second,
                                   double *mass, int *counts, int nx, int ny, int nz, int mx,
                                   int my, int mz, double ratio, double growth, double fx,
                                   double fy, double fz, double cx, double cy, double cz,
                                   bool two) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long yz = (long long)ny * nz;
    if (t >= (long long)nx * yz) return;
    CICGrid grid = {rho, vel, second, nx, ny, nz, mx, my, mz, ratio, growth,
                    {fx, fy, fz}, {cx, cy, cz}, two};
    grid.scatter_site((int)(t / yz), (int)(t / nz % ny), (int)(t % nz), mass, counts);
}

// Four fine samples per axis use one nearest coarse displacement. Accumulate
// their CIC contributions in a 3-cubed neighborhood, as in v3 cic_pertile.
// In 4.3.2 the group is centered on the coarse index (offsets -2..1).
extern "C" __global__ void scatter_tile4(const float *rho, const float *vel, const float *second,
                                         double *mass, int *counts, int fine, int coarse,
                                         double growth, double first, double correction, bool two) {
    const int tid = threadIdx.x;
    const long long tile = (long long)blockIdx.x * blockDim.x + tid;
    const long long size = (long long)coarse * coarse * coarse;
    if (tile >= size) return;
    const int center[] = {int(tile / coarse / coarse), int(tile / coarse % coarse),
                          int(tile % coarse)};

    int coordinates[3][4], offset[3][4], base[3], hits[3][3] = {};
    double weight[3][4][2];
    CICGrid grid = {rho, vel, second, fine, fine, fine, coarse, coarse, coarse,
                    (double)coarse / fine, growth, {first, first, first},
                    {correction, correction, correction}, two};
    bool compact = true;
    for (int axis = 0; axis < 3; ++axis) {
        const long long j = tile + axis * size;
        const double shift = vel[j] * first;
        const double shift2 = two ? second[j] * correction : 0.;
        for (int k = 0; k < 4; ++k) {
            int u = 4 * center[axis] + k - 2;
            if (u < 0) u += fine;
            coordinates[axis][k] = u;
            int cell;
            const double frac = grid.lower_and_fraction(axis, u, shift, shift2, cell);
            weight[axis][k][0] = 1. - frac;
            weight[axis][k][1] = frac;
            if (k == 0) base[axis] = cell;
            int dx = (cell - base[axis] + coarse) % coarse;
            offset[axis][k] = dx;
            if (dx > 1)
                compact = false;
            else {
                ++hits[axis][dx];
                ++hits[axis][dx + 1];
            }
        }
    }
    // Even for displacements large enough to break the three-cell span,
    // every original particle is deposited exactly once.
    if (!compact) {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    grid.scatter_site(coordinates[0][i], coordinates[1][j], coordinates[2][k], mass,
                                      counts);
        return;
    }
    __shared__ double sums[27][64];
#pragma unroll
    for (int c = 0; c < 27; ++c) sums[c][tid] = 0.;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k) {
                const long long idx =
                    ((long long)coordinates[0][i] * fine + coordinates[1][j]) * fine +
                    coordinates[2][k];
                const double particle_mass = 1. + growth * rho[idx];
#pragma unroll
                for (int dx = 0; dx < 2; ++dx) {
#pragma unroll
                    for (int dy = 0; dy < 2; ++dy) {
                        const double xy = weight[0][i][dx] * weight[1][j][dy];
#pragma unroll
                        for (int dz = 0; dz < 2; ++dz) {
                            const int x = offset[0][i] + dx, y = offset[1][j] + dy,
                                      z = offset[2][k] + dz;
                            const double term = (xy * weight[2][k][dz]) * particle_mass;
                            sums[(x * 3 + y) * 3 + z][tid] += term;
                        }
                    }
                }
            }
#pragma unroll
    for (int x = 0; x < 3; ++x) {
#pragma unroll
        for (int y = 0; y < 3; ++y) {
#pragma unroll
            for (int z = 0; z < 3; ++z) {
                const int count = hits[0][x] * hits[1][y] * hits[2][z];
                if (!count) continue;
                const long long dst =
                    ((long long)((base[0] + x) % coarse) * coarse + (base[1] + y) % coarse) *
                        coarse +
                    (base[2] + z) % coarse;
                atomicAdd(mass + dst, sums[(x * 3 + y) * 3 + z][tid]);
                atomicAdd(counts + dst, count);
            }
        }
    }
}

extern "C" __global__ void uncertain(const double *mass, const int *counts, const double *max_mass,
                                     bool *flags, long long total) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    double s = mass[i], r = (float)s;
    const float infinity = __int_as_float(0x7f800000);
    double lo = nextafterf((float)r, -infinity), hi = nextafterf((float)r, infinity);
    double distance = fmin(s - (r + (lo - r) * .5), (r + (hi - r) * .5) - s);
    // Both orders have error bounded by gamma(k)*sum(abs(contributions)).
    double k = counts[i], ku = k * 1.1102230246251565e-16;
    flags[i] = k > 1 && distance <= 2. * ku / (1. - ku) * k * (*max_mass);
}

// Return the wrapped interval in increasing order of original source index.
struct PeriodicRange {
    int start, count, wrap, n;
    __device__ PeriodicRange(int cell, int n_, double ratio, double shift) : n(n_) {
        int lo = (int)floor((cell - 1.) / ratio - shift) - 1;
        int hi = (int)ceil((cell + 1.) / ratio + shift) + 1;
        count = min(n, hi - lo + 1);
        start = count == n ? 0 : (lo % n + n) % n;
        wrap = max(0, start + count - n);
    }
    __device__ int at(int i) const { return i < wrap ? i : start + i - wrap; }
};

extern "C" __global__ void replay(const float *rho, const float *vel, const float *second,
                                  double *mass, const long long *indices, int num,
                                  const double *shift, int nx, int ny, int nz, int mx, int my,
                                  int mz, double ratio, double growth, double fx, double fy,
                                  double fz, double cx, double cy, double cz, bool two) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num) return;
    long long cell = indices[i];
    long long myz = (long long)my * mz;
    int target[] = {(int)(cell / myz), (int)(cell / mz % my), (int)(cell % mz)};
    const int mdims[] = {mx, my, mz};
    PeriodicRange xs(target[0], nx, ratio, shift[0]), ys(target[1], ny, ratio, shift[1]),
        zs(target[2], nz, ratio, shift[2]);
    CICGrid grid = {rho, vel, second, nx, ny, nz, mx, my, mz, ratio, growth,
                    {fx, fy, fz}, {cx, cy, cz}, two};
    double sum = 0.;
    for (int a = 0; a < xs.count; ++a)
        for (int b = 0; b < ys.count; ++b)
            for (int c = 0; c < zs.count; ++c) {
                int x = xs.at(a), y = ys.at(b), z = zs.at(c), base[3], bits = 0;
                double frac[3];
                grid.particle(x, y, z, base, frac);
                bool hit = true;
                for (int axis = 0; axis < 3; ++axis) {
                    int d = (target[axis] - base[axis] + mdims[axis]) % mdims[axis];
                    if (d > 1) {
                        hit = false;
                        break;
                    }
                    bits |= d << axis;
                }
                if (hit) {
                    long long t = ((long long)x * ny + y) * nz + z;
                    double w;
                    if (mx == 1 || my == 1 || mz == 1) {
                        for (int corner = 0; corner < 8; ++corner) {
                            long long dst = grid.corner(corner, base, frac, w);
                            if (dst == cell) sum += w * (1. + rho[t] * growth);
                        }
                    } else {
                        grid.corner(bits, base, frac, w);
                        sum += w * (1. + rho[t] * growth);
                    }
                }
            }
    mass[cell] = sum;
}

// As in v3 cic_repair_exact, threads search the periodic neighborhood together.
// One lane adds accepted weights in native x, y, z order. A parallel reduction
// of those weights could change a float at the rounding boundary being repaired.
extern "C" __global__ void replay_block(const float *rho, const float *vel, const float *second,
                                        double *mass, const long long *indices, int num,
                                        const double *shift, int nx, int ny, int nz, int mx,
                                        int my, int mz, double ratio, double growth, double fx,
                                        double fy, double fz, double cx, double cy, double cz,
                                        bool two) {
    const int site = blockIdx.x;
    if (site >= num) return;
    const long long cell = indices[site];
    const long long myz = (long long)my * mz;
    const int target[] = {(int)(cell / myz), (int)(cell / mz % my), (int)(cell % mz)};
    const int mdims[] = {mx, my, mz};
    PeriodicRange xs(target[0], nx, ratio, shift[0]), ys(target[1], ny, ratio, shift[1]),
        zs(target[2], nz, ratio, shift[2]);
    CICGrid grid = {rho, vel, second, nx, ny, nz, mx, my, mz, ratio, growth,
                    {fx, fy, fz}, {cx, cy, cz}, two};
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const long long yz = (long long)ys.count * zs.count;
    const long long total = xs.count * yz;
    __shared__ double weights[256];
    __shared__ unsigned hits[8];
    double sum = 0.;
    for (long long start = 0; start < total; start += blockDim.x) {
        const long long i = start + tid;
        bool hit = false;
        if (i < total) {
            const int x = xs.at(i / yz), y = ys.at(i / zs.count % ys.count),
                      z = zs.at(i % zs.count);
            int base[3], bits = 0;
            double frac[3];
            grid.particle(x, y, z, base, frac);
            hit = true;
            for (int a = 0; a < 3; ++a) {
                const int bit = (target[a] - base[a] + mdims[a]) % mdims[a];
                if (bit > 1) {
                    hit = false;
                    break;
                }
                bits |= bit << a;
            }
            if (hit) {
                double weight;
                grid.corner(bits, base, frac, weight);
                weights[tid] = weight * (1. + rho[((long long)x * ny + y) * nz + z] * growth);
            }
        }
        const unsigned mask = __ballot_sync(0xffffffffu, hit);
        if (lane == 0) hits[warp] = mask;
        __syncthreads();
        if (tid == 0) {
            for (int w = 0; w < 8; ++w) {
                unsigned pending = hits[w];
                while (pending) {
                    sum += weights[w * 32 + __ffs(pending) - 1];
                    pending &= pending - 1;
                }
            }
        }
        __syncthreads();
    }
    if (tid == 0) mass[cell] = sum;
}
