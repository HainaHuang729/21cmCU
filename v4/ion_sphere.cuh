// Shared spherical bubble painting and recombination commit for IONISE_ENTIRE_SPHERE=True.
// Mirrors src/bubble_helper_progs.c (update_in_sphere, check_region) and src/IonisationBox.c.
extern "C" __global__ void sphere_paint_sparse(int n_centers, const int *centers, int nx, int ny,
                                               int nz, int r_in, int r_out, float rsq, float *xh,
                                               int *min_painter) {
    int m = blockIdx.x;
    if (m >= n_centers) return;
    int c = centers[m];
    int nyz = ny * nz;
    int cx = c / nyz;
    int rem = c - cx * nyz;
    int cy = rem / nz;
    int cz = rem - cy * nz;

    int lx = -min(r_out, nx / 2), hx = min(r_out, (nx - 1) / 2);
    int ly = -min(r_out, ny / 2), hy = min(r_out, (ny - 1) / 2);
    int lz = -min(r_out, nz / 2), hz = min(r_out, (nz - 1) / 2);
    int wx_cnt = hx - lx + 1, wy_cnt = hy - ly + 1, wz_cnt = hz - lz + 1;
    int w_yz = wy_cnt * wz_cnt;
    int vol = wx_cnt * w_yz;

    for (int t = threadIdx.x; t < vol; t += blockDim.x) {
        int dx = lx + t / w_yz;
        int tre = t % w_yz;
        int dy = ly + tre / wz_cnt;
        int dz = lz + tre % wz_cnt;
        int mdx = abs(dx), mdy = abs(dy), mdz = abs(dz);
        bool inside = (r_in >= 0 && mdx <= r_in && mdy <= r_in && mdz <= r_in) ||
                      (rsq > ((float)(mdx * mdx) + (float)(mdy * mdy) + (float)(mdz * mdz)));
        if (!inside) continue;
        int wx = cx + dx;
        if (wx < 0)
            wx += nx;
        else if (wx >= nx)
            wx -= nx;
        int wy = cy + dy;
        if (wy < 0)
            wy += ny;
        else if (wy >= ny)
            wy -= ny;
        int wz = cz + dz;
        if (wz < 0)
            wz += nz;
        else if (wz >= nz)
            wz -= nz;
        int dst = (wx * ny + wy) * nz + wz;
        if (xh) xh[dst] = 0.f;
        if (min_painter && c < min_painter[dst]) {
            atomicMin(min_painter + dst, c);
        }
    }
}

extern "C" __global__ void sphere_paint_pull(int n, int n_centers, const int *centers, int nx,
                                             int ny, int nz, int r_in, float rsq, float *xh,
                                             int *min_painter) {
    __shared__ int s_c[256];
    int dst = blockIdx.x * blockDim.x + threadIdx.x;
    bool active = dst < n;
    int nyz = ny * nz;
    int wx = active ? dst / nyz : 0;
    int rem = active ? (dst - wx * nyz) : 0;
    int wy = active ? rem / nz : 0;
    int wz = active ? (rem - wy * nz) : 0;
    bool found = false;
    int painter = 0x7fffffff;

    for (int base = 0; base < n_centers; base += 256) {
        int idx = base + threadIdx.x;
        s_c[threadIdx.x] = idx < n_centers ? centers[idx] : -1;
        __syncthreads();
        if (active && !found) {
            int limit = min(256, n_centers - base);
            for (int k = 0; k < limit; ++k) {
                int c = s_c[k];
                int cx = c / nyz;
                int crem = c - cx * nyz;
                int cy = crem / nz;
                int cz = crem - cy * nz;
                int adx = abs(cx - wx), mdx = min(adx, nx - adx);
                int ady = abs(cy - wy), mdy = min(ady, ny - ady);
                int adz = abs(cz - wz), mdz = min(adz, nz - adz);
                if ((r_in >= 0 && mdx <= r_in && mdy <= r_in && mdz <= r_in) ||
                    (rsq > ((float)(mdx * mdx) + (float)(mdy * mdy) + (float)(mdz * mdz)))) {
                    found = true;
                    painter = c;
                    break;
                }
            }
        }
        __syncthreads();
    }
    if (active && found) {
        if (xh) xh[dst] = 0.f;
        if (min_painter) {
            min_painter[dst] = painter;
        }
    }
}

extern "C" __global__ void sphere_commit_gamma(int n_centers, const int *centers,
                                               const unsigned char *eligible,
                                               const float *gamma_cand, const int *min_painter,
                                               float radius, float *gamma12, float *mfp) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= n_centers) return;
    int c = centers[m];
    if (eligible[c] && min_painter[c] >= c) {
        gamma12[c] = gamma_cand[c];
        if (mfp) mfp[c] = radius;
    }
}
