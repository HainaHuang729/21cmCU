#define PS_MASS_FIXED_SCALE 70368744177664.0
#define PS_MASS_FIXED_INV (1.0 / 70368744177664.0)

extern "C" __global__ void scale_displacements(
    float *displacement, long size, float first, float second, int undo
) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 6*size) return;
    float factor = i < 3*size ? first : second;
    displacement[i] = undo ? __fdiv_rn(displacement[i],factor) : displacement[i]*factor;
}

__device__ inline double periodic(double x, int n) {
    while (x >= n) x -= n;
    while (x < 0.) x += n;
    return x;
}

__device__ inline int lower_cell(double x, int size, float &upper_weight) {
    int cell = (int)x;
    if (cell >= size) cell -= size;
    if (cell < 0) cell += size;
    double center = cell + 0.5;
    upper_weight = (float)fabs(x - center);
    if (x < center) {
        upper_weight = (float)(1.0 - upper_weight);
        if (--cell < 0) cell += size;
    }
    return cell;
}

extern "C" __global__ void cic(
    const float *initial_density, const float *displacement, long long *mass,
    int fine, int coarse, float initial_growth
) {
    long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long)fine*fine*fine) return;
    int k = index % fine;
    int j = (index / fine) % fine;
    int i = index / (fine*fine);
    int ratio = fine / coarse;
    long n = (long)coarse*coarse*coarse;
    long tile = ((long)(i/ratio)*coarse + j/ratio)*coarse + k/ratio;

    double x = (i + 0.5)/(double)fine;
    double y = (j + 0.5)/(double)fine;
    double z = (k + 0.5)/(double)fine;
    x += displacement[tile];
    y += displacement[tile+n];
    z += displacement[tile+2*n];
    x -= displacement[tile+3*n];
    y -= displacement[tile+4*n];
    z -= displacement[tile+5*n];

    float wx, wy, wz;
    int a = lower_cell(periodic(x*coarse,coarse),coarse,wx);
    int b = lower_cell(periodic(y*coarse,coarse),coarse,wy);
    int c = lower_cell(periodic(z*coarse,coarse),coarse,wz);
    int ix[2] = {a, (a+1) % coarse};
    int iy[2] = {b, (b+1) % coarse};
    int iz[2] = {c, (c+1) % coarse};
    float xw[2] = {(float)(1.0-wx), wx};
    float yw[2] = {(float)(1.0-wy), wy};
    float zw[2] = {(float)(1.0-wz), wz};
    double particle_mass = 1.0 + initial_growth*initial_density[index];
    for (int dz=0; dz<2; ++dz)
        for (int dy=0; dy<2; ++dy)
            for (int dx=0; dx<2; ++dx) {
                long target = ((long)ix[dx]*coarse + iy[dy])*coarse + iz[dz];
                {
                    long long q = llrint((particle_mass*(xw[dx]*yw[dy]*zw[dz])) * PS_MASS_FIXED_SCALE);
                    atomicAdd((unsigned long long *)(mass + target), (unsigned long long)q);
                }
            }
}

extern "C" __global__ void density_contrast(
    const long long *mass, float *contrast, long count, float mass_factor
) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        double m = (double)mass[i] * PS_MASS_FIXED_INV;
        contrast[i] = __fdiv_rn((float)m, mass_factor) - 1.0f;
    }
}

__device__ inline int cyclic_sorted_coord(int u, int g, int r, int coarse) {
    int w = 2 * r + 1;
    if (w >= coarse) return u;
    int lo = g - r;
    int hi = g + r;
    if (lo < 0) {
        return (u <= hi) ? u : (lo + coarse + (u - (hi + 1)));
    }
    if (hi >= coarse) {
        int first_cnt = hi - coarse + 1;
        return (u < first_cnt) ? u : (lo + (u - first_cnt));
    }
    return lo + u;
}

__device__ inline bool axis_weight_for_cell(
    int coord, int fine, int coarse, float d_1lpt, float d_2lpt, int target_g, float &w_out
) {
    double pos = (coord + 0.5) / (double)fine;
    pos += d_1lpt;
    pos -= d_2lpt;
    float w;
    int a = lower_cell(periodic(pos * coarse, coarse), coarse, w);
    if (a == target_g) {
        w_out = (float)(1.0 - w);
        return true;
    }
    if ((a + 1) % coarse == target_g) {
        w_out = w;
        return true;
    }
    return false;
}

extern "C" __global__ void density_contrast_guarded(
    const long long * __restrict__ mass,
    const float * __restrict__ displacement,
    float * __restrict__ contrast,
    int * __restrict__ suspect_cells,
    int * __restrict__ suspect_count,
    int * __restrict__ max_r,
    long count, int coarse, float mass_factor, int max_suspects
) {
    __shared__ int s_rx[256], s_ry[256], s_rz[256];
    int tid = threadIdx.x;
    long i = (long)blockIdx.x * blockDim.x + tid;
    int rx = 0, ry = 0, rz = 0;
    if (i < count) {
        double m = (double)mass[i] * PS_MASS_FIXED_INV;
        float mf = (float)m;
        contrast[i] = __fdiv_rn(mf, mass_factor) - 1.0f;
        float next_up = nextafterf(mf, __int_as_float(0x7f800000));
        float next_dn = nextafterf(mf, __int_as_float(0xff800000));
        double eps = 5e-11 * fmax(1.0, fabs(m) * (1.0 / 64.0));
        double dist_up = fabs(m - 0.5 * ((double)mf + (double)next_up));
        double dist_dn = fabs(m - 0.5 * ((double)mf + (double)next_dn));
        if (dist_up <= eps || dist_dn <= eps) {
            int slot = atomicAdd(suspect_count, 1);
            if (slot < max_suspects) suspect_cells[slot] = (int)i;
        }
        double sx = fabs(((double)displacement[i] - (double)displacement[i + 3 * count]) * coarse);
        double sy = fabs(((double)displacement[i + count] - (double)displacement[i + 4 * count]) * coarse);
        double sz = fabs(((double)displacement[i + 2 * count] - (double)displacement[i + 5 * count]) * coarse);
        rx = (int)ceil(sx) + 3;
        ry = (int)ceil(sy) + 3;
        rz = (int)ceil(sz) + 3;
    }
    s_rx[tid] = rx; s_ry[tid] = ry; s_rz[tid] = rz;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_rx[tid + stride] > s_rx[tid]) s_rx[tid] = s_rx[tid + stride];
            if (s_ry[tid + stride] > s_ry[tid]) s_ry[tid] = s_ry[tid + stride];
            if (s_rz[tid + stride] > s_rz[tid]) s_rz[tid] = s_rz[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicMax(&max_r[0], s_rx[0]);
        atomicMax(&max_r[1], s_ry[0]);
        atomicMax(&max_r[2], s_rz[0]);
    }
}

extern "C" __global__ void cic_repair_exact(
    const float * __restrict__ initial_density,
    const float * __restrict__ displacement,
    float * __restrict__ contrast,
    const int * __restrict__ suspect_cells,
    const int * __restrict__ suspect_count,
    const int * __restrict__ max_r,
    int fine, int coarse, float initial_growth, float mass_factor, int max_suspects
) {
    int total_s = *suspect_count;
    if (total_s <= 0) return;
    if (total_s > max_suspects) total_s = max_suspects;

    int ratio = fine / coarse;
    long n = (long)coarse * coarse * coarse;
    int rx = max_r[0], ry = max_r[1], rz = max_r[2];
    int wx = (2 * rx + 1 >= coarse) ? coarse : (2 * rx + 1);
    int wy = (2 * ry + 1 >= coarse) ? coarse : (2 * ry + 1);
    int wz = (2 * rz + 1 >= coarse) ? coarse : (2 * rz + 1);
    int yz_total = wy * wz;

    __shared__ int s_hits[1024];
    __shared__ int s_nhits;

    for (int s = blockIdx.x; s < total_s; s += gridDim.x) {
        int target = suspect_cells[s];
        int gz = target % coarse;
        int gy = (target / coarse) % coarse;
        int gx = target / (coarse * coarse);
        double exact_sum = 0.0;

        for (int ui = 0; ui < wx; ++ui) {
            int ti = cyclic_sorted_coord(ui, gx, rx, coarse);
            if (threadIdx.x == 0) s_nhits = 0;
            __syncthreads();

            for (int p = threadIdx.x; p < yz_total; p += blockDim.x) {
                int uj = p / wz;
                int uk = p % wz;
                int tj = cyclic_sorted_coord(uj, gy, ry, coarse);
                int tk = cyclic_sorted_coord(uk, gz, rz, coarse);
                long tile = ((long)ti * coarse + tj) * coarse + tk;

                float d0 = displacement[tile];
                float d3 = displacement[tile + 3 * n];
                bool hit_x = false;
                for (int di = 0; di < ratio; ++di) {
                    float dummy;
                    if (axis_weight_for_cell(ti * ratio + di, fine, coarse, d0, d3, gx, dummy)) {
                        hit_x = true; break;
                    }
                }
                if (!hit_x) continue;

                float d1 = displacement[tile + n];
                float d4 = displacement[tile + 4 * n];
                bool hit_y = false;
                for (int dj = 0; dj < ratio; ++dj) {
                    float dummy;
                    if (axis_weight_for_cell(tj * ratio + dj, fine, coarse, d1, d4, gy, dummy)) {
                        hit_y = true; break;
                    }
                }
                if (!hit_y) continue;

                float d2 = displacement[tile + 2 * n];
                float d5 = displacement[tile + 5 * n];
                bool hit_z = false;
                for (int dk = 0; dk < ratio; ++dk) {
                    float dummy;
                    if (axis_weight_for_cell(tk * ratio + dk, fine, coarse, d2, d5, gz, dummy)) {
                        hit_z = true; break;
                    }
                }
                if (hit_z) {
                    int pos = atomicAdd(&s_nhits, 1);
                    if (pos < 1024) s_hits[pos] = p;
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                int nh = s_nhits;
                if (nh > 0 && nh <= 1024) {
                    for (int a = 1; a < nh; ++a) {
                        int key = s_hits[a];
                        int b = a - 1;
                        while (b >= 0 && s_hits[b] > key) {
                            s_hits[b + 1] = s_hits[b];
                            --b;
                        }
                        s_hits[b + 1] = key;
                    }
                    for (int di = 0; di < ratio; ++di) {
                        int i = ti * ratio + di;
                        int h = 0;
                        while (h < nh) {
                            int uj0 = s_hits[h] / wz;
                            int h1 = h + 1;
                            while (h1 < nh && (s_hits[h1] / wz) == uj0) ++h1;
                            int tj = cyclic_sorted_coord(uj0, gy, ry, coarse);
                            for (int dj = 0; dj < ratio; ++dj) {
                                int j = tj * ratio + dj;
                                for (int idx_h = h; idx_h < h1; ++idx_h) {
                                    int uk0 = s_hits[idx_h] % wz;
                                    int tk = cyclic_sorted_coord(uk0, gz, rz, coarse);
                                    long tile = ((long)ti * coarse + tj) * coarse + tk;
                                    float wx_sel;
                                    if (!axis_weight_for_cell(i, fine, coarse, displacement[tile], displacement[tile + 3 * n], gx, wx_sel)) continue;
                                    float wy_sel;
                                    if (!axis_weight_for_cell(j, fine, coarse, displacement[tile + n], displacement[tile + 4 * n], gy, wy_sel)) continue;
                                    float d2 = displacement[tile + 2 * n];
                                    float d5 = displacement[tile + 5 * n];
                                    for (int dk = 0; dk < ratio; ++dk) {
                                        int k = tk * ratio + dk;
                                        float wz_sel;
                                        if (!axis_weight_for_cell(k, fine, coarse, d2, d5, gz, wz_sel)) continue;
                                        long fine_idx = ((long)i * fine + j) * fine + k;
                                        double pm = 1.0 + initial_growth * initial_density[fine_idx];
                                        exact_sum += pm * (wx_sel * wy_sel * wz_sel);
                                    }
                                }
                            }
                            h = h1;
                        }
                    }
                } else if (nh > 1024) {
                    for (int di = 0; di < ratio; ++di) {
                        int i = ti * ratio + di;
                        for (int uj = 0; uj < wy; ++uj) {
                            int tj = cyclic_sorted_coord(uj, gy, ry, coarse);
                            for (int dj = 0; dj < ratio; ++dj) {
                                int j = tj * ratio + dj;
                                for (int uk = 0; uk < wz; ++uk) {
                                    int tk = cyclic_sorted_coord(uk, gz, rz, coarse);
                                    long tile = ((long)ti * coarse + tj) * coarse + tk;
                                    float wx_sel, wy_sel;
                                    if (!axis_weight_for_cell(i, fine, coarse, displacement[tile], displacement[tile + 3 * n], gx, wx_sel)) continue;
                                    if (!axis_weight_for_cell(j, fine, coarse, displacement[tile + n], displacement[tile + 4 * n], gy, wy_sel)) continue;
                                    float d2 = displacement[tile + 2 * n];
                                    float d5 = displacement[tile + 5 * n];
                                    for (int dk = 0; dk < ratio; ++dk) {
                                        int k = tk * ratio + dk;
                                        float wz_sel;
                                        if (!axis_weight_for_cell(k, fine, coarse, d2, d5, gz, wz_sel)) continue;
                                        long fine_idx = ((long)i * fine + j) * fine + k;
                                        double pm = 1.0 + initial_growth * initial_density[fine_idx];
                                        exact_sum += pm * (wx_sel * wy_sel * wz_sel);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            contrast[target] = __fdiv_rn((float)exact_sum, mass_factor) - 1.0f;
        }
        __syncthreads();
    }
}

extern "C" __global__ void velocity_spectrum(
    const float2 *density, float2 *velocity, int side, double dk, float dlog_growth_dt
) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int half = side/2 + 1;
    if (i >= (long)side*side*half) return;
    int iz = i % half;
    int iy = (i / half) % side;
    int ix = i / (side*half);
    float kx = (float)((ix > side/2 ? ix-side : ix)*dk);
    float ky = (float)((iy > side/2 ? iy-side : iy)*dk);
    float kz = (float)(iz*dk);
    float k2 = kx*kx + ky*ky + kz*kz;
    // The derivative along z must have a real-valued inverse on its Nyquist plane.
    // FFTW's real transform discards the anti-Hermitian part there.
    if (k2 == 0.f || (side % 2 == 0 && iz == side/2)) {
        velocity[i] = make_float2(0.f,0.f);
        return;
    }
    float coeff = __fdiv_rn(dlog_growth_dt*kz,k2);
    double prefactor = (double)coeff/((long)side*side*side);
    float2 f = density[i];
    velocity[i] = make_float2((float)(-f.y*prefactor),(float)(f.x*prefactor));
}
extern "C" __global__ void cic_tiled(
    const float *initial_density, const float *displacement, long long *mass,
    int fine, int coarse, float initial_growth
) {
    int tg = blockIdx.x;
    int tiles_k = coarse >> 2;
    int tk0 = (tg % tiles_k) << 2;
    int tj = (tg / tiles_k) % coarse;
    int ti = tg / (coarse * tiles_k);
    int t = threadIdx.x;
    int sub = (t & 15) >> 2;
    int dk = t & 3;
    int dj = (t >> 4) & 3;
    int di = (t >> 6) & 3;

    __shared__ double s_acc[4][27];
    __shared__ float s_disp[4][6];
    __shared__ int s_base[4][3];

    if (t < 108) ((double *)s_acc)[t] = 0.0;
    long n = (long)coarse * coarse * coarse;
    if (t < 24) {
        int s = t / 6;
        int c = t % 6;
        long tile = ((long)ti * coarse + tj) * coarse + (tk0 + s);
        s_disp[s][c] = displacement[tile + (long)c * n];
    }
    __syncthreads();

    if (t < 4) {
        int i0 = ti * 4;
        int j0 = tj * 4;
        int k0 = (tk0 + t) * 4;
        double x0 = (i0 + 0.5) / (double)fine;
        double y0 = (j0 + 0.5) / (double)fine;
        double z0 = (k0 + 0.5) / (double)fine;
        x0 += s_disp[t][0];
        y0 += s_disp[t][1];
        z0 += s_disp[t][2];
        x0 -= s_disp[t][3];
        y0 -= s_disp[t][4];
        z0 -= s_disp[t][5];
        float dummy;
        s_base[t][0] = lower_cell(periodic(x0 * coarse, coarse), coarse, dummy);
        s_base[t][1] = lower_cell(periodic(y0 * coarse, coarse), coarse, dummy);
        s_base[t][2] = lower_cell(periodic(z0 * coarse, coarse), coarse, dummy);
    }
    __syncthreads();

    int i = ti * 4 + di;
    int j = tj * 4 + dj;
    int k = (tk0 + sub) * 4 + dk;
    long index = ((long)i * fine + j) * fine + k;
    double x = (i + 0.5) / (double)fine;
    double y = (j + 0.5) / (double)fine;
    double z = (k + 0.5) / (double)fine;
    x += s_disp[sub][0];
    y += s_disp[sub][1];
    z += s_disp[sub][2];
    x -= s_disp[sub][3];
    y -= s_disp[sub][4];
    z -= s_disp[sub][5];

    float wx, wy, wz;
    int a = lower_cell(periodic(x * coarse, coarse), coarse, wx);
    int b = lower_cell(periodic(y * coarse, coarse), coarse, wy);
    int c = lower_cell(periodic(z * coarse, coarse), coarse, wz);
    int ox = a - s_base[sub][0]; if (ox < 0) ox += coarse;
    int oy = b - s_base[sub][1]; if (oy < 0) oy += coarse;
    int oz = c - s_base[sub][2]; if (oz < 0) oz += coarse;

    float xw[2] = {(float)(1.0 - wx), wx};
    float yw[2] = {(float)(1.0 - wy), wy};
    float zw[2] = {(float)(1.0 - wz), wz};
    double particle_mass = 1.0 + initial_growth * initial_density[index];
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                int local_cell = ((ox + dx) * 3 + (oy + dy)) * 3 + (oz + dz);
                atomicAdd(&s_acc[sub][local_cell], particle_mass * (xw[dx] * yw[dy] * zw[dz]));
            }
    __syncthreads();

    if (t < 108) {
        int s = t / 27;
        int cell = t % 27;
        double v = s_acc[s][cell];
        if (v != 0.0) {
            int dx = cell / 9;
            int dy = (cell / 3) % 3;
            int dz = cell % 3;
            int gx = (s_base[s][0] + dx) % coarse;
            int gy = (s_base[s][1] + dy) % coarse;
            int gz = (s_base[s][2] + dz) % coarse;
            long long q = llrint(v * PS_MASS_FIXED_SCALE);
            atomicAdd((unsigned long long *)(mass + ((long)gx * coarse + gy) * coarse + gz), (unsigned long long)q);
        }
    }
}

extern "C" __global__ void cic_pertile(
    const float * __restrict__ initial_density,
    const float * __restrict__ displacement,
    long long * __restrict__ mass,
    int fine, int coarse, float initial_growth
) {
    long tile = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)coarse * coarse * coarse;
    if (tile >= n) return;

    int tk = tile % coarse;
    int tj = (tile / coarse) % coarse;
    int ti = tile / ((long)coarse * coarse);

    float d0 = displacement[tile];
    float d1 = displacement[tile + n];
    float d2 = displacement[tile + 2 * n];
    float d3 = displacement[tile + 3 * n];
    float d4 = displacement[tile + 4 * n];
    float d5 = displacement[tile + 5 * n];

    int i0 = ti * 4;
    int j0 = tj * 4;
    int k0 = tk * 4;

    int ax_c[4], ay_c[4], az_c[4];
    int ox_a[4], oy_a[4], oz_a[4];
    float xw0[4], xw1[4], yw0[4], yw1[4], zw0[4], zw1[4];

    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        double x = (i0 + m + 0.5) / (double)fine;
        x += d0; x -= d3;
        float wx;
        ax_c[m] = lower_cell(periodic(x * coarse, coarse), coarse, wx);
        xw0[m] = (float)(1.0 - wx);
        xw1[m] = wx;

        double y = (j0 + m + 0.5) / (double)fine;
        y += d1; y -= d4;
        float wy;
        ay_c[m] = lower_cell(periodic(y * coarse, coarse), coarse, wy);
        yw0[m] = (float)(1.0 - wy);
        yw1[m] = wy;

        double z = (k0 + m + 0.5) / (double)fine;
        z += d2; z -= d5;
        float wz;
        az_c[m] = lower_cell(periodic(z * coarse, coarse), coarse, wz);
        zw0[m] = (float)(1.0 - wz);
        zw1[m] = wz;
    }

    int base_a = ax_c[0], base_b = ay_c[0], base_c = az_c[0];
    bool fast_tile = true;
    #pragma unroll
    for (int m = 0; m < 4; ++m) {
        int ox = ax_c[m] - base_a; if (ox < 0) ox += coarse;
        int oy = ay_c[m] - base_b; if (oy < 0) oy += coarse;
        int oz = az_c[m] - base_c; if (oz < 0) oz += coarse;
        ox_a[m] = ox; oy_a[m] = oy; oz_a[m] = oz;
        if ((unsigned)ox > 1u || (unsigned)oy > 1u || (unsigned)oz > 1u) fast_tile = false;
    }

    __shared__ double s_acc[27][64];
    int tid = threadIdx.x;
    if (fast_tile) {
        #pragma unroll
        for (int c = 0; c < 27; ++c) s_acc[c][tid] = 0.0;
    }

    const float4 *dens4 = (const float4 *)initial_density;
    if (fast_tile) {
        for (int di = 0; di < 4; ++di) {
            int i = i0 + di;
            int ox = ox_a[di];
            float xw[2] = {xw0[di], xw1[di]};
            for (int dj = 0; dj < 4; ++dj) {
                int j = j0 + dj;
                int oy = oy_a[dj];
                float yw[2] = {yw0[dj], yw1[dj]};
                float4 v4 = dens4[((long)i * fine + j) * coarse + tk];
                float dvals[4] = {v4.x, v4.y, v4.z, v4.w};
                #pragma unroll
                for (int dk = 0; dk < 4; ++dk) {
                    int oz = oz_a[dk];
                    float zw[2] = {zw0[dk], zw1[dk]};
                    double particle_mass = 1.0 + initial_growth * dvals[dk];
                    #pragma unroll
                    for (int dz = 0; dz < 2; ++dz)
                        #pragma unroll
                        for (int dy = 0; dy < 2; ++dy)
                            #pragma unroll
                            for (int dx = 0; dx < 2; ++dx) {
                                int lc = ((ox + dx) * 3 + (oy + dy)) * 3 + (oz + dz);
                                s_acc[lc][tid] += particle_mass * (xw[dx] * yw[dy] * zw[dz]);
                            }
                }
            }
        }
        int ax[3] = {base_a, (base_a + 1) % coarse, (base_a + 2) % coarse};
        int ay[3] = {base_b, (base_b + 1) % coarse, (base_b + 2) % coarse};
        int az[3] = {base_c, (base_c + 1) % coarse, (base_c + 2) % coarse};
        #pragma unroll
        for (int dx = 0; dx < 3; ++dx)
            #pragma unroll
            for (int dy = 0; dy < 3; ++dy)
                #pragma unroll
                for (int dz = 0; dz < 3; ++dz) {
                    double v = s_acc[(dx * 3 + dy) * 3 + dz][tid];
                    if (v != 0.0) {
                        long long q = llrint(v * PS_MASS_FIXED_SCALE);
                        atomicAdd((unsigned long long *)(mass + ((long)ax[dx] * coarse + ay[dy]) * coarse + az[dz]), (unsigned long long)q);
                    }
                }
    } else {
        for (int di = 0; di < 4; ++di) {
            int i = i0 + di;
            int a = ax_c[di];
            int ix[2] = {a, (a + 1) % coarse};
            float xw[2] = {xw0[di], xw1[di]};
            for (int dj = 0; dj < 4; ++dj) {
                int j = j0 + dj;
                int b = ay_c[dj];
                int iy[2] = {b, (b + 1) % coarse};
                float yw[2] = {yw0[dj], yw1[dj]};
                float4 v4 = dens4[((long)i * fine + j) * coarse + tk];
                float dvals[4] = {v4.x, v4.y, v4.z, v4.w};
                for (int dk = 0; dk < 4; ++dk) {
                    int c = az_c[dk];
                    int iz[2] = {c, (c + 1) % coarse};
                    float zw[2] = {zw0[dk], zw1[dk]};
                    double particle_mass = 1.0 + initial_growth * dvals[dk];
                    for (int dz = 0; dz < 2; ++dz)
                        for (int dy = 0; dy < 2; ++dy)
                            for (int dx = 0; dx < 2; ++dx) {
                                long target = ((long)ix[dx] * coarse + iy[dy]) * coarse + iz[dz];
                                long long q = llrint((particle_mass * (xw[dx] * yw[dy] * zw[dz])) * PS_MASS_FIXED_SCALE);
                                atomicAdd((unsigned long long *)(mass + target), (unsigned long long)q);
                            }
                }
            }
        }
    }
}
