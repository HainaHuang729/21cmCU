#pragma STDC FP_CONTRACT OFF
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
#endif

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GNUC__) && !defined(__clang__)
#define P21_STRICT_FP __attribute__((optimize("no-fast-math", "fp-contract=off")))
#else
#define P21_STRICT_FP
#endif

/*
 * Bit-exact C/OpenMP CPU implementation of py21cmfast.rsds.rsds_shift
 * matching RegularGridInterpolator(linear, fill_value=None) +
 * cosmotile.cic.cloud_in_cell_los.
 *
 * Layout of f_in, d_in, f_out is row-major (n_los, nz): each column is
 * contiguous along LOS so OpenMP threads work independently with zero false
 * sharing and identical per-pixel IEEE-754 arithmetic order.
 */

P21_STRICT_FP static inline double p21_eval_fine_rsd(const double *d_col, int nz,
                                                      int n_rsd_subcells, int periodic,
                                                      int disp_prescaled, double fg) {
    const double scale = disp_prescaled ? 1.0 : (double)n_rsd_subcells;
    if (nz == 1) {
        return d_col[0] * scale;
    }

    if (periodic) {
        /* Extended grid: [-0.5, 0.5, ..., nz - 0.5, nz + 0.5] with d[nz-1], d[0..nz-1], d[0]. */
        long long k_ext = (long long)floor(fg + 0.5);
        if (k_ext < 0) k_ext = 0;
        if (k_ext > nz) k_ext = nz;
        const int idx0 = (k_ext == 0) ? (nz - 1) : (int)(k_ext - 1);
        const int idx1 = (k_ext + 1 == nz + 1) ? 0 : (int)k_ext;
        const double x0 = (double)k_ext - 0.5;
        const double t = fg - x0;
        const double v0 = d_col[idx0] * scale;
        const double v1 = d_col[idx1] * scale;
        return v0 * (1.0 - t) + v1 * t;
    }

    int k0;
    if (fg < 0.5) {
        k0 = 0;
    } else if (fg >= (double)nz - 0.5) {
        k0 = nz - 2;
    } else {
        k0 = (int)floor(fg - 0.5);
        if (k0 < 0) k0 = 0;
        if (k0 > nz - 2) k0 = nz - 2;
    }
    const double x0 = (double)k0 + 0.5;
    const double t = fg - x0;
    const double v0 = d_col[k0] * scale;
    const double v1 = d_col[k0 + 1] * scale;
    return v0 * (1.0 - t) + v1 * t;
}

P21_STRICT_FP static inline long long p21_wrap_idx(long long idx, long long n_fine) {
    long long r = idx % n_fine;
    return (r < 0) ? (r + n_fine) : r;
}

P21_STRICT_FP int p21_cpu_rsds_shift_f32(const float *f_in, const double *d_in, int n_los, int nz,
                                         int n_rsd_subcells, int periodic, int disp_prescaled,
                                         float *f_out) {
    if (!f_in || !d_in || !f_out || n_los <= 0 || nz <= 0 || n_rsd_subcells < 1) {
        return -1;
    }

    const long long n_fine = (long long)nz * (long long)n_rsd_subcells;
    const double step = (double)nz / (double)n_fine;
    int err = 0;

#pragma omp parallel if (n_los >= 64)
    {
        float *fine_out = (float *)malloc((size_t)n_fine * sizeof(float));
        if (!fine_out) {
#pragma omp atomic write
            err = -2;
        } else {
#pragma omp for schedule(static)
            for (int los = 0; los < n_los; los++) {
                if (err != 0) continue;
                const float *f_col = f_in + (size_t)los * (size_t)nz;
                const double *d_col = d_in + (size_t)los * (size_t)nz;
                float *out_col = f_out + (size_t)los * (size_t)nz;

                memset(fine_out, 0, (size_t)n_fine * sizeof(float));

                for (long long ii = 0; ii < n_fine; ii++) {
                    const int k = (int)(ii / n_rsd_subcells);
                    const float weight = f_col[k] / (float)n_rsd_subcells;
                    const double df0 = (double)ii * step;
                    const double df1 = (ii + 1 == n_fine) ? (double)nz : ((double)(ii + 1) * step);
                    const double fg = (df0 + df1) * 0.5;
                    const double rsd =
                        p21_eval_fine_rsd(d_col, nz, n_rsd_subcells, periodic, disp_prescaled, fg);

                    const double x = (double)ii + rsd;
                    const long long i0 = (long long)floor(x);
                    const long long ip = i0 + 1;
                    const double tx = (double)ip - x;
                    const double ddx = 1.0 - tx;

                    if (periodic) {
                        const long long wi0 = p21_wrap_idx(i0, n_fine);
                        const long long wip = p21_wrap_idx(ip, n_fine);
                        fine_out[wi0] = (float)((double)fine_out[wi0] + tx * (double)weight);
                        fine_out[wip] = (float)((double)fine_out[wip] + ddx * (double)weight);
                    } else {
                        if (i0 >= 0 && i0 < n_fine) {
                            fine_out[i0] = (float)((double)fine_out[i0] + tx * (double)weight);
                        }
                        if (ip >= 0 && ip < n_fine) {
                            fine_out[ip] = (float)((double)fine_out[ip] + ddx * (double)weight);
                        }
                    }
                }

                for (int k = 0; k < nz; k++) {
                    const size_t base = (size_t)k * (size_t)n_rsd_subcells;
                    float acc = 0.0f;
                    for (int s = 0; s < n_rsd_subcells; s++) {
                        acc += fine_out[base + (size_t)s];
                    }
                    out_col[k] = acc;
                }
            }
            free(fine_out);
        }
    }

    return err;
}

P21_STRICT_FP int p21_cpu_rsds_shift_f64(const double *f_in, const double *d_in, int n_los, int nz,
                                         int n_rsd_subcells, int periodic, int disp_prescaled,
                                         double *f_out) {
    if (!f_in || !d_in || !f_out || n_los <= 0 || nz <= 0 || n_rsd_subcells < 1) {
        return -1;
    }

    const long long n_fine = (long long)nz * (long long)n_rsd_subcells;
    const double step = (double)nz / (double)n_fine;
    int err = 0;

#pragma omp parallel if (n_los >= 64)
    {
        double *fine_out = (double *)malloc((size_t)n_fine * sizeof(double));
        if (!fine_out) {
#pragma omp atomic write
            err = -2;
        } else {
#pragma omp for schedule(static)
            for (int los = 0; los < n_los; los++) {
                if (err != 0) continue;
                const double *f_col = f_in + (size_t)los * (size_t)nz;
                const double *d_col = d_in + (size_t)los * (size_t)nz;
                double *out_col = f_out + (size_t)los * (size_t)nz;

                memset(fine_out, 0, (size_t)n_fine * sizeof(double));

                for (long long ii = 0; ii < n_fine; ii++) {
                    const int k = (int)(ii / n_rsd_subcells);
                    const double weight = f_col[k] / (double)n_rsd_subcells;
                    const double df0 = (double)ii * step;
                    const double df1 = (ii + 1 == n_fine) ? (double)nz : ((double)(ii + 1) * step);
                    const double fg = (df0 + df1) * 0.5;
                    const double rsd =
                        p21_eval_fine_rsd(d_col, nz, n_rsd_subcells, periodic, disp_prescaled, fg);

                    const double x = (double)ii + rsd;
                    const long long i0 = (long long)floor(x);
                    const long long ip = i0 + 1;
                    const double tx = (double)ip - x;
                    const double ddx = 1.0 - tx;

                    if (periodic) {
                        const long long wi0 = p21_wrap_idx(i0, n_fine);
                        const long long wip = p21_wrap_idx(ip, n_fine);
                        fine_out[wi0] += tx * weight;
                        fine_out[wip] += ddx * weight;
                    } else {
                        if (i0 >= 0 && i0 < n_fine) {
                            fine_out[i0] += tx * weight;
                        }
                        if (ip >= 0 && ip < n_fine) {
                            fine_out[ip] += ddx * weight;
                        }
                    }
                }

                for (int k = 0; k < nz; k++) {
                    const size_t base = (size_t)k * (size_t)n_rsd_subcells;
                    double acc = 0.0;
                    for (int s = 0; s < n_rsd_subcells; s++) {
                        acc += fine_out[base + (size_t)s];
                    }
                    out_col[k] = acc;
                }
            }
            free(fine_out);
        }
    }

    return err;
}
