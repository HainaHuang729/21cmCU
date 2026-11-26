#include <math.h>
#include <string.h>
#include <vector>

#include "cuda_common.cuh"
#include "cuda_tables.cuh"

extern "C" {
#include "../hmf.h"
}

namespace {

constexpr int kCondThreads = 128;
constexpr int kMaxGL = 128;
constexpr int kTailPanels = 64;
constexpr double kCritDensTransition = 1.2;

__constant__ double kGL32AllX[32] = {
    -9.97263861849481570e-01, -9.85611511545268271e-01, -9.64762255587506390e-01,
    -9.34906075937739667e-01, -8.96321155766052091e-01, -8.49367613732569970e-01,
    -7.94483795967942386e-01, -7.32182118740289711e-01, -6.63044266930215231e-01,
    -5.87715757240762304e-01, -5.06899908932229359e-01, -4.21351276130635333e-01,
    -3.31868602282127667e-01, -2.39287362252137065e-01, -1.44471961582796488e-01,
    -4.83076656877383173e-02, 4.83076656877383173e-02,  1.44471961582796488e-01,
    2.39287362252137065e-01,  3.31868602282127667e-01,  4.21351276130635333e-01,
    5.06899908932229359e-01,  5.87715757240762304e-01,  6.63044266930215231e-01,
    7.32182118740289711e-01,  7.94483795967942386e-01,  8.49367613732569970e-01,
    8.96321155766052091e-01,  9.34906075937739667e-01,  9.64762255587506390e-01,
    9.85611511545268271e-01,  9.97263861849481570e-01,
};

__constant__ double kGL32AllW[32] = {
    7.01861000947036177e-03, 1.62743947309064857e-02, 2.53920653092627284e-02,
    3.42738629130216327e-02, 4.28358980222269259e-02, 5.09980592623759596e-02,
    5.86840934785354609e-02, 6.58222227763615997e-02, 7.23457941088481993e-02,
    7.81938957870701168e-02, 8.33119242269466237e-02, 8.76520930044036861e-02,
    9.11738786957637104e-02, 9.38443990808043860e-02, 9.56387200792745973e-02,
    9.65400885147275761e-02, 9.65400885147275761e-02, 9.56387200792745973e-02,
    9.38443990808043860e-02, 9.11738786957637104e-02, 8.76520930044036861e-02,
    8.33119242269466237e-02, 7.81938957870701168e-02, 7.23457941088481993e-02,
    6.58222227763615997e-02, 5.86840934785354609e-02, 5.09980592623759596e-02,
    4.28358980222269259e-02, 3.42738629130216327e-02, 2.53920653092627284e-02,
    1.62743947309064857e-02, 7.01861000947036177e-03,
};

struct TableBuildScalingParams {
    double redshift;
    double growthf;
    double alpha_star_acg;
    double alpha_star_mcg;
    double alpha_esc;
    double lnf_star_acg;
    double lnf_star_mcg;
    double lnf_esc_acg;
    double lnf_esc_mcg;
    double lnMlim_Fstar_acg;
    double lnMlim_Fstar_mcg;
    double lnMlim_Fesc_acg;
    double lnMlim_Fesc_mcg;
    double mturn_acg_hom;
    double atomic_cooling_threshold;
    double lx_acg;
    double lx_mcg;
    double sfr_timescale;
    double omb_over_omm;
    double s_per_yr;
    double rhocrit_omm;
    double rhocrit_omb;
    double prefac_acg;
    double prefac_mcg;
    double prefac_xray_int;
    double clamp_min;
    int use_metallicity;
    int hmf_orig;
    int hmf_eff;
    double delta_c_sph;
    double delta_c_delos;
    double max_deltac_frac;
    double fract_float_err;
    double omm;
    double oml;
    double omr;
    double fstar_10_acg;
    double fesc_10_acg;
    double pop2_ion;
    double fstar_7_mcg;
    double fesc_7_mcg;
    double pop3_ion;
};

__device__ static inline double tb_eval_sigma(double lnM, const P21DevTable1Df &sig) {
    return p21_eval_rg1d_f(lnM, sig);
}

__device__ static inline double tb_eval_dsigmasqdm(double lnM, const P21DevTable1Df &dsig) {
    return -pow(10.0, p21_eval_rg1d_f(lnM, dsig));
}

__device__ static inline double tb_st_taylor_factor(double sig, double sig_cond, double growthf,
                                                    double delta_c_sph, double *zeroth_order) {
    constexpr double kJa = 0.73;
    constexpr double kAlpha = 0.81;
    constexpr double kBeta = 0.34;

    double del = delta_c_sph / growthf;
    double sigsq = sig * sig;
    double sigsq_inv = 1.0 / sigsq;
    double sigcsq = sig_cond * sig_cond;
    double sigdiff = (sig == sig_cond) ? 1e-6 : (sigsq - sigcsq);

    double t_array[6];
    t_array[0] = 1.0;
    for (int i = 1; i < 6; ++i) {
        t_array[i] = t_array[i - 1] * (-sigdiff) / (double)i * (kAlpha - (double)i + 1.0) *
                     sigsq_inv;
    }
    double result = 0.0;
    for (int i = 5; i >= 0; --i) {
        result += t_array[i];
    }

    double prefactor_1 = sqrt(kJa) * del;
    double prefactor_2 = kBeta * pow(sigsq_inv * (kJa * del * del), -kAlpha);
    result = prefactor_1 * (1.0 + prefactor_2 * result);
    if (zeroth_order != nullptr) {
        *zeroth_order = prefactor_1 * (1.0 + prefactor_2);
    }
    return result;
}

__device__ static inline double tb_conditional_hmf(
    int eff_hmf, double growthf, double lnM, double delta_cond, double sigma_cond,
    double delta_c_sph, double delta_c_delos, const P21DevTable1Df &sig_tbl,
    const P21DevTable1Df &dsig_tbl) {
    double sigma1 = tb_eval_sigma(lnM, sig_tbl);
    if (sigma1 < sigma_cond) {
        return 0.0;
    }
    double dsigmasqdm = tb_eval_dsigmasqdm(lnM, dsig_tbl);
    double sigdiff_inv =
        (sigma1 == sigma_cond) ? 1e6 : 1.0 / (sigma1 * sigma1 - sigma_cond * sigma_cond);

    if (eff_hmf == HMF_ST) {
        double delta_0 = delta_cond / growthf;
        double barrier = 0.0;
        double factor = tb_st_taylor_factor(sigma1, sigma_cond, growthf, delta_c_sph, &barrier) -
                        delta_0;
        return -dsigmasqdm * factor * pow(sigdiff_inv, 1.5) *
               exp(-(barrier - delta_0) * (barrier - delta_0) * 0.5 * sigdiff_inv) /
               sqrt(2.0 * M_PI);
    }
    if (eff_hmf == HMF_DELOS) {
        constexpr double kCoeffNu = 0.519;
        constexpr double kIndexNu = 0.582;
        constexpr double kExpFactor = -0.469;
        double dsigmadm = dsigmasqdm * 0.5;
        double nu = (delta_c_delos - delta_cond) * sqrt(sigdiff_inv) / growthf;
        if (nu <= 0.0) {
            return 0.0;
        }
        double dfdnu = kCoeffNu * pow(nu, kIndexNu) * exp(kExpFactor * nu * nu);
        return dfdnu * fabs(dsigmadm) * sigdiff_inv;
    }
    double del = (delta_c_sph - delta_cond) / growthf;
    return -del * dsigmasqdm * pow(sigdiff_inv, 1.5) * exp(-del * del * 0.5 * sigdiff_inv) /
           sqrt(2.0 * M_PI);
}

__device__ static inline double tb_unconditional_hmf(
    int hmf, double growthf, double lnM, double z, const TableBuildScalingParams &p,
    const P21DevTable1Df &sig_tbl, const P21DevTable1Df &dsig_tbl) {
    double sigma0 = tb_eval_sigma(lnM, sig_tbl);
    double dsigmasqdm = tb_eval_dsigmasqdm(lnM, dsig_tbl);
    if (hmf == HMF_DELOS) {
        double sigma_inv = 1.0 / sigma0;
        double dsigmadm = dsigmasqdm * (0.5 * sigma_inv);
        double nu = p.delta_c_delos * sigma_inv / growthf;
        double dfdnu = 0.519 * pow(nu, 0.582) * exp(-0.469 * nu * nu);
        return dfdnu * fabs(dsigmadm) * sigma_inv;
    }

    double sigma = sigma0 * growthf;
    double dsigmadm = dsigmasqdm * (growthf * growthf / (2.0 * sigma));
    if (hmf == HMF_PS) {
        return -sqrt(2.0 / M_PI) * (p.delta_c_sph / (sigma * sigma)) * dsigmadm *
               exp(-(p.delta_c_sph * p.delta_c_sph) / (2.0 * sigma * sigma));
    }
    if (hmf == HMF_ST) {
        constexpr double kA = 0.353;
        constexpr double kP = 0.175;
        constexpr double kJa = 0.73;
        double nuhat = sqrt(kA) * 0.0; /* unused, keep kJa below */
        (void)nuhat;
        nuhat = sqrt(kJa) * p.delta_c_sph / sigma;
        return -(dsigmadm / sigma) * sqrt(2.0 / M_PI) * kA * (1.0 + pow(nuhat, -2.0 * kP)) *
               nuhat * exp(-nuhat * nuhat / 2.0);
    }
    if (hmf == HMF_WATSON) {
        double f_sig = 0.282 * (pow(1.406 / sigma, 2.163) + 1.0) * exp(-1.210 / (sigma * sigma));
        return -(dsigmadm / sigma) * f_sig;
    }
    if (hmf == HMF_WATSON_Z) {
        double opz3 = pow(1.0 + z, 3.0);
        double opz4 = pow(1.0 + z, 4.0);
        double omz = p.omm * opz3 / (p.oml + p.omm * opz3 + p.omr * opz4);
        double a_z = omz * (0.990 * pow(1.0 + z, -3.216) + 0.074);
        double alpha_z = omz * (5.907 * pow(1.0 + z, -3.058) + 2.349);
        double beta_z = omz * (3.136 * pow(1.0 + z, -3.599) + 2.344);
        double f_sig = a_z * (pow(beta_z / sigma, alpha_z) + 1.0) * exp(-1.318 / (sigma * sigma));
        return -(dsigmadm / sigma) * f_sig;
    }
    if (hmf == HMF_YUNG24) {
        double a_z = 0.13765772 - 0.01003821 * z + 0.00102964 * z * z;
        double al_z = 1.06641384 + 0.02475576 * z - 0.00283342 * z * z;
        double b_z = 4.86693806 + 0.09212356 * z - 0.01426283 * z * z;
        double c_z = 1.19837952 - 0.00142967 * z - 0.00033074 * z * z;
        double f_sig = a_z * (pow(sigma / b_z, -al_z) + 1.0) * exp(-c_z / (sigma * sigma));
        return -(dsigmadm / sigma) * f_sig;
    }
    return 0.0;
}

__device__ static inline double tb_log_scaling_pl_limit(double lnM, double ln_norm, double alpha,
                                                        double ln_pivot, double ln_limit) {
    if ((alpha > 0.0 && lnM > ln_limit) || (alpha < 0.0 && lnM < ln_limit)) {
        return -ln_norm;
    }
    return alpha * (lnM - ln_pivot);
}

__device__ static inline double tb_nion_frac_acg(double lnM, double mturn_acg,
                                                 const TableBuildScalingParams &p) {
    double l10 = 10.0 * M_LN10;
    double fstar = tb_log_scaling_pl_limit(lnM, p.lnf_star_acg, p.alpha_star_acg, l10,
                                           p.lnMlim_Fstar_acg);
    double fesc =
        tb_log_scaling_pl_limit(lnM, p.lnf_esc_acg, p.alpha_esc, l10, p.lnMlim_Fesc_acg);
    return exp(fstar + fesc - mturn_acg / exp(lnM) + lnM);
}

__device__ static inline double tb_nion_frac_mcg(double lnM, double mturn_mcg, double atomic_thresh,
                                                 const TableBuildScalingParams &p) {
    double l7 = 7.0 * M_LN10;
    double fstar = tb_log_scaling_pl_limit(lnM, p.lnf_star_mcg, p.alpha_star_mcg, l7,
                                           p.lnMlim_Fstar_mcg);
    double fesc =
        tb_log_scaling_pl_limit(lnM, p.lnf_esc_mcg, p.alpha_esc, l7, p.lnMlim_Fesc_mcg);
    double M = exp(lnM);
    return exp(fstar + fesc - M / atomic_thresh - mturn_mcg / M + lnM);
}

__device__ static inline double tb_xray_frac_acg(double lnM, double mturn_acg,
                                                 const TableBuildScalingParams &p) {
    double M = exp(lnM);
    double Fstar = exp(tb_log_scaling_pl_limit(lnM, p.lnf_star_acg, p.alpha_star_acg, 10.0 * M_LN10,
                                               p.lnMlim_Fstar_acg) -
                       mturn_acg / M + p.lnf_star_acg);
    double stellar_mass = M * Fstar * p.omb_over_omm;
    double sfr = stellar_mass / p.sfr_timescale;
    double met = 0.0;
    if (p.use_metallicity) {
        double z_scale = pow(10.0, -0.056 * p.redshift + 0.064);
        double stellar_term = 1.0;
        if (stellar_mass > 0.0 && sfr > 0.0) {
            double M0 = 1.28825e10 * pow(sfr * p.s_per_yr, 0.56);
            stellar_term = pow(1.0 + pow(stellar_mass / M0, -2.1), -0.148);
        }
        met = 1.23 * stellar_term * z_scale;
    }
    double lx = p.use_metallicity ? (p.lx_acg / (1.0 + pow(met / 0.05, 0.64))) : p.lx_acg;
    return sfr * lx;
}

__device__ static inline double tb_xray_frac_mcg(double lnM, double mturn_mcg,
                                                 const TableBuildScalingParams &p) {
    double M = exp(lnM);
    double Fstar = exp(tb_log_scaling_pl_limit(lnM, p.lnf_star_mcg, p.alpha_star_mcg, 7.0 * M_LN10,
                                               p.lnMlim_Fstar_mcg) -
                       mturn_mcg / M - M / p.atomic_cooling_threshold + p.lnf_star_mcg);
    double stellar_mass = M * Fstar * p.omb_over_omm;
    double sfr = stellar_mass / p.sfr_timescale;
    double met = 0.0;
    if (p.use_metallicity) {
        double z_scale = pow(10.0, -0.056 * p.redshift + 0.064);
        double stellar_term = 1.0;
        if (stellar_mass > 0.0 && sfr > 0.0) {
            double M0 = 1.28825e10 * pow(sfr * p.s_per_yr, 0.56);
            stellar_term = pow(1.0 + pow(stellar_mass / M0, -2.1), -0.148);
        }
        met = 1.23 * stellar_term * z_scale;
    }
    double lx = p.use_metallicity ? (p.lx_mcg / (1.0 + pow(met / 0.05, 0.64))) : p.lx_mcg;
    return sfr * lx;
}

__device__ static inline double tb_scaling_frac(int kind, int is_mcg, double lnM, double mturn,
                                                const TableBuildScalingParams &p) {
    if (kind == 2) {
        return is_mcg ? tb_xray_frac_mcg(lnM, mturn, p) : tb_xray_frac_acg(lnM, mturn, p);
    }
    return is_mcg ? tb_nion_frac_mcg(lnM, mturn, p.atomic_cooling_threshold, p)
                  : tb_nion_frac_acg(lnM, mturn, p);
}

__device__ static inline double tb_delta_at(int i, int n_delta, double min_dens, double max_dens) {
    return min_dens + (double)((float)i / ((float)n_delta - 1.0f)) * (max_dens - min_dens);
}

__device__ static inline double tb_mturn_at(int j, int n_mturn, double l10_min, double l10_max) {
    return pow(10.0, l10_min + (double)((float)j / ((float)n_mturn - 1.0f)) * (l10_max - l10_min));
}

__device__ static inline float tb_clamp_log(double val, double clamp_min) {
    double lv = log(val);
    if (lv < clamp_min) {
        lv = clamp_min;
    }
    return (float)lv;
}

__global__ void cond_cmf_gl_kernel(int ngl, const double *gl_xi, const double *gl_wi, int n_delta,
                                   double min_dens, double max_dens, double sigma_cond,
                                   double dcrit_orig, TableBuildScalingParams p,
                                   P21DevTable1Df sig_tbl, P21DevTable1Df dsig_tbl,
                                   double *w_cmf_out) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)n_delta * (size_t)ngl;
    if (idx >= total) {
        return;
    }
    int i = (int)(idx / (size_t)ngl);
    int q = (int)(idx % (size_t)ngl);
    double delta = tb_delta_at(i, n_delta, min_dens, max_dens);
    if (delta > kCritDensTransition || delta > p.max_deltac_frac * dcrit_orig) {
        w_cmf_out[idx] = 0.0;
        return;
    }
    double c = tb_conditional_hmf(p.hmf_eff, p.growthf, gl_xi[q], delta, sigma_cond,
                                  p.delta_c_sph, p.delta_c_delos, sig_tbl, dsig_tbl);
    w_cmf_out[idx] = gl_wi[q] * c;
}

__global__ void cond_scaling_gl_kernel(int kind, int ngl, const double *gl_xi, int n_acg,
                                       double acg_l10_min, double acg_l10_max, int do_acg_2d,
                                       int n_mcg, double mcg_l10_min, double mcg_l10_max,
                                       int do_mcg_2d, int n_mturn, TableBuildScalingParams p,
                                       double *f_acg_out, double *f_mcg_out) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total_acg = n_acg * ngl;
    int total_mcg = do_mcg_2d ? (n_mcg * ngl) : 0;
    if (tid < total_acg) {
        int j = tid / ngl;
        int q = tid % ngl;
        double mturn = do_acg_2d ? tb_mturn_at(j, n_mturn, acg_l10_min, acg_l10_max)
                                 : p.mturn_acg_hom;
        f_acg_out[tid] = tb_scaling_frac(kind, 0, gl_xi[q], mturn, p);
    } else if (tid < total_acg + total_mcg) {
        int local = tid - total_acg;
        int j = local / ngl;
        int q = local % ngl;
        if (p.mturn_acg_hom > p.atomic_cooling_threshold * 1.001) {
            f_mcg_out[local] = 0.0;
        } else {
            double mturn = tb_mturn_at(j, n_mturn, mcg_l10_min, mcg_l10_max);
            f_mcg_out[local] = tb_scaling_frac(kind, 1, gl_xi[q], mturn, p);
        }
    }
}

__global__ void cond_dot_gl_kernel(int kind, int ngl, int n_delta, double min_dens, double max_dens,
                                   double lnM_min, double lnM_max, double lnM_cond,
                                   double dcrit_orig, int n_acg, int do_acg_2d, int n_mturn,
                                   double acg_l10_min, double acg_l10_max, int do_mcg_2d,
                                   double mcg_l10_min, double mcg_l10_max,
                                   TableBuildScalingParams p, const double *w_cmf,
                                   const double *f_acg, const double *f_mcg, float *acg_out,
                                   float *mcg_out) {
    int i = blockIdx.x;
    int j = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (i >= n_delta) {
        return;
    }

    double delta = tb_delta_at(i, n_delta, min_dens, max_dens);
    bool no_cond_domain = (lnM_min >= lnM_cond);
    bool above_dcrit = (delta > p.max_deltac_frac * dcrit_orig);
    bool gl_domain = (!no_cond_domain && !above_dcrit && delta <= kCritDensTransition);

    if (above_dcrit) {
        bool cond_in_range = (lnM_cond * (1.0 - p.fract_float_err) <= lnM_max);
        double M_cond = exp(lnM_cond);
        if (j < n_acg) {
            double val = 0.0;
            if (!no_cond_domain && cond_in_range) {
                double mturn = do_acg_2d ? tb_mturn_at(j, n_mturn, acg_l10_min, acg_l10_max)
                                         : p.mturn_acg_hom;
                val = (tb_scaling_frac(kind, 0, lnM_cond, mturn, p) / M_cond) * p.prefac_acg;
            }
            acg_out[(size_t)i * (size_t)n_acg + (size_t)j] = tb_clamp_log(val, p.clamp_min);
        }
        if (do_mcg_2d && j < n_mturn) {
            double val = 0.0;
            if (!no_cond_domain && cond_in_range &&
                p.mturn_acg_hom <= p.atomic_cooling_threshold * 1.001) {
                double mturn = tb_mturn_at(j, n_mturn, mcg_l10_min, mcg_l10_max);
                val = (tb_scaling_frac(kind, 1, lnM_cond, mturn, p) / M_cond) * p.prefac_mcg;
            }
            mcg_out[(size_t)i * (size_t)n_mturn + (size_t)j] = tb_clamp_log(val, p.clamp_min);
        }
        return;
    }

    if (no_cond_domain) {
        if (j < n_acg) {
            acg_out[(size_t)i * (size_t)n_acg + (size_t)j] = (float)p.clamp_min;
        }
        if (do_mcg_2d && j < n_mturn) {
            mcg_out[(size_t)i * (size_t)n_mturn + (size_t)j] = (float)p.clamp_min;
        }
        return;
    }

    if (!gl_domain) {
        /* Handled by cond_tail_gl32_kernel. */
        return;
    }

    const double *w_row = w_cmf + (size_t)i * (size_t)ngl;
    double acg_scale = p.prefac_acg * ((kind == 2) ? p.prefac_xray_int : 1.0);
    double mcg_scale = p.prefac_mcg * ((kind == 2) ? p.prefac_xray_int : 1.0);

    if (j < n_acg) {
        const double *fa = f_acg + (size_t)j * (size_t)ngl;
        double sum = 0.0;
        for (int q = 0; q < ngl; ++q) {
            sum += w_row[q] * fa[q];
        }
        acg_out[(size_t)i * (size_t)n_acg + (size_t)j] = tb_clamp_log(sum * acg_scale, p.clamp_min);
    }
    if (do_mcg_2d && j < n_mturn) {
        double sum = 0.0;
        if (p.mturn_acg_hom <= p.atomic_cooling_threshold * 1.001) {
            const double *fm = f_mcg + (size_t)j * (size_t)ngl;
            for (int q = 0; q < ngl; ++q) {
                sum += w_row[q] * fm[q];
            }
        }
        mcg_out[(size_t)i * (size_t)n_mturn + (size_t)j] = tb_clamp_log(sum * mcg_scale, p.clamp_min);
    }
}

__device__ static inline double warp_sum_32(double v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    }
    return v;
}

__device__ static inline double tb_panel_edge(double upper, double span, int p, int n_panels) {
    double t = 1.0 - (double)p / (double)n_panels;
    double t2 = t * t;
    return upper - span * (t2 * t2);
}

/* One warp (32 threads) integrates all ACG & MCG components for one high-delta row via 64-panel GL32. */
__global__ void cond_tail_gl32_kernel(int kind, int n_high, const int *high_indices, int n_delta,
                                      double min_dens, double max_dens, double lnM_min,
                                      double lnM_max, double lnM_cond, double sigma_cond,
                                      int n_acg, int do_acg_2d, int n_mturn, double acg_l10_min,
                                      double acg_l10_max, int do_mcg_2d, double mcg_l10_min,
                                      double mcg_l10_max, TableBuildScalingParams p,
                                      P21DevTable1Df sig_tbl, P21DevTable1Df dsig_tbl,
                                      float *acg_out, float *mcg_out) {
    __shared__ double sh_acg[64];
    __shared__ double sh_mcg[64];

    int hi_idx = blockIdx.x;
    int lane = threadIdx.x;
    if (hi_idx >= n_high) {
        return;
    }
    int i = high_indices[hi_idx];
    double delta = tb_delta_at(i, n_delta, min_dens, max_dens);
    double upper = fmin(lnM_max, lnM_cond);
    double span = upper - lnM_min;

    if (lane < n_acg) sh_acg[lane] = 0.0;
    if (lane < 64) {
        if (do_mcg_2d && lane < n_mturn) sh_mcg[lane] = 0.0;
    }
    if (n_acg > 32 && lane + 32 < n_acg) sh_acg[lane + 32] = 0.0;
    if (do_mcg_2d && n_mturn > 32 && lane + 32 < n_mturn) sh_mcg[lane + 32] = 0.0;
    __syncwarp();

    if (span > 0.0) {
        bool mcg_active =
            (do_mcg_2d != 0) && (p.mturn_acg_hom <= p.atomic_cooling_threshold * 1.001);
        for (int pan = 0; pan < kTailPanels; ++pan) {
            double a = tb_panel_edge(upper, span, pan, kTailPanels);
            double b = tb_panel_edge(upper, span, pan + 1, kTailPanels);
            if (b <= a) continue;
            double mid = 0.5 * (a + b);
            double half = 0.5 * (b - a);
            double lnM = mid + half * kGL32AllX[lane];
            double w_cmf = half * kGL32AllW[lane] *
                           tb_conditional_hmf(p.hmf_eff, p.growthf, lnM, delta, sigma_cond,
                                              p.delta_c_sph, p.delta_c_delos, sig_tbl, dsig_tbl);

            for (int j = 0; j < n_acg; ++j) {
                double mt = do_acg_2d ? tb_mturn_at(j, n_mturn, acg_l10_min, acg_l10_max)
                                      : p.mturn_acg_hom;
                double s = warp_sum_32(w_cmf * tb_scaling_frac(kind, 0, lnM, mt, p));
                if (lane == 0) {
                    sh_acg[j] += s;
                }
            }
            if (mcg_active) {
                for (int j = 0; j < n_mturn; ++j) {
                    double mt = tb_mturn_at(j, n_mturn, mcg_l10_min, mcg_l10_max);
                    double s = warp_sum_32(w_cmf * tb_scaling_frac(kind, 1, lnM, mt, p));
                    if (lane == 0) {
                        sh_mcg[j] += s;
                    }
                }
            }
        }
    }
    __syncwarp();

    double acg_scale = p.prefac_acg * ((kind == 2) ? p.prefac_xray_int : 1.0);
    double mcg_scale = p.prefac_mcg * ((kind == 2) ? p.prefac_xray_int : 1.0);
    for (int j = lane; j < n_acg; j += 32) {
        acg_out[(size_t)i * (size_t)n_acg + (size_t)j] =
            tb_clamp_log(sh_acg[j] * acg_scale, p.clamp_min);
    }
    if (do_mcg_2d) {
        for (int j = lane; j < n_mturn; j += 32) {
            mcg_out[(size_t)i * (size_t)n_mturn + (size_t)j] =
                tb_clamp_log(sh_mcg[j] * mcg_scale, p.clamp_min);
        }
    }
}

/* One warp (32 threads) integrates one redshift i in [0, n_z) for unconditional tables. */
__global__ void uncond_emissivity_gl32_kernel(
    int kind, int n_z, const double *z_vals, const double *growthf_vals, const double *lnMmin_vals,
    double lnMmax, const double *mturn_acg_hom_vals, const double *atomic_thresh_vals,
    const double *sfr_timescale_vals, int do_mcg, int n_mturn, double mcg_l10_min,
    double mcg_l10_max, TableBuildScalingParams p, P21DevTable1Df sig_tbl,
    P21DevTable1Df dsig_tbl, double *acg_out, double *mcg_out) {
    __shared__ double sh_acg;
    __shared__ double sh_mcg[64];

    int i = blockIdx.x;
    int lane = threadIdx.x;
    if (i >= n_z) {
        return;
    }

    if (lane == 0) sh_acg = 0.0;
    for (int j = lane; j < n_mturn; j += 32) {
        sh_mcg[j] = 0.0;
    }
    __syncwarp();

    double z = z_vals[i];
    double gf = growthf_vals[i];
    double lnM_lo = lnMmin_vals[i];
    double mt_acg = mturn_acg_hom_vals[i];
    double a_thresh = atomic_thresh_vals[i];
    double sfr_ts = sfr_timescale_vals[i];
    double span = lnMmax - lnM_lo;

    if (span > 0.0) {
        bool mcg_active = (do_mcg != 0) && (mt_acg <= a_thresh * 1.001);
        double dpan = span / (double)kTailPanels;
        for (int pan = 0; pan < kTailPanels; ++pan) {
            double a = lnM_lo + (double)pan * dpan;
            double b = (pan == kTailPanels - 1) ? lnMmax : (a + dpan);
            double mid = 0.5 * (a + b);
            double half = 0.5 * (b - a);
            double lnM = mid + half * kGL32AllX[lane];
            double w_umf = half * kGL32AllW[lane] *
                           tb_unconditional_hmf(p.hmf_orig, gf, lnM, z, p, sig_tbl, dsig_tbl);

            double sa = warp_sum_32(w_umf * tb_nion_frac_acg(lnM, mt_acg, p));
            if (lane == 0) {
                sh_acg += sa;
            }
            if (mcg_active) {
                for (int j = 0; j < n_mturn; ++j) {
                    double mt_m = tb_mturn_at(j, n_mturn, mcg_l10_min, mcg_l10_max);
                    double sm = warp_sum_32(w_umf * tb_nion_frac_mcg(lnM, mt_m, a_thresh, p));
                    if (lane == 0) {
                        sh_mcg[j] += sm;
                    }
                }
            }
        }
    }
    __syncwarp();

    double acg_pref = (kind == 0) ? (p.fstar_10_acg * p.fesc_10_acg * p.pop2_ion)
                                  : (p.rhocrit_omb * p.fstar_10_acg / sfr_ts);
    double mcg_pref = (kind == 0) ? (p.fstar_7_mcg * p.fesc_7_mcg * p.pop3_ion)
                                  : (p.rhocrit_omb * p.fstar_7_mcg / sfr_ts);

    if (lane == 0) {
        acg_out[i] = sh_acg * acg_pref;
    }
    if (do_mcg) {
        for (int j = lane; j < n_mturn; j += 32) {
            mcg_out[(size_t)i * (size_t)n_mturn + (size_t)j] = sh_mcg[j] * mcg_pref;
        }
    }
}

void build_tb_scaling_params(int kind, double redshift, double growthf, const ScalingConstants *sc,
                             TableBuildScalingParams *out) {
    memset(out, 0, sizeof(*out));
    ScalingConstants eff = *sc;
    if (kind == 1) {
        eff = evolve_scaling_constants_sfr(&eff);
    }

    out->redshift = redshift;
    out->growthf = growthf;
    out->alpha_star_acg = eff.alpha_star_acg;
    out->alpha_star_mcg = eff.alpha_star_mcg;
    out->alpha_esc = eff.alpha_esc;
    out->lnf_star_acg = log(eff.fstar_10_acg);
    out->lnf_star_mcg = log(eff.fstar_7_mcg);
    out->lnf_esc_acg = (eff.fesc_10_acg > 0.0) ? log(eff.fesc_10_acg) : 0.0;
    out->lnf_esc_mcg = (eff.fesc_7_mcg > 0.0) ? log(eff.fesc_7_mcg) : 0.0;
    out->lnMlim_Fstar_acg = (eff.Mlim_Fstar_acg > 0.0) ? log(eff.Mlim_Fstar_acg) : 0.0;
    out->lnMlim_Fstar_mcg = (eff.Mlim_Fstar_mcg > 0.0) ? log(eff.Mlim_Fstar_mcg) : 0.0;
    out->lnMlim_Fesc_acg = (eff.Mlim_Fesc_acg > 0.0) ? log(eff.Mlim_Fesc_acg) : 0.0;
    out->lnMlim_Fesc_mcg = (eff.Mlim_Fesc_mcg > 0.0) ? log(eff.Mlim_Fesc_mcg) : 0.0;
    out->mturn_acg_hom = eff.mturn_acg_homogeneous;
    out->atomic_cooling_threshold = eff.atomic_cooling_threshold;
    out->lx_acg = eff.lx_over_sfr_acg;
    out->lx_mcg = eff.lx_over_sfr_mcg;
    out->sfr_timescale = eff.sfr_timescale;
    out->omb_over_omm = cosmo_params_global->OMb / cosmo_params_global->OMm;
    out->s_per_yr = physconst.s_per_yr;
    out->rhocrit_omm = RHOcrit * cosmo_params_global->OMm;
    out->rhocrit_omb = RHOcrit * cosmo_params_global->OMb;

    if (kind == 0) {
        out->prefac_acg = sc->fstar_10_acg * sc->fesc_10_acg * sc->pop2_ion;
        out->prefac_mcg = sc->fstar_7_mcg * sc->fesc_7_mcg * sc->pop3_ion;
        out->prefac_xray_int = 1.0;
        out->clamp_min = -40.0;
    } else if (kind == 1) {
        out->prefac_acg = out->rhocrit_omb * sc->fstar_10_acg / sc->sfr_timescale;
        out->prefac_mcg = out->rhocrit_omb * sc->fstar_7_mcg / sc->sfr_timescale;
        out->prefac_xray_int = 1.0;
        out->clamp_min = -50.0;
    } else {
        out->prefac_acg = 1.0;
        out->prefac_mcg = 1.0;
        out->prefac_xray_int = out->rhocrit_omm;
        out->clamp_min = -50.0;
    }

    out->use_metallicity = astro_options_global->USE_METALLICITY ? 1 : 0;
    out->hmf_orig = matter_options_global->HMF;
    out->hmf_eff = (out->hmf_orig == HMF_PS || out->hmf_orig == HMF_ST ||
                    out->hmf_orig == HMF_DELOS)
                       ? out->hmf_orig
                       : HMF_PS;
    out->delta_c_sph = physconst.delta_c_sph;
    out->delta_c_delos = physconst.delta_c_delos;
    out->max_deltac_frac = MAX_DELTAC_FRAC;
    out->fract_float_err = FRACT_FLOAT_ERR;
    out->omm = cosmo_params_global->OMm;
    out->oml = cosmo_params_global->OMl;
    out->omr = cosmo_params_global->OMr;
    out->fstar_10_acg = sc->fstar_10_acg;
    out->fesc_10_acg = sc->fesc_10_acg;
    out->pop2_ion = sc->pop2_ion;
    out->fstar_7_mcg = sc->fstar_7_mcg;
    out->fesc_7_mcg = sc->fesc_7_mcg;
    out->pop3_ion = sc->pop3_ion;
}

}  // namespace

extern "C" int p21_cuda_fill_cond_emissivity_tables(
    int kind, double redshift, double growthf, double lnM_min, double lnM_max, double lnM_cond,
    double sigma_cond, int n_delta, double min_dens, double max_dens, int n_mturn, int do_acg_2d,
    double acg_l10_min, double acg_l10_max, int do_mcg_2d, double mcg_l10_min, double mcg_l10_max,
    const ScalingConstants *sc, float *acg_1d_out, float **acg_2d_out, float **mcg_2d_out) {
    if (!p21_cuda_get_enabled() || !sc || kind < 0 || kind > 2 || n_delta < 2 || n_mturn < 2) {
        return -1;
    }
    if (!uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES) ||
        !source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL)) {
        return -1;
    }
    if (astro_options_global->INTEGRATION_METHOD_ACGS != INTEGRATION_METHOD_GAUSS_LEGENDRE ||
        (do_mcg_2d &&
         astro_options_global->INTEGRATION_METHOD_MCGS != INTEGRATION_METHOD_GAUSS_LEGENDRE)) {
        return -1;
    }
    if ((do_acg_2d && !acg_2d_out) || (!do_acg_2d && !acg_1d_out) ||
        (do_mcg_2d && !mcg_2d_out) || n_mturn > 64) {
        return -1;
    }

    double gl_lo = 0.0, gl_hi = 0.0;
    const double *h_xi = nullptr;
    const double *h_wi = nullptr;
    int ngl = p21_get_gl_quadrature(&gl_lo, &gl_hi, &h_xi, &h_wi);
    if (ngl <= 0 || ngl > kMaxGL || !h_xi || !h_wi ||
        (float)gl_lo != (float)lnM_min || (float)gl_hi != (float)lnM_max) {
        return -1;
    }

    P21SamplerTables st{};
    p21_get_sampler_tables(&st);
    if (!st.sigma_1d || !st.dsigmasqdm_1d || !p21_cuda_ensure_init()) {
        return -1;
    }

    TableBuildScalingParams p{};
    build_tb_scaling_params(kind, redshift, growthf, sc, &p);

    const int n_acg = do_acg_2d ? n_mturn : 1;
    const double dcrit_orig = get_delta_crit(p.hmf_orig, sigma_cond, growthf);

    std::vector<int> high_indices;
    if (lnM_min < lnM_cond) {
        for (int i = 0; i < n_delta; ++i) {
            double delta =
                min_dens + (double)((float)i / ((float)n_delta - 1.0f)) * (max_dens - min_dens);
            if (delta > kCritDensTransition && delta <= p.max_deltac_frac * dcrit_orig) {
                high_indices.push_back(i);
            }
        }
    }

    cudaStream_t stream = p21_cuda_stream();
    P21DevTable1Df sig_tbl{};
    P21DevTable1Df dsig_tbl{};
    bool ok = p21_upload_rg1d_f(st.sigma_1d, &sig_tbl, stream) &&
              p21_upload_rg1d_f(st.dsigmasqdm_1d, &dsig_tbl, stream);

    const size_t gl_bytes = (size_t)ngl * sizeof(double);
    const size_t wcmf_bytes = (size_t)n_delta * (size_t)ngl * sizeof(double);
    const size_t facg_bytes = (size_t)n_acg * (size_t)ngl * sizeof(double);
    const size_t fmcg_bytes = do_mcg_2d ? ((size_t)n_mturn * (size_t)ngl * sizeof(double)) : 0ULL;
    const size_t out_acg_elems = (size_t)n_delta * (size_t)n_acg;
    const size_t out_mcg_elems = do_mcg_2d ? ((size_t)n_delta * (size_t)n_mturn) : 0ULL;

    double *d_xi = (double *)p21_cuda_pool_alloc(gl_bytes, stream);
    double *d_wi = (double *)p21_cuda_pool_alloc(gl_bytes, stream);
    double *d_wcmf = (double *)p21_cuda_pool_alloc(wcmf_bytes, stream);
    double *d_facg = (double *)p21_cuda_pool_alloc(facg_bytes, stream);
    double *d_fmcg = fmcg_bytes ? (double *)p21_cuda_pool_alloc(fmcg_bytes, stream) : nullptr;
    float *d_acg_out = (float *)p21_cuda_pool_alloc(out_acg_elems * sizeof(float), stream);
    float *d_mcg_out =
        out_mcg_elems ? (float *)p21_cuda_pool_alloc(out_mcg_elems * sizeof(float), stream) : nullptr;
    int *d_high_idx =
        high_indices.empty()
            ? nullptr
            : (int *)p21_cuda_pool_alloc(high_indices.size() * sizeof(int), stream);

    if (!d_xi || !d_wi || !d_wcmf || !d_facg || (fmcg_bytes && !d_fmcg) || !d_acg_out ||
        (out_mcg_elems && !d_mcg_out) || (!high_indices.empty() && !d_high_idx)) {
        ok = false;
    }

    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(d_xi, h_xi, gl_bytes, cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(d_wi, h_wi, gl_bytes, cudaMemcpyHostToDevice, stream));
        if (ok && !high_indices.empty()) {
            ok = P21_CUDA_OK(cudaMemcpyAsync(d_high_idx, high_indices.data(),
                                             high_indices.size() * sizeof(int),
                                             cudaMemcpyHostToDevice, stream));
        }
    }

    if (ok) {
        int total_cmf = n_delta * ngl;
        cond_cmf_gl_kernel<<<(total_cmf + kCondThreads - 1) / kCondThreads, kCondThreads, 0,
                             stream>>>(ngl, d_xi, d_wi, n_delta, min_dens, max_dens, sigma_cond,
                                       dcrit_orig, p, sig_tbl, dsig_tbl, d_wcmf);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok) {
        int total_scale = n_acg * ngl + (do_mcg_2d ? n_mturn * ngl : 0);
        cond_scaling_gl_kernel<<<(total_scale + kCondThreads - 1) / kCondThreads, kCondThreads, 0,
                                 stream>>>(kind, ngl, d_xi, n_acg, acg_l10_min, acg_l10_max,
                                           do_acg_2d, n_mturn, mcg_l10_min, mcg_l10_max, do_mcg_2d,
                                           n_mturn, p, d_facg, d_fmcg);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok) {
        int max_j = (do_mcg_2d && n_mturn > n_acg) ? n_mturn : n_acg;
        dim3 grid((unsigned int)n_delta, (unsigned int)((max_j + 63) / 64));
        cond_dot_gl_kernel<<<grid, 64, 0, stream>>>(
            kind, ngl, n_delta, min_dens, max_dens, lnM_min, lnM_max, lnM_cond, dcrit_orig, n_acg,
            do_acg_2d, n_mturn, acg_l10_min, acg_l10_max, do_mcg_2d, mcg_l10_min, mcg_l10_max, p,
            d_wcmf, d_facg, d_fmcg, d_acg_out, d_mcg_out);
        ok = P21_CUDA_OK(cudaGetLastError());
    }
    if (ok && !high_indices.empty()) {
        cond_tail_gl32_kernel<<<(int)high_indices.size(), 32, 0, stream>>>(
            kind, (int)high_indices.size(), d_high_idx, n_delta, min_dens, max_dens, lnM_min,
            lnM_max, lnM_cond, sigma_cond, n_acg, do_acg_2d, n_mturn, acg_l10_min, acg_l10_max,
            do_mcg_2d, mcg_l10_min, mcg_l10_max, p, sig_tbl, dsig_tbl, d_acg_out, d_mcg_out);
        ok = P21_CUDA_OK(cudaGetLastError());
    }

    std::vector<float> h_acg(out_acg_elems);
    std::vector<float> h_mcg(out_mcg_elems);
    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(h_acg.data(), d_acg_out, out_acg_elems * sizeof(float),
                                         cudaMemcpyDeviceToHost, stream));
        if (ok && out_mcg_elems > 0) {
            ok = P21_CUDA_OK(cudaMemcpyAsync(h_mcg.data(), d_mcg_out,
                                             out_mcg_elems * sizeof(float), cudaMemcpyDeviceToHost,
                                             stream));
        }
        if (ok) {
            ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
        }
    }

    p21_cuda_pool_free(d_high_idx, stream);
    p21_cuda_pool_free(d_mcg_out, stream);
    p21_cuda_pool_free(d_acg_out, stream);
    p21_cuda_pool_free(d_fmcg, stream);
    p21_cuda_pool_free(d_facg, stream);
    p21_cuda_pool_free(d_wcmf, stream);
    p21_cuda_pool_free(d_wi, stream);
    p21_cuda_pool_free(d_xi, stream);
    p21_free_dev_table(&dsig_tbl, stream);
    p21_free_dev_table(&sig_tbl, stream);

    if (!ok) {
        return -1;
    }
    for (float v : h_acg) {
        if (!isfinite(v)) return -1;
    }
    for (float v : h_mcg) {
        if (!isfinite(v)) return -1;
    }

    if (do_acg_2d) {
        for (int i = 0; i < n_delta; ++i) {
            memcpy(acg_2d_out[i], h_acg.data() + (size_t)i * (size_t)n_mturn,
                   (size_t)n_mturn * sizeof(float));
        }
    } else {
        memcpy(acg_1d_out, h_acg.data(), (size_t)n_delta * sizeof(float));
    }
    if (do_mcg_2d) {
        for (int i = 0; i < n_delta; ++i) {
            memcpy(mcg_2d_out[i], h_mcg.data() + (size_t)i * (size_t)n_mturn,
                   (size_t)n_mturn * sizeof(float));
        }
    }

    p21_cuda_stats_mut()->cond_table_runs++;
    return 0;
}

extern "C" int p21_cuda_fill_uncond_emissivity_tables(
    int kind, int n_z, const double *z_vals, const double *growthf_vals, const double *lnMmin_vals,
    double lnMmax, const double *mturn_acg_hom_vals, const double *atomic_thresh_vals,
    const double *sfr_timescale_vals, int do_mcg, int n_mturn, double mcg_l10_min,
    double mcg_l10_max, const ScalingConstants *sc, double *acg_1d_out, double **mcg_2d_out) {
    if (!p21_cuda_get_enabled() || !sc || kind < 0 || kind > 1 || n_z < 1 || !z_vals ||
        !growthf_vals || !lnMmin_vals || !mturn_acg_hom_vals || !atomic_thresh_vals ||
        !sfr_timescale_vals || !acg_1d_out || (do_mcg && (!mcg_2d_out || n_mturn < 2 || n_mturn > 64))) {
        return -1;
    }
    if (!uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES) ||
        !source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL) ||
        matter_options_global->HMF == HMF_REED07) {
        return -1;
    }

    P21SamplerTables st{};
    p21_get_sampler_tables(&st);
    if (!st.sigma_1d || !st.dsigmasqdm_1d || !p21_cuda_ensure_init()) {
        return -1;
    }

    TableBuildScalingParams p{};
    build_tb_scaling_params(kind, sc->redshift, sc->growth_factor, sc, &p);

    cudaStream_t stream = p21_cuda_stream();
    P21DevTable1Df sig_tbl{};
    P21DevTable1Df dsig_tbl{};
    bool ok = p21_upload_rg1d_f(st.sigma_1d, &sig_tbl, stream) &&
              p21_upload_rg1d_f(st.dsigmasqdm_1d, &dsig_tbl, stream);

    const size_t z_bytes = (size_t)n_z * sizeof(double);
    const size_t mcg_elems = do_mcg ? ((size_t)n_z * (size_t)n_mturn) : 0ULL;

    double *d_z = (double *)p21_cuda_pool_alloc(z_bytes, stream);
    double *d_gf = (double *)p21_cuda_pool_alloc(z_bytes, stream);
    double *d_lnMmin = (double *)p21_cuda_pool_alloc(z_bytes, stream);
    double *d_mt_hom = (double *)p21_cuda_pool_alloc(z_bytes, stream);
    double *d_athresh = (double *)p21_cuda_pool_alloc(z_bytes, stream);
    double *d_sfr_ts = (double *)p21_cuda_pool_alloc(z_bytes, stream);
    double *d_acg_out = (double *)p21_cuda_pool_alloc(z_bytes, stream);
    double *d_mcg_out =
        mcg_elems ? (double *)p21_cuda_pool_alloc(mcg_elems * sizeof(double), stream) : nullptr;

    if (!d_z || !d_gf || !d_lnMmin || !d_mt_hom || !d_athresh || !d_sfr_ts || !d_acg_out ||
        (mcg_elems && !d_mcg_out)) {
        ok = false;
    }

    if (ok) {
        ok = P21_CUDA_OK(cudaMemcpyAsync(d_z, z_vals, z_bytes, cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(
                 cudaMemcpyAsync(d_gf, growthf_vals, z_bytes, cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(
                 cudaMemcpyAsync(d_lnMmin, lnMmin_vals, z_bytes, cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(d_mt_hom, mturn_acg_hom_vals, z_bytes,
                                         cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(d_athresh, atomic_thresh_vals, z_bytes,
                                         cudaMemcpyHostToDevice, stream)) &&
             P21_CUDA_OK(cudaMemcpyAsync(d_sfr_ts, sfr_timescale_vals, z_bytes,
                                         cudaMemcpyHostToDevice, stream));
    }

    if (ok) {
        uncond_emissivity_gl32_kernel<<<n_z, 32, 0, stream>>>(
            kind, n_z, d_z, d_gf, d_lnMmin, lnMmax, d_mt_hom, d_athresh, d_sfr_ts, do_mcg,
            n_mturn, mcg_l10_min, mcg_l10_max, p, sig_tbl, dsig_tbl, d_acg_out, d_mcg_out);
        ok = P21_CUDA_OK(cudaGetLastError());
    }

    std::vector<double> h_acg((size_t)n_z);
    std::vector<double> h_mcg(mcg_elems);
    if (ok) {
        ok = P21_CUDA_OK(
            cudaMemcpyAsync(h_acg.data(), d_acg_out, z_bytes, cudaMemcpyDeviceToHost, stream));
        if (ok && mcg_elems > 0) {
            ok = P21_CUDA_OK(cudaMemcpyAsync(h_mcg.data(), d_mcg_out, mcg_elems * sizeof(double),
                                             cudaMemcpyDeviceToHost, stream));
        }
        if (ok) {
            ok = P21_CUDA_OK(cudaStreamSynchronize(stream));
        }
    }

    p21_cuda_pool_free(d_mcg_out, stream);
    p21_cuda_pool_free(d_acg_out, stream);
    p21_cuda_pool_free(d_sfr_ts, stream);
    p21_cuda_pool_free(d_athresh, stream);
    p21_cuda_pool_free(d_mt_hom, stream);
    p21_cuda_pool_free(d_lnMmin, stream);
    p21_cuda_pool_free(d_gf, stream);
    p21_cuda_pool_free(d_z, stream);
    p21_free_dev_table(&dsig_tbl, stream);
    p21_free_dev_table(&sig_tbl, stream);

    if (!ok) {
        return -1;
    }
    for (double v : h_acg) {
        if (!isfinite(v)) return -1;
    }
    for (double v : h_mcg) {
        if (!isfinite(v)) return -1;
    }

    memcpy(acg_1d_out, h_acg.data(), z_bytes);
    if (do_mcg) {
        for (int i = 0; i < n_z; ++i) {
            memcpy(mcg_2d_out[i], h_mcg.data() + (size_t)i * (size_t)n_mturn,
                   (size_t)n_mturn * sizeof(double));
        }
    }

    p21_cuda_stats_mut()->cond_table_runs++;
    return 0;
}
