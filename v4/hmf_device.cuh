// Shared halo mass function, conditional mass function, and Gamma-approx helpers for v4.3.2.
// HMF codes match InputParameters.h:
//   0: PS, 1: ST, 2: WATSON, 3: WATSON-Z, 4: DELOS, 5: REED07, 6: YUNG24.

__device__ inline double hmf_sheth_delc_fixed(double del, double sig) {
    return sqrt(0.73) * del * (1. + 0.34 * pow(sig * sig / (0.73 * del * del), 0.81));
}

__device__ inline double hmf_delta_crit(int hmf, double sig, double growthf, double delta_c_sph,
                                        double delta_c_delos) {
    if (hmf == 4) return delta_c_delos;
    if (hmf == 1) return hmf_sheth_delc_fixed(delta_c_sph / growthf, sig) * growthf;
    return delta_c_sph;
}

__device__ inline double hmf_conditional(int hmf, double growthf, double delta_cond,
                                         double sigma_cond, double sigma1, double pos_dsigmasqdm,
                                         double delta_c_sph, double delta_c_delos) {
    if (sigma1 < sigma_cond) return 0.;
    double sigdiff_inv =
        sigma1 == sigma_cond ? 1.e6 : 1. / (sigma1 * sigma1 - sigma_cond * sigma_cond);
    if (hmf == 1) {
        double variance = sigma1 == sigma_cond ? 1.e-6 : sigma1 * sigma1 - sigma_cond * sigma_cond;
        double inv = 1. / variance;
        double terms[6] = {1.};
        for (int j = 1; j < 6; ++j)
            terms[j] = terms[j - 1] * (-variance) / j * (.81 - j + 1) * (1. / (sigma1 * sigma1));
        double sum = 0.;
        for (int j = 5; j >= 0; --j) sum += terms[j];
        double del = delta_c_sph / growthf;
        double c = sqrt(.73) * del;
        double b = .34 * pow((1. / (sigma1 * sigma1)) * (.73 * del * del), -.81);
        double offset = delta_cond / growthf;
        double barrier = c * (1. + b) - offset;
        return pos_dsigmasqdm * (c * (1. + b * sum) - offset) * pow(inv, 1.5) *
               exp(-barrier * barrier * .5 * inv) / sqrt(2. * 3.14159265358979323846);
    }
    if (hmf == 4) {
        double dsigmadm = -0.5 * pos_dsigmasqdm;
        double nu = (delta_c_delos - delta_cond) * sqrt(sigdiff_inv) / growthf;
        return 0.519 * pow(nu, 0.582) * exp(-0.469 * nu * nu) * fabs(dsigmadm) * sigdiff_inv;
    }
    double del = (delta_c_sph - delta_cond) / growthf;
    return pos_dsigmasqdm * del * pow(sigdiff_inv, 1.5) * exp(-del * del * 0.5 * sigdiff_inv) /
           sqrt(2. * 3.14159265358979323846);
}

__device__ inline double hmf_unconditional(int hmf, double z, double growthf, double lnM,
                                           double sigma0, double pos_dsigmasqdm, double omm,
                                           double oml, double omr, double delta_c_sph,
                                           double delta_c_delos) {
    const double sqrt2pi = 0.79788456080286535588;  // sqrt(2 / pi)
    double sigma = sigma0 * growthf;
    double pos_dsigmadm = pos_dsigmasqdm * (growthf * growthf / (2. * sigma));
    if (hmf == 0) {
        return sqrt2pi * (delta_c_sph / (sigma * sigma)) * pos_dsigmadm *
               exp(-(delta_c_sph * delta_c_sph) / (2. * sigma * sigma));
    }
    if (hmf == 1) {
        double nuhat = sqrt(0.73) * delta_c_sph / sigma;
        return (pos_dsigmadm / sigma) * sqrt2pi * 0.353 * (1. + pow(nuhat, -0.35)) * nuhat *
               exp(-0.5 * nuhat * nuhat);
    }
    if (hmf == 2) {
        double f_sig = 0.282 * (pow(1.406 / sigma, 2.163) + 1.) * exp(-1.210 / (sigma * sigma));
        return (pos_dsigmadm / sigma) * f_sig;
    }
    if (hmf == 3) {
        double zp1 = 1. + z;
        double omz = omm * pow(zp1, 3.) / (oml + omm * pow(zp1, 3.) + omr * pow(zp1, 4.));
        double Az = omz * (0.990 * pow(zp1, -3.216) + 0.074);
        double az = omz * (5.907 * pow(zp1, -3.058) + 2.349);
        double bz = omz * (3.136 * pow(zp1, -3.599) + 2.344);
        double f_sig = Az * (pow(bz / sigma, az) + 1.) * exp(-1.318 / (sigma * sigma));
        return (pos_dsigmadm / sigma) * f_sig;
    }
    if (hmf == 4) {
        double sigma_inv = 1. / sigma0;
        double dsigmadm_abs = 0.5 * pos_dsigmasqdm * sigma_inv;
        double nu = delta_c_delos * sigma_inv / growthf;
        return 0.519 * pow(nu, 0.582) * exp(-0.469 * nu * nu) * dsigmadm_abs * sigma_inv;
    }
    if (hmf == 5) {
        // Note: hmf.c effective_spectral_index computes -exp(lnM)*dsigmasqdm/(2*sigma0^2).
        double dlnsdlnm = exp(lnM) * pos_dsigmasqdm / (2. * sigma0 * sigma0);
        double neff = -3. * (2. * dlnsdlnm + 1.);
        double nu = delta_c_sph / sigma;
        double lnsigma = -log(sigma);
        double G1 = exp(-pow(lnsigma - 0.4, 2.) / (2. * 0.6 * 0.6));
        double G2 = exp(-pow(lnsigma - 0.75, 2.) / (2. * 0.2 * 0.2));
        double a_prefac = 0.764 / 1.08;
        double f_sig =
            0.3222 * sqrt(2. * a_prefac / 3.14159265358979323846) *
            (1. + pow(1. / (a_prefac * nu * nu), 0.3) + 0.6 * G1 + 0.4 * G2) * nu *
            exp(-1.08 * a_prefac * nu * nu / 2. - 0.03 * pow(nu, 0.6) / pow(neff + 3., 2.));
        return (pos_dsigmadm / sigma) * f_sig;
    }
    double Az = 0.13765772 + (-0.01003821) * z + 0.00102964 * z * z;
    double az = 1.06641384 + 0.02475576 * z + (-0.00283342) * z * z;
    double bz = 4.86693806 + 0.09212356 * z + (-0.01426283) * z * z;
    double cz = 1.19837952 + (-0.00142967) * z + (-0.00033074) * z * z;
    double f_sig = Az * (pow(sigma / bz, -az) + 1.) * exp(-cz / (sigma * sigma));
    return (pos_dsigmadm / sigma) * f_sig;
}

__device__ inline double hmf_gamma_inc_upper(double a, double x) {
    if (isinf(x)) return 0.;
    if (x <= 0.) return a > 0. ? tgamma(a) : nan("");
    if (x >= 1.5 && (a <= 0.5 || x >= a + 1.0)) {
        double b0 = x + 1.0 - a;
        double c = 1e300;
        double d = fabs(b0) < 1e-300 ? 1e300 : 1.0 / b0;
        double h = d;
        for (int i = 1; i <= 200; ++i) {
            double an = -i * (i - a);
            double bn = b0 + 2.0 * i;
            d = bn + an * d;
            if (fabs(d) < 1e-300) d = 1e-300;
            c = bn + an / c;
            if (fabs(c) < 1e-300) c = 1e-300;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (fabs(del - 1.0) <= 1e-15) break;
        }
        return exp(-x + a * log(x)) * h;
    }
    if (a > 0.5) {
        double ap = a;
        double sum = 1.0 / a;
        double del = sum;
        for (int k = 1; k <= 300; ++k) {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (fabs(del) <= fabs(sum) * 1e-15) break;
        }
        double P = sum * exp(-x + a * log(x) - lgamma(a));
        return exp(lgamma(a)) * (1.0 - P);
    }
    int shifts = (int)floor(0.5 - a);
    double b = a + shifts;
    double lnx = log(x);
    double blnx = b * lnx;
    double xbm1_over_b =
        fabs(blnx) < 1e-7 ? lnx * (1.0 + blnx * (0.5 + blnx / 6.0)) : expm1(blnx) / b;
    double g1pm1_over_b;
    if (fabs(b) < 1e-4) {
        g1pm1_over_b = -0.57721566490153286061 +
                       b * (0.98905599532797255539 +
                            b * (-0.90747907608088628902 + b * 0.98172808683440018723));
    } else {
        g1pm1_over_b = (tgamma(1.0 + b) - 1.0) / b;
    }
    double xb = pow(x, b);
    double series = 0.0;
    double neg_xk_fact = 1.0;
    for (int k = 1; k <= 35; ++k) {
        neg_xk_fact *= (-x) / k;
        double term = neg_xk_fact / (k + b);
        series += term;
        if (fabs(term) <= 1e-16 * (fabs(series) + 1.0)) break;
    }
    double val = (g1pm1_over_b - xbm1_over_b) - xb * series;
    for (int s = 0; s < shifts; ++s) {
        b -= 1.0;
        val = (val - exp(-x + b * lnx)) / b;
    }
    return val;
}

__device__ inline double hmf_fcoll_approx(double numin, double beta) {
    double gg = hmf_gamma_inc_upper(0.5 + beta, 0.5 * numin);
    return gg * pow(2.0, 0.5 + beta) / sqrt(2.0 * 3.14159265358979323846);
}

__device__ inline double hmf_fcoll_approx_condition(double numin, double nucondition, double beta) {
    return (hmf_fcoll_approx(numin, beta) - hmf_fcoll_approx(nucondition, beta)) +
           hmf_fcoll_approx(nucondition, 0.0) * pow(nucondition, beta);
}

template <class SigmaLookup>
__device__ inline double hmf_integral_gamma_approx(SigmaLookup sigma_lookup, double lnM_lo,
                                                   double lnM_hi, double growthf, double delta,
                                                   double sigma_c, double delta_c_sph,
                                                   double lnMturn_l, double Mturn_upper,
                                                   double index_base, int abs_gamma_type) {
    const double MPIVOT1 = 1.5e9;
    const double MPIVOT2 = 5.3e5;
    const double AINDEX1 = 9.0;
    const double AINDEX2 = 13.6;
    const double AINDEX3 = 21.0;

    double lnM_lo_limit = lnM_lo;
    double lnM_hi_limit = lnM_hi;
    double lnMturn_u = log(Mturn_upper);
    double lnMp1 = log(MPIVOT1);
    double lnMp2 = log(MPIVOT2);

    if (abs_gamma_type >= 3 && lnMturn_l > lnM_lo_limit) lnM_lo_limit = lnMturn_l;
    if (abs_gamma_type == 4 && lnMturn_u < lnM_hi_limit) lnM_hi_limit = lnMturn_u;

    if (lnM_lo_limit >= lnM_hi_limit || sigma_lookup(lnM_lo_limit) <= sigma_c) {
        return 0.0;
    }

    double delta_arg = pow((delta_c_sph - delta) / growthf, 2.0);
    double beta1 = index_base * AINDEX1 * 0.5;
    double beta2 = index_base * AINDEX2 * 0.5;
    double beta3 = index_base * AINDEX3 * 0.5;

    double sigma_pivot1 = sigma_lookup(lnMp1);
    double sigma_pivot2 = sigma_lookup(lnMp2);
    double sigma_lo_limit = sigma_lookup(lnM_lo_limit);
    double sigma_hi_limit = sigma_lookup(lnM_hi_limit);

    double nu_pivot1_umf = delta_arg / (sigma_pivot1 * sigma_pivot1);
    double nu_pivot2_umf = delta_arg / (sigma_pivot2 * sigma_pivot2);
    double nu_condition = delta_arg / (sigma_c * sigma_c);

    double nu_pivot1 = delta_arg / (sigma_pivot1 * sigma_pivot1 - sigma_c * sigma_c);
    double nu_pivot2 = delta_arg / (sigma_pivot2 * sigma_pivot2 - sigma_c * sigma_c);
    double nu_lo_limit = delta_arg / (sigma_lo_limit * sigma_lo_limit - sigma_c * sigma_c);
    double nu_hi_limit = delta_arg / (sigma_hi_limit * sigma_hi_limit - sigma_c * sigma_c);

    double fcoll = 0.0;
    if (abs_gamma_type == 4) {
        if (nu_hi_limit <= nu_pivot2) {
            fcoll += hmf_fcoll_approx(nu_lo_limit, beta3) * pow(nu_pivot2_umf, -beta3);
            fcoll -= hmf_fcoll_approx(nu_hi_limit, beta3) * pow(nu_pivot2_umf, -beta3);
        } else {
            fcoll -= hmf_fcoll_approx(nu_hi_limit, beta2) * pow(nu_pivot1_umf, -beta2);
            if (nu_lo_limit > nu_pivot2) {
                fcoll += hmf_fcoll_approx(nu_lo_limit, beta2) * pow(nu_pivot1_umf, -beta2);
            } else {
                fcoll += hmf_fcoll_approx(nu_pivot2, beta2) * pow(nu_pivot1_umf, -beta2);
                fcoll +=
                    (hmf_fcoll_approx(nu_lo_limit, beta3) - hmf_fcoll_approx(nu_pivot2, beta3)) *
                    pow(nu_pivot2_umf, -beta3);
            }
        }
    } else {
        if (nu_lo_limit >= nu_condition) {
            return hmf_fcoll_approx(nu_lo_limit, 0.0);
        }
        if (nu_lo_limit >= nu_pivot1) {
            fcoll += hmf_fcoll_approx_condition(nu_lo_limit, nu_condition, beta1) *
                     pow(nu_pivot1_umf, -beta1);
        } else {
            fcoll += hmf_fcoll_approx_condition(nu_pivot1, nu_condition, beta1) *
                     pow(nu_pivot1_umf, -beta1);
            if (nu_lo_limit > nu_pivot2) {
                fcoll +=
                    (hmf_fcoll_approx(nu_lo_limit, beta2) - hmf_fcoll_approx(nu_pivot1, beta2)) *
                    pow(nu_pivot1_umf, -beta2);
            } else {
                fcoll += (hmf_fcoll_approx(nu_pivot2, beta2) - hmf_fcoll_approx(nu_pivot1, beta2)) *
                         pow(nu_pivot1_umf, -beta2);
                fcoll +=
                    (hmf_fcoll_approx(nu_lo_limit, beta3) - hmf_fcoll_approx(nu_pivot2, beta3)) *
                    pow(nu_pivot2_umf, -beta3);
            }
        }
    }
    if (fcoll <= 0.0) fcoll = 1e-40;
    return fcoll;
}
