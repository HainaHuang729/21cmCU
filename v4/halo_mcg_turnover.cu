// Inputs are the previous J21, v_cb in km/s, Gamma12, and first-ionization
// redshift. Emit log10 of both turnover masses in solar masses.
extern "C" __global__ void log_turnover(int n, const double *p, const float *lw, const float *vcb,
                                        const float *gamma, const float *z_reion, double *acg,
                                        double *mcg, float *acg_grid, float *mcg_grid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float z = (float)p[0];
    double j21 = lw ? lw[i] : 0.f;
    float velocity = vcb ? vcb[i] : (float)p[3];
    double base = 3.314e7 * pow(1. + z, -1.5);
    double f_lw = 1. + p[5] * pow(j21, p[6]);
    double f_vcb = pow(1. + p[7] * velocity / p[4], p[8]);
    float m_mcg = fmax(base * f_lw * f_vcb, p[2]);
    if (acg) {
        float m_photo =
            reionization_feedback(z, gamma ? gamma[i] : 0.f, z_reion ? z_reion[i] : 0.f);
        float m_acg = fmaxf((float)p[1], m_photo);
        acg[i] = log10((double)m_acg);
        acg_grid[i] = glibc_log10f(m_acg);
        m_mcg = fmaxf(m_mcg, m_photo);
    }
    mcg[i] = log10((double)m_mcg);
    mcg_grid[i] = glibc_log10f(m_mcg);
}
