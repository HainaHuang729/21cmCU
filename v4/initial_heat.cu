// First redshift, with no stellar Lyman-alpha background.
extern "C" __global__ void initial_heat(int size, const double *p, const double *rec,
                                        const float *density, float *xe, float *tk, float *ts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    double z = p[0], temp = (float)rec[1], electron = (float)rec[0];
    double cmb = p[2] * (1. + z), hscale = pow(1. + z, 3.) * (1. + density[i]);
    double xc = p[3] / cmb *
                ((1. - electron) * p[5] * hscale * kappa_10(temp, 0) +
                 electron * p[6] * hscale * kappa_10_elec(temp, 0) +
                 electron * p[5] * hscale * kappa_10_pH(temp, 0)) /
                p[4];
    ts[i] = (1. + xc) / (1. / cmb + xc / temp);
    tk[i] = rec[1] * (1. + (p[1] ? (float)(.58 - .006 * (z - 10.)) : 0.) * density[i]);
    xe[i] = rec[0];
}
