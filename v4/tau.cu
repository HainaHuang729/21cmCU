// Optical depth for photons observed at zp and emitted at zpp.
struct OpticalDepth {
    const double *p, *ion_history, *mcg_history;
    double min_z, step, nu;
    int bins;
    __device__ double operator()(double z) const {
        int i = (int)floor((z - min_z) / step);
        if (i < 0 || i + 1 >= bins) return nan("");
        double t = (z - (min_z + i * step)) / step;
        double ion = ion_history[i] * (1. - t) + ion_history[i + 1] * t;
        double mini = mcg_history ? mcg_history[i] * (1. - t) + mcg_history[i + 1] * t : 0.;
        double neutral =
            (ion < 1.e-20 && mini < 1.e-20) ? 1. : fmax(1.e-4, 1. - (ion + mini) / (1. - p[1]));
        double zp1 = 1.f + (float)z;
        double x = sqrt(p[3]) * pow(zp1, -1.5);
        double dx = -1.5 * sqrt(p[3]) * pow(zp1, -2.5);
        double expansion = sqrt(1. + x * x);
        double dt = p[4] * dx * (1. + x / expansion) / (x + expansion);
        double hz = nu / (1. + p[0]) * (1. + z);
        double cross = p[12] * (1. - p[1]) * hydrogenic(hz, p[8] * p[17], 1.) +
                       p[13] * (1. - p[1]) * helium(hz / p[17], p[9]) +
                       p[13] * p[1] * hydrogenic(hz, p[10] * p[17], 2.);
        return dt * p[2] * pow(1. + z, 3.) * neutral * cross;
    }
    __device__ double residual(double trial, double zpp) {
        nu = trial;
        unsigned char intervals = 0;
        return integrate_frequency(*this, zpp, p[0], intervals, .005) - 1.;
    }
};

extern "C" __global__ void tau_cutoffs(const double *p, const double *edges, const double *history,
                                       const double *molecular, int nr, int bins, double min_z,
                                       double step, double *result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr) return;
    double floor_hz = p[14] * p[17], zpp = edges[i];
    OpticalDepth f = {
        p, history, molecular ? molecular + (long long)i * bins : nullptr, min_z, step, 0., bins};
    if (p[1] > .9999) {
        result[i] = floor_hz;
        return;
    }
    double a = p[9] * p[17], b = 1.e6 * p[17], c = b;
    double fa = f.residual(a, zpp);
    if (fa < 0.) {
        result[i] = fmax(floor_hz, a);
        return;
    }
    double fb = f.residual(b, zpp), fc = fb, d = b - a, e = d;
    if (!(fa >= 0 && fb < 0)) {
        result[i] = nan("");
        return;
    }
    for (int iter = 0; iter < 100; iter++) {
        if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if (fabs(fc) < fabs(fb)) {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        double tol = .5 * 2.220446049250313e-16 * fabs(b), mid = .5 * (c - b);
        if (fb == 0 || fabs(mid) <= tol) break;
        if (fabs(e) < tol || fabs(fa) <= fabs(fb)) {
            d = mid;
            e = mid;
        } else {
            double s = fb / fa, num, den;
            if (a == c) {
                num = 2. * mid * s;
                den = 1. - s;
            } else {
                double q = fa / fc, r = fb / fc;
                num = s * (2. * mid * q * (q - r) - (b - a) * (r - 1.));
                den = (q - 1.) * (r - 1.) * (s - 1.);
            }
            if (num > 0)
                den = -den;
            else
                num = -num;
            if (2. * num < fmin(3. * mid * den - fabs(tol * den), fabs(e * den))) {
                e = d;
                d = num / den;
            } else {
                d = mid;
                e = mid;
            }
        }
        a = b;
        fa = fb;
        b += fabs(d) > tol ? d : copysign(tol, mid);
        fb = f.residual(b, zpp);
        if (!isfinite(fb)) {
            result[i] = nan("");
            return;
        }
        double other = ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) ? a : c;
        if (fabs(other - b) < .02 * fmin(fabs(other), fabs(b))) break;
    }
    result[i] = fmax(floor_hz, b);
}
