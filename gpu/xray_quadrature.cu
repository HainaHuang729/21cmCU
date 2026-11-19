// 7-point Gauss / 15-point Kronrod embedded rule (QUADPACK constants).
__device__ __constant__ double kronrod_nodes[] = {
    0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
    0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
    0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
    0.207784955007898467600689403773245
};
__device__ __constant__ double kronrod_weights[] = {
    0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
    0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
    0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
    0.204432940075298892414161999234649, 0.209482141084727828012999174891714
};
__device__ __constant__ double gauss_weights[] = {
    0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
    0.381830050505118944950369775488975, 0.417959183673469387755102040816327
};

struct SecondaryTable {
    const float *energy;
    const float *ion;
    const float *values;
    int energy_count;
    int ion_count;
    int row;
    int ion_bin;
    float ion_offset;

    __device__ float interpolate(float e, int field) const {
        if (e < energy[0]) return field == 0 ? 1.f : 0.f;
        if (e > .999 * energy[energy_count - 1]) e = .999 * energy[energy_count - 1];
        const int bin = e < 1008.88
            ? int(log(e / 10.) / 1.98026273e-2)
            : 232 + int(log(e / 1008.88) / 9.53101798e-2);
        if (bin < 0 || bin >= energy_count - 1) return nanf("");
        const float *lo = values + (field * ion_count + ion_bin) * energy_count;
        const float *hi = lo + energy_count;
        const float width = energy[bin + 1] - energy[bin];
        const float offset = e - energy[bin];
        const float v = lo[bin] + ((lo[bin + 1] - lo[bin]) / width) * offset;
        const float w = hi[bin] + ((hi[bin + 1] - hi[bin]) / width) * offset;
        return v + ((w - v) / (ion[ion_bin + 1] - ion[ion_bin])) * ion_offset;
    }
};

struct FrequencyIntegrand {
    const double *p;
    SecondaryTable secondary_table;
    int channel;

    __device__ double term(double nu, double threshold, double cross, double abundance,
                           float fraction) const {
        const double dnu = nu - threshold;
        const float e = dnu / p[17];
        if (channel == 0) {
            const double planck = p[11] / p[17];
            return secondary_table.interpolate(e, 0) * planck * dnu * abundance * fraction * cross;
        }
        if (channel == 1) {
            const float extra = secondary_table.interpolate(e, 2)
                              + secondary_table.interpolate(e, 3)
                              + secondary_table.interpolate(e, 4) + 1;
            return extra * abundance * fraction * cross;
        }
        return secondary_table.interpolate(e, 1) * abundance * fraction * cross;
    }

    __device__ double operator()(double nu) const {
        const float xe = secondary_table.ion[secondary_table.row];
        const double energy = nu / p[17];
        const double hi = term(nu, p[8] * p[17], hydrogenic(nu, p[8] * p[17], 1.), p[12], 1 - xe);
        const double hei = term(nu, p[9] * p[17], helium(energy, p[9]), p[13], 1 - xe);
        const double heii = term(nu, p[10] * p[17], hydrogenic(nu, p[10] * p[17], 2.), p[13], xe);
        return ((hi + hei) + heii) * pow(nu / (p[14] * p[17]), -p[16] - 1.);
    }
};

struct Estimate {
    double value;
    double error;
    double abs_deviation;
};

__device__ Estimate integrate_interval(const FrequencyIntegrand &f, double a, double b) {
    constexpr double eps = 2.2204460492503131e-16;
    const double center = .5 * (a + b);
    const double width = .5 * (b - a);
    const double middle = f(center);
    double gauss = middle * gauss_weights[3];
    double kronrod = middle * kronrod_weights[7];
    double abs_integral = fabs(kronrod);
    double left[7], right[7];
    for (int i = 0; i < 7; ++i) {
        const int j = (i < 3) ? 2 * i + 1 : 2 * (i - 3);
        const double delta = width * kronrod_nodes[j];
        left[j] = f(center - delta);
        right[j] = f(center + delta);
        const double pair = left[j] + right[j];
        if (i < 3) gauss += gauss_weights[i] * pair;
        kronrod += kronrod_weights[j] * pair;
        abs_integral += kronrod_weights[j] * (fabs(left[j]) + fabs(right[j]));
    }
    const double mean = .5 * kronrod;
    double deviation = kronrod_weights[7] * fabs(middle - mean);
    for (int j = 0; j < 7; ++j)
        deviation += kronrod_weights[j] * (fabs(left[j] - mean) + fabs(right[j] - mean));
    deviation *= fabs(width);
    abs_integral *= fabs(width);
    double error = fabs((kronrod - gauss) * width);
    if (deviation != 0. && error != 0.)
        error = deviation * fmin(1., pow(200. * error / deviation, 1.5));
    return {kronrod * width, fmax(error, 50. * eps * abs_integral), deviation};
}

struct Interval {
    double lo;
    double hi;
    Estimate integral;
};

__device__ double integrate_frequency(const FrequencyIntegrand &f, double lo, double hi,
                                      unsigned char &interval_count) {
    constexpr int capacity = 128;
    constexpr double relative_error = .01;  // Same as the production native frequency integral.
    Interval intervals[capacity];
    intervals[0] = {lo, hi, integrate_interval(f, lo, hi)};
    double value = intervals[0].integral.value;
    double error = intervals[0].integral.error;
    int used = 1;
    bool split = error > relative_error * fabs(value)
              || (error != 0. && error == intervals[0].integral.abs_deviation);
    while (split && used < capacity) {
        int worst = 0;
        for (int i = 1; i < used; ++i)
            if (intervals[i].integral.error > intervals[worst].integral.error) worst = i;
        const Interval old = intervals[worst];
        const double midpoint = .5 * (old.lo + old.hi);
        const Interval left = {old.lo, midpoint, integrate_interval(f, old.lo, midpoint)};
        const Interval right = {midpoint, old.hi, integrate_interval(f, midpoint, old.hi)};
        value += (left.integral.value + right.integral.value) - old.integral.value;
        error += (left.integral.error + right.integral.error) - old.integral.error;
        if (left.integral.error >= right.integral.error) {
            intervals[worst] = left;
            intervals[used++] = right;
        } else {
            intervals[worst] = right;
            intervals[used++] = left;
        }
        split = error > relative_error * fabs(value);
    }
    interval_count = split ? 0 : used;
    if (split) return nan("");
    double sum = 0.;
    for (int i = 0; i < used; ++i) sum += intervals[i].integral.value;
    return sum;
}

extern "C" __global__ void xray_frequency_adaptive(const double *params, const double *lower_hz,
    const float *energy, const float *ion, const float *values, int nr, int ni, int ne,
    double *result, unsigned char *subintervals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 3 * ni * nr) return;
    const int radius = i % nr;
    const int row = (i / nr) % ni;
    const int channel = i / (nr * ni);
    const float capped_ion = fminf(ion[row], .999 * ion[ni - 1]);
    const int bin = min(row, ni - 2);
    const SecondaryTable table{energy, ion, values, ne, ni, row, bin, capped_ion - ion[bin]};
    const FrequencyIntegrand integrand{params, table, channel};
    unsigned char used = 0;
    const double value = integrate_frequency(integrand, lower_hz[radius], params[15] * params[17], used);
    result[i] = channel == 2 ? value * params[18] : value;
    subintervals[i] = used;
}
