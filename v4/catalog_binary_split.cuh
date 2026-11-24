// 21cmFAST 4.3.2 Parkinson binary splitting. Active branch masses and
// barriers round to float after each step, as in the native queue.
struct CatalogBinarySplit {
    const float *sigma, *slope;
    const double *j_table;
    int sigma_bins, j_bins;
    double sigma_origin, sigma_width, j_origin, j_width;
    double initial_barrier, target_barrier, minimum, minimum_sigma;
    double factor, gamma1, gamma2;

    __device__ double variance(double log_mass) const {
        double value = catalog_float_table(sigma, log_mass, sigma_origin, sigma_width, sigma_bins);
        return value * value;
    }

    __device__ double alpha(double mass, double log_mass, double variance) const {
        double log_slope =
            catalog_float_table(slope, log_mass, sigma_origin, sigma_width, sigma_bins);
        return (mass / (2. * variance)) * reference_pow(10., log_slope);
    }

    __device__ double unresolved(double sigma_start, double variance_gap, double g1,
                                 double step) const {
        double u = sigma_start / sqrt(variance_gap);
        double maximum = j_origin + (j_bins - 1) * j_width;
        double j;
        if (fabs(gamma1) < 1e-7) {
            j = u;
        } else if (u < j_origin) {
            j = reference_pow(u, 1. - gamma1) / (1. - gamma1);
        } else if (u > maximum) {
            j = (j_table[j_bins - 1] + u) + (.5 * gamma1) * (1. / maximum - 1. / u);
        } else {
            double bin = floor((u - j_origin) / j_width);
            if (!(bin >= 0. && bin < j_bins - 1)) return nan("");
            int index = (int)bin;
            double t = ((u - j_origin) - j_width * index) / j_width;
            j = j_table[index] * (1. - t) + j_table[index + 1] * t;
        }
        return (((.7978845608028654 * g1) * j) / sigma_start) * step;
    }
};

__device__ double catalog_split_volume(double variance, double gap) {
    return variance / (gap * sqrt(gap));
}

__device__ bool draw_catalog_binary(const CatalogBinarySplit &model, MassSampleWords &words,
                                    double mass, float2 *branches, float *masses, int capacity,
                                    int &size) {
    if (capacity <= 0) return false;
    int active = 0, end = 1;
    size = 0;
    branches[0] = make_float2((float)model.initial_barrier, (float)mass);
    double variance_minimum = model.minimum_sigma * model.minimum_sigma;
    while (active < end) {
        double barrier = branches[active].x, parent = branches[active].y;
        double delta = model.target_barrier - barrier;
        double half = .5 * parent;
        double log_half = reference_log(half);
        double sigma = catalog_float_table(model.sigma, reference_log(parent), model.sigma_origin,
                                           model.sigma_width, model.sigma_bins);
        double variance = sigma * sigma;
        double sigma_half = catalog_float_table(model.sigma, log_half, model.sigma_origin,
                                                model.sigma_width, model.sigma_bins);
        double variance_half = sigma_half * sigma_half;
        double gap_half = variance_half - variance;
        double gap_min = variance_minimum - variance;
        double g1 = model.factor * reference_pow(barrier / sigma, model.gamma2);
        double q_min = model.minimum / parent;
        double step = .14142135623730953 * sqrt(gap_half);
        double q = 0., upper = 0.;
        double alpha_half, eta, beta, b, difference;
        if (q_min < .5) {
            alpha_half = model.alpha(half, log_half, variance_half);
            double v_min = catalog_split_volume(variance_minimum, gap_min);
            double v_half = catalog_split_volume(variance_half, gap_half);
            double log_q = reference_log(2. * q_min);
            beta = reference_log(v_min / v_half) / log_q;
            b = reference_pow(2., beta) * v_half;
            double mu = model.gamma1 < 0. ? -reference_log(model.minimum_sigma / sigma_half) / log_q
                                          : alpha_half;
            eta = beta - 1. - model.gamma1 * mu;
            difference = reference_pow(.5, eta) - reference_pow(q_min, eta);
            double g2 = g1 * reference_pow(sigma_half / sigma, model.gamma1) *
                        reference_pow(.5, mu * model.gamma1);
            upper = .7978845608028654 * b * difference / eta * alpha_half * g2;
            step = fmin(step, .1 / upper);
        }
        bool save = step >= delta;
        if (save) step = delta;
        double unresolved = model.unresolved(sigma, gap_min, g1, step);
        if (!isfinite(unresolved) || !isfinite(step)) return false;
        if (q_min < .5) {
            double u;
            if (!catalog_uniform(words, u)) return false;
            if (u < upper * step) {
                if (!catalog_uniform(words, u)) return false;
                q = reference_pow(reference_pow(q_min, eta) + difference * u, 1. / eta);
                double submass = q * parent;
                double log_submass = reference_log(submass);
                double subvariance = model.variance(log_submass);
                double alpha = model.alpha(submass, log_submass, subvariance);
                double ratio = (alpha / alpha_half) *
                               (catalog_split_volume(subvariance, subvariance - variance) /
                                (b * reference_pow(q, beta)));
                if (!catalog_uniform(words, u)) return false;
                if (u > ratio) q = 0.;
            }
        }
        double larger = (1. - unresolved - q) * parent;
        double smaller = q * parent;
        if (save) {
            if (larger > model.minimum) {
                if (size == capacity) return false;
                masses[size++] = (float)larger;
            }
            if (smaller > model.minimum) {
                if (size == capacity) return false;
                masses[size++] = (float)smaller;
            }
            ++active;
        } else {
            float next = (float)(step + barrier);
            if (!(next > barrier)) return false;
            if (smaller > model.minimum) {
                if (end == capacity) return false;
                branches[end++] = make_float2(next, (float)smaller);
            }
            if (larger > model.minimum) {
                branches[active] = make_float2(next, (float)larger);
            } else {
                ++active;
            }
        }
    }
    return true;
}
