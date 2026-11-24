// GSL 2.7's two upper-tail algorithms, driven by the original integer words.
__device__ bool catalog_gaussian_tail(MassSampleWords &words, double limit, double &value) {
    if (!isfinite(limit)) return false;
    if (limit < 1.) {
        do {
            double y, radius_squared;
            if (!catalog_polar_pair(words, y, radius_squared)) return false;
            value = y * sqrt((-2. * reference_log(radius_squared)) / radius_squared);
        } while (value < limit);
    } else {
        double u;
        do {
            double v;
            if (!catalog_uniform(words, u)) return false;
            do {
                if (!catalog_uniform(words, v)) return false;
            } while (v == 0.);
            value = sqrt(limit * limit - 2. * reference_log(v));
        } while (value * u > limit);
    }
    return true;
}

// GSL linear interpolation on the reversed native sigma table.
__device__ double catalog_sigma_inverse(const float *values, int bins, double origin, double width,
                                        double sigma) {
    if (!(sigma >= values[bins - 1] && sigma <= values[0])) return nan("");
    int high = 0, low = bins - 1;
    while (low - high > 1) {
        int mid = high + (low - high) / 2;
        if (sigma < values[mid])
            high = mid;
        else
            low = mid;
    }
    double x_lo = values[low], x_hi = values[high];
    double y_lo = origin + low * width, y_hi = origin + high * width;
    if (!(x_hi > x_lo)) return nan("");
    return y_lo + ((sigma - x_lo) / (x_hi - x_lo)) * (y_hi - y_lo);
}

__device__ double catalog_partition_barrier(double sigma, double conditional_sigma, double growth,
                                            double delta_c) {
    double variance =
        sigma == conditional_sigma ? 1e-6 : sigma * sigma - conditional_sigma * conditional_sigma;
    double inverse_sigma = 1. / (sigma * sigma);
    double t1 = -variance * (.81 * inverse_sigma);
    double t2 = (.09499999999999997 * inverse_sigma) * (t1 * variance);
    double t3 = (.3966666666666666 * inverse_sigma) * (t2 * variance);
    double t4 = (.5475 * inverse_sigma) * (t3 * variance);
    double t5 = (t4 * variance) * (.638 * inverse_sigma);
    double sum = ((t3 + t5) + t1) + ((t2 + t4) + 1.);
    double critical = delta_c / growth;
    double b = .34 * reference_pow((.73 * inverse_sigma) * (critical * critical), -.81);
    return (.8544003745317531 * critical) * (1. + b * sum);
}

struct CatalogPartition {
    const float *sigma;
    int bins;
    double origin, width, growth_in, growth_out, delta_c, minimum, minimum_sigma, correction;
    int hmf;

    __device__ double sigma_for(double mass) const {
        return catalog_float_table(sigma, reference_log(mass), origin, width, bins);
    }
};

__device__ bool draw_catalog_partition(const CatalogPartition &model, MassSampleWords &words,
                                       double mass, float *masses, int capacity, int &size) {
    double remaining = mass;
    double from =
        (catalog_barrier(model.sigma_for(mass), model.growth_in, model.delta_c, 0., model.hmf) /
         model.growth_in) *
        model.growth_out;
    double minimum_variance = model.minimum_sigma * model.minimum_sigma;
    size = 0;
    while (remaining > model.minimum) {
        if (size == capacity) return false;
        double sigma = model.sigma_for(remaining);
        double variance = sigma * sigma;
        double delta =
            (catalog_barrier(sigma, model.growth_out, model.delta_c, 0., model.hmf) - from) /
            (remaining / mass);
        double term = ((delta * delta) / model.growth_out) / model.growth_out;
        double limit = sqrt(term / (minimum_variance - variance));
        double target, proposal, acceptance;
        double condition = delta / model.growth_out;
        double bound = model.hmf == 0 ? 0.
                                      : catalog_partition_barrier(model.minimum_sigma, sigma,
                                                                  model.growth_out, model.delta_c) -
                                            condition;
        do {
            double sample;
            if (!catalog_gaussian_tail(words, limit, sample)) return false;
            sample *= model.correction;
            target = sqrt(term / (sample * sample) + variance);
            if (model.hmf == 0) break;
            acceptance =
                bound / (catalog_partition_barrier(target, sigma, model.growth_out, model.delta_c) -
                         condition);
            if (!catalog_uniform(words, proposal)) return false;
        } while (proposal > acceptance);
        double ln_mass =
            catalog_sigma_inverse(model.sigma, model.bins, model.origin, model.width, target);
        double picked = reference_exp(ln_mass);
        if (!isfinite(picked) || !(picked > 0.) || !(remaining - picked < remaining)) return false;
        masses[size++] = (float)picked;
        remaining -= picked;
    }
    return true;
}
