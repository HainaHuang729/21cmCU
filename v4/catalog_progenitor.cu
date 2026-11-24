// A native worker consumes the masses and placement words in condition order.
// Commit a prefix only after all draws for its final condition fit in the window.
__device__ bool position_draws(MassSampleWords &cursor, double *draws, int size) {
    for (int axis = 0; axis < size; ++axis) {
        unsigned word;
        if (!cursor.take(word)) return false;
        draws[axis] = word * 2.3283064365386963e-10;
    }
    return true;
}

__device__ bool catalog_polar_pair(MassSampleWords &cursor, double &y, double &radius_squared) {
    unsigned word;
    double x;
    do {
        do {
            if (!cursor.take(word)) return false;
        } while (!word);
        x = 2. * (word * 2.3283064365386963e-10) - 1.;
        do {
            if (!cursor.take(word)) return false;
        } while (!word);
        y = 2. * (word * 2.3283064365386963e-10) - 1.;
        radius_squared = x * x + y * y;
    } while (radius_squared > 1. || radius_squared == 0.);
    return true;
}

__device__ bool placement_draws(MassSampleWords &cursor, double *draws, bool progenitor) {
    if (!progenitor && !position_draws(cursor, draws, 3)) return false;
    for (int axis = 0; axis < 3; ++axis) {
        if (!catalog_polar_pair(cursor, draws[4 + axis], draws[7 + axis])) return false;
    }
    draws[3] = 0.;
    return !progenitor || position_draws(cursor, draws, 4);
}

__device__ bool catalog_uniform(MassSampleWords &cursor, double &value) {
    unsigned word;
    if (!cursor.take(word)) return false;
    value = word * 2.3283064365386963e-10;
    return true;
}

__device__ double catalog_tangent(unsigned word, const double *tangents, int index) {
#ifdef CATALOG_DEVICE_TAN
    return catalog_word_tangent(word);
#else
    return tangents[index];
#endif
}

// GSL_DEVICE_DISTRIBUTIONS

// CATALOG_BINARY_SPLIT

__device__ double catalog_sheth_barrier(double sigma, double growth, double delta_c);
__device__ double catalog_barrier(double sigma, double growth, double delta_c, double delta_c_delos,
                                  int hmf);

// CATALOG_PARTITION

__device__ bool draw_grid_poisson(MassSampleWords &cursor, double mean, int &count,
                                  const double *tangents = nullptr) {
    // GSL uses the product method through mean 10 inclusive. Larger means
    // switch to a gamma/binomial decomposition on the original stream.
    if (!isfinite(mean)) return false;
    count = 0;
    while (mean > 10.) {
#ifdef CATALOG_HIGH_POISSON
#ifndef CATALOG_DEVICE_TAN
        if (!tangents) return false;
#endif
        if (mean > 2000000000.) return false;
        unsigned m = (unsigned)(mean * .875);
        double gamma = catalog_gamma_int(cursor, m, tangents);
        if (!(gamma > 0.)) return false;
        if (gamma >= mean) {
            unsigned pick = gsl_ran_binomial(cursor, mean / gamma, m - 1);
            if (pick == 0xffffffffu || pick > 0x7fffffffu - (unsigned)count) return false;
            count += pick;
            return true;
        }
        if (m > 0x7fffffffu - (unsigned)count) return false;
        count += m;
        mean -= gamma;
#else
        return false;
#endif
    }
    double threshold = reference_exp(-mean);
    double product = 1.;
    --count;
    do {
        unsigned word;
        if (!cursor.take(word)) return false;
        product *= word * 2.3283064365386963e-10;
        ++count;
    } while (product > threshold);
    return true;
}

extern "C" __global__ void draw_catalog_prefix(
    int count, const double *conditions, const unsigned *words, int word_count,
    const double *tangents, const double *table, const double *axes, int nx, int ny,
    double minimum_mass, int progenitor, float *scratch, int max_masses, int capacity, int *offsets,
    float *masses, double *draws, int *result, const float *tree_sigma, const float *tree_slope,
    const double *tree_j, const double *tree_parameters, float2 *binary_branches) {
    MassSampleWords cursor;
    cursor.words = words;
    cursor.count = word_count;
    MassSampleTable inverse = {};
    if (table) inverse = {table, nx, ny, axes[0], axes[1], axes[2], axes[3], axes[4]};
    int stored = 0;
    offsets[0] = 0;
    result[0] = result[1] = result[2] = result[3] = 0;
    for (int condition = 0; condition < count; ++condition) {
        const double *p = conditions + 4 * condition;
        int mode = (int)p[3];
        if (mode < 0) return;
        int size = 0;
        if (mode == 1) {
            scratch[0] = (float)p[2];
            size = 1;
        } else if (mode == 4) {
            if (!draw_grid_poisson(cursor, p[2], size, tangents) || size > max_masses) return;
            for (int i = 0; i < size; ++i) {
                unsigned word;
                double mass;
                if (!cursor.take(word) || !inverse.mass(p[0], p[1], word, mass)) return;
                scratch[i] = (float)mass;
            }
        } else if (mode == 5) {
            const double *v = tree_parameters;
            CatalogBinarySplit model = {tree_sigma, tree_slope, tree_j, (int)v[0], (int)v[1], v[2],
                                        v[3],       v[4],       v[5],   v[6],      v[7],      v[8],
                                        v[9],       v[10],      v[11],  v[12]};
            if (!draw_catalog_binary(model, cursor, p[1], binary_branches, scratch, max_masses,
                                     size))
                return;
        } else if (mode == 6) {
            const double *v = tree_parameters;
            CatalogPartition model = {tree_sigma, (int)v[0], v[1], v[2], v[3],     v[4],
                                      v[5],       v[6],      v[7], v[8], (int)v[9]};
            if (!draw_catalog_partition(model, cursor, p[1], scratch, max_masses, size)) return;
        } else if (mode >= 2 && !draw_limited_catalog_mass(inverse, cursor, p[0], p[1], p[2],
                                                           mode == 2, max_masses, scratch, size)) {
            return;
        }
        for (int i = 0; i < size; ++i) {
            if (scratch[i] < minimum_mass) continue;
            if (stored == capacity ||
                !placement_draws(cursor, draws + 10 * stored, progenitor != 0))
                return;
            masses[stored++] = scratch[i];
        }
        offsets[condition + 1] = stored;
        result[0] = condition + 1;
        result[1] = cursor.consumed;
        result[3] = stored;
        if (mode == 4 && p[2] > 10.) ++result[2];
    }
}

// Match the optimized native EvaluateRGTable1D operation order.
__device__ double catalog_moment(double delta, const double *moments, int index, int column,
                                 double origin, double width) {
    double weight = ((delta - origin) - width * index) / width;
    return moments[2 * index + column] * (1. - weight) + moments[2 * (index + 1) + column] * weight;
}

extern "C" __global__ void prepare_catalog_grid(int count, const float *density,
                                                const float *overlap, const double *moments,
                                                int bins, double origin, double width,
                                                double growth, double mass, double minimum,
                                                double collapse, double density_min, double rho_m,
                                                double *conditions) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    double delta = density[i] * growth;
    double available = 1. - overlap[i];
    double target = 0.;
    int mode = -1;  // Defer a condition outside the tabulated domain to the native sampler.
    if (isfinite(delta) && isfinite(available)) {
        if (delta <= density_min) {
            mode = 0;
        } else if (delta > collapse) {
            mode = 1;
            target = mass * available;
        } else if (mass < minimum) {
            mode = delta == collapse ? 1 : 4;
        } else {
            double bin = floor((delta - origin) / width);
            if (bin >= 0. && bin < bins - 1) {
                mode = delta == collapse ? 1 : 4;
                double moment = catalog_moment(delta, moments, (int)bin, mode == 1, origin, width);
                target = moment * mass;
                if (mode == 4) target /= rho_m;
                target *= available;
            }
        }
    }
    double *row = conditions + 4 * i;
    row[0] = delta;
    row[1] = mass;
    row[2] = target;
    row[3] = mode;
}

// get_delta_crit(HMF_ST) from the native release, including reassociation
// around the scalar pow call. The order affects threshold comparisons.
__device__ double catalog_sheth_barrier(double sigma, double growth, double delta_c) {
    double critical = delta_c / growth;
    double ratio = ((sigma * sigma) * 1.36986301369863) / (critical * critical);
    double ellipsoidal = .34 * reference_pow(ratio, .81);
    return (growth * .8544003745317531) * (ellipsoidal * critical + critical);
}

__device__ double catalog_barrier(double sigma, double growth, double delta_c, double delta_c_delos,
                                  int hmf) {
    if (hmf == 1) return catalog_sheth_barrier(sigma, growth, delta_c);
    if (hmf == 4) return delta_c_delos;
    return delta_c;
}

extern "C" __global__ void prepare_catalog_progenitors(
    int count, const float *masses, const double *moments, int bins, double origin, double width,
    const float *sigma_table, int sigma_bins, double sigma_origin, double sigma_width,
    double growth_in, double growth_out, double delta_c, double minimum, double maximum,
    double collapse_fraction, double density_min, double rho_m, double mass_correction,
    int sampling_method, int hmf, double delta_c_delos, double *conditions) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    double mass = masses[i];
    double *row = conditions + 4 * i;
    row[0] = 0.;
    row[1] = mass;
    row[2] = 0.;
    row[3] = -1.;
    if (!(mass > 0. && mass >= minimum && mass <= maximum)) return;

    double log_mass = reference_log(mass);
    double sigma =
        catalog_float_table(sigma_table, log_mass, sigma_origin, sigma_width, sigma_bins);
    if (!(sigma > 0.)) return;
    double delta =
        (catalog_barrier(sigma, growth_in, delta_c, delta_c_delos, hmf) / growth_in) * growth_out;
    double barrier = catalog_barrier(sigma, growth_out, delta_c, delta_c_delos, hmf);
    double collapse = collapse_fraction * barrier;
    if (!isfinite(delta) || !isfinite(collapse)) return;

    int mode;
    double target = 0.;
    if (delta <= density_min) {
        mode = 0;
    } else if (delta > collapse) {
        mode = 1;
        target = mass;
    } else if (delta < collapse && sampling_method == 2) {
        mode = 5;
    } else if (delta < collapse && sampling_method == 3) {
        mode = 6;
    } else {
        double bin = floor((log_mass - origin) / width);
        if (!(bin >= 0. && bin < bins - 1)) return;
        if (delta == collapse) {
            mode = 1;
        } else if (sampling_method == 1) {
            mode = 4;
        } else {
            mode = sigma * 7. * growth_out < barrier ? 2 : 3;
        }
        target = catalog_moment(log_mass, moments, (int)bin, mode != 4, origin, width) * mass;
        if (mode == 4) target /= rho_m;
        if (mode == 2 || mode == 3) target *= mass_correction;
    }
    row[0] = log_mass;
    row[2] = target;
    row[3] = mode;
}
