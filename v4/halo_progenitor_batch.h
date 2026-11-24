enum { PROGENITOR_WINDOW = 16384, PROGENITOR_OUTPUT = 65536, PROGENITOR_DRAWS = 10 };
int sample_grid_enabled, sample_progenitor_enabled, sample_high_poisson_enabled;
unsigned long long catalog_condition_builds[2], catalog_condition_reuses[2];
int (*prepare_cuda_grid)(int, int, const float *, const float *, const double *);
int (*prepare_cuda_progenitors)(int, int, const float *, const double *, int);
int (*sample_progenitor_window)(int, const double *, int, int, const unsigned int *, int, int, int,
                                const RGTable2D *, double, int, int, int *, float *, double *,
                                int *);

typedef struct {
    int count, next;
    // Sampling may commit only a prefix. Keep the fixed condition block until
    // the worker reaches its end, independently of the available random words.
    index_huge condition_first;
    int condition_count, gpu_condition_count;
    bool device_conditions, tree;
    unsigned long long built, reused;
    double parameters[PROGENITOR_WINDOW][4];
    struct HaloSamplingConstants conditions[PROGENITOR_WINDOW];
    int offsets[PROGENITOR_WINDOW + 1];
    float masses[PROGENITOR_OUTPUT];
    double draws[PROGENITOR_OUTPUT][PROGENITOR_DRAWS];
} ProgenitorWindow;

static int catalog_condition_offset(const ProgenitorWindow *window, index_huge index) {
    if (index < window->condition_first ||
        index - window->condition_first >= (index_huge)window->condition_count)
        return -1;
    return index - window->condition_first;
}

static void load_catalog_condition(ProgenitorWindow *window, index_huge index, double value,
                                   struct HaloSamplingConstants *target) {
    int offset = catalog_condition_offset(window, index);
    if (offset >= 0 && !window->device_conditions) {
        *target = window->conditions[offset];
        ++window->reused;
    } else {
        stoc_set_consts_cond(target, value);
        ++window->built;
    }
}

static void record_catalog_conditions(const ProgenitorWindow *window, int progenitor) {
#pragma omp atomic update
    catalog_condition_builds[progenitor] += window->built;
#pragma omp atomic update
    catalog_condition_reuses[progenitor] += window->reused;
}

static CatalogRandomState *catalog_window_state(gsl_rng *rng) {
    if (!sample_progenitor_window || !catalog_inverse_gpu || rng->type != &catalog_random_type)
        return NULL;
    CatalogRandomState *state = rng->state;
    if (state->recovery) return NULL;
    if (state->cursor == CATALOG_RANDOM_WORDS) {
        catalog_random_get(state);
        if (state->recovery) return NULL;
        --state->cursor;
    }
    return state;
}

static void submit_catalog_window(ProgenitorWindow *window, CatalogRandomState *state, int offset,
                                  int capacity, bool progenitor) {
    int count = window->gpu_condition_count - offset;
    if (count <= 0) return;
    int result[2] = {0};
    const double *parameters = window->device_conditions ? NULL : window->parameters[offset];
    if (sample_progenitor_window(count, parameters, offset, state->worker, state->words,
                                 state->cursor, CATALOG_RANDOM_WORDS - state->cursor, progenitor,
                                 window->tree ? NULL : catalog_inverse_target,
                                 simulation_options_global->SAMPLER_MIN_MASS, MAX_HALO_CELL,
                                 capacity, window->offsets, window->masses, window->draws[0],
                                 result))
        return;
    window->count = result[0];
    state->cursor += result[1];
}

// Retain a condition even if this build cannot sample it on the device. CPU
// fallback still needs the native constants and must begin at the same word.
static bool store_catalog_condition(ProgenitorWindow *window, int offset,
                                    const struct HaloSamplingConstants *condition, int mode,
                                    double target) {
    window->conditions[offset] = *condition;
    window->parameters[offset][0] = condition->cond_val;
    window->parameters[offset][1] = condition->M_cond;
    window->parameters[offset][2] = target;
    window->parameters[offset][3] = mode;
    window->condition_count = offset + 1;
    ++window->built;
    if (!sample_high_poisson_enabled && mode == 4 && !(target <= 10.)) return false;
    window->gpu_condition_count = offset + 1;
    return true;
}

static void prepare_grid_window(ProgenitorWindow *window, const float *density,
                                const float *overlap, index_huge first, index_huge end,
                                size_huge remaining, const struct HaloSamplingConstants *initial,
                                gsl_rng *rng) {
    window->count = window->next = 0;
    if (!sample_grid_enabled || remaining == 0) return;
    CatalogRandomState *state = catalog_window_state(rng);
    if (!state) return;
    int offset = catalog_condition_offset(window, first);
    int capacity = remaining < PROGENITOR_OUTPUT ? remaining : PROGENITOR_OUTPUT;
    if (offset >= 0) {
        submit_catalog_window(window, state, offset, capacity, false);
        return;
    }
    int count = end - first < PROGENITOR_WINDOW ? end - first : PROGENITOR_WINDOW;
    window->condition_first = first;
    window->condition_count = window->gpu_condition_count = 0;
    window->device_conditions = false;
    if (prepare_cuda_grid && count > 0) {
        double bounds[] = {
            initial->growth_out,
            initial->M_cond,
            initial->M_min,
            MAX_DELTAC_FRAC * get_delta_crit(matter_options_global->HMF, initial->sigma_cond,
                                             initial->growth_out),
            DELTA_MIN,
            RHOcrit * cosmo_params_global->OMm};
        if (!prepare_cuda_grid(count, state->worker, density + first, overlap + first, bounds)) {
            window->device_conditions = true;
            window->condition_count = window->gpu_condition_count = count;
            submit_catalog_window(window, state, 0, capacity, false);
            return;
        }
    }
    struct HaloSamplingConstants condition = *initial;
    for (int i = 0; i < count; ++i) {
        index_huge cell = first + i;
        stoc_set_consts_cond(&condition, density[cell] * initial->growth_out);
        double available = 1. - overlap[cell];
        int mode = 4;
        double target = condition.expected_N * available;
        if (condition.delta <= DELTA_MIN) {
            mode = 0;
        } else if (condition.delta >= MAX_DELTAC_FRAC * get_delta_crit(matter_options_global->HMF,
                                                                       condition.sigma_cond,
                                                                       condition.growth_out)) {
            mode = 1;
            target = condition.expected_M * available;
        }
        if (!store_catalog_condition(window, i, &condition, mode, target)) break;
    }
    submit_catalog_window(window, state, 0, capacity, false);
}

static void prepare_progenitor_window(ProgenitorWindow *window, const HaloCatalog *parents,
                                      index_huge first, index_huge end,
                                      const struct HaloSamplingConstants *initial, gsl_rng *rng) {
    window->count = window->next = 0;
    bool number_limited = matter_options_global->SAMPLE_METHOD == SAMPLE_NUMBER_LIMITED;
    bool binary = matter_options_global->SAMPLE_METHOD == SAMPLE_BINARY_SPLIT;
    bool partition = matter_options_global->SAMPLE_METHOD == SAMPLE_PARTITION;
    if (!sample_progenitor_enabled || (!number_limited && !binary && !partition &&
                                       matter_options_global->SAMPLE_METHOD != SAMPLE_MASS_LIMITED))
        return;
    CatalogRandomState *state = catalog_window_state(rng);
    if (!state) return;
    int offset = catalog_condition_offset(window, first);
    if (offset >= 0) {
        submit_catalog_window(window, state, offset, PROGENITOR_OUTPUT, true);
        return;
    }
    int count = end - first < PROGENITOR_WINDOW ? end - first : PROGENITOR_WINDOW;
    window->condition_first = first;
    window->condition_count = window->gpu_condition_count = 0;
    window->device_conditions = false;
    window->tree = binary || partition;
    if (prepare_cuda_progenitors && count > 0) {
        double bounds[] = {initial->growth_in,
                           initial->growth_out,
                           physconst.delta_c_sph,
                           initial->M_min,
                           initial->M_max_tables,
                           MAX_DELTAC_FRAC,
                           DELTA_MIN,
                           RHOcrit * cosmo_params_global->OMm,
                           simulation_options_global->HALOMASS_CORRECTION,
                           initial->sigma_min,
                           simulation_options_global->PARKINSON_G0,
                           simulation_options_global->PARKINSON_y1,
                           simulation_options_global->PARKINSON_y2,
                           simulation_options_global->SAMPLER_MIN_MASS,
                           (double)matter_options_global->HMF,
                           physconst.delta_c_delos};
        if (!prepare_cuda_progenitors(count, state->worker, parents->halo_masses + first, bounds,
                                      partition ? 3
                                      : binary  ? 2
                                                : number_limited)) {
            window->device_conditions = true;
            window->condition_count = window->gpu_condition_count = count;
            submit_catalog_window(window, state, 0, PROGENITOR_OUTPUT, true);
            return;
        }
    }
    if (window->tree) return;
    struct HaloSamplingConstants condition = *initial;
    for (int i = 0; i < count; ++i) {
        double mass = parents->halo_masses[first + i];
        if (mass < initial->M_min || mass > initial->M_max_tables) break;
        stoc_set_consts_cond(&condition, mass);
        int mode;
        double target = condition.expected_M;
        if (condition.delta <= DELTA_MIN) {
            mode = 0;
        } else if (condition.delta >= MAX_DELTAC_FRAC * get_delta_crit(matter_options_global->HMF,
                                                                       condition.sigma_cond,
                                                                       condition.growth_out)) {
            mode = 1;
        } else if (number_limited) {
            mode = 4;
            target = condition.expected_N;
        } else {
            target *= simulation_options_global->HALOMASS_CORRECTION;
            bool rare = condition.sigma_cond * 7. * condition.growth_out < condition.delta_crit;
            mode = rare ? 2 : 3;
        }
        if (!store_catalog_condition(window, i, &condition, mode, target)) break;
    }
    submit_catalog_window(window, state, 0, PROGENITOR_OUTPUT, true);
}

// GSL's integer gamma tail uses the host's scalar tangent. Evaluate it once
// per buffered word when that window first needs the high-mean branch.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-fast-math", "fp-contract=off", "no-tree-vectorize")))
#endif
void catalog_word_tangents(int count, const unsigned int *words, double *out) {
    for (int i = 0; i < count; ++i) {
        volatile double angle = 3.14159265358979323846 * (words[i] * 0x1p-32);
        out[i] = tan(angle);
    }
}
