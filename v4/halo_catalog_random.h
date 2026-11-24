/* GSL still owns the distribution algorithms and their rejection rules.
 * The existing CUDA engines supply each worker's words in native order. */
enum { CATALOG_RANDOM_WORDS = 624 * 256 };
int (*catalog_random_words)(int worker, unsigned int seed, int first, int count,
                            unsigned int *words);

typedef struct {
    gsl_rng rng;
    gsl_rng *recovery;
    int worker, cursor;
    unsigned int seed;
    unsigned long long supplied;
    unsigned int *words;
} CatalogRandomState;

static const gsl_rng_type *catalog_worker_rng_type(int worker) {
    switch (worker % 5) {
        case 0:
            return gsl_rng_mt19937;
        case 1:
            return gsl_rng_gfsr4;
        case 2:
            return gsl_rng_cmrg;
        case 3:
            return gsl_rng_mrg;
        default:
            return gsl_rng_taus2;
    }
}

static bool catalog_worker_is_m31(int worker) {
    int k = worker % 5;
    return k == 2 || k == 3;
}

static void catalog_random_set(void *opaque, unsigned long seed) {
    CatalogRandomState *state = opaque;
    state->seed = seed;
    state->cursor = CATALOG_RANDOM_WORDS;
    state->supplied = 0;
}

static unsigned long catalog_random_get(void *opaque) {
    CatalogRandomState *state = opaque;
    if (state->cursor == CATALOG_RANDOM_WORDS && !state->recovery) {
        if (catalog_random_words(state->worker, state->seed, state->supplied == 0,
                                 CATALOG_RANDOM_WORDS, state->words) != 0) {
            /* Let the native loop unwind with its original trajectory. The
             * failed callback is then reported by the catalogue entry point. */
            state->recovery = gsl_rng_alloc(catalog_worker_rng_type(state->worker));
            if (!state->recovery) Throw(MemoryAllocError);
            gsl_rng_set(state->recovery, state->seed);
            for (unsigned long long i = 0; i < state->supplied; ++i) gsl_rng_get(state->recovery);
        } else {
            state->supplied += CATALOG_RANDOM_WORDS;
            state->cursor = 0;
        }
    }
    if (state->recovery) return gsl_rng_get(state->recovery);
    return state->words[state->cursor++];
}

static double catalog_random_double(void *opaque) { return catalog_random_get(opaque) * 0x1p-32; }

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-fast-math", "fp-contract=off")))
#endif
static double catalog_random_double_m31(void *opaque) {
    return (double)catalog_random_get(opaque) / 2147483647.0;
}

static const gsl_rng_type catalog_random_type = {
    "cuda-catalog-words", 0xffffffffUL,         0, sizeof(CatalogRandomState), catalog_random_set,
    catalog_random_get,   catalog_random_double};

static const gsl_rng_type catalog_random_type_m31 = {
    "cuda-catalog-words-m31",   2147483646UL,       0,
    sizeof(CatalogRandomState), catalog_random_set, catalog_random_get,
    catalog_random_double_m31};

void sample_n_unique_integers(unsigned int n, random_huge seed, unsigned int array[]);

static void catalog_seed_random(gsl_rng **rng, random_huge seed, CatalogRandomState *states) {
    if (!states) {
        seed_rng_threads_fast(rng, seed);
        return;
    }
    unsigned int seeds[1000];
    int threads = simulation_options_global->N_THREADS;
    if (threads < 1 || threads > 1000) Throw(ValueError);
    sample_n_unique_integers(threads, seed, seeds);
    for (int i = 0; i < threads; ++i) {
        CatalogRandomState *state = &states[i];
        state->worker = i;
        state->words = malloc((size_t)CATALOG_RANDOM_WORDS * sizeof(unsigned int));
        if (!state->words) Throw(MemoryAllocError);
        state->rng.type =
            catalog_worker_is_m31(i) ? &catalog_random_type_m31 : &catalog_random_type;
        state->rng.state = state;
        catalog_random_set(state, seeds[i]);
        rng[i] = &state->rng;
    }
}

static void catalog_destroy_random(CatalogRandomState *states) {
    if (!states) return;
    for (int i = 0; i < simulation_options_global->N_THREADS; ++i) {
        if (states[i].recovery) gsl_rng_free(states[i].recovery);
        free(states[i].words);
        states[i].words = NULL;
    }
    free(states);
}
