#include <sched.h>
/* A row's published suffix [first, ny) is immutable for this sampler call. */
typedef struct {
    struct cached_rf_params params;
    double lower;
    int first;
    unsigned char locked;
} CatalogRow;
typedef struct {
    CatalogRandomState *random;
    CatalogRow *rows;
    double *targets;
    bool enabled;
} CatalogDemand;
void (*catalog_inverse_work)(unsigned long long);
int (*catalog_inverse_gpu)(int, int, const double *, const double *, const double *, const int *,
                           double **);
void fix_mass_sample(gsl_rng *, double, int *, double *, float *);
int stoc_partition_sample(struct HaloSamplingConstants *, gsl_rng *, int *, float *);
int stoc_split_sample(struct HaloSamplingConstants *, gsl_rng *, int *, float *);

// Only the writer may extend past the earliest published bin. Already-published
// entries never change, so the common interpolation lookup takes no lock.
static void catalog_require(CatalogDemand *demand, int i, int kmin) {
    if (i < 0 || i >= catalog_inverse_target->nx_bin || kmin < 0 ||
        kmin + 1 >= catalog_inverse_target->ny_bin)
        Throw(TableEvaluationError);
    CatalogRow *row = &demand->rows[i];
    int first = __atomic_load_n(&row->first, __ATOMIC_ACQUIRE);
    if (first < 0) Throw(TableGenerationError);
    if (first <= kmin) return;

    while (__atomic_test_and_set(&row->locked, __ATOMIC_ACQUIRE)) sched_yield();
    first = __atomic_load_n(&row->first, __ATOMIC_RELAXED);
    if (first > kmin) {
        gsl_function function = {.function = cached_dndm_inv_f, .params = &row->params};
        gsl_root_fsolver *solver = gsl_root_fsolver_alloc(gsl_root_fsolver_brent);
        if (!solver) {
            __atomic_clear(&row->locked, __ATOMIC_RELEASE);
            Throw(MemoryAllocError);
        }
        // Capture the invariant native M_cond outside the GSL loop. With the upstream
        // compiler flags a mutable struct dereference changes division rounding.
        const double mass = row->params.M_cond;
        double upper = row->params.lnM_cond, lower = row->lower;
        int status = GSL_SUCCESS;
        for (int k = first - 1; k >= kmin; --k) {
            row->params.rf_target = demand->targets[k];
            gsl_root_fsolver_set(solver, &function, lower, upper);
            int iteration = 0;
            do {
                ++iteration;
                gsl_root_fsolver_iterate(solver);
                double guess = gsl_root_fsolver_root(solver);
                double lo = gsl_root_fsolver_x_lower(solver);
                double hi = gsl_root_fsolver_x_upper(solver);
                status = gsl_root_test_interval(lo, hi, 1e-4, 0.);
                if (status == GSL_SUCCESS) {
                    lower = lo;
                    catalog_inverse_target->z_arr[i][k] = exp(guess) / mass;
                }
            } while (status == GSL_CONTINUE && iteration < MAX_ITER_RF);
            // Upstream reports an unconverged interval and advances using the last valid lower
            // bound.
            if (status != GSL_SUCCESS) {
                LOG_ERROR("gsl RF error occured! %d", status);
                if (status > 0) break;
            }
        }
        gsl_root_fsolver_free(solver);
        if (status > 0) {
            __atomic_clear(&row->locked, __ATOMIC_RELEASE);
            CATCH_GSL_ERROR(status);
        }
        row->lower = lower;
        __atomic_store_n(&row->first, kmin, __ATOMIC_RELEASE);
    }
    __atomic_clear(&row->locked, __ATOMIC_RELEASE);
}

/* Existing table hooks */
/* The axes and per-condition barrier are prepared by the same native v4
 * helpers used by initialize_dndm_tables and initialize_dndm_inverse_table. */
int (*catalog_tables_gpu)(int, const double *, const double *);

static void catalog_table(double xmin, double xmax, double ymin, double ymax, double growth,
                          double param, bool from_catalog) {
    int nx = simulation_options_global->N_COND_INTERP;
    double conditions[nx][4];
    double ln_mass = param;
    double sigma = from_catalog ? 0 : evaluate_sigma(ln_mass);
    double bounds[] = {xmin,
                       xmax,
                       ymin,
                       ymax,
                       growth,
                       physconst.delta_c_sph,
                       RHOcrit,
                       cosmo_params_global->OMm,
                       matter_options_global->HMF,
                       physconst.delta_c_delos};
    for (int i = 0; i < nx; ++i) {
        double x = xmin + (xmax - xmin) * (double)i / (double)(nx - 1);
        double delta = x;
        if (from_catalog) {
            ln_mass = x;
            sigma = evaluate_sigma(x);
            delta = get_delta_crit(matter_options_global->HMF, sigma, param) / param * growth;
        }
        conditions[i][0] = ln_mass;
        conditions[i][1] = sigma;
        conditions[i][2] = delta;
        conditions[i][3] =
            MAX_DELTAC_FRAC * get_delta_crit(matter_options_global->HMF, sigma, growth);
    }
    if (catalog_tables_gpu(nx, bounds, conditions[0])) Throw(TableGenerationError);
}

static bool catalog_uses_gpu_tables(void) {
    return matter_options_global->SOURCE_MODEL == SOURCE_MODEL_CHMF_SAMPLER &&
           uses_interpolation_tables(matter_options_global->USE_INTERPOLATION_TABLES);
}

static void catalog_dndm(double xmin, double xmax, double ymin, double ymax, double growth,
                         double param, bool from_catalog) {
    if (catalog_uses_gpu_tables())
        catalog_table(xmin, xmax, ymin, ymax, growth, param, from_catalog);
    else
        initialize_dndm_tables(xmin, xmax, ymin, ymax, growth, param, from_catalog);
}

static void catalog_inverse(double xmin, double xmax, double ymin, double growth, double param,
                            bool from_catalog, const float *samples, size_huge count,
                            CatalogDemand *demand) {
    if (catalog_uses_gpu_tables())
        catalog_cached_inverse(xmin, xmax, ymin, growth, param, from_catalog, samples, count,
                               demand);
    else
        initialize_dndm_inverse_table(xmin, xmax, ymin, growth, param, from_catalog);
}
