#include "dft.h"

#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cexcept.h"
#include "cosmology.h"
#include "exceptions.h"
#include "indexing.h"
#include "logger.h"

#define P21_FFTW_PLAN_CACHE_CAP 32

typedef struct {
    bool valid;
    bool is_c2r;
    bool use_wisdom;
    int dim;
    int dim_los;
    int n_threads;
    int align64;
    uint64_t lru_tick;
    fftwf_plan plan;
} P21FftwPlanCacheEntry;

static P21FftwPlanCacheEntry g_fftw_plan_cache[P21_FFTW_PLAN_CACHE_CAP];
static uint64_t g_fftw_cache_tick = 0;
static uint64_t g_fftw_cache_hits = 0;
static uint64_t g_fftw_cache_misses = 0;
static pthread_mutex_t g_fftw_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

static bool fftw_plan_cache_enabled(void) {
    const char *env = getenv("P21_FFTW_PLAN_CACHE");
    if (env != NULL && strcmp(env, "0") == 0) {
        return false;
    }
    return true;
}

void p21_dft_clear_plan_cache(void) {
    pthread_mutex_lock(&g_fftw_cache_mutex);
    for (int i = 0; i < P21_FFTW_PLAN_CACHE_CAP; i++) {
        if (g_fftw_plan_cache[i].valid && g_fftw_plan_cache[i].plan != NULL) {
            fftwf_destroy_plan(g_fftw_plan_cache[i].plan);
        }
        memset(&g_fftw_plan_cache[i], 0, sizeof(g_fftw_plan_cache[i]));
    }
    g_fftw_cache_tick = 0;
    g_fftw_cache_hits = 0;
    g_fftw_cache_misses = 0;
    pthread_mutex_unlock(&g_fftw_cache_mutex);
}

#define P21_CPU_SCRATCH_POOL_CAP 24

typedef struct {
    bool valid;
    void *ptr;
    size_t n_bytes;
} P21CpuScratchEntry;

static P21CpuScratchEntry g_cpu_scratch_pool[P21_CPU_SCRATCH_POOL_CAP];
static size_t g_cpu_scratch_pooled_bytes = 0;
static uint64_t g_cpu_scratch_hits = 0;
static uint64_t g_cpu_scratch_misses = 0;
static pthread_mutex_t g_cpu_scratch_mutex = PTHREAD_MUTEX_INITIALIZER;

static bool cpu_scratch_pool_enabled(void) {
    const char *env = getenv("P21_CPU_SCRATCH_POOL");
    if (env != NULL && strcmp(env, "0") == 0) {
        return false;
    }
    return true;
}

static size_t cpu_scratch_max_bytes(void) {
    const char *env = getenv("P21_CPU_SCRATCH_POOL_MB");
    if (env != NULL && *env != '\0') {
        char *endp = NULL;
        unsigned long long mb = strtoull(env, &endp, 10);
        if (endp != env && mb > 0) {
            return (size_t)mb * 1024ULL * 1024ULL;
        }
    }
    return 2048ULL * 1024ULL * 1024ULL;
}

void *p21_dft_scratch_acquire(size_t n_bytes) {
    if (!cpu_scratch_pool_enabled()) {
        return fftwf_malloc(n_bytes);
    }

    pthread_mutex_lock(&g_cpu_scratch_mutex);
    for (int i = 0; i < P21_CPU_SCRATCH_POOL_CAP; i++) {
        if (g_cpu_scratch_pool[i].valid && g_cpu_scratch_pool[i].n_bytes == n_bytes) {
            void *ptr = g_cpu_scratch_pool[i].ptr;
            g_cpu_scratch_pool[i].valid = false;
            g_cpu_scratch_pool[i].ptr = NULL;
            g_cpu_scratch_pool[i].n_bytes = 0;
            g_cpu_scratch_pooled_bytes -= n_bytes;
            g_cpu_scratch_hits++;
            pthread_mutex_unlock(&g_cpu_scratch_mutex);
            return ptr;
        }
    }
    g_cpu_scratch_misses++;
    pthread_mutex_unlock(&g_cpu_scratch_mutex);
    return fftwf_malloc(n_bytes);
}

void p21_dft_scratch_release(void *ptr, size_t n_bytes) {
    if (ptr == NULL) {
        return;
    }
    if (!cpu_scratch_pool_enabled()) {
        fftwf_free(ptr);
        return;
    }

    const size_t max_bytes = cpu_scratch_max_bytes();
    pthread_mutex_lock(&g_cpu_scratch_mutex);
    if (g_cpu_scratch_pooled_bytes + n_bytes <= max_bytes) {
        for (int i = 0; i < P21_CPU_SCRATCH_POOL_CAP; i++) {
            if (!g_cpu_scratch_pool[i].valid) {
                g_cpu_scratch_pool[i].valid = true;
                g_cpu_scratch_pool[i].ptr = ptr;
                g_cpu_scratch_pool[i].n_bytes = n_bytes;
                g_cpu_scratch_pooled_bytes += n_bytes;
                pthread_mutex_unlock(&g_cpu_scratch_mutex);
                return;
            }
        }
    }
    pthread_mutex_unlock(&g_cpu_scratch_mutex);
    fftwf_free(ptr);
}

void p21_dft_clear_scratch_pool(void) {
    pthread_mutex_lock(&g_cpu_scratch_mutex);
    for (int i = 0; i < P21_CPU_SCRATCH_POOL_CAP; i++) {
        if (g_cpu_scratch_pool[i].valid && g_cpu_scratch_pool[i].ptr != NULL) {
            fftwf_free(g_cpu_scratch_pool[i].ptr);
        }
        g_cpu_scratch_pool[i].valid = false;
        g_cpu_scratch_pool[i].ptr = NULL;
        g_cpu_scratch_pool[i].n_bytes = 0;
    }
    g_cpu_scratch_pooled_bytes = 0;
    g_cpu_scratch_hits = 0;
    g_cpu_scratch_misses = 0;
    pthread_mutex_unlock(&g_cpu_scratch_mutex);
}

void p21_dft_get_scratch_stats(unsigned long long *hits_out, unsigned long long *misses_out,
                               int *pooled_out) {
    pthread_mutex_lock(&g_cpu_scratch_mutex);
    if (hits_out != NULL) *hits_out = (unsigned long long)g_cpu_scratch_hits;
    if (misses_out != NULL) *misses_out = (unsigned long long)g_cpu_scratch_misses;
    if (pooled_out != NULL) {
        int count = 0;
        for (int i = 0; i < P21_CPU_SCRATCH_POOL_CAP; i++) {
            if (g_cpu_scratch_pool[i].valid) count++;
        }
        *pooled_out = count;
    }
    pthread_mutex_unlock(&g_cpu_scratch_mutex);
}

void p21_dft_get_cache_stats(unsigned long long *hits_out, unsigned long long *misses_out,
                             int *entries_out) {
    pthread_mutex_lock(&g_fftw_cache_mutex);
    if (hits_out != NULL) *hits_out = (unsigned long long)g_fftw_cache_hits;
    if (misses_out != NULL) *misses_out = (unsigned long long)g_fftw_cache_misses;
    if (entries_out != NULL) {
        int count = 0;
        for (int i = 0; i < P21_FFTW_PLAN_CACHE_CAP; i++) {
            if (g_fftw_plan_cache[i].valid) count++;
        }
        *entries_out = count;
    }
    pthread_mutex_unlock(&g_fftw_cache_mutex);
}

// FFTW stores the number of threads in each plan at planning time, so this must be called
// before every plan is created. fftwf_init_threads() is idempotent, and calling it here also
// guards against an earlier clean_fftw() having reset the threading state.
static void set_fftw_threads(int n_threads) {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(n_threads);
}

static int execute_dft_cube(bool is_c2r, bool use_wisdom, int dim, int dim_los, int n_threads,
                            fftwf_complex *box) {
    // The wisdom is loaded into memory by CreateFFTWWisdoms() when the inputs are broadcast,
    // and kept there until clean_fftw(), so there's no need to import it here.
    unsigned flag = use_wisdom ? FFTW_WISDOM_ONLY : FFTW_ESTIMATE;
    fftwf_plan plan = NULL;

    // FFTW new-array plan reuse is bit-exact only when both creation and execution pointers
    // have identical SIMD alignment modulo 64 bytes (covers SSE2/AVX2/AVX-512).
    const int align64 = (int)((uintptr_t)box & 63ULL);
    bool can_cache = fftw_plan_cache_enabled() && ((align64 & 15) == 0);
    if (!can_cache) {
        set_fftw_threads(n_threads);
        plan = is_c2r
                   ? fftwf_plan_dft_c2r_3d(dim, dim, dim_los, (fftwf_complex *)box, (float *)box,
                                           flag)
                   : fftwf_plan_dft_r2c_3d(dim, dim, dim_los, (float *)box, (fftwf_complex *)box,
                                           flag);
        if (plan == NULL && flag == FFTW_WISDOM_ONLY) {
            LOG_ERROR("No FFTW wisdom for %s DIM=%d DIM_LOS=%d NTHREADS=%d.",
                      is_c2r ? "c2r" : "r2c", dim, dim_los, n_threads);
            Throw(ValueError);
        }
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
        return 0;
    }

    pthread_mutex_lock(&g_fftw_cache_mutex);
    int hit_idx = -1;
    int free_idx = -1;
    int lru_idx = 0;
    uint64_t lru_min = UINT64_MAX;
    for (int i = 0; i < P21_FFTW_PLAN_CACHE_CAP; i++) {
        P21FftwPlanCacheEntry *e = &g_fftw_plan_cache[i];
        if (!e->valid) {
            if (free_idx < 0) free_idx = i;
            continue;
        }
        if (e->is_c2r == is_c2r && e->use_wisdom == use_wisdom && e->dim == dim &&
            e->dim_los == dim_los && e->n_threads == n_threads && e->align64 == align64) {
            hit_idx = i;
            break;
        }
        if (e->lru_tick < lru_min) {
            lru_min = e->lru_tick;
            lru_idx = i;
        }
    }

    if (hit_idx >= 0) {
        g_fftw_cache_hits++;
        g_fftw_plan_cache[hit_idx].lru_tick = ++g_fftw_cache_tick;
        plan = g_fftw_plan_cache[hit_idx].plan;
    } else {
        g_fftw_cache_misses++;
        set_fftw_threads(n_threads);
        plan = is_c2r
                   ? fftwf_plan_dft_c2r_3d(dim, dim, dim_los, (fftwf_complex *)box, (float *)box,
                                           flag)
                   : fftwf_plan_dft_r2c_3d(dim, dim, dim_los, (float *)box, (fftwf_complex *)box,
                                           flag);
        if (plan == NULL && flag == FFTW_WISDOM_ONLY) {
            pthread_mutex_unlock(&g_fftw_cache_mutex);
            LOG_ERROR("No FFTW wisdom for %s DIM=%d DIM_LOS=%d NTHREADS=%d.",
                      is_c2r ? "c2r" : "r2c", dim, dim_los, n_threads);
            Throw(ValueError);
        }
        int slot = (free_idx >= 0) ? free_idx : lru_idx;
        if (g_fftw_plan_cache[slot].valid && g_fftw_plan_cache[slot].plan != NULL) {
            fftwf_destroy_plan(g_fftw_plan_cache[slot].plan);
        }
        g_fftw_plan_cache[slot].valid = (plan != NULL);
        g_fftw_plan_cache[slot].is_c2r = is_c2r;
        g_fftw_plan_cache[slot].use_wisdom = use_wisdom;
        g_fftw_plan_cache[slot].dim = dim;
        g_fftw_plan_cache[slot].dim_los = dim_los;
        g_fftw_plan_cache[slot].n_threads = n_threads;
        g_fftw_plan_cache[slot].align64 = align64;
        g_fftw_plan_cache[slot].lru_tick = ++g_fftw_cache_tick;
        g_fftw_plan_cache[slot].plan = plan;
    }

    if (is_c2r) {
        fftwf_execute_dft_c2r(plan, (fftwf_complex *)box, (float *)box);
    } else {
        fftwf_execute_dft_r2c(plan, (float *)box, (fftwf_complex *)box);
    }
    pthread_mutex_unlock(&g_fftw_cache_mutex);
    return 0;
}

int dft_c2r_cube(bool use_wisdom, int dim, int dim_los, int n_threads, fftwf_complex *box) {
    return execute_dft_cube(true, use_wisdom, dim, dim_los, n_threads, box);
}

int dft_r2c_cube(bool use_wisdom, int dim, int dim_los, int n_threads, fftwf_complex *box) {
    return execute_dft_cube(false, use_wisdom, dim, dim_los, n_threads, box);
}

int CreateFFTWWisdoms() {
    int status;

    Try {  // This Try wraps the entire function so we don't indent.
        fftwf_plan plan;

        char wisdom_filename[500];

        omp_set_num_threads(simulation_options_global->N_THREADS);
        set_fftw_threads(simulation_options_global->N_THREADS);

        // allocate array for the k-space and real-space boxes
        fftwf_complex *HIRES_box =
            (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
        fftwf_complex *LOWRES_box =
            (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

        sprintf(wisdom_filename, "%s/r2c_DIM%d_DIM%d_NTHREADS%d", config_settings.wisdoms_path,
                simulation_options_global->DIM, (int)D_PARA, simulation_options_global->N_THREADS);
        if (fftwf_import_wisdom_from_filename(wisdom_filename) == 0) {
            plan = fftwf_plan_dft_r2c_3d(simulation_options_global->DIM,
                                         simulation_options_global->DIM, D_PARA, (float *)HIRES_box,
                                         (fftwf_complex *)HIRES_box, FFTW_PATIENT);
            fftwf_export_wisdom_to_filename(wisdom_filename);
            fftwf_destroy_plan(plan);
        }

        sprintf(wisdom_filename, "%s/c2r_DIM%d_DIM%d_NTHREADS%d", config_settings.wisdoms_path,
                simulation_options_global->DIM, (int)D_PARA, simulation_options_global->N_THREADS);
        if (fftwf_import_wisdom_from_filename(wisdom_filename) == 0) {
            plan = fftwf_plan_dft_c2r_3d(
                simulation_options_global->DIM, simulation_options_global->DIM, D_PARA,
                (fftwf_complex *)HIRES_box, (float *)HIRES_box, FFTW_PATIENT);
            fftwf_export_wisdom_to_filename(wisdom_filename);
            fftwf_destroy_plan(plan);
        }

        sprintf(wisdom_filename, "%s/r2c_DIM%d_DIM%d_NTHREADS%d", config_settings.wisdoms_path,
                simulation_options_global->HII_DIM, (int)HII_D_PARA,
                simulation_options_global->N_THREADS);
        if (fftwf_import_wisdom_from_filename(wisdom_filename) == 0) {
            plan = fftwf_plan_dft_r2c_3d(
                simulation_options_global->HII_DIM, simulation_options_global->HII_DIM, HII_D_PARA,
                (float *)LOWRES_box, (fftwf_complex *)LOWRES_box, FFTW_PATIENT);
            fftwf_export_wisdom_to_filename(wisdom_filename);
            fftwf_destroy_plan(plan);
        }

        sprintf(wisdom_filename, "%s/c2r_DIM%d_DIM%d_NTHREADS%d", config_settings.wisdoms_path,
                simulation_options_global->HII_DIM, (int)HII_D_PARA,
                simulation_options_global->N_THREADS);
        if (fftwf_import_wisdom_from_filename(wisdom_filename) == 0) {
            plan = fftwf_plan_dft_c2r_3d(
                simulation_options_global->HII_DIM, simulation_options_global->HII_DIM, HII_D_PARA,
                (fftwf_complex *)LOWRES_box, (float *)LOWRES_box, FFTW_PATIENT);
            fftwf_export_wisdom_to_filename(wisdom_filename);
            fftwf_destroy_plan(plan);
        }

        // deallocate
        fftwf_free(HIRES_box);
        fftwf_free(LOWRES_box);
        p21_dft_clear_plan_cache();

    }  // End of Try{}

    Catch(status) { return (status); }
    return (0);
}

// Free FFTW's global state: its threads, and the wisdom it keeps in memory. FFTW records
// wisdom for every plan it creates (including FFTW_ESTIMATE ones), so that creating the same
// plan again is cheap. This is called from Python when the global state of the backend is
// freed, rather than at the end of each Compute* function, so that the wisdom is kept between
// calls (the plans themselves are still created and destroyed in every FFT).
// There must not be any FFTW plans alive when this is called.
void clean_fftw() {
    p21_dft_clear_plan_cache();
    p21_dft_clear_scratch_pool();
    fftwf_cleanup_threads();
    fftwf_cleanup();
}

// Test function: run `n_repeat` forward+backward FFTs of a cubic box of side `dim`
// using `n_threads`, so that the threading behaviour of the FFTs can be tested in isolation.
int test_dft_cube(int dim, int n_threads, int n_repeat) {
    int i, status = 0;
    unsigned long long ii, n_kspace = (unsigned long long)dim * dim * (dim / 2 + 1);
    fftwf_complex *box = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * n_kspace);

    // FFTW's OpenMP backend uses the default team size, so match the Compute* functions
    omp_set_num_threads(n_threads);

    for (ii = 0; ii < n_kspace; ii++) box[ii] = (float)(ii % 7);

    for (i = 0; i < n_repeat; i++) {
        status = dft_r2c_cube(false, dim, dim, n_threads, box);
        if (status != 0) break;
        status = dft_c2r_cube(false, dim, dim, n_threads, box);
        if (status != 0) break;
    }
    fftwf_free(box);
    return status;
}
