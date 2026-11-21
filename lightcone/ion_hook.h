typedef struct {
    int phase, side, threads, filter, filtered, last, no_rng, nl, nh;
    float z, prev_z, radius, lo, hi, ls, li, hs, hi_inv, crit, pixel, mmin, eff, tre, dt, dz, density_limit;
    double dk, cut, npoisson, norm, gamma, fmin, mean, dz_table, max_log;
    float *density, *xe, *tk, *old_rec, *old_z, *xhi, *nrec, *zre, *mfp, *gamma12, *temperature, *fcoll;
    const float *low, *high;
} IonGPU;
typedef int (*IonHook)(IonGPU *);
static IonHook ion_hook = NULL;
void SetIonGPUHook(void *hook) { ion_hook = (IonHook)hook; }

void IonPoissonDraws(int threads, double mean, unsigned long long n, int *out) {
    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
        gsl_rng_set(rng, tid);
        #pragma omp for schedule(static)
        for (unsigned long long i=0; i<n; ++i)
            out[i] = (int)gsl_ran_poisson(rng,mean);
        gsl_rng_free(rng);
    }
}

void IonRecombinationKnots(double *ln_gamma, double *values, double *derivatives) {
    memcpy(ln_gamma,lnGamma_values,sizeof(lnGamma_values));
    memcpy(values,RR_table,sizeof(RR_table));
    for (int i=0; i<RR_Z_NPTS; ++i)
        for (int j=0; j<RR_lnGamma_NPTS; ++j)
            derivatives[i*RR_lnGamma_NPTS+j] =
                gsl_spline_eval_deriv(RR_spline[i],lnGamma_values[j],RR_acc[i]);
}

static fftwf_complex *ion_ref[3], *ion_ref_work, *ion_ref_density;
static float *ion_ref_fcoll;
static int ion_ref_side;
static bool ion_ref_density_ready, ion_ref_xerec_ready, ion_ref_xe_ready, ion_ref_rec_ready, ion_ref_mean_ready;
static unsigned long long ion_ref_inverses[3];

static fftwf_plan ion_plan_r2c = NULL;
static fftwf_plan ion_plan_c2r = NULL;
static fftwf_plan ion_plan_r2c_mt = NULL;
static fftwf_plan ion_plan_c2r_mt = NULL;
static int ion_plan_mt_threads = 0;
static fftwf_complex *ion_thread_buf[16];
static int ion_thread_slots = 0;
static int ion_pruned_side = 0;
static fftwf_plan ion_pruned_pz_plane = NULL;
static fftwf_plan ion_pruned_px[256];
static fftwf_plan ion_pruned_py[256];

static void ion_destroy_pruned_plans(void) {
    if (ion_pruned_pz_plane) { fftwf_destroy_plan(ion_pruned_pz_plane); ion_pruned_pz_plane = NULL; }
    for (int i = 0; i < 256; ++i) {
        if (ion_pruned_px[i]) { fftwf_destroy_plan(ion_pruned_px[i]); ion_pruned_px[i] = NULL; }
        if (ion_pruned_py[i]) { fftwf_destroy_plan(ion_pruned_py[i]); ion_pruned_py[i] = NULL; }
    }
    ion_pruned_side = 0;
}
static float *ion_kmag_table = NULL;
static int ion_kmag_side = 0;
static double ion_kmag_dk = 0.0;

static int ion_sched_n = 0;
static float ion_sched_r[128];
static int ion_sched_f[128];
static float ion_batch_lo[128];
static float ion_batch_hi[128];
static bool ion_batch_ready = false;

void IonReferenceReset(void) {
    ion_ref_density_ready=false;
    ion_ref_xerec_ready=false;
    ion_ref_xe_ready=false;
    ion_ref_rec_ready=false;
    ion_ref_mean_ready=false;
    ion_batch_ready=false;
}

void IonReferenceDestroyPlans(void) {
    if (ion_plan_r2c) { fftwf_destroy_plan(ion_plan_r2c); ion_plan_r2c = NULL; }
    if (ion_plan_c2r) { fftwf_destroy_plan(ion_plan_c2r); ion_plan_c2r = NULL; }
    if (ion_plan_r2c_mt) { fftwf_destroy_plan(ion_plan_r2c_mt); ion_plan_r2c_mt = NULL; }
    if (ion_plan_c2r_mt) { fftwf_destroy_plan(ion_plan_c2r_mt); ion_plan_c2r_mt = NULL; }
    ion_plan_mt_threads = 0;
    for (int f = 0; f < 3; ++f) {
        if (ion_ref[f]) { fftwf_free(ion_ref[f]); ion_ref[f] = NULL; }
    }
    if (ion_ref_work) { fftwf_free(ion_ref_work); ion_ref_work = NULL; }
    if (ion_ref_density) { fftwf_free(ion_ref_density); ion_ref_density = NULL; }
    if (ion_ref_fcoll) { fftwf_free(ion_ref_fcoll); ion_ref_fcoll = NULL; }
    for (int t = 0; t < 16; ++t) {
        if (ion_thread_buf[t]) { fftwf_free(ion_thread_buf[t]); ion_thread_buf[t] = NULL; }
    }
    ion_thread_slots = 0;
    if (ion_kmag_table) { free(ion_kmag_table); ion_kmag_table = NULL; }
    ion_kmag_side = 0;
    ion_kmag_dk = 0.0;
    ion_destroy_pruned_plans();
    ion_ref_side = 0;
    IonReferenceReset();
}

void IonReferenceInverseCounts(unsigned long long *out) {
    memcpy(out,ion_ref_inverses,sizeof(ion_ref_inverses));
}

void IonReferenceSetSchedule(int count, const float *radii, const int *filtered) {
    int n = count < 0 ? 0 : (count > 128 ? 128 : count);
    ion_sched_n = n;
    for (int i = 0; i < n; ++i) {
        ion_sched_r[i] = radii[i];
        ion_sched_f[i] = filtered[i];
    }
    ion_batch_ready = false;
}

int IonReferenceScheduleCount(void) {
    return ion_sched_n;
}

static int IonEnsurePlans(int side, int nthreads, int create_cached_plans) {
    struct UserParams *user_params = user_params_ufunc;
    int n = side * side * side;
    size_t bytes = sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS;
    ps_destroy_ion_fftw_plans_cb = IonReferenceDestroyPlans;
    if (ion_ref_side != side) {
        ion_destroy_pruned_plans();
        if (ion_plan_r2c) { fftwf_destroy_plan(ion_plan_r2c); ion_plan_r2c = NULL; }
        if (ion_plan_c2r) { fftwf_destroy_plan(ion_plan_c2r); ion_plan_c2r = NULL; }
        if (ion_plan_r2c_mt) { fftwf_destroy_plan(ion_plan_r2c_mt); ion_plan_r2c_mt = NULL; }
        if (ion_plan_c2r_mt) { fftwf_destroy_plan(ion_plan_c2r_mt); ion_plan_c2r_mt = NULL; }
        ion_plan_mt_threads = 0;
        for (int f = 0; f < 3; ++f) {
            if (ion_ref[f]) fftwf_free(ion_ref[f]);
            ion_ref[f] = fftwf_malloc(bytes);
        }
        if (ion_ref_work) fftwf_free(ion_ref_work);
        if (ion_ref_density) fftwf_free(ion_ref_density);
        if (ion_ref_fcoll) fftwf_free(ion_ref_fcoll);
        ion_ref_work = fftwf_malloc(bytes);
        ion_ref_density = fftwf_malloc(bytes);
        ion_ref_fcoll = fftwf_malloc((size_t)n * sizeof(*ion_ref_fcoll));
        for (int t = 0; t < 16; ++t) {
            if (ion_thread_buf[t]) { fftwf_free(ion_thread_buf[t]); ion_thread_buf[t] = NULL; }
        }
        ion_thread_slots = 0;
        ion_ref_side = side;
        IonReferenceReset();
    }
    if (!ion_ref_work || !ion_ref_density || !ion_ref_fcoll ||
        !ion_ref[0] || !ion_ref[1] || !ion_ref[2]) return 1;
    if (create_cached_plans) {
        const unsigned pflags = ps_fftw_plan_flags();
        if (!ion_plan_r2c || !ion_plan_c2r) {
            fftwf_init_threads();
            fftwf_plan_with_nthreads(1);
        }
        if (!ion_plan_r2c) {
            ion_plan_r2c = fftwf_plan_dft_r2c_3d(side, side, side, (float *)ion_ref_work, ion_ref_work, pflags);
            if (!ion_plan_r2c) return 1;
        }
        if (!ion_plan_c2r) {
            ion_plan_c2r = fftwf_plan_dft_c2r_3d(side, side, side, ion_ref_work, (float *)ion_ref_work, pflags);
            if (!ion_plan_c2r) return 1;
        }
        const int mt = nthreads < 1 ? 1 : (nthreads > 8 ? 8 : nthreads);
        if (!ion_plan_r2c_mt || !ion_plan_c2r_mt || ion_plan_mt_threads != mt) {
            if (ion_plan_r2c_mt) { fftwf_destroy_plan(ion_plan_r2c_mt); ion_plan_r2c_mt = NULL; }
            if (ion_plan_c2r_mt) { fftwf_destroy_plan(ion_plan_c2r_mt); ion_plan_c2r_mt = NULL; }
            fftwf_init_threads();
            fftwf_plan_with_nthreads(mt);
            ion_plan_r2c_mt = fftwf_plan_dft_r2c_3d(side, side, side, (float *)ion_ref_work, ion_ref_work, pflags);
            ion_plan_c2r_mt = fftwf_plan_dft_c2r_3d(side, side, side, ion_ref_work, (float *)ion_ref_work, pflags);
            fftwf_plan_with_nthreads(1);
            if (!ion_plan_r2c_mt || !ion_plan_c2r_mt) {
                if (ion_plan_r2c_mt) { fftwf_destroy_plan(ion_plan_r2c_mt); ion_plan_r2c_mt = NULL; }
                if (ion_plan_c2r_mt) { fftwf_destroy_plan(ion_plan_c2r_mt); ion_plan_c2r_mt = NULL; }
                ion_plan_mt_threads = 0;
                return 1;
            }
            ion_plan_mt_threads = mt;
        }
        int need_slots = nthreads < 2 ? 2 : (nthreads > 16 ? 16 : nthreads);
        while (ion_thread_slots < need_slots) {
            ion_thread_buf[ion_thread_slots] = fftwf_malloc(bytes);
            if (!ion_thread_buf[ion_thread_slots]) return 1;
            ++ion_thread_slots;
        }
    }
    return 0;
}

static int IonEnsureKmag(int side, double dk, int nthreads) {
    if (ion_kmag_side == side && ion_kmag_dk == dk && ion_kmag_table) return 0;
    int mid = side / 2;
    int kz_len = mid + 1;
    size_t total = (size_t)side * side * kz_len;
    if (ion_kmag_table) { free(ion_kmag_table); ion_kmag_table = NULL; }
    ion_kmag_table = (float *)malloc(total * sizeof(float));
    if (!ion_kmag_table) return 1;
    int nt = nthreads < 1 ? 1 : (nthreads > 16 ? 16 : nthreads);
    #pragma omp parallel for collapse(2) num_threads(nt)
    for (int nx = 0; nx < side; ++nx) {
        for (int ny = 0; ny < side; ++ny) {
            float kx = (nx > mid) ? (nx - side) * dk : nx * dk;
            float ky = (ny > mid) ? (ny - side) * dk : ny * dk;
            float *out = ion_kmag_table + ((size_t)nx * side + ny) * kz_len;
            for (int nz = 0; nz <= mid; ++nz) {
                float kz = nz * dk;
                out[nz] = (float)sqrt(kx * kx + ky * ky + kz * kz);
            }
        }
    }
    ion_kmag_side = side;
    ion_kmag_dk = dk;
    return 0;
}

static void IonFilterCopy(const fftwf_complex *src, fftwf_complex *dst, int side, float radius, int filtered) {
    const size_t kn = (size_t)side * side * (size_t)(side / 2 + 1);
    if (!filtered) {
        memcpy(dst, src, sizeof(fftwf_complex) * kn);
    } else if (global_params.HII_FILTER == 1) {
        const float *km = ion_kmag_table;
        for (size_t idx = 0; idx < kn; ++idx) {
            float kR = km[idx] * radius;
            kR *= 0.413566994;
            if (kR > 1) dst[idx] = 0;
            else dst[idx] = src[idx];
        }
    } else {
        memcpy(dst, src, sizeof(fftwf_complex) * kn);
        filter_box(dst, 1, global_params.HII_FILTER, radius);
    }
}

static void IonFilterCopyMT(const fftwf_complex *src, fftwf_complex *dst, int side, float radius, int filtered, int nt) {
    const size_t kn = (size_t)side * side * (size_t)(side / 2 + 1);
    if (!filtered) {
        PSParallelMemcpy(dst, src, sizeof(fftwf_complex) * kn, nt);
    } else if (global_params.HII_FILTER == 1) {
        const float *km = ion_kmag_table;
        const float *s = (const float *)src;
        float *d = (float *)dst;
        const double step = 0.413566994 * (double)radius * ion_kmag_dk;
        const int mid = side / 2;
        const int kz_len = mid + 1;
        const int nmax = (step > 0.0) ? ((int)floor(1.001 / step) + 1) : side;
        if (nmax < mid - 1) {
            #pragma omp parallel num_threads(nt)
            {
                int tid = omp_get_thread_num();
                int nthr = omp_get_num_threads();
                size_t nbytes = sizeof(fftwf_complex) * kn;
                size_t b0 = (nbytes * (size_t)tid) / (size_t)nthr;
                size_t b1 = (nbytes * (size_t)(tid + 1)) / (size_t)nthr;
                if (b1 > b0) memset((char *)d + b0, 0, b1 - b0);
            }
            const int nz_lim = nmax < mid ? nmax : mid;
            #pragma omp parallel for collapse(2) num_threads(nt)
            for (int ix_off = -nmax; ix_off <= nmax; ++ix_off) {
                for (int iy_off = -nmax; iy_off <= nmax; ++iy_off) {
                    const int nx = (ix_off < 0) ? (ix_off + side) : ix_off;
                    const int ny = (iy_off < 0) ? (iy_off + side) : iy_off;
                    const size_t row = ((size_t)nx * (size_t)side + (size_t)ny) * (size_t)kz_len;
                    for (int nz = 0; nz <= nz_lim; ++nz) {
                        const size_t idx = row + (size_t)nz;
                        float kR = km[idx] * radius;
                        kR *= 0.413566994;
                        if (!(kR > 1)) {
                            d[2 * idx] = s[2 * idx];
                            d[2 * idx + 1] = s[2 * idx + 1];
                        }
                    }
                }
            }
        } else {
            #pragma omp parallel for schedule(static) num_threads(nt)
            for (size_t idx = 0; idx < kn; ++idx) {
                float kR = km[idx] * radius;
                kR *= 0.413566994;
                if (kR > 1) {
                    d[2 * idx] = 0.f;
                    d[2 * idx + 1] = 0.f;
                } else {
                    d[2 * idx] = s[2 * idx];
                    d[2 * idx + 1] = s[2 * idx + 1];
                }
            }
        }
    } else {
        IonFilterCopy(src, dst, side, radius, filtered);
    }
}

int TestIonFilterCopyMTCompare(int side, double dk, float radius, int filtered, int nt, const fftwf_complex *src, fftwf_complex *dst_sparse, fftwf_complex *dst_full) {
    struct UserParams *user_params = user_params_ufunc;
    (void)user_params;
    if (IonEnsureKmag(side, dk, nt)) return 1;
    IonFilterCopyMT(src, dst_sparse, side, radius, filtered, nt);
    const size_t kn = (size_t)side * side * (size_t)(side / 2 + 1);
    if (!filtered) {
        memcpy(dst_full, src, sizeof(fftwf_complex) * kn);
    } else {
        const float *km = ion_kmag_table;
        for (size_t idx = 0; idx < kn; ++idx) {
            float kR = km[idx] * radius;
            kR *= 0.413566994;
            if (kR > 1) dst_full[idx] = 0;
            else dst_full[idx] = src[idx];
        }
    }
    return 0;
}

static int ion_ensure_pruned_plans(int side, int nz_cnt) {
    if (nz_cnt < 2 || nz_cnt >= 256 || !ion_ref_work) return 1;
    if (ion_pruned_side != side) {
        ion_destroy_pruned_plans();
        ion_pruned_side = side;
    }
    const int H = side / 2 + 1;
    const int P = 2 * H;
    int n1[1] = {side};
    fftwf_init_threads();
    fftwf_plan_with_nthreads(1);
    if (!ion_pruned_pz_plane) {
        int inembed_z[1] = {H};
        int onembed_z[1] = {P};
        ion_pruned_pz_plane = fftwf_plan_many_dft_c2r(
            1, n1, side, ion_ref_work, inembed_z, 1, H,
            (float *)ion_ref_work, onembed_z, 1, P, FFTW_ESTIMATE);
        if (!ion_pruned_pz_plane) return 1;
    }
    if (!ion_pruned_px[nz_cnt]) {
        ion_pruned_px[nz_cnt] = fftwf_plan_many_dft(
            1, n1, nz_cnt, ion_ref_work, NULL, side * H, 1,
            ion_ref_work, NULL, side * H, 1, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!ion_pruned_px[nz_cnt]) return 1;
    }
    if (!ion_pruned_py[nz_cnt]) {
        ion_pruned_py[nz_cnt] = fftwf_plan_many_dft(
            1, n1, nz_cnt, ion_ref_work, NULL, H, 1,
            ion_ref_work, NULL, H, 1, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!ion_pruned_py[nz_cnt]) return 1;
    }
    return 0;
}

static int ion_can_use_pruned(int side, float radius, int filtered, int *out_nmax, int *out_nz_cnt) {
    if (!filtered || global_params.HII_FILTER != 1) return 0;
    if (user_params_ufunc && user_params_ufunc->USE_FFTW_WISDOM) return 0;
    if (!ion_kmag_table || ion_kmag_dk <= 0.0) return 0;
    const double step = 0.413566994 * (double)radius * ion_kmag_dk;
    if (step <= 0.0) return 0;
    const int nmax = (int)floor(1.001 / step) + 1;
    if (nmax < 0 || nmax > side / 4) return 0;
    const int H = side / 2 + 1;
    int nz_cnt = (nmax + 1 < 2) ? 2 : (nmax + 1);
    if (nz_cnt > H) nz_cnt = H;
    if (nz_cnt >= 256) return 0;
    *out_nmax = nmax;
    *out_nz_cnt = nz_cnt;
    return 1;
}

static void ion_pruned_fill_and_xpass(
    const fftwf_complex *src, fftwf_complex *work,
    int side, float radius, int nmax, int nz_cnt, int nt, fftwf_plan px
) {
    const int mid = side / 2;
    const int H = mid + 1;
    const size_t kn = (size_t)side * (size_t)side * (size_t)H;
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        int nthr = omp_get_num_threads();
        size_t nbytes = sizeof(fftwf_complex) * kn;
        size_t b0 = (nbytes * (size_t)tid) / (size_t)nthr;
        size_t b1 = (nbytes * (size_t)(tid + 1)) / (size_t)nthr;
        if (b1 > b0) memset((char *)work + b0, 0, b1 - b0);
    }
    const float *km = ion_kmag_table;
    const float *s = (const float *)src;
    float *d = (float *)work;
    const int nz_lim = (nmax < mid) ? nmax : mid;
    #pragma omp parallel for collapse(2) num_threads(nt)
    for (int ix_off = -nmax; ix_off <= nmax; ++ix_off) {
        for (int iy_off = -nmax; iy_off <= nmax; ++iy_off) {
            const int nx = (ix_off < 0) ? (ix_off + side) : ix_off;
            const int ny = (iy_off < 0) ? (iy_off + side) : iy_off;
            const size_t row = ((size_t)nx * (size_t)side + (size_t)ny) * (size_t)H;
            for (int nz = 0; nz <= nz_lim; ++nz) {
                const size_t idx = row + (size_t)nz;
                float kR = km[idx] * radius;
                kR *= 0.413566994;
                if (!(kR > 1)) {
                    d[2 * idx] = s[2 * idx];
                    d[2 * idx + 1] = s[2 * idx + 1];
                }
            }
        }
    }
    #pragma omp parallel for num_threads(nt)
    for (int iy_off = -nmax; iy_off <= nmax; ++iy_off) {
        const int ny = (iy_off < 0) ? (iy_off + side) : iy_off;
        fftwf_complex *ptr = work + (size_t)ny * (size_t)H;
        fftwf_execute_dft(px, ptr, ptr);
    }
}

int TestIonPrunedC2RCompare(int side, double dk, float radius, int nt, const void *src_ptr, float *out_pruned, float *out_full) {
    const fftwf_complex *src = (const fftwf_complex *)src_ptr;
    const int nthr = ps_clamp_threads(nt);
    const int H = side / 2 + 1;
    const int P = 2 * H;
    const size_t kn = (size_t)side * (size_t)side * (size_t)H;
    if (IonEnsureKmag(side, dk, nthr)) return 1;
    fftwf_complex *full_buf = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * kn);
    fftwf_complex *pr_buf = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * kn);
    if (!full_buf || !pr_buf) {
        if (full_buf) fftwf_free(full_buf);
        if (pr_buf) fftwf_free(pr_buf);
        return 1;
    }
    IonFilterCopyMT(src, full_buf, side, radius, 1, nthr);
    fftwf_init_threads();
    fftwf_plan_with_nthreads(nthr);
    fftwf_plan p3 = fftwf_plan_dft_c2r_3d(side, side, side, full_buf, (float *)full_buf, FFTW_ESTIMATE);
    if (!p3) { fftwf_free(full_buf); fftwf_free(pr_buf); return 1; }
    fftwf_execute_dft_c2r(p3, full_buf, (float *)full_buf);
    const float *fbuf = (const float *)full_buf;
    #pragma omp parallel for collapse(2) num_threads(nthr)
    for (int x = 0; x < side; ++x) {
        for (int y = 0; y < side; ++y) {
            memcpy(&out_full[((size_t)x * side + y) * side],
                   &fbuf[((size_t)x * side + y) * P],
                   (size_t)side * sizeof(float));
        }
    }

    int nmax = 0, nz_cnt = 0;
    if (ion_can_use_pruned(side, radius, 1, &nmax, &nz_cnt)) {
        fftwf_plan_with_nthreads(1);
        int n1[1] = {side}, iz[1] = {H}, oz[1] = {P};
        fftwf_plan pz = fftwf_plan_many_dft_c2r(1, n1, side, pr_buf, iz, 1, H, (float *)pr_buf, oz, 1, P, FFTW_ESTIMATE);
        fftwf_plan px = fftwf_plan_many_dft(1, n1, nz_cnt, pr_buf, NULL, side * H, 1, pr_buf, NULL, side * H, 1, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftwf_plan py = fftwf_plan_many_dft(1, n1, nz_cnt, pr_buf, NULL, H, 1, pr_buf, NULL, H, 1, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!pz || !px || !py) {
            if (pz) fftwf_destroy_plan(pz);
            if (px) fftwf_destroy_plan(px);
            if (py) fftwf_destroy_plan(py);
            fftwf_destroy_plan(p3);
            fftwf_free(full_buf); fftwf_free(pr_buf);
            return 1;
        }
        ion_pruned_fill_and_xpass(src, pr_buf, side, radius, nmax, nz_cnt, nthr, px);
        #pragma omp parallel for num_threads(nthr)
        for (int x = 0; x < side; ++x) {
            fftwf_complex *plane = pr_buf + (size_t)x * (size_t)side * (size_t)H;
            fftwf_execute_dft(py, plane, plane);
            fftwf_execute_dft_c2r(pz, plane, (float *)plane);
            const float *fplane = (const float *)plane;
            for (int y = 0; y < side; ++y) {
                memcpy(&out_pruned[((size_t)x * side + y) * side],
                       &fplane[(size_t)y * P],
                       (size_t)side * sizeof(float));
            }
        }
        fftwf_destroy_plan(pz);
        fftwf_destroy_plan(px);
        fftwf_destroy_plan(py);
    } else {
        IonFilterCopyMT(src, pr_buf, side, radius, 1, nthr);
        fftwf_execute_dft_c2r(p3, pr_buf, (float *)pr_buf);
        const float *pbuf = (const float *)pr_buf;
        #pragma omp parallel for collapse(2) num_threads(nthr)
        for (int x = 0; x < side; ++x) {
            for (int y = 0; y < side; ++y) {
                memcpy(&out_pruned[((size_t)x * side + y) * side],
                       &pbuf[((size_t)x * side + y) * P],
                       (size_t)side * sizeof(float));
            }
        }
    }
    fftwf_destroy_plan(p3);
    fftwf_free(full_buf);
    fftwf_free(pr_buf);
    return 0;
}

static int IonReferencePrepareFields(IonGPU *a, int f_start, int f_end) {
    struct UserParams *user_params = user_params_ufunc;
    int side = a->side, n = side * side * side;
    int use_cached = (ps_keep_fftw_plans && !user_params->USE_FFTW_WISDOM) ? 1 : 0;
    if (IonEnsurePlans(side, a->threads, use_cached)) return 1;
    if (use_cached && IonEnsureKmag(side, a->dk, a->threads)) return 1;
    const float *inputs[] = {a->density, a->xe, a->old_rec};
    if (use_cached) {
        const int r_fft_stride = 2 * ((int)HII_MID_PARA + 1);
        const size_t row_bytes = (size_t)side * sizeof(float);
        const unsigned long long kpx = HII_KSPACE_NUM_PIXELS;
        const double inv_n = (double)n;
        const int mt = a->threads < 1 ? 1 : (a->threads > 8 ? 8 : a->threads);
        fftwf_plan r2c = ion_plan_r2c_mt ? ion_plan_r2c_mt : ion_plan_r2c;
        for (int f = f_start; f < f_end; ++f) {
            float *data = (float *)ion_ref[f];
            const float *src = inputs[f];
            #pragma omp parallel for collapse(2) num_threads(mt)
            for (int x = 0; x < side; ++x)
                for (int y = 0; y < side; ++y)
                    memcpy(&data[((size_t)x * side + y) * r_fft_stride],
                           &src[((size_t)x * side + y) * side],
                           row_bytes);
            fftwf_execute_dft_r2c(r2c, data, ion_ref[f]);
            float *fdata = (float *)ion_ref[f];
            const unsigned long long fpx = 2ULL * kpx;
            #pragma omp parallel for num_threads(mt)
            for (unsigned long long i = 0; i < fpx; ++i)
                fdata[i] = (float)((double)fdata[i] / inv_n);
        }
    } else {
        for (int f = f_start; f < f_end; ++f) {
            float *data = (float *)ion_ref[f];
            #pragma omp parallel for num_threads(a->threads)
            for (int x = 0; x < side; ++x)
                for (int y = 0; y < side; ++y)
                    for (int z = 0; z < side; ++z)
                        data[HII_R_FFT_INDEX(x,y,z)] = inputs[f][HII_R_INDEX(x,y,z)];
            if (dft_r2c_cube(user_params->USE_FFTW_WISDOM, side, side, a->threads, ion_ref[f])) return 1;
            #pragma omp parallel for num_threads(a->threads)
            for (long i = 0; i < (long)HII_KSPACE_NUM_PIXELS; ++i)
                ion_ref[f][i] /= (double)n;
        }
    }
    return 0;
}

static int IonReferencePrepareDensity(IonGPU *a) {
    if (ion_ref_density_ready) return 0;
    if (IonReferencePrepareFields(a, 0, 1)) return 1;
    ion_ref_density_ready = true;
    return 0;
}

static int IonReferencePrepareXe(IonGPU *a) {
    if (ion_ref_xe_ready) return 0;
    if (IonReferencePrepareFields(a, 1, 2)) return 1;
    ion_ref_xe_ready = true;
    return 0;
}

static int IonReferencePrepareXeRec(IonGPU *a) {
    if (ion_ref_xerec_ready) return 0;
    if (!ion_ref_xe_ready) {
        if (IonReferencePrepareFields(a, 1, 2)) return 1;
        ion_ref_xe_ready = true;
    }
    if (!ion_ref_rec_ready) {
        if (IonReferencePrepareFields(a, 2, 3)) return 1;
        ion_ref_rec_ready = true;
    }
    ion_ref_xerec_ready = true;
    return 0;
}

static int IonReferencePrepare(IonGPU *a) {
    if (IonReferencePrepareDensity(a)) return 1;
    if (IonReferencePrepareXeRec(a)) return 1;
    return 0;
}

static int IonReferenceInverse(IonGPU *a, int field, fftwf_complex *target) {
    struct UserParams *user_params = user_params_ufunc;
    int use_cached = (ps_keep_fftw_plans && !user_params->USE_FFTW_WISDOM) ? 1 : 0;
    if (use_cached) {
        if (IonEnsurePlans(a->side, a->threads, 1) || IonEnsureKmag(a->side, a->dk, a->threads)) return 1;
        IonFilterCopy(ion_ref[field], target, a->side, a->radius, a->filtered);
        fftwf_execute_dft_c2r(ion_plan_c2r, target, (float *)target);
    } else {
        memcpy(target, ion_ref[field], sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
        if (a->filtered) filter_box(target, 1, global_params.HII_FILTER, a->radius);
        if (dft_c2r_cube(user_params->USE_FFTW_WISDOM, a->side, a->side, a->threads, target)) return 1;
    }
    ++ion_ref_inverses[field];
    return 0;
}

int IonReferenceBatchDensity(IonGPU *a, float *out_all, int max_scales) {
    struct UserParams *user_params = user_params_ufunc;
    int side = a->side;
    size_t n = (size_t)side * side * side;
    if (ion_sched_n <= 0 || ion_sched_n > max_scales) return 1;
    if (IonEnsurePlans(side, a->threads, 1) || IonEnsureKmag(side, a->dk, a->threads)) return 1;
    if (IonReferencePrepareDensity(a)) return 1;
    const int r_fft_stride = 2 * ((int)HII_MID_PARA + 1);
    const float dmin = (float)(-1. + FRACT_FLOAT_ERR);
    if (!user_params->USE_FFTW_WISDOM && ion_plan_c2r) {
        int nt = a->threads < 1 ? 1 : (a->threads > ion_thread_slots ? ion_thread_slots : a->threads);
        #pragma omp parallel num_threads(nt)
        {
            int tid = omp_get_thread_num();
            fftwf_complex *buf = ion_thread_buf[tid];
            #pragma omp for schedule(dynamic, 1)
            for (int s = 0; s < ion_sched_n; ++s) {
                IonFilterCopy(ion_ref[0], buf, side, ion_sched_r[s], ion_sched_f[s]);
                fftwf_execute_dft_c2r(ion_plan_c2r, buf, (float *)buf);
                const float *fbuf = (const float *)buf;
                float *dst = out_all + (size_t)s * n;
                float lo = 0.f, hi = 0.f;
                for (int x = 0; x < side; ++x) {
                    for (int y = 0; y < side; ++y) {
                        const float *row = &fbuf[((size_t)x * side + y) * r_fft_stride];
                        float *drow = &dst[((size_t)x * side + y) * side];
                        for (int z = 0; z < side; ++z) {
                            float v = fmaxf(row[z], dmin);
                            drow[z] = v;
                            lo = fminf(lo, v);
                            hi = fmaxf(hi, v);
                        }
                    }
                }
                ion_batch_lo[s] = lo;
                ion_batch_hi[s] = hi;
            }
        }
    } else {
        for (int s = 0; s < ion_sched_n; ++s) {
            IonFilterCopy(ion_ref[0], ion_ref_density, side, ion_sched_r[s], ion_sched_f[s]);
            if (dft_c2r_cube(user_params->USE_FFTW_WISDOM, side, side, a->threads, ion_ref_density)) return 1;
            const float *fbuf = (const float *)ion_ref_density;
            float *dst = out_all + (size_t)s * n;
            float lo = 0.f, hi = 0.f;
            #pragma omp parallel for collapse(2) num_threads(a->threads) reduction(min:lo) reduction(max:hi)
            for (int x = 0; x < side; ++x) {
                for (int y = 0; y < side; ++y) {
                    const float *row = &fbuf[((size_t)x * side + y) * r_fft_stride];
                    float *drow = &dst[((size_t)x * side + y) * side];
                    for (int z = 0; z < side; ++z) {
                        float v = fmaxf(row[z], dmin);
                        drow[z] = v;
                        lo = fminf(lo, v);
                        hi = fmaxf(hi, v);
                    }
                }
            }
            ion_batch_lo[s] = lo;
            ion_batch_hi[s] = hi;
        }
    }
    ion_ref_inverses[0] += (unsigned long long)ion_sched_n;
    ion_batch_ready = true;
    return 0;
}

int IonReferenceBatchScaleInfo(int s, float radius, int filtered, float *lo_out, float *hi_out) {
    if (!ion_batch_ready || s < 0 || s >= ion_sched_n) return 1;
    if (ion_sched_r[s] != radius || ion_sched_f[s] != filtered) return 1;
    *lo_out = ion_batch_lo[s];
    *hi_out = ion_batch_hi[s];
    return 0;
}

int IonReferenceSingleDensityMT(IonGPU *a, float *out_delta) {
    struct UserParams *user_params = user_params_ufunc;
    int side = a->side;
    int use_cached = (ps_keep_fftw_plans && !user_params->USE_FFTW_WISDOM) ? 1 : 0;
    if (IonEnsurePlans(side, a->threads, use_cached)) return 1;
    if (use_cached && IonEnsureKmag(side, a->dk, a->threads)) return 1;
    if (IonReferencePrepareDensity(a)) return 1;
    const int r_fft_stride = 2 * ((int)HII_MID_PARA + 1);
    const float dmin = (float)(-1. + FRACT_FLOAT_ERR);
    float lo = 0.f, hi = 0.f;
    if (use_cached && ion_plan_c2r_mt) {
        const int mt = ps_clamp_threads(a->threads);
        const int H = side / 2 + 1;
        int nmax = 0, nz_cnt = 0;
        if (ion_can_use_pruned(side, a->radius, a->filtered, &nmax, &nz_cnt) &&
            ion_ensure_pruned_plans(side, nz_cnt) == 0) {
            ion_pruned_fill_and_xpass(ion_ref[0], ion_ref_work, side, a->radius, nmax, nz_cnt, mt, ion_pruned_px[nz_cnt]);
            const fftwf_plan py = ion_pruned_py[nz_cnt];
            const fftwf_plan pz = ion_pruned_pz_plane;
            #pragma omp parallel for num_threads(mt) reduction(min:lo) reduction(max:hi)
            for (int x = 0; x < side; ++x) {
                fftwf_complex *plane = ion_ref_work + (size_t)x * (size_t)side * (size_t)H;
                fftwf_execute_dft(py, plane, plane);
                fftwf_execute_dft_c2r(pz, plane, (float *)plane);
                const float *fplane = (const float *)plane;
                for (int y = 0; y < side; ++y) {
                    const float *row = fplane + (size_t)y * (size_t)r_fft_stride;
                    float *dst = &out_delta[((size_t)x * side + y) * side];
                    for (int z = 0; z < side; ++z) {
                        float v = fmaxf(row[z], dmin);
                        dst[z] = v;
                        lo = fminf(lo, v);
                        hi = fmaxf(hi, v);
                    }
                }
            }
        } else {
            IonFilterCopyMT(ion_ref[0], ion_ref_work, side, a->radius, a->filtered, mt);
            fftwf_execute_dft_c2r(ion_plan_c2r_mt, ion_ref_work, (float *)ion_ref_work);
            const float *fbuf = (const float *)ion_ref_work;
            #pragma omp parallel for collapse(2) num_threads(mt) reduction(min:lo) reduction(max:hi)
            for (int x = 0; x < side; ++x) {
                for (int y = 0; y < side; ++y) {
                    const float *row = &fbuf[((size_t)x * side + y) * r_fft_stride];
                    float *dst = &out_delta[((size_t)x * side + y) * side];
                    for (int z = 0; z < side; ++z) {
                        float v = fmaxf(row[z], dmin);
                        dst[z] = v;
                        lo = fminf(lo, v);
                        hi = fmaxf(hi, v);
                    }
                }
            }
        }
        ++ion_ref_inverses[0];
    } else {
        if (IonReferenceInverse(a, 0, ion_ref_work)) return 1;
        const float *fbuf = (const float *)ion_ref_work;
        #pragma omp parallel for collapse(2) num_threads(a->threads) reduction(min:lo) reduction(max:hi)
        for (int x = 0; x < side; ++x) {
            for (int y = 0; y < side; ++y) {
                const float *row = &fbuf[((size_t)x * side + y) * r_fft_stride];
                float *dst = &out_delta[((size_t)x * side + y) * side];
                for (int z = 0; z < side; ++z) {
                    float v = fmaxf(row[z], dmin);
                    dst[z] = v;
                    lo = fminf(lo, v);
                    hi = fmaxf(hi, v);
                }
            }
        }
    }
    a->lo = lo;
    a->hi = hi;
    return 0;
}

int IonReferenceXeRecBoxOpt(IonGPU *a, float *out_xe, float *out_rec, int skip_rec) {
    struct UserParams *user_params = user_params_ufunc;
    int side = a->side;
    if (skip_rec) {
        if (IonReferencePrepareXe(a)) return 1;
    } else {
        if (IonReferencePrepareXeRec(a)) return 1;
    }
    const int r_fft_stride = 2 * ((int)HII_MID_PARA + 1);
    const int f_end = skip_rec ? 2 : 3;
    if (ps_keep_fftw_plans && !user_params->USE_FFTW_WISDOM && ion_plan_c2r_mt) {
        const int mt = ps_clamp_threads(a->threads);
        const int H = side / 2 + 1;
        int nmax = 0, nz_cnt = 0;
        const int can_pruned = (ion_can_use_pruned(side, a->radius, a->filtered, &nmax, &nz_cnt) &&
                                ion_ensure_pruned_plans(side, nz_cnt) == 0) ? 1 : 0;
        for (int f = 1; f < f_end; ++f) {
            float *out = (f == 1) ? out_xe : out_rec;
            if (can_pruned) {
                ion_pruned_fill_and_xpass(ion_ref[f], ion_ref_work, side, a->radius, nmax, nz_cnt, mt, ion_pruned_px[nz_cnt]);
                const fftwf_plan py = ion_pruned_py[nz_cnt];
                const fftwf_plan pz = ion_pruned_pz_plane;
                #pragma omp parallel for num_threads(mt)
                for (int x = 0; x < side; ++x) {
                    fftwf_complex *plane = ion_ref_work + (size_t)x * (size_t)side * (size_t)H;
                    fftwf_execute_dft(py, plane, plane);
                    fftwf_execute_dft_c2r(pz, plane, (float *)plane);
                    const float *fplane = (const float *)plane;
                    for (int y = 0; y < side; ++y) {
                        const float *row = fplane + (size_t)y * (size_t)r_fft_stride;
                        float *dst = &out[((size_t)x * side + y) * side];
                        if (f == 1) {
                            for (int z = 0; z < side; ++z)
                                dst[z] = fminf(fmaxf(row[z], 0.f), 0.999f);
                        } else {
                            for (int z = 0; z < side; ++z)
                                dst[z] = fmaxf(row[z], 0.f);
                        }
                    }
                }
            } else {
                IonFilterCopyMT(ion_ref[f], ion_ref_work, side, a->radius, a->filtered, mt);
                fftwf_execute_dft_c2r(ion_plan_c2r_mt, ion_ref_work, (float *)ion_ref_work);
                const float *fbuf = (const float *)ion_ref_work;
                #pragma omp parallel for collapse(2) num_threads(mt)
                for (int x = 0; x < side; ++x) {
                    for (int y = 0; y < side; ++y) {
                        const float *row = &fbuf[((size_t)x * side + y) * r_fft_stride];
                        float *dst = &out[((size_t)x * side + y) * side];
                        if (f == 1) {
                            for (int z = 0; z < side; ++z)
                                dst[z] = fminf(fmaxf(row[z], 0.f), 0.999f);
                        } else {
                            for (int z = 0; z < side; ++z)
                                dst[z] = fmaxf(row[z], 0.f);
                        }
                    }
                }
            }
            ++ion_ref_inverses[f];
        }
    } else {
        for (int f = 1; f < f_end; ++f) {
            if (IonReferenceInverse(a, f, ion_ref_work)) return 1;
            const float *fbuf = (const float *)ion_ref_work;
            float *out = (f == 1) ? out_xe : out_rec;
            #pragma omp parallel for collapse(2) num_threads(a->threads)
            for (int x = 0; x < side; ++x) {
                for (int y = 0; y < side; ++y) {
                    const float *row = &fbuf[((size_t)x * side + y) * r_fft_stride];
                    float *dst = &out[((size_t)x * side + y) * side];
                    if (f == 1) {
                        for (int z = 0; z < side; ++z)
                            dst[z] = fminf(fmaxf(row[z], 0.f), 0.999f);
                    } else {
                        for (int z = 0; z < side; ++z)
                            dst[z] = fmaxf(row[z], 0.f);
                    }
                }
            }
        }
    }
    return 0;
}

int IonReferenceXeRecBox(IonGPU *a, float *out_xe, float *out_rec) {
    struct UserParams *user_params = user_params_ufunc;
    int side = a->side;
    if (IonReferencePrepareXeRec(a)) return 1;
    const int r_fft_stride = 2 * ((int)HII_MID_PARA + 1);
    if (!user_params->USE_FFTW_WISDOM && ion_plan_c2r_mt) {
        const int mt = a->threads < 1 ? 1 : (a->threads > 8 ? 8 : a->threads);
        for (int f = 1; f < 3; ++f) {
            IonFilterCopyMT(ion_ref[f], ion_ref_work, side, a->radius, a->filtered, mt);
            fftwf_execute_dft_c2r(ion_plan_c2r_mt, ion_ref_work, (float *)ion_ref_work);
            const float *fbuf = (const float *)ion_ref_work;
            float *out = (f == 1) ? out_xe : out_rec;
            #pragma omp parallel for collapse(2) num_threads(mt)
            for (int x = 0; x < side; ++x) {
                for (int y = 0; y < side; ++y) {
                    const float *row = &fbuf[((size_t)x * side + y) * r_fft_stride];
                    float *dst = &out[((size_t)x * side + y) * side];
                    if (f == 1) {
                        for (int z = 0; z < side; ++z)
                            dst[z] = fminf(fmaxf(row[z], 0.f), 0.999f);
                    } else {
                        for (int z = 0; z < side; ++z)
                            dst[z] = fmaxf(row[z], 0.f);
                    }
                }
            }
        }
        ++ion_ref_inverses[1];
        ++ion_ref_inverses[2];
    } else {
        for (int f = 1; f < 3; ++f) {
            if (IonReferenceInverse(a, f, ion_ref_work)) return 1;
            const float *fbuf = (const float *)ion_ref_work;
            float *out = (f == 1) ? out_xe : out_rec;
            #pragma omp parallel for collapse(2) num_threads(a->threads)
            for (int x = 0; x < side; ++x) {
                for (int y = 0; y < side; ++y) {
                    const float *row = &fbuf[((size_t)x * side + y) * r_fft_stride];
                    float *dst = &out[((size_t)x * side + y) * side];
                    if (f == 1) {
                        for (int z = 0; z < side; ++z)
                            dst[z] = fminf(fmaxf(row[z], 0.f), 0.999f);
                    } else {
                        for (int z = 0; z < side; ++z)
                            dst[z] = fmaxf(row[z], 0.f);
                    }
                }
            }
        }
    }
    return 0;
}

int IonReferenceCells(IonGPU *a, const long *indices, long count,
                      float *out, float *fc, int evaluate) {
    struct UserParams *user_params=user_params_ufunc;
    int side=a->side;
    if(!ion_ref_density_ready || (evaluate && !ion_ref_mean_ready))return 1;
    if(evaluate==1 && IonReferencePrepareXeRec(a))return 1;
    #pragma omp parallel for num_threads(a->threads)
    for(long i=0;i<count;++i) {
        long k=indices[i];
        float v=((float*)ion_ref_density)[HII_R_FFT_INDEX(k/(side*side),(k/side)%side,k%side)];
        out[i]=fmaxf(v,-1.+FRACT_FLOAT_ERR);
        if(evaluate)fc[i]=ion_ref_fcoll[k];
    }
    /* Replay mode 2 needs only density and collapse. */
    if(evaluate==1)
        for(int f=1;f<3;++f) {
            if(IonReferenceInverse(a,f,ion_ref_work))return 1;
            #pragma omp parallel for num_threads(a->threads)
            for(long i=0;i<count;++i) {
                long k=indices[i];
                float v=((float*)ion_ref_work)[HII_R_FFT_INDEX(k/(side*side),(k/side)%side,k%side)];
                out[f*count+i]=f==1?fminf(fmaxf(v,0.),.999):fmaxf(v,0.);
            }
        }
    return 0;
}

int IonReferenceBounds(IonGPU *a) {
    ion_ref_mean_ready=false;
    if(IonReferencePrepare(a) || IonReferenceInverse(a,0,ion_ref_density))return 1;
    struct UserParams *user_params=user_params_ufunc;
    int side=a->side;
    float lo=0.,hi=0.;
    #pragma omp parallel for num_threads(a->threads) reduction(min:lo) reduction(max:hi)
    for(int x=0;x<side;++x)
        for(int y=0;y<side;++y)for(int z=0;z<side;++z) {
            float v=fmaxf(((float*)ion_ref_density)[HII_R_FFT_INDEX(x,y,z)],-1.+FRACT_FLOAT_ERR);
            lo=fminf(lo,v);hi=fmaxf(hi,v);
        }
    a->lo=lo;a->hi=hi;
    return 0;
}

int IonReferenceMean(IonGPU *a, double *out) {
    struct UserParams *user_params=user_params_ufunc;
    int side=a->side, bad=0;
    double result=0.;
    #pragma omp parallel for num_threads(a->threads) reduction(+:result,bad)
    for(int x=0;x<side;++x)
        for(int y=0;y<side;++y)for(int z=0;z<side;++z) {
            float v=fmaxf(((float*)ion_ref_density)[HII_R_FFT_INDEX(x,y,z)],-1.+FRACT_FLOAT_ERR);
            float fc=0,dummy=0;
            bad+=EvaluateSplineTable(false,1,v,0.,0.,&fc,&dummy);
            ion_ref_fcoll[HII_R_INDEX(x,y,z)]=fc;
            result+=(double)fc;
        }
    *out=result;
    ion_ref_mean_ready=(bad==0);
    return bad;
}


int IonReferenceBatchDensityPrefetch(
    int side, double dk, int nthreads, const float *density,
    float *out_all, int max_scales,
    int *out_n, float *out_r, int *out_f, float *out_lo, float *out_hi
) {
    struct UserParams *user_params = user_params_ufunc;
    if (!user_params || user_params->USE_FFTW_WISDOM || !ps_keep_fftw_plans ||
        ion_ref_side != side || ion_kmag_side != side || ion_kmag_dk != dk ||
        !ion_kmag_table || !ion_plan_c2r || (!ion_plan_r2c_mt && !ion_plan_r2c) ||
        ion_sched_n <= 0 || ion_sched_n > max_scales || ion_thread_slots < 1 ||
        !density || !out_all || !out_n || !out_r || !out_f || !out_lo || !out_hi) {
        return 9;
    }
    const int ns = ion_sched_n;
    *out_n = ns;
    for (int s = 0; s < ns; ++s) {
        out_r[s] = ion_sched_r[s];
        out_f[s] = ion_sched_f[s];
    }
    const int kz_len = side / 2 + 1;
    const int r_fft_stride = 2 * kz_len;
    const size_t n = (size_t)side * side * side;
    const size_t kn = (size_t)side * side * kz_len;
    const size_t row_bytes = (size_t)side * sizeof(float);
    const double inv_n = (double)n;
    const int r2c_mt = nthreads < 1 ? 1 : (nthreads > 8 ? 8 : nthreads);
    const int nt = nthreads < 1 ? 1 : (nthreads > ion_thread_slots ? ion_thread_slots : nthreads);
    float *data = (float *)ion_ref[0];
    #pragma omp parallel for collapse(2) num_threads(nt)
    for (int x = 0; x < side; ++x)
        for (int y = 0; y < side; ++y)
            memcpy(&data[((size_t)x * side + y) * r_fft_stride],
                   &density[((size_t)x * side + y) * side],
                   row_bytes);
    fftwf_plan r2c = ion_plan_r2c_mt ? ion_plan_r2c_mt : ion_plan_r2c;
    fftwf_execute_dft_r2c(r2c, data, ion_ref[0]);
    float *fdata = (float *)ion_ref[0];
    const unsigned long long fpx = 2ULL * (unsigned long long)kn;
    #pragma omp parallel for num_threads(r2c_mt)
    for (unsigned long long i = 0; i < fpx; ++i)
        fdata[i] = (float)((double)fdata[i] / inv_n);

    const float dmin = (float)(-1. + FRACT_FLOAT_ERR);
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        fftwf_complex *buf = ion_thread_buf[tid];
        #pragma omp for schedule(dynamic, 1)
        for (int s = 0; s < ns; ++s) {
            IonFilterCopy(ion_ref[0], buf, side, out_r[s], out_f[s]);
            fftwf_execute_dft_c2r(ion_plan_c2r, buf, (float *)buf);
            const float *fbuf = (const float *)buf;
            float *dst = out_all + (size_t)s * n;
            float lo = 0.f, hi = 0.f;
            for (int x = 0; x < side; ++x) {
                for (int y = 0; y < side; ++y) {
                    const float *row = &fbuf[((size_t)x * side + y) * r_fft_stride];
                    float *drow = &dst[((size_t)x * side + y) * side];
                    for (int z = 0; z < side; ++z) {
                        float v = fmaxf(row[z], dmin);
                        drow[z] = v;
                        lo = fminf(lo, v);
                        hi = fmaxf(hi, v);
                    }
                }
            }
            out_lo[s] = lo;
            out_hi[s] = hi;
        }
    }
    return 0;
}

int IonReferenceCommitPrefetch(
    int count, const float *r, const int *f, const float *lo, const float *hi
) {
    if (count != ion_sched_n || count <= 0 || count > 128) return 1;
    for (int s = 0; s < count; ++s) {
        if (r[s] != ion_sched_r[s] || f[s] != ion_sched_f[s]) return 1;
    }
    for (int s = 0; s < count; ++s) {
        ion_batch_lo[s] = lo[s];
        ion_batch_hi[s] = hi[s];
    }
    ion_batch_ready = true;
    ion_ref_density_ready = true;
    ion_ref_xerec_ready = false;
    ion_ref_inverses[0] += (unsigned long long)count;
    return 0;
}

void IonReferenceRecombRows(const float *density, const long *ids, long count, float z, int *rows) {
    #pragma omp parallel for
    for(long i=0;i<count;++i) {
        float d=1.+density[ids[i]];
        float effective=pow(d,1./3.);
        effective*=1+z;
        int row=(int)((effective-1.)/RR_DEL_Z+0.5);
        rows[i]=row<0?0:row>=RR_Z_NPTS?RR_Z_NPTS-1:row;
    }
}
