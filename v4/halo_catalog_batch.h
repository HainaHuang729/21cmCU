/* The next mass draw consumes the same per-thread GSL state. Select accepted
 * polar pairs on that stream and batch their transforms with halo placement. */
enum { CATALOG_BATCH = 65536, INPUT_STRIDE = 17, OUTPUT_STRIDE = 7 };
int (*place_on_gpu)(int, int, double, double, const double *, const double *, float *);

typedef struct {
    double rows[CATALOG_BATCH][INPUT_STRIDE];
    float results[CATALOG_BATCH][OUTPUT_STRIDE];
    index_huge first;
    int size, failed;
} CatalogBatch;

static void draw_property_pairs(gsl_rng *rng, double *row) {
    for (int axis = 0; axis < 3; ++axis) {
        double x, y, r2;
        do {
            x = 2. * gsl_rng_uniform_pos(rng) - 1.;
            y = 2. * gsl_rng_uniform_pos(rng) - 1.;
            /* Installed GSL computes mul, mul, add before the rejection test. */
            volatile double xx = x * x, yy = y * y;
            r2 = xx + yy;
        } while (r2 > 1. || r2 == 0.);
        row[8 + axis] = y;
        row[11 + axis] = r2;
    }
}

/* libdevice and the system libm can round log() differently in double.
 * Re-evaluate only ambiguous float roundings before storing the catalogue. */
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-fast-math", "fp-contract=off")))
#endif
static float exact_property(const double *row, int axis, const double *corr) {
    double r2 = row[11 + axis];
    double draw = row[8 + axis] * sqrt((-2. * log(r2)) / r2);
    if (!corr) return (float)draw;
    double rho = corr[axis];
    return (float)(sqrt(1. - rho * rho) * draw + rho * row[14 + axis]);
}

static void flush_halos(CatalogBatch *batch, HaloCatalog *out, const double *corr) {
    if (!batch->size || batch->failed) return;
    batch->failed = place_on_gpu(batch->size, corr != NULL, simulation_options_global->BOX_LEN,
                                 BOXLEN_PARA, corr, batch->rows[0], batch->results[0]);
    if (batch->failed) return;
    float *properties[] = {out->star_rng, out->sfr_rng, out->xray_rng};
    for (int i = 0; i < batch->size; ++i) {
        index_huge idx = batch->first + i;
        float *result = batch->results[i];
        memcpy(out->halo_coords + 3 * idx, result, 3 * sizeof(float));
        for (int axis = 0; axis < 3; ++axis) {
            properties[axis][idx] = ((int)result[6] & (1 << axis))
                                        ? exact_property(batch->rows[i], axis, corr)
                                        : result[3 + axis];
        }
    }
    batch->size = 0;
}

static void queue_halo(CatalogBatch *batch, HaloCatalog *out, index_huge idx, bool writable,
                       const double centre[3], double scale, gsl_rng *rng, const double *corr,
                       const double *parent, const double *prepared) {
    double discard[INPUT_STRIDE];
    double *row = writable && !batch->failed ? batch->rows[batch->size] : discard;
    if (prepared) {
        memcpy(row + 4, prepared, 10 * sizeof(double));
    } else {
        if (corr) draw_property_pairs(rng, row);
        for (int axis = 0; axis < 3; ++axis) row[4 + axis] = gsl_rng_uniform(rng);
        row[7] = corr ? gsl_rng_uniform(rng) : 0.;
        /* Upstream still draws progenitor properties on overflow. A grid halo
         * consumes only its three position uniforms in that case. */
        if (!corr && writable) draw_property_pairs(rng, row);
    }
    if (!writable || batch->failed) return;
    if (!batch->size) batch->first = idx;
    memcpy(row, centre, 3 * sizeof(double));
    row[3] = scale;
    for (int axis = 0; axis < 3; ++axis) row[14 + axis] = corr ? parent[axis] : 0.;
    if (++batch->size == CATALOG_BATCH) flush_halos(batch, out, corr);
}
