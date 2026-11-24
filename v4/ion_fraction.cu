// The upstream caller supplies normalized ACG and MCG photon counts and
// recombinations per baryon at the current filter radius (center method).
extern "C" __global__ void ion_fraction_center(const double *acg, const double *mcg,
                                               const double *xe, const double *rec, float *neutral,
                                               int n, int last) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (exceeds_ionization_barrier(acg[i] + mcg[i], xe[i], rec[i])) {
        neutral[i] = 0;
    } else if (last && neutral[i] > 1.e-30) {
        double residual = (1. - acg[i] - mcg[i]) - xe[i];
        neutral[i] = residual < 0 ? 0 : residual > 1 ? 1 : residual;
    }
}
