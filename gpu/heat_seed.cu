extern "C" __global__ void heat_seed(const float *density, float *xe, float *tk,
    float *ts, long n, float growth, float adiabatic, float kinetic,
    float electrons, double radiative, double coupling) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float delta = density[i] * growth;
    const double xc = coupling * (1. + delta);
    xe[i] = electrons;
    tk[i] = kinetic * (1. + adiabatic * delta);
    ts[i] = (1. + xc) / (1./radiative + xc/kinetic);
}
