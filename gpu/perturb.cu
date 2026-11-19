extern "C" __global__ void scale_displacements(
    float *displacement, long size, float first, float second, int undo
) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 6*size) return;
    float factor = i < 3*size ? first : second;
    displacement[i] = undo ? __fdiv_rn(displacement[i],factor) : displacement[i]*factor;
}

__device__ inline double periodic(double x, int n) {
    while (x >= n) x -= n;
    while (x < 0.) x += n;
    return x;
}

__device__ inline int lower_cell(double x, int size, float &upper_weight) {
    int cell = (int)x;
    double center = cell + 0.5;
    upper_weight = (float)fabs(x - center);
    if (x < center) {
        upper_weight = (float)(1.0 - upper_weight);
        if (--cell < 0) cell += size;
    }
    return cell;
}

extern "C" __global__ void cic(
    const float *initial_density, const float *displacement, double *mass,
    int fine, int coarse, float initial_growth
) {
    long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long)fine*fine*fine) return;
    int k = index % fine;
    int j = (index / fine) % fine;
    int i = index / (fine*fine);
    int ratio = fine / coarse;
    long n = (long)coarse*coarse*coarse;
    long tile = ((long)(i/ratio)*coarse + j/ratio)*coarse + k/ratio;

    double x = (i + 0.5)/(double)fine;
    double y = (j + 0.5)/(double)fine;
    double z = (k + 0.5)/(double)fine;
    x += displacement[tile];
    y += displacement[tile+n];
    z += displacement[tile+2*n];
    x -= displacement[tile+3*n];
    y -= displacement[tile+4*n];
    z -= displacement[tile+5*n];

    float wx, wy, wz;
    int a = lower_cell(periodic(x*coarse,coarse),coarse,wx);
    int b = lower_cell(periodic(y*coarse,coarse),coarse,wy);
    int c = lower_cell(periodic(z*coarse,coarse),coarse,wz);
    int ix[2] = {a, (a+1) % coarse};
    int iy[2] = {b, (b+1) % coarse};
    int iz[2] = {c, (c+1) % coarse};
    float xw[2] = {(float)(1.0-wx), wx};
    float yw[2] = {(float)(1.0-wy), wy};
    float zw[2] = {(float)(1.0-wz), wz};
    double particle_mass = 1.0 + initial_growth*initial_density[index];
    for (int dz=0; dz<2; ++dz)
        for (int dy=0; dy<2; ++dy)
            for (int dx=0; dx<2; ++dx) {
                long target = ((long)ix[dx]*coarse + iy[dy])*coarse + iz[dz];
                atomicAdd(mass + target, particle_mass*(xw[dx]*yw[dy]*zw[dz]));
            }
}

extern "C" __global__ void density_contrast(
    const double *mass, float *contrast, long count, float mass_factor
) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) contrast[i] = __fdiv_rn((float)mass[i],mass_factor) - 1.0f;
}

extern "C" __global__ void velocity_spectrum(
    const float2 *density, float2 *velocity, int side, double dk, float dlog_growth_dt
) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int half = side/2 + 1;
    if (i >= (long)side*side*half) return;
    int iz = i % half;
    int iy = (i / half) % side;
    int ix = i / (side*half);
    float kx = (float)((ix > side/2 ? ix-side : ix)*dk);
    float ky = (float)((iy > side/2 ? iy-side : iy)*dk);
    float kz = (float)(iz*dk);
    float k2 = kx*kx + ky*ky + kz*kz;
    // The derivative along z must have a real-valued inverse on its Nyquist plane.
    // FFTW's real transform discards the anti-Hermitian part there.
    if (k2 == 0.f || (side % 2 == 0 && iz == side/2)) {
        velocity[i] = make_float2(0.f,0.f);
        return;
    }
    float coeff = __fdiv_rn(dlog_growth_dt*kz,k2);
    double prefactor = (double)coeff/((long)side*side*side);
    float2 f = density[i];
    velocity[i] = make_float2((float)(-f.y*prefactor),(float)(f.x*prefactor));
}
