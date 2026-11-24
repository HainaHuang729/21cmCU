// Uniform draws use the same GSL stream and order as the native sampler.
// Each thread places one accepted halo in the sampler's original array order.
extern "C" __global__ void place_halos(const double *rows, float *out, int count, int sphere,
                                       double box_xy, double box_z, double rho_star, double rho_sfr,
                                       double rho_xray) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double *row = rows + 17 * i;
    float *result = out + 7 * i;
    int rounding_mask = 0;
    double corr[] = {rho_star, rho_sfr, rho_xray};
    double direction[3];
    double radial_scale = 0.;
    if (sphere) {
        for (int axis = 0; axis < 3; ++axis) direction[axis] = 2. * row[4 + axis] - 1.;
        double norm = sqrt(direction[0] * direction[0] + direction[1] * direction[1] +
                           direction[2] * direction[2]);
        radial_scale = (row[3] * row[7]) / norm;
    }
    for (int axis = 0; axis < 3; ++axis) {
        double position = sphere ? row[axis] + radial_scale * direction[axis]
                                 : (row[axis] + (row[4 + axis] - 0.5)) * row[3];
        double length = axis == 2 ? box_z : box_xy;
        while (position >= length) position -= length;
        while (position < 0.) position += length;
        result[axis] = (float)position;

        double r2 = row[11 + axis];
        double draw = row[8 + axis] * sqrt((-2. * log(r2)) / r2);
        double rho = corr[axis];
        double random_term = sphere ? sqrt(1. - rho * rho) * draw : draw;
        double parent_term = sphere ? rho * row[14 + axis] : 0.;
        double value = random_term + parent_term;
        float rounded = (float)value;
        result[3 + axis] = rounded;
        // Only an error reaching a float midpoint can change the stored word.
        // Bound the effect of double libm rounding in the two summed terms,
        // including cancellation when their signs differ.
        float infinity = __int_as_float(0x7f800000);
        float neighbour = nextafterf(rounded, (double)rounded < value ? infinity : -infinity);
        double midpoint = ((double)rounded + (double)neighbour) * .5;
        double error = 32. * 2.2204460492503131e-16 * (fabs(random_term) + fabs(parent_term));
        if (fabs(value - midpoint) <= error) rounding_mask |= 1 << axis;
    }
    result[6] = rounding_mask;
}
