// One v4 thermal step from upstream source rates and coupling coefficients.
// Each row contains the previous state and rates, followed by the spin inputs.
struct ThermalStepInput {
    double previous_xe, previous_tk, ionization_rate, heating_rate, dz;
    double radiation_temperature, inverse_radiation_temperature;
    double lya_flux, lya_coupling, cmb_coupling, collision_coupling;
};

extern "C" __global__ void thermal_step(const ThermalStepInput *rows, int n, float *xe, float *tk,
                                        float *ts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const ThermalStepInput &cell = rows[i];
    xe[i] = advance_xe(cell.previous_xe, cell.ionization_rate, cell.dz);
    double temperature = advance_temperature(cell.previous_tk, cell.heating_rate, cell.dz);
    if (temperature < 0.) temperature = cell.radiation_temperature;
    tk[i] = temperature;
    double inverse = 1. / temperature;
    double spin;
    if (cell.lya_flux > 1.e-20) {
        spin = cell.radiation_temperature;
        double previous = 0.;
        while (fabs(spin - previous) / spin > 1.e-3) {
            previous = spin;
            spin = spin_next(previous, inverse, inverse * inverse, cell.lya_coupling,
                             cell.cmb_coupling, cell.inverse_radiation_temperature,
                             cell.collision_coupling);
        }
    } else {
        spin = (cell.cmb_coupling + cell.collision_coupling) /
               (cell.cmb_coupling * cell.inverse_radiation_temperature +
                cell.collision_coupling * inverse);
    }
    ts[i] = fabs(spin);
}
