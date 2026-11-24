#include "Constants.h"
#include "InputParameters.h"
#include "cosmology.h"
#include "exceptions.h"
#include "thermochem.h"
#include <math.h>
#include <stddef.h>

// These are exported by the explicitly version-checked 4.3.2 native library.
extern double transfer_function_CLASS(double, int, int);

// gpu_dispatch is provided by the scoped adapter that loads this source.
static bool supported_power_spectrum(void) {
    int spectrum = matter_options_global->POWER_SPECTRUM;
    return spectrum >= POWER_SPECTRUM_EH && spectrum <= POWER_SPECTRUM_CLASS;
}

int gpu_init_ps(double *constants) {
    bool classy = matter_options_global->POWER_SPECTRUM == POWER_SPECTRUM_CLASS;
    if (!supported_power_spectrum()) return 2;
    bool gaussian = matter_options_global->FILTER == FILTER_GAUSSIAN;
    if (!gaussian && matter_options_global->FILTER != FILTER_TOPHAT) return 2;

    double cosmology[] = {
        cosmo_params_global->OMm,
        cosmo_params_global->hlittle,
        cosmo_params_global->OMb,
        cosmo_params_global->OMn,
        physconst.T_cmb,
        cosmo_tables_global->USE_SIGMA_8,
        0,
        0,
        uses_v_cb(matter_options_global->V_CB_MODEL),
        matter_options_global->POWER_SPECTRUM,
        gaussian ? pow(2 * M_PI, 1.5) * cosmo_params_global->OMm * RHOcrit * 3 : 0.};
    Table1D *density = classy ? cosmo_tables_global->transfer_density : NULL;
    cosmology[6] = classy ? density->size : 0;
    Table1D *velocity = classy && matter_options_global->V_CB_MODEL == V_CB_MODEL_FLUCTS
                            ? cosmo_tables_global->transfer_vcb
                            : NULL;
    cosmology[7] = velocity ? velocity->size : 0;
    void *out[] = {constants, classy ? density->x_values : NULL, classy ? density->y_values : NULL,
                   velocity ? velocity->y_values : NULL};
    if (gpu_dispatch(25, 1, cosmology, out)) return 1;
    if (cosmo_tables_global->USE_SIGMA_8) {
        double normalization[] = {0,
                                  cosmo_tables_global->ps_norm,
                                  cosmo_params_global->POWER_INDEX,
                                  cosmo_params_global->OMm * RHOcrit,
                                  0,
                                  0,
                                  8. / cosmo_params_global->hlittle};
        if (gaussian || matter_options_global->POWER_SPECTRUM != POWER_SPECTRUM_EH)
            normalization[6] = MtoR(RtoM(normalization[6]));
        if (gpu_dispatch(19, 1, normalization, NULL)) return 1;
    }
    // Other native routines and free_ps() share this native interpolator.
    if (classy) transfer_function_CLASS(1., 0, 0);
    return 0;
}

int gpu_sigma(float min_mass, float max_mass, int n) {
    double low = log(min_mass);
    double width = (log(max_mass) - low) / (n - 1.);
    double params[] = {n,
                       cosmo_tables_global->ps_norm,
                       cosmo_params_global->POWER_INDEX,
                       cosmo_params_global->OMm * RHOcrit,
                       low,
                       width,
                       0};
    if (matter_options_global->FILTER == FILTER_GAUSSIAN ||
        matter_options_global->POWER_SPECTRUM != POWER_SPECTRUM_EH) {
        // Four-byte masses and native MtoR set the integration limits. A one-ULP
        // radius change can move the adaptive derivative integral to another panel.
        double radii[n];
        for (int i = 0; i < n; ++i) radii[i] = MtoR((float)exp(low + i * width));
        void *axes[] = {radii};
        return gpu_dispatch(19, n, params, axes);
    }
    return gpu_dispatch(19, n, params, NULL);
}

double gpu_sigma_z0(double mass) {
    if (!supported_power_spectrum() || (matter_options_global->FILTER != FILTER_TOPHAT &&
                                        matter_options_global->FILTER != FILTER_GAUSSIAN))
        return sigma_z0(mass);

    double result;
    double params[] = {0,
                       cosmo_tables_global->ps_norm,
                       cosmo_params_global->POWER_INDEX,
                       cosmo_params_global->OMm * RHOcrit,
                       0,
                       0,
                       MtoR(mass)};
    void *out[] = {&result};
    if (gpu_dispatch(28, 1, params, out)) Throw(ValueError);
    return result;
}

int gpu_mhr(void) {
    double params[] = {No, cosmo_params_global->Y_He, alpha_B(1.e4)};
    return gpu_dispatch(20, 1, params, NULL);
}
