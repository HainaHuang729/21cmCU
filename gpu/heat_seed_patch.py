"""Offload the two high-redshift thermal steps for the fixed configuration."""
def patch_heat(source):
    def patch(a,b):
        nonlocal source
        if source.count(a)!=1:
            raise ValueError(a)
        source=source.replace(a,b,1)

    anchor='void SetGPUHeatHook(void *p){gpu_heat_hook=(GPUHeatHook)p;}'
    patch(anchor,anchor+"""
typedef int (*GPUHeatSeedHook)(int,float,float,float,float,float,double,double,
                               float*,float*,float*,float*);
static GPUHeatSeedHook gpu_heat_seed_hook=NULL;
void SetGPUHeatSeedHook(void *hook) { gpu_heat_seed_hook=(GPUHeatSeedHook)hook; }
""")
    anchor="""        growthfac = growth_factor_zp * inverse_growth_factor_z;
        // read file"""
    patch(anchor,anchor+"""
        if (gpu_heat_seed_hook && user_params->NON_CUBIC_FACTOR==1.0) {
            const double trad = T_cmb*(1. + (float)redshift);
            const double coll = xcoll((float)redshift,TK,0.,xe);
            if (gpu_heat_seed_hook(user_params->HII_DIM,redshift,growthfac,cT_ad,TK,xe,trad,coll,
                perturbed_field->density,this_spin_temp->x_e_box,this_spin_temp->Tk_box,
                this_spin_temp->Ts_box)) Throw(ValueError);
        } else {
""")
    anchor="""        if(!flag_options->M_MIN_in_Mass) {
            M_MIN = (float)TtoM(redshift, astro_params->X_RAY_Tvir_MIN, mu_for_Ts);
            LOG_DEBUG("Attempting"""
    patch(anchor,"        }\n"+anchor)
    anchor='        /* The existing CPU no-light path still reads delNL0. */'
    patch(anchor,"""        const bool gpu_no_light = resident_filter && NO_LIGHT && gpu_heat_hook &&
            flag_options->USE_MASS_DEPENDENT_ZETA && user_params->USE_INTERPOLATION_TABLES &&
            !flag_options->USE_MINI_HALOS && !user_params->MINIMIZE_MEMORY;
"""+anchor)
    patch('if (resident_filter && NO_LIGHT &&\n','if (resident_filter && NO_LIGHT && !gpu_no_light &&\n')
    patch('&& !user_params->MINIMIZE_MEMORY && !NO_LIGHT) {',
          '&& !user_params->MINIMIZE_MEMORY && (!NO_LIGHT || gpu_no_light)) {')
    patch('a.init=initialization_required;', 'a.init=initialization_required | (gpu_no_light ? 2 : 0);')
    return source
