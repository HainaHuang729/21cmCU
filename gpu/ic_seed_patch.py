"""Run the device IC hook before the CPU per-grid Fourier generator."""
from ic_reference_patch import patch_reference


def patch_seed(source):
    source=patch_reference(source)
    a="int ComputeInitialConditions("
    assert source.count(a)==1
    source=source.replace(a,"""
typedef int (*ICSeedGPUHook)(unsigned long long,int,int,int,double,double,float,const double *,float **);
static ICSeedGPUHook ic_seed_gpu_hook;
void SetICSeedGPUHook(void *ptr) { ic_seed_gpu_hook=(ICSeedGPUHook)ptr; }
"""+a,1)
    a="    Broadcast_struct_global_UF(user_params,cosmo_params);"
    assert source.count(a)==1
    source=source.replace(a,a+"""
    if (ic_seed_gpu_hook) {
        if (user_params->PERTURB_ON_HIGH_RES || user_params->NON_CUBIC_FACTOR != 1.0
            || !user_params->USE_2LPT || user_params->USE_RELATIVE_VELOCITIES
            || user_params->POWER_SPECTRUM != 6 || user_params->DIM % user_params->HII_DIM)
            Throw(ValueError);
        init_ps();
        const double params[]={theta_cmb,omhh,alpha_nu,sound_horizon,beta_c,f_nu,N_nu,
            TWOPI*PI*sigma_norm*sigma_norm,cosmo_params->POWER_INDEX,cosmo_params->KP,cosmo_params->MS};
        float *outputs[]={boxes->hires_density,boxes->lowres_density,
            boxes->lowres_vx,boxes->lowres_vy,boxes->lowres_vz,
            boxes->lowres_vx_2LPT,boxes->lowres_vy_2LPT,boxes->lowres_vz_2LPT};
        const int result=ic_seed_gpu_hook(random_seed,user_params->DIM,user_params->HII_DIM,
            user_params->NO_RNG,DELTA_K,VOLUME,L_FACTOR*user_params->BOX_LEN/(user_params->HII_DIM+0.0),
            params,outputs);
        free_ps();
        if (result) Throw(ValueError);
    } else {
""",1)
    a="    } // End of Try{}"
    assert source.count(a)==1
    return source.replace(a,"    }\n"+a,1)
