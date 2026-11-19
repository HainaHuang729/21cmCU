"""Register the CUDA brightness backend inside the normal C entry point."""

def patch_brightness(source):
    key="int ComputeBrightnessTemp("
    assert source.count(key)==1
    source=source.replace(key,"""
typedef int (*BrightnessGPUHook)(int,int,float,double,float,float,float,double,
                                 float*,float*,float*,float*,float*);
static BrightnessGPUHook brightness_hook=NULL;
void SetBrightnessGPUHook(void *hook) { brightness_hook=(BrightnessGPUHook)hook; }

"""+key,1)
    key="    char wisdom_filename[500];"
    assert source.count(key)==1
    return source.replace(key,"""    if (brightness_hook && user_params->NON_CUBIC_FACTOR==1 &&
        flag_options->USE_TS_FLUCT && !flag_options->SUBCELL_RSD) {
        init_ps();
        float rad=T_cmb*(1+redshift), h=hubble(redshift);
        float scale=27*(cosmo_params->OMb*cosmo_params->hlittle*cosmo_params->hlittle/0.023)*
          sqrt((0.15/cosmo_params->OMm/cosmo_params->hlittle/cosmo_params->hlittle)*(1.+redshift)/10.0);
        if (brightness_hook(user_params->HII_DIM,flag_options->APPLY_RSDS,redshift,DELTA_K,
                            scale,rad,h,fabs(global_params.MAX_DVDR*h),perturb_field->density,
                            perturb_field->velocity_z,ionized_box->xH_box,spin_temp->Ts_box,
                            box->brightness_temp)) Throw(ValueError);
        goto brightness_success;
    }

"""+key,1).replace("    } // End of try","    brightness_success: ;\n    } // End of try",1)
