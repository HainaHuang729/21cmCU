"""Register the CUDA brightness backend inside the normal C entry point."""

def patch_brightness(source):
    key="int ComputeBrightnessTemp("
    assert source.count(key)==1
    source=source.replace(key,"""
typedef int (*BrightnessGPUHook)(int,int,float,double,float,float,float,double,
                                 float*,float*,float*,float*,float*);
static BrightnessGPUHook brightness_hook=NULL;
void SetBrightnessGPUHook(void *hook) { brightness_hook=(BrightnessGPUHook)hook; }

int ComputeBrightnessFastFFTW(
    int side, int rsd, float redshift, double dk, float factor,
    float trad, float hubble, double limit,
    const float *density, const float *velocity,
    const float *neutral, const float *ts, float *out
) {
    struct UserParams *user_params = user_params_ufunc;
    if (!user_params || user_params->USE_FFTW_WISDOM || user_params->HII_DIM != side || user_params->NON_CUBIC_FACTOR != 1.0)
        return 2;
    const int nt = user_params->N_THREADS < 1 ? 1 : (user_params->N_THREADS > 8 ? 8 : user_params->N_THREADS);
    if (rsd) {
        if (EnsurePerturbFFTWPlans(user_params, 0)) return 1;
        float *fbox = (float *)ps_perturb_fftw_box;
        const size_t row_bytes = (size_t)side * sizeof(float);
        #pragma omp parallel for collapse(2) num_threads(nt)
        for (int i = 0; i < side; ++i)
            for (int j = 0; j < side; ++j)
                memcpy(&fbox[HII_R_FFT_INDEX(i,j,0)], &velocity[HII_R_INDEX(i,j,0)], row_bytes);
        fftwf_execute(ps_perturb_plan_r2c);
        #pragma omp parallel for num_threads(nt)
        for (int n_x = 0; n_x < side; ++n_x) {
            for (int n_y = 0; n_y < side; ++n_y) {
                for (int n_z = 0; n_z <= HII_MIDDLE_PARA; ++n_z) {
                    float k_z = n_z * DELTA_K;
                    ps_perturb_fftw_box[HII_C_INDEX(n_x,n_y,n_z)] *= k_z * I / (float)HII_TOT_NUM_PIXELS;
                }
            }
        }
        fftwf_execute(ps_perturb_plan_c2r);
    }
    const float *fbox = rsd ? (const float *)ps_perturb_fftw_box : NULL;
    double ave = 0.0;
    #pragma omp parallel for reduction(+:ave) num_threads(nt)
    for (int i = 0; i < side; ++i) {
        for (int j = 0; j < side; ++j) {
            for (int k = 0; k < side; ++k) {
                const unsigned long long idx = HII_R_INDEX(i,j,k);
                float bt = factor * neutral[idx] * (1 + density[idx]);
                float pixel_Ts_factor = (1 - trad / ts[idx]);
                bt *= pixel_Ts_factor;
                if (rsd) {
                    double dvdx = clip(fbox[HII_R_FFT_INDEX(i,j,k)], -limit, limit);
                    bt /= (dvdx / hubble + 1.0);
                }
                out[idx] = bt;
                ave += bt;
            }
        }
    }
    if (!ps_keep_fftw_plans) {
        ps_destroy_cached_fftw_plans();
        fftwf_cleanup_threads();
        fftwf_cleanup();
    }
    return isfinite(ave) ? 0 : 1;
}

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
