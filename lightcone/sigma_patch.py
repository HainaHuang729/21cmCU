"""Route the top-hat Eisenstein-Hu variance integrals through CUDA."""
def patch_sigma(source):
    def patch(a,b):
        nonlocal source
        assert source.count(a)==1,a
        source=source.replace(a,b,1)

    a='double sigma_z0(double M){'
    patch(a,"""
typedef int (*GPUSigmaMoments)(int,const double *,const double *,const double *,double *,double *);
static GPUSigmaMoments sigma_hook;
void SetGPUSigmaMoments(void *ptr) { sigma_hook=(GPUSigmaMoments)ptr; }
static bool UseGPUSigmaMoments(void) {
    return sigma_hook && user_params_ps->POWER_SPECTRUM==6 && global_params.FILTER==0;
}
static void GPUSigmaEvaluate(int n,const double *radius,const double *mass,
                             double *variance,double *derivative) {
    const double params[]={theta_cmb,omhh,alpha_nu,sound_horizon,beta_c,f_nu,N_nu,
        1.,cosmo_params_ps->POWER_INDEX,cosmo_params_ps->KP,cosmo_params_ps->MS};
    if (sigma_hook(n,params,radius,mass,variance,derivative)) Throw(ValueError);
}
"""+a+"""
    if (UseGPUSigmaMoments()) {
        const double radius=MtoR(M);
        double integral,derivative;
        GPUSigmaEvaluate(1,&radius,&M,&integral,&derivative);
        return sigma_norm*sqrt(integral);
    }
""")
    a='    Radius_8 = 8.0/cosmo_params_ps->hlittle;'
    patch(a,a+"""
    if (UseGPUSigmaMoments()) {
        const double mass=0.;
        double variance,derivative;
        gsl_integration_workspace_free(w);
        GPUSigmaEvaluate(1,&Radius_8,&mass,&variance,&derivative);
        sigma_norm=cosmo_params_ps->SIGMA_8/sqrt(variance);
        return R_CUTOFF;
    }
""")
    a='double dsigmasqdm_z0(double M){'
    patch(a,a+"""
    if (UseGPUSigmaMoments()) {
        const double radius=MtoR(M);
        double integral,derivative;
        GPUSigmaEvaluate(1,&radius,&M,&integral,&derivative);
        return sigma_norm*sigma_norm*derivative;
    }
""")
    a='#pragma omp parallel shared(Mass_InterpTable,Sigma_InterpTable,dSigmadm_InterpTable)'
    patch(a,"""
    if (UseGPUSigmaMoments()) {
        double radius[NMass],mass[NMass],integral[NMass],derivative[NMass];
        for(i=0;i<NMass;++i) {
            Mass_InterpTable[i]=log(M_Min)+(float)i/(NMass-1)*(log(M_Max)-log(M_Min));
            mass[i]=exp(Mass_InterpTable[i]);
            radius[i]=MtoR(mass[i]);
        }
        GPUSigmaEvaluate(NMass,radius,mass,integral,derivative);
        for(i=0;i<NMass;++i) {
            Sigma_InterpTable[i]=sigma_norm*sqrt(integral[i]);
            dSigmadm_InterpTable[i]=log10(-sigma_norm*sigma_norm*derivative[i]);
        }
    } else
"""+a)
    a = 'double init_ps(){'
    patch(a, '''
typedef struct {
    int valid;
    int power_spectrum;
    double vals[11];
    double r_cutoff;
    double omhh;
    double theta_cmb;
    double f_nu;
    double f_baryon;
    double sigma_norm;
} PSInitCache;
static PSInitCache ps_init_cache = {0};
''' + a)
    a = '    double kstart, kend;\n\n    //we start the interpolator if using CLASS:'
    patch(a, a + '''
    const char *ps_cache_env = getenv("PS_CACHE_INIT_PS");
    const int use_ps_cache = (!ps_cache_env || ps_cache_env[0] != '0') && (user_params_ps->POWER_SPECTRUM != 5);
    double ps_key_vals[11] = {
        (double)cosmo_params_ps->OMm,
        (double)cosmo_params_ps->OMb,
        (double)cosmo_params_ps->hlittle,
        (double)cosmo_params_ps->SIGMA_8,
        (double)cosmo_params_ps->POWER_INDEX,
        (double)cosmo_params_ps->KP,
        (double)cosmo_params_ps->MS,
        (double)global_params.OMn,
        (double)global_params.M_WDM,
        (double)global_params.g_x,
        (double)T_cmb
    };
    if (use_ps_cache && ps_init_cache.valid &&
        ps_init_cache.power_spectrum == user_params_ps->POWER_SPECTRUM &&
        memcmp(ps_init_cache.vals, ps_key_vals, sizeof(ps_key_vals)) == 0) {
        gsl_integration_workspace_free(w);
        R_CUTOFF = ps_init_cache.r_cutoff;
        omhh = ps_init_cache.omhh;
        theta_cmb = ps_init_cache.theta_cmb;
        f_nu = ps_init_cache.f_nu;
        f_baryon = ps_init_cache.f_baryon;
        TFset_parameters();
        sigma_norm = ps_init_cache.sigma_norm;
        return R_CUTOFF;
    }
''')
    a = '    sigma_norm = cosmo_params_ps->SIGMA_8/sqrt(result); //takes care of volume factor'
    patch(a, a + '''
    if (use_ps_cache) {
        ps_init_cache.power_spectrum = user_params_ps->POWER_SPECTRUM;
        memcpy(ps_init_cache.vals, ps_key_vals, sizeof(ps_key_vals));
        ps_init_cache.r_cutoff = R_CUTOFF;
        ps_init_cache.omhh = omhh;
        ps_init_cache.theta_cmb = theta_cmb;
        ps_init_cache.f_nu = f_nu;
        ps_init_cache.f_baryon = f_baryon;
        ps_init_cache.sigma_norm = sigma_norm;
        ps_init_cache.valid = 1;
    }''')
    return source
