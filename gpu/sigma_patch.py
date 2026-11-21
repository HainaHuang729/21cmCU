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
    return sigma_hook && user_params_ps->POWER_SPECTRUM==0 && !global_params.P_CUTOFF && global_params.FILTER==0;
}
static void GPUSigmaEvaluate(int n,const double *radius,const double *mass,
                             double *variance,double *derivative) {
    const double params[]={theta_cmb,omhh,alpha_nu,sound_horizon,beta_c,f_nu,N_nu,
        1.,cosmo_params_ps->POWER_INDEX};
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
    return source
