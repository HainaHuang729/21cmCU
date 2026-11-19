"""Switch the fixed Sheth-Tormen integral to the GPU."""
def patch_hmf(source):
    def patch(a,b):
        nonlocal source
        assert source.count(a)==1,a
        source=source.replace(a,b,1)
    a='double Nion_General(double z, double M_Min, double MassTurnover, double Alpha_star, double Alpha_esc, double Fstar10, double Fesc10, double Mlim_Fstar, double Mlim_Fesc){'
    patch(a,"""
typedef int (*GPUGlobalCollapse)(int,int,const double *,const double *,const float *,const float *,double *);
static GPUGlobalCollapse global_collapse_hook;
void SetGPUGlobalCollapse(void *ptr) { global_collapse_hook=(GPUGlobalCollapse)ptr; }

static bool UseGPUGlobalCollapse(void) {
    return global_collapse_hook && user_params_ps->HMF==1 && user_params_ps->USE_INTERPOLATION_TABLES &&
        Sigma_InterpTable && dSigmadm_InterpTable;
}
static void GPUGlobalEvaluate(int n,const double *g,double Mmin,double turn,double as,double ae,
                        double fs,double fe,double mlstar,double mlesc,double *out) {
    const double p[]={log(Mmin),log(fmax(global_params.M_MAX_INTEGRAL,Mmin*100)),turn,as,ae,fs,fe,
                     mlstar,mlesc,MinMass,mass_bin_width,inv_mass_bin_width};
    if(global_collapse_hook(n,NMass,p,g,Sigma_InterpTable,dSigmadm_InterpTable,out))
        Throw(TableGenerationError);
}
"""+a+"""
    if (UseGPUGlobalCollapse()) {
        const double g=dicke(z);
        double result;
        GPUGlobalEvaluate(1,&g,M_Min,MassTurnover,Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc,&result);
        return result;
    }
""")
    a='#pragma omp parallel shared(z_val,Nion_z_val,zmin,zmax, MassTurn, Alpha_star, Alpha_esc, Fstar10, Fesc10, Mlim_Fstar, Mlim_Fesc)'
    assert a in source
    source=source.replace(a,"""
    if(UseGPUGlobalCollapse()) {
        double growth[Nbin];
        for(int i=0;i<Nbin;++i){
            z_val[i]=zmin+(double)i/(Nbin-1.)*(zmax-zmin);
            growth[i]=dicke(z_val[i]);
        }
        GPUGlobalEvaluate(Nbin,growth,Mmin,MassTurn,Alpha_star,Alpha_esc,Fstar10,Fesc10,
                          Mlim_Fstar,Mlim_Fesc,Nion_z_val);
        return;
    }

"""+a,1)
    a='#pragma omp parallel shared(z_X_val,SFRD_val,zmin,zmax, MassTurn, Alpha_star, Fstar10, Mlim_Fstar)'
    assert a in source
    source=source.replace(a,"""
    if(UseGPUGlobalCollapse()) {
        double growth[Nbin];
        for(int i=0;i<Nbin;++i){
            z_X_val[i]=zmin+(double)i/(Nbin-1.)*(zmax-zmin);
            growth[i]=dicke(z_X_val[i]);
        }
        GPUGlobalEvaluate(Nbin,growth,Mmin,MassTurn,Alpha_star,0.,Fstar10,1.,Mlim_Fstar,0.,SFRD_val);
        return;
    }

"""+a,1)
    return source
