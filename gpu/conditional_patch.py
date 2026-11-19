"""Divert native conditional-collapse integrals to the GPU in the fixed model."""
def patch_condition(source):
    anchor="void initialise_Nion_General_spline(float z"
    assert source.count(anchor)==1
    source=source.replace(anchor,"""
typedef int (*GPUConditionalTable)(int,int,int,int,int,const double *,const float *,
    const float *,const float *,const float *,const float *,const float *,float *,float *);
static GPUConditionalTable conditional_hook;
void SetGPUConditionalTable(void *ptr) { conditional_hook=(GPUConditionalTable)ptr; }
static int conditional_modes = 3;
void SetGPUConditionalMode(int bits) { conditional_modes=bits; }

"""+anchor,1)
    anchor="#pragma omp parallel shared(log10_overdense_spline_SFR,log10_Nion_spline,"
    assert source.count(anchor)==1
    source=source.replace(anchor,"""
    if (conditional_hook && (conditional_modes & (1 << 0)) && !(FAST_FCOLL_TABLES && global_params.USE_FAST_ATOMIC)) {
        float low[NSFR_low], high[NSFR_high];
        for (int i=0;i<NSFR_low;++i) {
            const double overdense=log10(1.+overdense_small_low)+(double)i/(NSFR_low-1.)*
                (log10(1.+overdense_small_high)-log10(1.+overdense_small_low));
            log10_overdense_spline_SFR[i]=overdense;
            low[i]=pow(10.,overdense)-1.;
        }
        for (int i=0;i<NSFR_high;++i)
            high[i]=Overdense_spline_SFR[i]=overdense_large_low+(float)i/(NSFR_high-1.)*
                (overdense_large_high-overdense_large_low);
        const double p[]={MinMass,mass_bin_width,inv_mass_bin_width,growthf,sigma2,Deltac,
            Mmin,Mmax,MassTurnover,Alpha_star,Alpha_esc,Fstar10,Fesc10,Mlim_Fstar,Mlim_Fesc};
        if (conditional_hook(0,NSFR_low,NSFR_high,NMass,NGL_SFR,p,Sigma_InterpTable,dSigmadm_InterpTable,
                             low,high,xi_SFR+1,wi_SFR+1,log10_Nion_spline,Nion_spline))
            Throw(ValueError);
        return;
    }

"""+anchor,1)
    anchor="#pragma omp parallel shared(log10_SFRD_z_low_table,growthf,Mmax,sigma2,overdense_low_table,"
    front,tail=source.split("void initialise_SFRD_Conditional_table(",1)
    next_func=tail.index("void initialise_SFRD_Conditional_table_MINI(")
    body,rest=tail[:next_func],tail[next_func:]
    quadrature="        initialiseGL_Nion_Xray(NGL_SFR, MassTurnover/50., Mmax);"
    assert body.count(quadrature)==1
    body=body.replace(quadrature,
        "        if (!conditional_hook || !(conditional_modes & (1 << 1)) || (FAST_FCOLL_TABLES && global_params.USE_FAST_ATOMIC))\n"
        "    "+quadrature,1)
    assert body.count(anchor)==1
    body=body.replace(anchor,"""
        if (conditional_hook && (conditional_modes & (1 << 1)) && !(FAST_FCOLL_TABLES && global_params.USE_FAST_ATOMIC)) {
            float low[NSFR_low];
            for (int i=0;i<NSFR_low;++i) low[i]=overdense_low_table[i]-1.;
            const double p[]={MinMass,mass_bin_width,inv_mass_bin_width,growthf[j],sigma2,Deltac,
                Mmin,Mmax,MassTurnover,Alpha_star,0.,Fstar10,1.,Mlim_Fstar,0.};
            if (conditional_hook(1,NSFR_low,NSFR_high,NMass,NGL_SFR,p,Sigma_InterpTable,dSigmadm_InterpTable,
                low,overdense_high_table,xi_SFR_Xray+1,wi_SFR_Xray+1,log10_SFRD_z_low_table[j],SFRD_z_high_table[j]))
                Throw(ValueError);
            continue;
        }

"""+anchor,1)
    return front+"void initialise_SFRD_Conditional_table("+body+rest
