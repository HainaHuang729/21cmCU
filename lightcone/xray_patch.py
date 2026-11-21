"""Replace the fixed-path optical-depth and X-ray frequency integrals."""
def patch_xray(source):
    a='int ComputeTsBox('
    assert source.count(a)==1
    # The helper is needed by the internal thermal routine, which is defined before ComputeTsBox.
    s="""typedef int (*GPUXRayTables)(int,int,int,int,const double *,const float *,const double *,
    const float *,const float *,const float *,const float *,const float *,const float *,const float *,
    double **,double **,double **,int,double *);
static GPUXRayTables xray_gpu_hook;
static int gpu_xray_parity;
void SetGPUXRayParity(int mode) { gpu_xray_parity=mode; }
void SetGPUXRayTables(void *ptr) { xray_gpu_hook=(GPUXRayTables)ptr; }

"""+source
    a='            if(user_params->USE_INTERPOLATION_TABLES) {\n                if(flag_options->USE_MINI_HALOS){\n                    lower_int_limit'
    assert s.count(a)==1
    s=s.replace(a,'''            if(user_params->USE_INTERPOLATION_TABLES && !(xray_gpu_hook &&
                flag_options->USE_MASS_DEPENDENT_ZETA && !flag_options->USE_MINI_HALOS)) {
                if(flag_options->USE_MINI_HALOS){
                    lower_int_limit''',1)
    a='        } // end loop over R_ct filter steps'
    assert s.count(a)==1
    return s.replace(a,a+"""

        if (xray_gpu_hook && user_params->USE_INTERPOLATION_TABLES &&
            flag_options->USE_MASS_DEPENDENT_ZETA && !flag_options->USE_MINI_HALOS) {
            const double p[]={zp,x_e_ave,N_b0*C*sqrt(cosmo_params->OMm+cosmo_params->OMl)/Ho,
                cosmo_params->OMm,cosmo_params->OMl,determine_zpp_min,zpp_bin_width,
                global_params.Pop2_ion*astro_params->F_STAR10*astro_params->F_ESC10,
                NUIONIZATION/NU_over_EV,HeI_NUIONIZATION/NU_over_EV,HeII_NUIONIZATION/NU_over_EV,
                hplank*NU_over_EV,f_H,f_He,astro_params->NU_X_THRESH,
                global_params.NU_X_MAX,astro_params->X_RAY_SPEC_INDEX,NU_over_EV,
                C/FOURPI/Ly_alpha_HZ/hubble(zp)};
            const int xr_n = global_params.NUM_FILTER_STEPS_FOR_Ts;
            double parity_cutoffs[xr_n];
            if (gpu_xray_parity == 1 || gpu_xray_parity == 4 || gpu_xray_parity == 5) {
                for (int r=0; r<xr_n; ++r) {
                    parity_cutoffs[r] = fmax(
                        nu_tau_one(zp,zpp_for_evolve_list[r],x_e_ave,filling_factor_of_HI_zp),
                        astro_params->NU_X_THRESH*NU_over_EV);
                    if (filling_factor_of_HI_zp < 0) filling_factor_of_HI_zp=0;
                }
            }
            if (xray_gpu_hook(global_params.NUM_FILTER_STEPS_FOR_Ts,x_int_NXHII,x_int_NENERGY,
                zpp_interp_points_SFR,p,zpp_for_evolve_list,Nion_z_val,x_int_Energy,x_int_XHII,
                &x_int_fheat[0][0],&x_int_n_Lya[0][0],&x_int_nion_HI[0][0],&x_int_nion_HeI[0][0],
                &x_int_nion_HeII[0][0],freq_int_heat_tbl,freq_int_ion_tbl,freq_int_lya_tbl,gpu_xray_parity,parity_cutoffs))
                Throw(TableGenerationError);
            if (gpu_xray_parity >= 3) {
                double **tables[]={freq_int_heat_tbl,freq_int_ion_tbl,freq_int_lya_tbl};
                for (int c=0; c<3; ++c)
                    for (int i=0; i<x_int_NXHII; ++i)
                        for (int r=0; r<xr_n; ++r)
                            if (!isfinite(tables[c][i][r]))
                                tables[c][i][r]=integrate_over_nu(zp,x_int_XHII[i],parity_cutoffs[r],c);
            }
            if (gpu_xray_parity == 5) {
                double max_relative[3]={0.,0.,0.};
                int failures[3]={0,0,0};
                for (int r=0; r<xr_n; ++r) {
                    for (int i=0; i<x_int_NXHII; ++i) {
                        const double gpu[]={freq_int_heat_tbl[i][r],freq_int_ion_tbl[i][r],freq_int_lya_tbl[i][r]};
                        for (int c=0; c<3; ++c) {
                            const double ref=integrate_over_nu(zp,x_int_XHII[i],parity_cutoffs[r],c);
                            const double error=fabs(gpu[c]-ref);
                            const double relative=ref != 0. ? error/fabs(ref) : (error == 0. ? 0. : INFINITY);
                            max_relative[c]=fmax(max_relative[c],relative);
                            failures[c]+=relative>1.e-4;
                        }
                    }
                }
                printf("XRAY_TABLE_CHECK %.9g %d %d %d %d %.17g %.17g %.17g\\n",
                    zp,xr_n*x_int_NXHII,failures[0],failures[1],failures[2],
                    max_relative[0],max_relative[1],max_relative[2]);
            }
            if (gpu_xray_parity == 2) {
                for (int r=0; r<xr_n; ++r) {
                    for (int i=0; i<x_int_NXHII; ++i) {
                        freq_int_heat_tbl[i][r]=integrate_over_nu(zp,x_int_XHII[i],parity_cutoffs[r],0);
                        freq_int_ion_tbl[i][r]=integrate_over_nu(zp,x_int_XHII[i],parity_cutoffs[r],1);
                        freq_int_lya_tbl[i][r]=integrate_over_nu(zp,x_int_XHII[i],parity_cutoffs[r],2);
                    }
                }
            }
            if (filling_factor_of_HI_zp < 0) filling_factor_of_HI_zp=0;
        }
""",1)
