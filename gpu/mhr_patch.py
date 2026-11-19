"""Divert the dense MHR rate table quadrature to the GPU."""
def patch_mhr(source):
    a='void init_MHR(){'
    assert source.count(a)==1
    s=source.replace(a,"""
typedef int (*GPUMHRTable)(int,int,const double *,const double *,double,double,int,const double *);
static GPUMHRTable gpu_mhr_table;
void SetGPUMHRTable(void *ptr){gpu_mhr_table=(GPUMHRTable)ptr;}
"""+a,1)
    a='  // now the recombination rate look up tables'
    assert s.count(a)==2
    s=s.replace(a,"""
  if (gpu_mhr_table) {
    double pdf[A_NPTS][3];
    for (int i=0;i<A_NPTS;++i) {
      const double z=2.+i;
      pdf[i][0]=z;
      pdf[i][1]=C_MHR(z);
      pdf[i][2]=beta_MHR(z);
    }
    double params[RR_Z_NPTS][5];
    for (int i=0;i<RR_Z_NPTS;++i) {
      const float z=i*RR_DEL_Z;
      params[i][0]=z;
      params[i][1]=0.;
      params[i][2]=C_MHR(z);
      params[i][3]=beta_MHR(z);
      params[i][4]=No*pow(1.+z,3);
    }
    for(int j=0;j<RR_lnGamma_NPTS;++j) lnGamma_values[j]=RR_lnGamma_min+j*RR_DEL_lnGamma;
    if(gpu_mhr_table(RR_Z_NPTS,RR_lnGamma_NPTS,&params[0][0],lnGamma_values,
        alpha_B(1.e4),1.+1./(4./global_params.Y_He-3.),A_NPTS,&pdf[0][0])) Throw(ValueError);
    return;
  }

"""+a,1)
    a='  init_A_MHR();'
    assert s.count(a)==1
    s=s.replace(a,'  if (!gpu_mhr_table) init_A_MHR();',1)
    a='  free_A_MHR();'
    assert s.count(a)==1
    s=s.replace(a,'  if (!gpu_mhr_table) free_A_MHR();',1)
    a='  free_beta_MHR();'
    assert s.count(a)==1
    return s.replace(a,a+'\n  if (gpu_mhr_table) return;',1)
