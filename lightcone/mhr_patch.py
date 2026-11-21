"""Divert the dense MHR rate table quadrature to the GPU or OpenMP host."""
def patch_mhr(source):
    a = 'void init_MHR(){'
    assert source.count(a) == 1
    s = source.replace(a, """
static double recombination_rate_ws(double z, double gamma12_bg, double T4, int usecaseB,
                                    double A, double C_0, double beta,
                                    gsl_integration_workspace *w, int *status_out);
typedef int (*GPUMHRTable)(int,int,const double *,const double *,double,double,int,const double *);
static GPUMHRTable gpu_mhr_table;
void SetGPUMHRTable(void *ptr){gpu_mhr_table=(GPUMHRTable)ptr;}
""" + a, 1)

    a = '  // now the recombination rate look up tables'
    assert s.count(a) == 2
    s = s.replace(a, """
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

  {
    const char *omp_env = getenv("PS_OMP_MHR");
    if (!omp_env || omp_env[0] != '0') {
      float gamma_vals[RR_lnGamma_NPTS];
      for (gamma_ct = 0; gamma_ct < RR_lnGamma_NPTS; ++gamma_ct) {
        lnGamma_values[gamma_ct] = RR_lnGamma_min + gamma_ct * RR_DEL_lnGamma;
        gamma_vals[gamma_ct] = (float)exp(lnGamma_values[gamma_ct]);
      }
      double A_z[RR_Z_NPTS], C_z[RR_Z_NPTS], beta_z[RR_Z_NPTS];
      for (z_ct = 0; z_ct < RR_Z_NPTS; ++z_ct) {
        float zz = z_ct * RR_DEL_Z;
        A_z[z_ct] = A_MHR(zz);
        C_z[z_ct] = C_MHR(zz);
        beta_z[z_ct] = beta_MHR(zz);
      }
      gsl_set_error_handler_off();
      int nt = omp_get_max_threads();
      if (nt < 1) nt = 1;
      if (nt > 16) nt = 16;
      int err_status = 0;
      #pragma omp parallel num_threads(nt)
      {
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
        if (!w) {
          #pragma omp atomic write
          err_status = 1;
        } else {
          #pragma omp for schedule(dynamic, 1)
          for (int zi = 0; zi < RR_Z_NPTS; ++zi) {
            float zz = zi * RR_DEL_Z;
            for (int gi = 0; gi < RR_lnGamma_NPTS; ++gi) {
              int st = 0;
              double val = recombination_rate_ws(
                  zz, gamma_vals[gi], 1, 1, A_z[zi], C_z[zi], beta_z[zi], w, &st);
              RR_table[zi][gi] = val;
              if (st != 0) {
                #pragma omp critical
                {
                  if (err_status == 0) err_status = st;
                }
              }
            }
          }
          gsl_integration_workspace_free(w);
        }
      }
      if (err_status != 0) GSL_ERROR(err_status);
      for (z_ct = 0; z_ct < RR_Z_NPTS; ++z_ct) {
        RR_acc[z_ct] = gsl_interp_accel_alloc();
        RR_spline[z_ct] = gsl_spline_alloc(gsl_interp_cspline, RR_lnGamma_NPTS);
        gsl_spline_init(RR_spline[z_ct], lnGamma_values, RR_table[z_ct], RR_lnGamma_NPTS);
      }
      return;
    }
  }

""" + a, 1)

    aux_anchor = 'double aux_function(double del, void *params){'
    assert s.count(aux_anchor) == 1
    s = s.replace(aux_anchor, r"""static double recombination_rate_ws(double z, double gamma12_bg, double T4, int usecaseB,
                                    double A, double C_0, double beta,
                                    gsl_integration_workspace *w, int *status_out) {
  double result = 0.0, error = 0.0, lower_limit, upper_limit;
  gsl_function F;
  double rel_tol = 0.01;
  RR_par p = {z, gamma12_bg, T4, A, C_0, beta, No*pow(1+z, 3), usecaseB};
  F.function = &MHR_rr;
  F.params = &p;
  lower_limit = log(0.01);
  upper_limit = log(200);
  int status = gsl_integration_qag(&F, lower_limit, upper_limit, 0, rel_tol,
                                   1000, GSL_INTEG_GAUSS61, w, &result, &error);
  if (status != 0) {
      LOG_ERROR("gsl integration error occured!");
      LOG_ERROR("(function argument): lower_limit=%e upper_limit=%e rel_tol=%e result=%e error=%e",lower_limit,upper_limit,rel_tol,result,error);
      LOG_ERROR("data: z=%e gamma12_bg=%e T4=%e A_MHR(z)=%e",z,gamma12_bg,T4,A);
      LOG_ERROR("data: C_MHR(z)=%e beta_MHR(z)=%e nH=%e usecaseB=%d\n",C_0,beta,No*pow(1+z,3),usecaseB);
  }
  *status_out = status;
  return result;
}

""" + aux_anchor, 1)

    a = '  init_A_MHR();'
    assert s.count(a) == 1
    s = s.replace(a, '  if (!gpu_mhr_table) init_A_MHR();', 1)
    a = '  free_A_MHR();'
    assert s.count(a) == 1
    s = s.replace(a, '  if (!gpu_mhr_table) free_A_MHR();', 1)
    a = '  free_beta_MHR();'
    assert s.count(a) == 1
    return s.replace(a, a + '\n  if (gpu_mhr_table) return;', 1)
