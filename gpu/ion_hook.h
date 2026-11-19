typedef struct {
    int phase, side, threads, filter, filtered, last, no_rng, nl, nh;
    float z, prev_z, radius, lo, hi, ls, li, hs, hi_inv, crit, pixel, mmin, eff, tre, dt, dz, density_limit;
    double dk, cut, npoisson, norm, gamma, fmin, mean, dz_table, max_log;
    float *density, *xe, *tk, *old_rec, *old_z, *xhi, *nrec, *zre, *mfp, *gamma12, *temperature, *fcoll;
    const float *low, *high;
} IonGPU;
typedef int (*IonHook)(IonGPU *);
static IonHook ion_hook = NULL;
void SetIonGPUHook(void *hook) { ion_hook = (IonHook)hook; }

void IonPoissonDraws(int threads, double mean, unsigned long long n, int *out) {
    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
        gsl_rng_set(rng, tid);
        #pragma omp for schedule(static)
        for (unsigned long long i=0; i<n; ++i)
            out[i] = (int)gsl_ran_poisson(rng,mean);
        gsl_rng_free(rng);
    }
}

void IonRecombinationKnots(double *ln_gamma, double *values, double *derivatives) {
    memcpy(ln_gamma,lnGamma_values,sizeof(lnGamma_values));
    memcpy(values,RR_table,sizeof(RR_table));
    for (int i=0; i<RR_Z_NPTS; ++i)
        for (int j=0; j<RR_lnGamma_NPTS; ++j)
            derivatives[i*RR_lnGamma_NPTS+j] =
                gsl_spline_eval_deriv(RR_spline[i],lnGamma_values[j],RR_acc[i]);
}

static fftwf_complex *ion_ref[3], *ion_ref_work, *ion_ref_density;
static float *ion_ref_fcoll;
static int ion_ref_side;
static bool ion_ref_ready, ion_ref_mean_ready;
static unsigned long long ion_ref_inverses[3];

void IonReferenceReset(void) {
    ion_ref_ready=false;
    ion_ref_mean_ready=false;
}

void IonReferenceInverseCounts(unsigned long long *out) {
    memcpy(out,ion_ref_inverses,sizeof(ion_ref_inverses));
}

static int IonReferencePrepare(IonGPU *a) {
    struct UserParams *user_params=user_params_ufunc;
    int side=a->side, n=side*side*side;
    size_t bytes=sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS;
    if(ion_ref_side!=side) {
        for(int f=0;f<3;++f) {
            if(ion_ref[f])fftwf_free(ion_ref[f]);
            ion_ref[f]=fftwf_malloc(bytes);
        }
        if(ion_ref_work)fftwf_free(ion_ref_work);
        if(ion_ref_density)fftwf_free(ion_ref_density);
        if(ion_ref_fcoll)fftwf_free(ion_ref_fcoll);
        ion_ref_work=fftwf_malloc(bytes);
        ion_ref_density=fftwf_malloc(bytes);
        ion_ref_fcoll=fftwf_malloc((size_t)n*sizeof(*ion_ref_fcoll));
        ion_ref_side=side;
        IonReferenceReset();
    }
    if(!ion_ref_work||!ion_ref_density||!ion_ref_fcoll||
       !ion_ref[0]||!ion_ref[1]||!ion_ref[2])return 1;
    if(ion_ref_ready)return 0;
    const float *inputs[]={a->density,a->xe,a->old_rec};
    for(int f=0;f<3;++f) {
        float *data=(float*)ion_ref[f];
        #pragma omp parallel for num_threads(a->threads)
        for(int x=0;x<side;++x)
            for(int y=0;y<side;++y)for(int z=0;z<side;++z)
                data[HII_R_FFT_INDEX(x,y,z)]=inputs[f][HII_R_INDEX(x,y,z)];
        if(dft_r2c_cube(user_params->USE_FFTW_WISDOM,side,side,a->threads,ion_ref[f]))return 1;
        #pragma omp parallel for num_threads(a->threads)
        for(long i=0;i<HII_KSPACE_NUM_PIXELS;++i)ion_ref[f][i]/=(double)n;
    }
    ion_ref_ready=true;
    return 0;
}

static int IonReferenceInverse(IonGPU *a, int field, fftwf_complex *target) {
    struct UserParams *user_params=user_params_ufunc;
    memcpy(target,ion_ref[field],sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    if(a->filtered)filter_box(target,1,global_params.HII_FILTER,a->radius);
    if(dft_c2r_cube(user_params->USE_FFTW_WISDOM,a->side,a->side,a->threads,target))return 1;
    ++ion_ref_inverses[field];
    return 0;
}

int IonReferenceCells(IonGPU *a, const long *indices, long count,
                      float *out, float *fc, int evaluate) {
    struct UserParams *user_params=user_params_ufunc;
    int side=a->side;
    if(!ion_ref_ready || (evaluate && !ion_ref_mean_ready))return 1;
    for(long i=0;i<count;++i) {
        long k=indices[i];
        float v=((float*)ion_ref_density)[HII_R_FFT_INDEX(k/(side*side),(k/side)%side,k%side)];
        out[i]=fmaxf(v,-1.+FRACT_FLOAT_ERR);
        if(evaluate)fc[i]=ion_ref_fcoll[k];
    }
    if(evaluate)
        for(int f=1;f<3;++f) {
            if(IonReferenceInverse(a,f,ion_ref_work))return 1;
            for(long i=0;i<count;++i) {
                long k=indices[i];
                float v=((float*)ion_ref_work)[HII_R_FFT_INDEX(k/(side*side),(k/side)%side,k%side)];
                out[f*count+i]=f==1?fminf(fmaxf(v,0.),.999):fmaxf(v,0.);
            }
        }
    return 0;
}

int IonReferenceBounds(IonGPU *a) {
    ion_ref_mean_ready=false;
    if(IonReferencePrepare(a) || IonReferenceInverse(a,0,ion_ref_density))return 1;
    struct UserParams *user_params=user_params_ufunc;
    int side=a->side;
    float lo=0.,hi=0.;
    #pragma omp parallel for num_threads(a->threads) reduction(min:lo) reduction(max:hi)
    for(int x=0;x<side;++x)
        for(int y=0;y<side;++y)for(int z=0;z<side;++z) {
            float v=fmaxf(((float*)ion_ref_density)[HII_R_FFT_INDEX(x,y,z)],-1.+FRACT_FLOAT_ERR);
            lo=fminf(lo,v);hi=fmaxf(hi,v);
        }
    a->lo=lo;a->hi=hi;
    return 0;
}

int IonReferenceMean(IonGPU *a, double *out) {
    struct UserParams *user_params=user_params_ufunc;
    int side=a->side, bad=0;
    double result=0.;
    #pragma omp parallel for num_threads(a->threads) reduction(+:result,bad)
    for(int x=0;x<side;++x)
        for(int y=0;y<side;++y)for(int z=0;z<side;++z) {
            float v=fmaxf(((float*)ion_ref_density)[HII_R_FFT_INDEX(x,y,z)],-1.+FRACT_FLOAT_ERR);
            float fc=0,dummy=0;
            bad+=EvaluateSplineTable(false,1,v,0.,0.,&fc,&dummy);
            ion_ref_fcoll[HII_R_INDEX(x,y,z)]=fc;
            result+=(double)fc;
        }
    *out=result;
    ion_ref_mean_ready=(bad==0);
    return bad;
}

void IonReferenceRecombRows(const float *density, const long *ids, long count, float z, int *rows) {
    #pragma omp parallel for
    for(long i=0;i<count;++i) {
        float d=1.+density[ids[i]];
        float effective=pow(d,1./3.);
        effective*=1+z;
        int row=(int)((effective-1.)/RR_DEL_Z+0.5);
        rows[i]=row<0?0:row>=RR_Z_NPTS?RR_Z_NPTS-1:row;
    }
}
