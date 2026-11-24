extern "C" __global__ void bubble_filter(const float2 *s, float2 *o,
    int side, double dk, float radius, int filtered) {
    int i=blockIdx.x*blockDim.x+threadIdx.x, h=side/2+1;
    if(i>=side*side*h) return;
    int x=i/(side*h), y=(i/h)%side, z=i%h;
    float kx=(float)((x>side/2?x-side:x)*dk);
    float ky=(float)((y>side/2?y-side:y)*dk);
    float kz=(float)(z*dk);
    float kr=sqrtf(kx*kx+ky*ky+kz*kz)*radius;
    kr=(float)(kr*0.413566994);
    double n=(double)side*side*side;
    o[i]=(filtered && kr>1.f) ? make_float2(0.f,0.f) :
         make_float2((float)(s[i].x/n),(float)(s[i].y/n));
}
extern "C" __global__ void collapse(
    const float *density, const float *lo, const float *hi,
    float *out, int n, int nl, int nh, float ls,float li,float hs,float hi_inv,
    float crit,double cut,int *bad) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    float d=density[i];
    if (d < crit) {
        if (d <= -1.f) {out[i]=0.f;return;}
        float v = (glibc_log10f((float)(d + 1.0)) - ls) * li;
        int j = (int)floorf(v);
        if (j < 0 || j + 1 > nl - 1) {
            atomicAdd(bad, 1);
            v = fminf(fmaxf(v, 0.f), (float)(nl - 1));
            j = min((int)floorf(v), nl - 2);
        }
        float f = lo[j] * (1.f + (float)j - v) + lo[j + 1] * (v - (float)j);
        out[i] = glibc_expf(f);
    } else {
        if ((double)d < cut) {
            float v = (d - hs) * hi_inv;
            int j = (int)floorf(v);
            if (j < 0 || j + 1 > nh - 1) {
                atomicAdd(bad, 1);
                v = fminf(fmaxf(v, 0.f), (float)(nh - 1));
                j = min((int)floorf(v), nh - 2);
            }
            out[i] = hi[j] * (1.f + (float)j - v) + hi[j + 1] * (v - (float)j);
        } else {
            out[i] = 1.f;
        }
    }
}
// The final neutral fraction may be close to zero after the stochastic
// assignment even when the unadjusted collapse estimate was far from it.
__device__ double partial_collapse(double coll, float mass, float mmin, int draw,
                                  float pixel, float density, double npoisson) {
    coll = fmin(coll,1.);
    if ((float)(mass/mmin)<npoisson)
        coll = (draw*(mass/(float)npoisson))/(pixel*(1.+density));
    if (mass<mmin/5.) coll=0.;
    return fmin(coll,1.);
}

__device__ __forceinline__ bool exceeds_ionization_barrier(double photons, double xe,
                                                           double rec_per_baryon) {
    return photons > (1. - xe) * (1. + rec_per_baryon);
}

extern "C" __global__ void assign_ion(
    const float *density,const float *xe,const float *rec,const float *fc,const float *old_z,
    const float *tk,const int *draw,float *xhi,float *zre,float *mfp,float *gamma12,float *temp,
    int n,int last,float radius,float z,float pixel,float mmin,float eff,float tre,
    double mean_st,double gpref,double fmin_coll,double npoisson,const double *sum_coll) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    float d=density[i], e=xe[i];
    const double norm=mean_st/fmax(*sum_coll/(double)n,fmin_coll);
    double coll=norm*(double)fc[i];
    float mass=last?(float)(coll*(double)pixel*(1.+d)):0.f;
    if(coll<=fmin_coll)coll=fmin_coll;
    double recycled=(double)rec[i]/(1.+d);
    if(exceeds_ionization_barrier(coll*eff, e, recycled)){
        if(xhi[i]>1.e-7){gamma12[i]=(float)(gpref*coll);mfp[i]=radius;}
        zre[i]=old_z[i]<0.f?z:old_z[i];
        xhi[i]=0.f;
    } else if(last&&xhi[i]>1.e-30) {
        coll=partial_collapse(coll,mass,mmin,draw[i],pixel,d,npoisson);
        float neutral=(float)(1.-coll*eff);
        temp[i]=neutral<=0.f?tre:neutral>=1.f?tk[i]:(float)(tk[i]*neutral+tre*(1.-neutral));
        xhi[i]=fminf(fmaxf(neutral-e,0.f),1.f);
    }
}
// Match the scalar precision of the caller's native ionization model.
#ifdef RECOMB_V432
typedef double recomb_scalar;
#else
typedef float recomb_scalar;
#endif
extern "C" __global__ void recombine(
    const float *density,const float *gamma12,const float *xhi,
    const float *old,float *out,const double *knots,const double *values,const double *deriv,
    int n,recomb_scalar z,recomb_scalar dt,recomb_scalar dz,double dz_table,double maxlog,int *nonfinite) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    double z_eff=z;
#ifdef RECOMB_V432
    // A single homogeneous update uses the stored redshift without density scaling.
    if (density)
#endif
    {
        recomb_scalar d=(recomb_scalar)(1.+density[i]);
        recomb_scalar ze=(recomb_scalar)pow((double)d,1./3.);
        ze*=1.f+z;
        z_eff=ze-1.;
    }
    double iz=z_eff/dz_table+0.5;
    int row=max(0,min(299,(int)iz));
    double g=log((double)gamma12[i]);
    float res;
    if(g < -10.) {
        res = old[i];
    } else {
#ifdef RECOMB_V432
        if (g >= knots[249]) g=maxlog;
#else
        g=fmin(g,maxlog);
#endif
        int j=max(0,min(248,(int)((g-knots[0])/(knots[1]-knots[0]))));
        if(g<knots[j])--j;
        if(g>knots[j+1])++j;
        int off=250*row+j;
        double w=knots[j+1]-knots[j], t=(g-knots[j])/w, t2=t*t,t3=t2*t;
        double rate=(2*t3-3*t2+1)*values[off]+(t3-2*t2+t)*w*deriv[off]
                    +(-2*t3+3*t2)*values[off+1]+(t3-t2)*w*deriv[off+1];
        res = (float)(old[i]+rate*dt*dz*(1.-xhi[i]));
    }
    out[i] = res;
    if (!isfinite(res)) atomicCAS(nonfinite, 0, 1);
}

extern "C" __global__ void rec_boundary(const float *density,const float *gamma,
    unsigned char *out,int n,float z,double step) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    float ze=(float)pow((double)(float)(1.+density[i]),1./3.);
    ze*=1.f+z;
    double iz=(ze-1.)/step+0.5;
    out[i]=gamma[i]>0.f && fabs(iz-nearbyint(iz))<1.e-4;
}

extern "C" __global__ void spline_candidates(const float *density,unsigned char *flags,
    int n,float crit,double cut) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) flags[i]=(fabs((double)density[i]-cut)<4.e-6 || fabs((double)density[i]-crit)<4.e-6);
}
extern "C" __global__ void ion_candidates(
    const float *d,const float *e,const float *r,const float *fc,const float *xhi,
    const int *draw,unsigned char *flags,int n,int last,float pixel,float mmin,float eff,
    double norm,double floor_coll,double npoisson) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    double f=norm*fc[i], right=(1.-e[i])*(1.+r[i]/(1.+d[i]));
    float mass=(float)(f*pixel*(1.+d[i]));
    f=fmax(f,floor_coll);
    bool close=fabs(f*eff-right)<2.e-3*fmax(1.,right);
    bool partial=last && xhi[i]>1.e-30 &&
        (fabs((double)mass/mmin-0.2)<2.e-4 || fabs((double)mass/mmin-npoisson)<2.e-4 ||
         ((f*eff+e[i]>.80 ||
           partial_collapse(f,mass,mmin,draw[i],pixel,d[i],npoisson)*eff+e[i]>.80) &&
          f*eff<=right+2.e-4));
    flags[i]=close || partial || (xhi[i]>1.e-7 && f*eff>right);
}

extern "C" __global__ void ion_temperature(const float *density, const float *xhi,
    const float *zre, const float *neutral_temperature, float *output,
    int n, float z, float tre, float density_limit, int *nonfinite) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!(zre[i] > 0.f && xhi[i] < 1.e-30)) {
        if (!isfinite(output[i])) atomicCAS(nonfinite, 0, 2);
        return;
    }
    float delta = density[i], result = 1.f;
    const double scale = (1. + z) / (1. + zre[i]);
    if (fabsf(z - zre[i]) >= 1.e-4) {
        float delta_re = (float)(delta * (1. + z) / (1. + zre[i]));
        if (delta_re <= -1.f) delta_re = -1. + density_limit;
        if (delta <= -1.f) delta = -1. + density_limit;
        result = pow((1. + delta) / (1. + delta_re), 1.1333);
        result *= pow(scale, 3.4);
        result *= expf(pow((1. + z)/7.1, 2.5) - pow((1. + zre[i])/7.1, 2.5));
    }
    result *= pow((double)tre, 1.7);
    result += pow(1.e4 * ((1. + z)/4.), 1.7) * (1.f + delta);
    const float val = fmaxf((float)pow((double)result, 0.5882), neutral_temperature[i]);
    output[i] = val;
    if (!isfinite(val)) atomicCAS(nonfinite, 0, 2);
}

__device__ __forceinline__ void bubble_filter_fields(const float2 *s, float2 *o,
    int side, double dk, float radius, int filtered, int nfields) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x, h=side/2+1;
    const int stride=side*side*h;
    if(i>=stride) return;
    const int x=i/(side*h), y=(i/h)%side, z=i%h;
    const float kx=(float)((x>side/2?x-side:x)*dk);
    const float ky=(float)((y>side/2?y-side:y)*dk);
    const float kz=(float)(z*dk);
    float kr=sqrtf(kx*kx+ky*ky+kz*kz)*radius;
    kr=(float)(kr*0.413566994);
    const double n=(double)side*side*side;
    for (int field=0;field<nfields;++field) {
        const int j=i+field*stride;
        o[j]=(filtered && kr>1.f) ? make_float2(0.f,0.f) :
             make_float2((float)(s[j].x/n),(float)(s[j].y/n));
    }
}

extern "C" __global__ void bubble_filter_triplet(const float2 *s, float2 *o,
    int side, double dk, float radius, int filtered) {
    bubble_filter_fields(s, o, side, dk, radius, filtered, 3);
}

extern "C" __global__ void bubble_filter_pair(const float2 *s, float2 *o,
    int side, double dk, float radius, int filtered) {
    bubble_filter_fields(s, o, side, dk, radius, filtered, 2);
}

extern "C" __global__ void min_ion_margin_pass1(
    const float *delta, const float *xe, const float *rec, const float *fc, const float *xhi,
    const double *sum_coll, double *partials, int n,
    double mean_st, double floor_coll, float eff) {
    __shared__ double shared[256];
    const double norm = mean_st / fmax(*sum_coll / (double)n, floor_coll);
    const float norm_f = (float)norm;
    const float floor_f = (float)floor_coll;
    double best = 1.e100;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if (xhi[i] <= 1.e-7f) continue;
        const float d = delta[i];
        const float e = xe[i];
        const float r = rec[i];
        const float f = fc[i];
        const float right_f = (1.f - e) * (1.f + r / (1.f + d));
        const float left_f = fmaxf(norm_f * f, floor_f) * eff;
        const float dist_f = fabsf(left_f - right_f) / fmaxf(1.f, right_f);
        if (dist_f < 1.e-2f) {
            const double right = (1. - (double)e) * (1. + (double)r / (1. + (double)d));
            const double left = fmax(norm * (double)f, floor_coll) * (double)eff;
            const double dist = fabs(left - right) / fmax(1., right);
            best = fmin(best, dist);
        } else {
            best = fmin(best, (double)dist_f);
        }
    }
    shared[threadIdx.x] = best;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] = fmin(shared[threadIdx.x], shared[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = shared[0];
    }
}

extern "C" __global__ void min_ion_margin_pass2(const double *partials, double *out, int nblocks) {
    __shared__ double shared[256];
    double best = 1.e100;
    for (int i = threadIdx.x; i < nblocks; i += blockDim.x) {
        best = fmin(best, partials[i]);
    }
    shared[threadIdx.x] = best;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] = fmin(shared[threadIdx.x], shared[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[0] = shared[0];
    }
}

extern "C" __global__ void collapse_with_sum(
    const float *density, const float *lo, const float *hi,
    float *out, double *partials, long n, int nl, int nh, float ls, float li, float hs, float hi_inv,
    float crit, double cut, int *bad) {
    double acc = 0.;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        float d = density[i];
        float val;
        if (d < crit) {
            if (d <= -1.f) {
                val = 0.f;
            } else {
                float v = (glibc_log10f((float)(d + 1.0)) - ls) * li;
                int j = (int)floorf(v);
                if (j < 0 || j + 1 > nl - 1) {
                    atomicAdd(bad, 1);
                    v = fminf(fmaxf(v, 0.f), (float)(nl - 1));
                    j = min((int)floorf(v), nl - 2);
                }
                float f = lo[j] * (1.f + (float)j - v) + lo[j + 1] * (v - (float)j);
                val = glibc_expf(f);
            }
        } else if ((double)d < cut) {
            float v = (d - hs) * hi_inv;
            int j = (int)floorf(v);
            if (j < 0 || j + 1 > nh - 1) {
                atomicAdd(bad, 1);
                v = fminf(fmaxf(v, 0.f), (float)(nh - 1));
                j = min((int)floorf(v), nh - 2);
            }
            val = hi[j] * (1.f + (float)j - v) + hi[j + 1] * (v - (float)j);
        } else {
            val = 1.f;
        }
        out[i] = val;
        acc += (double)val;
    }
    __shared__ double shared[256];
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) shared[threadIdx.x] += shared[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = shared[0];
}

extern "C" __global__ void sum_double_pass2(const double *partials, double *out, int nblocks) {
    __shared__ double shared[256];
    double total = 0.;
    for (int i = threadIdx.x; i < nblocks; i += blockDim.x) total += partials[i];
    shared[threadIdx.x] = total;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) shared[threadIdx.x] += shared[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = shared[0];
}

extern "C" __global__ void ion_init_arrays(
    float * __restrict__ xhi, float * __restrict__ zre,
    float * __restrict__ gamma, float * __restrict__ mfp, float * __restrict__ temp, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    xhi[i] = 1.f;
    zre[i] = -1.f;
    gamma[i] = 0.f;
    mfp[i] = 0.f;
    temp[i] = 0.f;
}

extern "C" __global__ void screen_ion_scale_pass1(
    const float * __restrict__ delta, const float * __restrict__ xe,
    const float * __restrict__ rec, const float * __restrict__ fc,
    const float * __restrict__ xhi, const double * __restrict__ sum_coll,
    int *out_flag, int n, double mean_st, double floor_coll, float eff,
    double cut, double ratio_thresh) {
    if (*out_flag) return;
    const double norm = mean_st / fmax(*sum_coll / (double)n, floor_coll);
    const float cut_lo = (float)(cut - 1.e-3);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if ((i & 511) == 0 && *out_flag) return;
        if (xhi[i] <= 1.e-7f) continue;
        const float df = delta[i];
        if (df >= cut_lo) {
            atomicCAS(out_flag, 0, 1);
            return;
        }
        const double d = (double)df;
        const double right = (1. - (double)xe[i]) * (1. + (double)rec[i] / (1. + d));
        const double left = fmax(norm * (double)fc[i], floor_coll) * (double)eff;
        if (left >= ratio_thresh * right) {
            atomicCAS(out_flag, 0, 1);
            return;
        }
    }
}

extern "C" __global__ void min_ion_margin_amp_pass1(
    const float * __restrict__ delta, const float * __restrict__ xe,
    const float * __restrict__ rec, const float * __restrict__ fc,
    const float * __restrict__ xhi, const double * __restrict__ sum_coll,
    double *partials, int n, double mean_st, double floor_coll, float eff,
    double xe_scale, double rec_scale) {
    __shared__ double shared[256];
    const double norm = mean_st / fmax(*sum_coll / (double)n, floor_coll);
    double best = 1.e100;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if (xhi[i] <= 1.e-7f) continue;
        const double d = (double)delta[i];
        const double e = (double)xe[i];
        const double r = (double)rec[i];
        const double f = (double)fc[i];
        const double inv_opd = 1. / (1. + d);
        const double right = (1. - e) * (1. + r * inv_opd);
        const double left = fmax(norm * f, floor_coll) * (double)eff;
        const double err_right = xe_scale * (1. + r * inv_opd) + (1. - e) * rec_scale * inv_opd;
        const double ratio = (err_right > 0.) ? fabs(left - right) / err_right : 1.e100;
        best = fmin(best, ratio);
    }
    shared[threadIdx.x] = best;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] = fmin(shared[threadIdx.x], shared[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = shared[0];
    }
}

extern "C" __global__ void ion_clamp_fields(float *fields, int n, int has_delta, int has_xe, int has_rec) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t offset = 0;
    if (has_delta) {
        fields[offset + (size_t)i] = fmaxf(fields[offset + (size_t)i], -1.f + 1.e-7f);
        offset += (size_t)n;
    }
    if (has_xe) {
        fields[offset + (size_t)i] = fminf(fmaxf(fields[offset + (size_t)i], 0.f), 0.999f);
        offset += (size_t)n;
    }
    if (has_rec) {
        fields[offset + (size_t)i] = fmaxf(fields[offset + (size_t)i], 0.f);
    }
}
