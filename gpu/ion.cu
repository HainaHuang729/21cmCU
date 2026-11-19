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
    if (d>=cut) {out[i]=1.f;return;}
    bool low=d<crit;
    float v=low?(log10f((float)(d+1.))-ls)*li:(d-hs)*hi_inv;
    int j=(int)floorf(v), size=low?nl:nh;
    if(j<0||j+1>=size) {
        atomicAdd(bad,1);
        v=fminf(fmaxf(v,0.f),(float)(size-1));
        j=min((int)floorf(v),size-2);
    }
    const float *t=low?lo:hi;
    float f=t[j]*(1.f+(float)j-v)+t[j+1]*(v-(float)j);
    out[i]=low?expf(f):f;
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
    if(coll*eff>(1.-e)*(1.+recycled)){
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
extern "C" __global__ void recombine(
    const float *density,const float *gamma12,const float *xhi,
    const float *old,float *out,const double *knots,const double *values,const double *deriv,
    int n,float z,float dt,float dz,double dz_table,double maxlog) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    float d=(float)(1.+density[i]);
    float ze=(float)pow((double)d,1./3.);
    ze*=1.f+z;
    double iz=(ze-1.)/dz_table+0.5;
    int row=max(0,min(299,(int)iz));
    double g=log((double)gamma12[i]);
    if(g < -10.) {out[i]=old[i];return;}
    g=fmin(g,maxlog);
    int j=max(0,min(248,(int)((g-knots[0])/(knots[1]-knots[0]))));
    if(g<knots[j])--j;
    if(g>knots[j+1])++j;
    int off=250*row+j;
    double w=knots[j+1]-knots[j], t=(g-knots[j])/w, t2=t*t,t3=t2*t;
    double rate=(2*t3-3*t2+1)*values[off]+(t3-2*t2+t)*w*deriv[off]
                +(-2*t3+3*t2)*values[off+1]+(t3-t2)*w*deriv[off+1];
    out[i]=(float)(old[i]+rate*dt*dz*(1.-xhi[i]));
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
    int n, float z, float tre, float density_limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !(zre[i] > 0.f && xhi[i] < 1.e-30)) return;
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
    output[i] = fmaxf((float)pow((double)result, 0.5882), neutral_temperature[i]);
}

extern "C" __global__ void bubble_filter_triplet(const float2 *s, float2 *o,
    int side, double dk, float radius, int filtered) {
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
    for (int field=0;field<3;++field) {
        const int j=i+field*stride;
        o[j]=(filtered && kr>1.f) ? make_float2(0.f,0.f) :
             make_float2((float)(s[j].x/n),(float)(s[j].y/n));
    }
}
