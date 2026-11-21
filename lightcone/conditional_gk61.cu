// Reproduce the native conditional-mass interpolation and its float intermediates.
__device__ double native_mass_integrand(double logm, float density, const double *p,
                                         const float *sigma, const float *slope, int ns) {
    const double m = exp(logm);
    const float lm = (float)log(m);
    const float inv = (float)p[2];
    const int k = (int)floorf((lm-(float)p[0])*inv);
    if (k < 0 || k + 1 >= ns) return nan("");
    const float base = (float)p[0] + (float)p[1]*(float)k;
    const float diff = lm-base;
    const float s = sigma[k] + diff*(sigma[k+1]-sigma[k])*inv;
    const float derivative = slope[k] + diff*(slope[k+1]-slope[k])*inv;
    const float s1 = s*s;
    const float s2 = (float)p[4]*(float)p[4];
    if (s1<s2) return 0.;
    const float dm = (float)(-pow(10.,(double)derivative));
    const float normalized_dm = (float)(dm/(2.*s1));
    const float gap = (float)p[5]-density;
    const float growth = (float)p[3];
    const double var = s1==s2 ? 1.e-6 : (double)(s1-s2);
    const float conditional = (float)(-(gap/growth)*(2.*s1*normalized_dm)*
         exp((double)(-gap*gap)/(2.*growth*growth*var))/pow(var,1.5));
    const double star=(p[9]>0.&&m>p[13])||(p[9]<0.&&m<p[13])?1./p[11]:pow(m/1.e10,p[9]);
    const double escape=(p[10]>0.&&m>p[14])||(p[10]<0.&&m<p[14])?1./p[12]:pow(m/1.e10,p[10]);
    return m*exp(-p[8]/m)*star*escape*conditional/sqrt(2.*3.14159265358979323846);
}

struct Panel {double lo,hi,area,error;};
__device__ Panel evaluate_panel(double lo,double hi,float density,const double* p,
                                 const float* sigma,const float* slope,int ns,const double* c) {
    const double mid=.5*(lo+hi), half=.5*(hi-lo);
    const double center=native_mass_integrand(mid,density,p,sigma,slope,ns);
    double left[30],right[30];
    double kron=center*c[61], gauss=0., absolute=fabs(kron);
    // The native rule sums the embedded Gauss abscissae first.
    for (int parity=1;parity>=0;--parity) {
        for (int i=parity;i<30;i+=2) {
            const double off=half*c[i];
            left[i]=native_mass_integrand(mid-off,density,p,sigma,slope,ns);
            right[i]=native_mass_integrand(mid+off,density,p,sigma,slope,ns);
            const double pair=left[i]+right[i];
            kron+=c[31+i]*pair;
            absolute+=c[31+i]*(fabs(left[i])+fabs(right[i]));
            if (parity) gauss+=c[62+i/2]*pair;
        }
    }
    const double avg=kron*.5;
    double asc=c[61]*fabs(center-avg);
    for(int i=0;i<30;++i) asc+=c[31+i]*(fabs(left[i]-avg)+fabs(right[i]-avg));
    asc*=half;
    absolute*=half;
    double error=fabs((kron-gauss)*half);
    if(asc!=0.&&error!=0.) error=asc*fmin(1.,pow(200.*error/asc,1.5));
    if(absolute>2.2250738585072014e-308/1.1102230246251565e-14) error=fmax(error,1.1102230246251565e-14*absolute);
    return {lo,hi,kron*half,error};
}

extern "C" __global__ void conditional_heat_qag(
    const double *p, const float *sigma, const float *slope,
    const float *density, const double *rule, float *out,
    Panel *workspace, int *panels, unsigned long long *stats,
    int nr, int ns, int heat) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nr)return;
    const float d=density[i];
    if((double)d>p[5]) {
        out[i] = heat ? 1.e10f : 1.f;
        if (panels) panels[i]=0;
        return;
    }
    constexpr int limit=1000;
    Panel *rows=workspace+i*limit;
    rows[0]=evaluate_panel(p[6],p[7],d,p,sigma,slope,ns,rule);
    double area=rows[0].area, error=rows[0].error;
    int count=1;
    while (error>0.01*fabs(area) && count<limit) {
        int k=0;
        for(int j=1;j<count;++j) if(rows[j].error>rows[k].error) k=j;
        const Panel prev=rows[k];
        const double mid=.5*(prev.lo+prev.hi);
        Panel left=evaluate_panel(prev.lo,mid,d,p,sigma,slope,ns,rule);
        Panel right=evaluate_panel(mid,prev.hi,d,p,sigma,slope,ns,rule);
        area+=left.area+right.area-prev.area;
        error+=left.error+right.error-prev.error;
        if(right.error>left.error) {Panel tmp=left;left=right;right=tmp;}
        rows[k]=left;
        rows[count++]=right;
    }
    const bool valid=isfinite(area)&&isfinite(error)&&error<=0.01*fabs(area);
    if (panels) panels[i]=valid?count:-count;
    if (stats) {
        atomicAdd(&stats[0], (unsigned long long)count);
        if (!valid) atomicAdd(&stats[1], 1ULL);
    }
    double sum=0.;
    for(int j=0;j<count;++j) sum+=rows[j].area;
    if (!valid) {
        out[i] = nanf("");
    } else if (heat) {
        out[i] = (float)((float)sum*1.e10);
    } else {
        float r = (float)sum;
        out[i] = (r < 0.f) ? 1.e-40f : r;
    }
}

__device__ Panel evaluate_panel_warp(double lo,double hi,float density,const double* p,
                                 const float* sigma,const float* slope,int ns,const double* c) {
    const double mid=.5*(lo+hi), half=.5*(hi-lo);
    const int lane=threadIdx.x;
    __shared__ double left[30],right[30],center;
    if(lane<30) {
        const double off=half*c[lane];
        left[lane]=native_mass_integrand(mid-off,density,p,sigma,slope,ns);
        right[lane]=native_mass_integrand(mid+off,density,p,sigma,slope,ns);
    } else if(lane==30) {
        center=native_mass_integrand(mid,density,p,sigma,slope,ns);
    }
    __syncwarp();
    if(lane!=0) return {0.,0.,0.,0.};
    double kron=center*c[61], gauss=0., absolute=fabs(kron);
    // Preserve the reference's summation order and stopping condition.
    for(int parity=1;parity>=0;--parity) {
        for(int i=parity;i<30;i+=2) {
            const double pair=left[i]+right[i];
            kron+=c[31+i]*pair;
            absolute+=c[31+i]*(fabs(left[i])+fabs(right[i]));
            if (parity) gauss+=c[62+i/2]*pair;
        }
    }
    const double avg=kron*.5;
    double asc=c[61]*fabs(center-avg);
    for(int i=0;i<30;++i) asc+=c[31+i]*(fabs(left[i]-avg)+fabs(right[i]-avg));
    asc*=half;
    absolute*=half;
    double error=fabs((kron-gauss)*half);
    if(asc!=0.&&error!=0.) error=asc*fmin(1.,pow(200.*error/asc,1.5));
    if(absolute>2.2250738585072014e-308/1.1102230246251565e-14) error=fmax(error,1.1102230246251565e-14*absolute);
    return {lo,hi,kron*half,error};
}

extern "C" __global__ void conditional_heat_qag_warp(
    const double *p, const float *sigma, const float *slope,
    const float *density, const double *rule, float *out,
    Panel *workspace, int *panels, unsigned long long *stats,
    int nr, int ns, int heat) {
    const int i=blockIdx.x, lane=threadIdx.x;
    if(i>=nr)return;
    const float d=density[i];
    if((double)d>p[5]) {
        if(lane==0) {
            out[i] = heat ? 1.e10f : 1.f;
            if (panels) panels[i]=0;
        }
        return;
    }
    constexpr int limit=1000;
    Panel *rows=workspace+i*limit;
    __shared__ double lo,mid,hi;
    __shared__ int active;
    Panel first=evaluate_panel_warp(p[6],p[7],d,p,sigma,slope,ns,rule);
    double area=first.area,error=first.error;
    int count=1,k=0;
    if(lane==0) rows[0]=first;
    while(true) {
        if(lane==0) {
            active=error>0.01*fabs(area) && count<limit;
            if(active) {
                k=0;
                for(int j=1;j<count;++j) if(rows[j].error>rows[k].error) k=j;
                lo=rows[k].lo;
                hi=rows[k].hi;
                mid=.5*(lo+hi);
            }
        }
        __syncwarp();
        if(!active) break;
        Panel left=evaluate_panel_warp(lo,mid,d,p,sigma,slope,ns,rule);
        __syncwarp();
        Panel right=evaluate_panel_warp(mid,hi,d,p,sigma,slope,ns,rule);
        if(lane==0) {
            const Panel prev=rows[k];
            area+=left.area+right.area-prev.area;
            error+=left.error+right.error-prev.error;
            if(right.error>left.error) {Panel tmp=left;left=right;right=tmp;}
            rows[k]=left;
            rows[count++]=right;
        }
        __syncwarp();
    }
    if(lane==0) {
        const bool valid=isfinite(area)&&isfinite(error)&&error<=0.01*fabs(area);
        if (panels) panels[i]=valid?count:-count;
        if (stats) {
            atomicAdd(&stats[0], (unsigned long long)count);
            if (!valid) atomicAdd(&stats[1], 1ULL);
        }
        double sum=0.;
        for(int j=0;j<count;++j) sum+=rows[j].area;
        if (!valid) {
            out[i] = nanf("");
        } else if (heat) {
            out[i] = (float)((float)sum*1.e10);
        } else {
            float r = (float)sum;
            out[i] = (r < 0.f) ? 1.e-40f : r;
        }
    }
}
