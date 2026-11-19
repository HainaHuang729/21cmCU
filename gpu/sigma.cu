__device__ double raw_power(double k,const double *p) {
    const double root_alpha=sqrt(p[2]);
    const double q=k*p[0]*p[0]/p[1];
    const double qeff=q/(root_alpha+(1.-root_alpha)/(1.+pow(.43*k*p[3],4.)));
    double transfer=log(2.718281828459045+1.84*p[4]*root_alpha*qeff);
    transfer /= transfer + qeff*qeff*(14.4+325./(1.+60.5*pow(qeff,1.11)));
    const double qnu=3.92*q/sqrt(p[5]/p[6]);
    transfer *= 1. + (1.2*pow(p[5],.64)*pow(p[6],.3+.6*p[5]))/
                    (pow(qnu,-1.6)+pow(qnu,.8));
    const double pk=(k<=p[9]?pow(k,p[8]):pow(k,p[10])*pow(p[9],p[8]-p[10]));
    return pk*transfer*transfer;

}

extern "C" __global__ void sigma_moments(const double *p,const double *radii,const double *masses,
    const double *qx,const double *qw,int count,double *result,double *derivative) {
    const int i=blockIdx.x, thread=threadIdx.x;
    if(i>=count) return;
    const double r=radii[i], mass=masses[i];
    double var=0., deriv=0.;
    for(int j=thread;j<2240;j+=blockDim.x) {
        const int piece=j/16, n=j%16;
        double lo,hi;
        if(piece<16) {
            lo=piece?ldexp(1.,piece-16):0.;
            hi=ldexp(1.,piece-15);
        } else {
            lo=1.+(piece-16)*349./124.;
            hi=1.+(piece-15)*349./124.;
        }
        const double x=.5*((hi-lo)*qx[n]+hi+lo);
        const double k=x/r;
        double w,dwdx;
        if (x<.025) {
            const double x2=x*x;
            w=1.-x2/10.+x2*x2/280.;
            dwdx=-x/5.+x*x2/70.;
        } else {
            w=3.*(sin(x)-x*cos(x))/(x*x*x);
            dwdx=3.*sin(x)/(x*x)-3.*w/x;
        }
        const double common=.5*(hi-lo)*qw[n]*k*k*raw_power(k,p);
        var+=common*w*w;
        if(mass>0.) deriv+=common*2.*w*dwdx*x/(3.*mass);
    }
    __shared__ double v[128],d[128];
    v[thread]=var;d[thread]=deriv;
    __syncthreads();
    for(int step=blockDim.x/2;step;step/=2) {
        if(thread<step){v[thread]+=v[thread+step];d[thread]+=d[thread+step];}
        __syncthreads();
    }
    if(!thread){result[i]=v[0]/r;derivative[i]=d[0]/r;}
}
