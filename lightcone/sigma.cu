__device__ double raw_power(double k,const double *p) {
    const double root_alpha=sqrt(p[2]);
    const double q=k*p[0]*p[0]/p[1];
    const double qeff=q/(root_alpha+(1.-root_alpha)/(1.+pow(.43*k*p[3],4.)));
    double transfer=log(2.71828182846+1.84*p[4]*root_alpha*qeff);
    transfer /= transfer + qeff*qeff*(14.4+325./(1.+60.5*pow(qeff,1.11)));
    const double qnu=3.92*q/sqrt(p[5]/p[6]);
    transfer *= 1. + (1.2*pow(p[5],.64)*pow(p[6],.3+.6*p[5]))/
                    (pow(qnu,-1.6)+pow(qnu,.8));
    const double pk=pow(k,p[8]);
    return pk*transfer*transfer;

}

struct SigmaIntegrand {
    const double *p;
    double radius, mass;
    bool derivative;
    __device__ double operator()(double k) const {
        const double x=k*radius;
        const double w=x<1.e-4 ? 1. : 3.*(sin(x)/pow(x,3)-cos(x)/pow(x,2));
        const double power=k*k*raw_power(k,p);
        if (!derivative) return power*w*w;
        const double dw=x<1.e-10 ? 0. : 9.*cos(x)*k/pow(x,3)+3.*sin(x)*(1.-3./(x*x))/(x*radius);
        return power*2.*w*dw*(radius/(3.*mass));
    }
};

extern "C" __global__ void sigma_moments(const double *p,const double *radii,const double *masses,
    const double *rule,int count,double *result,double *derivative) {
    const int i=blockIdx.x;
    if(i>=count) return;
    const double r=radii[i],mass=masses[i],lo=1.e-99/r,hi=350./r;
    double value=table_integral(lo,hi,1.e-6,SigmaIntegrand{p,r,mass,false},rule);
    if(!threadIdx.x) result[i]=value;
    value=mass>0. ? table_integral(lo,hi,1.e-6,SigmaIntegrand{p,r,mass,true},rule) : 0.;
    if(!threadIdx.x) derivative[i]=value;
}
