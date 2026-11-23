struct CollapseIntegrand {
    const double *p;
    const float *sigma, *deriv;
    int ns;
    double gf;
    __device__ double operator()(double logmass) const {
        const double mass=exp(logmass), lm=log(mass);
        const int bin=min(ns-2,max(0,(int)floor((lm-p[9])*p[11])));
        const float base=(float)p[9]+(float)p[10]*(float)bin;
        const double t=(lm-base)*p[11];
        const double s=(sigma[bin]+t*(sigma[bin+1]-sigma[bin]))*gf;
        const double derivative=-pow(10.,deriv[bin]+t*(deriv[bin+1]-deriv[bin]));
        const double v=sqrt(.73)*1.68/s;
        const double density=-mass*derivative*gf*gf/(2.*s*s)*sqrt(2./3.141592653589793)
             *.353*(1.+pow(v,-.175*2.))*v*pow(2.71828182846,-.5*v*v);
        const double fstar=(p[3]>0.&&mass>p[7])||(p[3]<0.&&mass<p[7])?
             1./p[5]:pow(mass/1.e10,p[3]);
        const double fesc=(p[4]>0.&&mass>p[8])||(p[4]<0.&&mass<p[8])?
             1./p[6]:pow(mass/1.e10,p[4]);
        return density*exp(-p[2]/mass)*fstar*fesc;
    }
};

extern "C" __global__ void global_collapse(const double *p,const double *growth,
    const float *sigma,const float *deriv,const double *rule,int ns,int count,double *out) {
    const int row=blockIdx.x;
    if(row>=count)return;
    double result=table_integral(p[0],p[1],.001,CollapseIntegrand{p,sigma,deriv,ns,growth[row]},rule);
    if(!threadIdx.x) out[row]=result;
}
