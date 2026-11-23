struct RecombinationRate {
    const double *p;
    double g,self_shield,alpha,helium;
    __device__ double operator()(double log_density) const {
        const double del=exp(log_density);
        const double x=del/self_shield;
        const double gamma=g*(.98*pow(1.+pow(x,1.64),-2.28)+.02*pow(1.+x,-.84))*1.e-12;
        const double density=p[4]*del;
        const double y=gamma/(density*helium*alpha);
        const double approx=helium*density*alpha/gamma;
        const double b=-2.-y;
        const double chi=approx<1.e-30 ? 0. : (approx<1.e-5 ? approx : (-b-sqrt(b*b-4.))/2.);
        const double frac=1.-chi;
        const double h=(pow(del,-2./3.)-p[2])/(2.*7.61/(3.*(1.+p[0])));
        const double pdf=p[1]*exp(-.5*h*h)*pow(del,p[3]);
        return 1.e15*density*pdf*alpha*frac*frac*del*del;
    }
};

extern "C" __global__ void mhr_rates(const double *rows,const double *gammas,
    const double *rule,int nz,int ng,double alpha,double helium,double *rates) {
    const int i=blockIdx.x;
    if(i>=nz*ng)return;
    const double *p=rows+5*(i/ng);
    const double g=(float)exp(gammas[i%ng]);
    const double shield=26.7*pow((1.+p[0])/10.,-3.)*pow(g,2./3.);
    double value=table_integral(log(.01),log(200.),.01,RecombinationRate{p,g,shield,alpha,helium},rule);
    if (!threadIdx.x) rates[i]=value;
}

extern "C" __global__ void natural_derivatives(const double *x,const double *values,
    int nz,int n,double *deriv) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=nz||n>256)return;
    double upper[256], second[256];
    const double *y=values+(long)row*n;
    double *out=deriv+(long)row*n;
    upper[0]=second[0]=0.;
    for(int j=1;j<n-1;++j) {
        const double a=x[j]-x[j-1], b=x[j+1]-x[j];
        const double diag=2.*(a+b)-a*upper[j-1];
        const double rhs=6.*((y[j+1]-y[j])/b-(y[j]-y[j-1])/a);
        upper[j]=b/diag;
        second[j]=(rhs-a*second[j-1])/diag;
    }
    second[n-1]=0.;
    for(int j=n-2;j>0;--j)second[j]-=upper[j]*second[j+1];
    for(int j=0;j<n-1;++j){
        const double h=x[j+1]-x[j];
        out[j]=(y[j+1]-y[j])/h-h*(2.*second[j]+second[j+1])/6.;
    }
    const double h=x[n-1]-x[n-2];
    out[n-1]=(y[n-1]-y[n-2])/h+h*(second[n-2]+2.*second[n-1])/6.;
}

struct DensityPDF {
    const double *p;
    __device__ double operator()(double density) const {
        const double spread=2.*7.61/(3.*(1.+p[0]));
        const double diff=pow(density,-2./3.)-p[1];
        return exp(-diff*diff/(2.*spread*spread))*pow(density,p[2]);
    }
};

extern "C" __global__ void mhr_pdf_normalization(const double *params,const double *rule,
    int count,double *norm) {
    const int row=blockIdx.x;
    if(row>=count)return;
    double value=table_integral(1.e-25,1.e25,.001,DensityPDF{params+3*row},rule);
    if(!threadIdx.x) norm[row]=1./value;
}

extern "C" __global__ void mhr_pdf_interpolate(double *rows,const double *z,
    const double *norm,const double *slopes,int nz,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=nz) return;
    double *row=rows+5*i;
    const double query=fmin(fmax(row[0],z[0]),z[n-1]);
    const int j=min(n-2,max(0,(int)floor(query-z[0])));
    const double t=query-z[j],u=1.-t;
    row[1]=u*u*((1.+2.*t)*norm[j]+t*slopes[j])+
           t*t*((3.-2.*t)*norm[j+1]-u*slopes[j+1]);
}
