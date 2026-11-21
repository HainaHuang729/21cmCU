extern "C" __global__ void mhr_rates(const double *rows,const double *gammas,
    const double *nodes,const double *weights,int nz,int ng,int nq,double alpha,double helium,
    double *rates) {
    const int i=blockIdx.x, lane=threadIdx.x;
    if(i>=nz*ng)return;
    const double *p=rows+5*(i/ng);
    const double g=(float)exp(gammas[i%ng]);
    const double self_shield=26.7*pow((1.+p[0])/10.,-3.)*pow(g,2./3.);
    const double lo=log(.01),hi=log(200.);
    double sum=0.;
    for(int j=lane;j<nq;j+=blockDim.x) {
        const double del=exp(.5*(hi-lo)*nodes[j]+.5*(hi+lo));
        const double x=del/self_shield;
        const double gamma=g*(.98*pow(1.+pow(x,1.64),-2.28)+.02*pow(1.+x,-.84))*1.e-12;
        const double density=p[4]*del;
        const double y=gamma/(density*helium*alpha);
        const double chi=y>1.e5?1./y:2./(2.+y+sqrt(y*(y+4.)));
        const double frac=1.-chi;
        const double h=(pow(del,-2./3.)-p[2])/(2.*7.61/(3.*(1.+p[0])));
        const double pdf=p[1]*exp(-.5*h*h)*pow(del,p[3]);
        sum+=weights[j]*1.e15*density*pdf*alpha*frac*frac*del*del;
    }
    __shared__ double partial[64];
    partial[lane]=sum;
    __syncthreads();
    for(int step=blockDim.x/2;step;step/=2){
        if(lane<step)partial[lane]+=partial[lane+step];
        __syncthreads();
    }
    if(!lane)rates[i]=partial[0]*.5*(hi-lo);
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

extern "C" __global__ void mhr_pdf_normalization(const double *params,
    const double *nodes,const double *weights,double *norm) {
    const int row=blockIdx.x,lane=threadIdx.x;
    const double *p=params+3*row;
    const double spread=2.*7.61/(3.*(1.+p[0]));
    const int intervals=116, order=32;
    const double extent=25.*log(10.), width=2.*extent/intervals;
    double sum=0.;
    for (int j=lane;j<intervals*order;j+=blockDim.x) {
        const int interval=j/order,k=j%order;
        const double log_density=-extent+(interval+.5*(1.+nodes[k]))*width;
        const double diff=(exp(-2./3.*log_density)-p[1])/spread;
        sum+=weights[k]*exp(-.5*diff*diff+(1.+p[2])*log_density);
    }
    __shared__ double partial[128];
    partial[lane]=sum;
    __syncthreads();
    for (int step=blockDim.x/2;step;step/=2) {
        if (lane<step) partial[lane]+=partial[lane+step];
        __syncthreads();
    }
    if (!lane) norm[row]=1./(.5*width*partial[0]);
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
