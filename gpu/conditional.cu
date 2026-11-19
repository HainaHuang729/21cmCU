__device__ double conditional_integrand(double logmass, float delta, const double *p,
        const float *sigma, const float *derivative, int nsigma) {
    const double mass=exp(logmass);
    const float lm=(float)logmass;
    const int bin=min(nsigma-2,max(0,(int)floorf((lm-(float)p[0])*(float)p[2])));
    const float base=(float)p[0]+(float)p[1]*(float)bin;
    const float offset=(lm-base)*(float)p[2];
    const float s=sigma[bin]+offset*(sigma[bin+1]-sigma[bin]);
    const float deriv=derivative[bin]+offset*(derivative[bin+1]-derivative[bin]);
    const float s2=(float)p[4];
    float variance=s*s-s2*s2;
    if (variance<0.f) return 0.;
    if (variance==0.f) variance=1.e-6f;
    const float gap=(float)p[5]-delta;
    const float growth=(float)p[3];
    const double conditional=(gap/growth)*pow(10.,(double)deriv)*
         exp(-(double)(gap*gap)/(2.*growth*growth*variance))/(variance*sqrt((double)variance));
    const double star=(p[9]>0. && mass>p[13])||(p[9]<0. && mass<p[13]) ?
        1./p[11] : pow(mass/1.e10,p[9]);
    const double escape=(p[10]>0. && mass>p[14])||(p[10]<0. && mass<p[14]) ?
        1./p[12] : pow(mass/1.e10,p[10]);
    return mass*exp(-p[8]/mass)*star*escape*conditional*0.39894228040143267794;
}

extern "C" __global__ void conditional_table(const double *p,const float *sigma,
    const float *derivative,const float *density,const double *x,const double *w,
    const double *gl_x,const double *gl_w,float *out,int nsigma,int nq,int high,int heat) {
    const int i=blockIdx.x, lane=threadIdx.x;
    const float d=density[i];
    if ((!high && d>p[5]*.9999) || (high && d>p[5])) {
        if (!lane) out[i]=heat ? (high?1.e10f:23.02585093f) : (high?1.f:0.f);
        return;
    }
    double sum=0.;
    if (high) {
        // Composite Gauss-Legendre with geometric subdivision near the high-mass endpoint.
        const int divisions=14, order=16;
        const double width=p[7]-p[6];
        for (int j=lane;j<(divisions+1)*order;j+=blockDim.x) {
            const int piece=j/order, k=j%order;
            const double hi=ldexp(1.,-piece),lo=piece==divisions?0.:hi*.5;
            const double u=.5*((hi-lo)*gl_x[k]+hi+lo);
            const double mass=p[7]-width*u;
            sum+=.5*width*(hi-lo)*gl_w[k]*conditional_integrand(mass,d,p,sigma,derivative,nsigma);
        }
    } else {
        const double half_width=.5*(p[7]-p[6]);
        const double midpoint=.5*(p[7]+p[6]);
        for (int j=lane;j<nq;j+=blockDim.x) {
            const float knot=(float)(midpoint+half_width*x[j]);
            const float weight=(float)(half_width*w[j]);
            sum+=(double)weight*conditional_integrand(knot,d,p,sigma,derivative,nsigma);
        }
    }
    __shared__ double values[128];
    values[lane]=sum;
    __syncthreads();
    for(int offset=blockDim.x/2;offset;offset/=2) {
        if (lane<offset) values[lane]+=values[lane+offset];
        __syncthreads();
    }
    if (lane==0) {
        const double v=values[0];
        if(high) out[i]=(float)(fmax(v,0.)*(heat?1.e10:1.));
        else out[i]=(float)(log(fmax(v,1.e-38))+(heat?log(1.e10):0.));
    }
}
