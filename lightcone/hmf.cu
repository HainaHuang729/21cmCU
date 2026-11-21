extern "C" __global__ void global_collapse(const double *p,const double *growth,
    const float *sigma,const float *deriv,const double *nodes,const double *weights,int ns,
    double *out) {
    const int row=blockIdx.x,lane=threadIdx.x;
    const double gf=growth[row],step=(p[1]-p[0])/128.;
    double sum=0.;
    for(int i=lane;i<1024;i+=blockDim.x) {
        const int piece=i/8,j=i%8;
        const double lm=p[0]+step*(piece+.5*(1.+nodes[j]));
        const double mass=exp(lm);
        const int bin=min(ns-2,max(0,(int)floor((lm-p[9])*p[11])));
        const float base=(float)p[9]+(float)p[10]*(float)bin;
        const double t=(lm-base)*p[11];
        const double s=(sigma[bin]+t*(sigma[bin+1]-sigma[bin]))*gf;
        const double derivative=-pow(10.,deriv[bin]+t*(deriv[bin+1]-deriv[bin]));
        const double v=sqrt(.73)*1.68/s;
        const double density=-mass*derivative*gf*gf/(2.*s*s)*sqrt(2./3.141592653589793)
             *.353*(1.+pow(v,-.175*2.))*v*exp(-.5*v*v);
        const double fstar=(p[3]>0.&&mass>p[7])||(p[3]<0.&&mass<p[7])?
             1./p[5]:pow(mass/1.e10,p[3]);
        const double fesc=(p[4]>0.&&mass>p[8])||(p[4]<0.&&mass<p[8])?
             1./p[6]:pow(mass/1.e10,p[4]);
        sum+=weights[j]*density*exp(-p[2]/mass)*fstar*fesc;
    }
    __shared__ double partial[128];
    partial[lane]=sum;
    __syncthreads();
    for(int k=blockDim.x/2;k;k/=2){
        if(lane<k)partial[lane]+=partial[lane+k];
        __syncthreads();
    }
    if(!lane)out[row]=partial[0]*.5*step;
}
