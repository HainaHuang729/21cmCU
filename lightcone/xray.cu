__device__ double hydrogenic(double E,double threshold,double Z) {
    if(E<threshold)return 0.;
    const double e=sqrt(E/threshold-1.);
    const double ratio=e>1.e-8?atan(e)/e:1.;
    const double h=threshold/E;
    return (6.3e-18/(Z*Z))*h*h*h*h*exp(4.-4.*ratio)/
           (e>1.e-8?-expm1(-6.283185307179586/e):1.);
}

__device__ double helium(double E,double threshold) {
    if(E<threshold)return 0.;
    const double x=E/13.61-.4434, y=sqrt(x*x+2.136*2.136);
    return 9.492e-16*((x-1.)*(x-1.)+2.039*2.039)*
           pow(y,.5*3.188-5.5)*pow(1.+sqrt(y/1.469),-3.188);
}

__device__ double cross_sum(double E,double xe,const double *p) {
    return p[12]*(1.-xe)*hydrogenic(E,p[8],1.)+
           p[13]*(1.-xe)*helium(E,p[9])+p[13]*xe*hydrogenic(E,p[10],2.);
}

extern "C" __global__ void find_tau_energy(const double *p,const float *radii_z,
    const double *fcoll,int ncoll,const double *node,const double *weight,int n,int nq,double *out) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const double width=(double)radii_z[i]-p[0];
    if (width<=0. || p[1]>.9999) {out[i]=p[14];return;}
    double z[24],w[24];
    for(int j=0;j<nq;++j) {
        z[j]=1.+p[0]+width*.5*(node[j]+1.);
        const double t=(z[j]-1.-p[5])/p[6];
        const int m=min(ncoll-2,max(0,(int)floor(t)));
        const double f=fcoll[m]+(t-m)*(fcoll[m+1]-fcoll[m]);
        const double neutral=fmax(1.e-4,1.-p[7]*f/(1.-p[1]));
        w[j]=.5*width*weight[j]*p[2]*z[j]*z[j]*neutral/sqrt(p[3]*z[j]*z[j]*z[j]+p[4]);
    }
    double lo=log(p[9]), hi=log(1.e6);
    for(int k=0;k<22;++k) {
        const double mid=.5*(lo+hi), E=exp(mid)/(1.+p[0]);
        double tau=0.;
        for(int j=0;j<nq;++j) tau+=w[j]*cross_sum(E*z[j],p[1],p);
        if(tau>1.)lo=mid; else hi=mid;
    }
    out[i]=fmax(p[14],exp(.5*(hi+lo)));
}

__device__ double secondary(const float *data,const float *energy,const float *ion,
    int ni,int ne,int row,double E,int field) {
    if(E<energy[0])return field==0?1.:0.;
    E=fmin(E,.999*energy[ne-1]);
    int a=0,b=ne-1;
    while(b-a>1){int mid=(a+b)/2;if(E<energy[mid])b=mid;else a=mid;}
    double x=fmin((double)ion[row],.999*ion[ni-1]);
    const int m=min(row,ni-2);
    const double t=(x-ion[m])/(ion[m+1]-ion[m]),u=(E-energy[a])/(energy[b]-energy[a]);
    const float *lo=data+(field*ni+m)*ne,*hi=lo+ne;
    const double v=lo[a]+u*(lo[b]-lo[a]),w=hi[a]+u*(hi[b]-hi[a]);
    return v+t*(w-v);
}

extern "C" __global__ void xray_rate(const double *p,const double *grid,const float *energy,
    const float *ion,const float *data,int n,int ne,int ni,double *out) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=ni*n)return;
    const int row=i/n;
    const double E=grid[i%n],xe=(double)ion[row];
    const double spectrum=pow(E/p[14],-p[16]-1.)*p[17];
    const double cross[3]={p[12]*(1.-xe)*hydrogenic(E,p[8],1.),
                          p[13]*(1.-xe)*helium(E,p[9]),p[13]*xe*hydrogenic(E,p[10],2.)};
    double heat=0., ionization=0., lya=0.;
    for(int j=0;j<3;++j) {
        if(cross[j]==0.)continue;
        const double e=E-p[8+j];
        heat+=cross[j]*secondary(data,energy,ion,ni,ne,row,e,0)*p[11]*e;
        lya+=cross[j]*secondary(data,energy,ion,ni,ne,row,e,1);
        double secondary_ion=1.;
        for(int k=2;k<5;++k)secondary_ion+=secondary(data,energy,ion,ni,ne,row,e,k);
        ionization+=cross[j]*secondary_ion;
    }
    out[i]=spectrum*heat;
    out[i+ni*n]=spectrum*ionization;
    out[i+2*ni*n]=spectrum*lya;
}
