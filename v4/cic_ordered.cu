// The CPU adds contributions in particle order within each OpenMP slab.
// Revisit cells whose double-precision summation error could change the
// rounded float grid. Both passes use the same position and weight arithmetic.
struct CICGrid {
    const float *rho, *vel, *second;
    int n, m;
    double growth, first, correction;
    bool two;
    __device__ void particle(int x,int y,int z,int *base,double *fraction) const {
        const int coords[]={x,y,z};
        const double ratio=(double)m/n;
        long long source=0;
        for(int a=0;a<3;++a)
            source=source*m+((int)(coords[a]*ratio+.5))%m;
        for(int a=0;a<3;++a) {
            long long j=source+(long long)a*m*m*m;
            double displacement=vel[j]*first;
            double p=coords[a]+displacement;
            if(two) {
                double shift=second[j]*correction;
                p=a==0 ? p-shift : coords[a]+(displacement-shift);
            }
            p*=ratio;
            int ix=(int)floor(p);
            base[a]=(ix%m+m)%m;
            fraction[a]=p-ix;
        }
    }
    __device__ long long corner(int bits,const int *base,const double *frac,double &w) const {
        long long cell=0;
        w=1.;
        for(int a=0;a<3;++a) {
            int bit=(bits>>a)&1;
            int ix=base[a]+bit;
            cell=cell*m+(ix==m?0:ix);
            w*=bit?frac[a]:1.-frac[a];
        }
        return cell;
    }
};

extern "C" __global__ void scatter(const float *rho,const float *vel,const float *second,
    double *mass,int *counts,int n,int m,double growth,double first,double correction,bool two) {
    long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=(long long)n*n*n)return;
    CICGrid grid={rho,vel,second,n,m,growth,first,correction,two};
    int base[3];double frac[3];
    grid.particle(t/n/n,t/n%n,t%n,base,frac);
    for(int b=0;b<8;++b) {
        double w;
        long long dst=grid.corner(b,base,frac,w);
        atomicAdd(mass+dst,w*(1.+rho[t]*growth));
        atomicAdd(counts+dst,1);
    }
}

extern "C" __global__ void uncertain(const double *mass,const int *counts,
    const double *max_mass,bool *flags,long long total) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=total)return;
    double s=mass[i],r=(float)s;
    const float infinity=__int_as_float(0x7f800000);
    double lo=nextafterf((float)r,-infinity),hi=nextafterf((float)r,infinity);
    double distance=fmin(s-(r+(lo-r)*.5), (r+(hi-r)*.5)-s);
    // Both orders have error bounded by gamma(k)*sum(abs(contributions)).
    double k=counts[i],ku=k*1.1102230246251565e-16;
    flags[i]=k>1 && distance<=2.*ku/(1.-ku)*k*(*max_mass);
}

// Return the wrapped interval in increasing order of original source index.
struct PeriodicRange {
    int start,count,wrap,n;
    __device__ PeriodicRange(int cell,int n_,int m,double shift):n(n_) {
        int lo=(int)floor((cell-1.)*n/m-shift)-1;
        int hi=(int)ceil((cell+1.)*n/m+shift)+1;
        count=min(n,hi-lo+1);
        start=count==n?0:(lo%n+n)%n;
        wrap=max(0,start+count-n);
    }
    __device__ int at(int i) const {return i<wrap ? i : start+i-wrap;}
};

extern "C" __global__ void replay(const float *rho,const float *vel,const float *second,
    double *mass,const long long *indices,int num,const double *shift,
    int n,int m,double growth,double first,double correction,bool two) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=num)return;
    long long cell=indices[i];
    int target[]={int(cell/m/m),int(cell/m%m),int(cell%m)};
    PeriodicRange xs(target[0],n,m,*shift),ys(target[1],n,m,*shift),zs(target[2],n,m,*shift);
    CICGrid grid={rho,vel,second,n,m,growth,first,correction,two};
    double sum=0.;
    for(int a=0;a<xs.count;++a)for(int b=0;b<ys.count;++b)for(int c=0;c<zs.count;++c) {
        int x=xs.at(a),y=ys.at(b),z=zs.at(c),base[3],bits=0;
        double frac[3];grid.particle(x,y,z,base,frac);
        bool hit=true;
        for(int axis=0;axis<3;++axis) {
            int d=(target[axis]-base[axis]+m)%m;
            if(d>1) {hit=false;break;}
            bits|=d<<axis;
        }
        if(hit){
            long long t=((long long)x*n+y)*n+z;
            double w;
            if(m==1) {
                for(int corner=0;corner<8;++corner){grid.corner(corner,base,frac,w);sum+=w*(1.+rho[t]*growth);}
            } else {grid.corner(bits,base,frac,w);sum+=w*(1.+rho[t]*growth);}
        }
    }
    mass[cell]=sum;
}
