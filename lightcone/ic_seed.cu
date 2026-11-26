extern "C" __global__ void scale_initial_modes(float2 *modes, int n, double dk,
    double volume, const double *p) {
    const long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    const int h=n/2+1;
    if (i >= (long)n*n*h) return;
    const int x=i/(n*h), y=(i/h)%n, z=i%h;
    const float kx=(float)((x>n/2?x-n:x)*dk);
    const float ky=(float)((y>n/2?y-n:y)*dk);
    const float kz=(float)(z*dk);
    const double k=sqrtf(kx*kx+ky*ky+kz*kz);
    if (i==0) { modes[i]=make_float2(0,0); return; }

    // Eisenstein-Hu transfer function, matching POWER_SPECTRUM=0.
    const double root_alpha=sqrt(p[2]);
    const double q=k*p[0]*p[0]/p[1];
    const double qeff=q/(root_alpha+(1.-root_alpha)/(1.+pow(.43*k*p[3],4.)));
    double transfer=log(2.718281828459045+1.84*p[4]*root_alpha*qeff);
    transfer /= transfer + qeff*qeff*(14.4+325./(1.+60.5*pow(qeff,1.11)));
    const double qnu=3.92*q/sqrt(p[5]/p[6]);
    transfer *= 1. + (1.2*pow(p[5],.64)*pow(p[6],.3+.6*p[5]))/
                    (pow(qnu,-1.6)+pow(qnu,.8));
    const double pk=pow(k,p[8]);
    const float power=(float)(pk*transfer*transfer*p[7]);
    const double amplitude=sqrt(volume*power*0.5);
    const float2 value=modes[i];
    modes[i]=make_float2((float)(value.x*amplitude),(float)(value.y*amplitude));
}

extern "C" __global__ void enforce_real_planes(float2 *modes, int n) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=2*n*n)return;
    const int h=n/2+1, q=index/2, z=(index%2)?n/2:0;
    const int x=q/n, y=q%n;
    const int target=q*h+z;
    if (x>0 && x<n/2) {
        const float2 value=modes[((n-x)*n+(n-y)%n)*h+z];
        modes[target]=make_float2(value.x,-value.y);
    } else if (x==0 || x==n/2) {
        if (y>0 && y<n/2) {
            const float2 value=modes[(x*n+n-y)*h+z];
            modes[target]=make_float2(value.x,-value.y);
        } else if (y==0 || y==n/2) modes[target].y=0.f;
    }
}

extern "C" __global__ void enforce_real_planes_general(float2 *modes, int nx, int ny, int nz) {
    const long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    const long long plane_sz=(long long)nx*ny;
    if (index >= 2*plane_sz) return;
    const int mx=nx/2, my=ny/2, mz=nz/2, hz=nz/2+1;
    const int plane_k=(int)(index%2);
    if (plane_k && mz==0) return;
    const int k=plane_k ? mz : 0;
    const long long q=index/2;
    const int i=(int)(q/ny), j=(int)(q%ny);
    const long long pos=((long long)i*ny+j)*hz+k;
    if (i==0 && j==0 && k==0) {
        modes[0]=make_float2(0.f,0.f);
    } else if ((i==0 || i==mx) && (j==0 || j==my)) {
        modes[pos].y=0.f;
    } else if (i>0 && i<mx) {
        if (j==0 || j==my) {
            const float2 v=modes[((long long)(nx-i)*ny+j)*hz+k];
            modes[pos]=make_float2(v.x,-v.y);
        } else if (j<my || j>ny-my) {
            const int sy=ny-j;
            const float2 v=modes[((long long)(nx-i)*ny+sy)*hz+k];
            modes[pos]=make_float2(v.x,-v.y);
        }
    } else if ((i==0 || i==mx) && j>0 && j<my) {
        const float2 v=modes[((long long)i*ny+(ny-j))*hz+k];
        modes[pos]=make_float2(v.x,-v.y);
    }
}
